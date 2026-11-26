/*
 * CAT-lite partial-volume estimation.
 *
 * A port of ../brainchop-test/cat-lite.js, which is itself a browser-side
 * approximation of CAT's AMAP/PVE idea. It jointly considers the model's GM,
 * WM and CSF priors and the subject's normalized T1, estimates a coarse local
 * bias field, and evaluates pure and mixed tissue classes. It is NOT CAT12.
 *
 * The constants are upstream's and are compiled in rather than exposed: they
 * were tuned against the 24-channel model, and a CLI knob would invite tuning
 * them per subject, which is how a reproducible pipeline stops being one.
 */

#include "cat_lite.h"

#if BC_HAS_PVE

#include "bc_threads.h"
#include "bwlabel.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_EPSILON 1e-8

/* Upstream's model-entry values; see brainchop-parameters.js for the entry. */
#define CL_MIN_SUPPORT 0.03
#define CL_PRIOR_POWER 3.0
#define CL_MIN_SEPARATION 0.035
#define CL_SIGMA_FLOOR 0.025
#define CL_BIAS_BLOCK 16
#define CL_BIAS_PASSES 3
#define CL_PRIOR_STRENGTH 0.8
#define CL_INTENSITY_STRENGTH 0.75
#define CL_MIXEL_PRIOR 0.35
#define CL_CSF_WM_MIXEL_PRIOR 6.0
#define CL_SPATIAL_WEIGHT 0.25
#define CL_STAT_PURITY 0.55
#define CL_PARTIAL_VOLUME_SIGMA BC_PVE_BLUR_SIGMA

static double cl_clamp(double v, double low, double high) {
	return v < low ? low : (v > high ? high : v);
}

/*
 * Separable Gaussian, matching the WGSL in the page's probability runner.
 *
 * The denominator deliberately accumulates weights for samples that fall
 * OUTSIDE the volume, because the shader adds to weight_sum before its bounds
 * test and only the numerator is skipped. That darkens a two-voxel border.
 * Reproducing it is the point: moving the bounds test first is more defensible
 * and does not match the reference.
 */
typedef struct {
	const float *source;
	float *destination;
	const double *weight;
	int radius, axis;
	const size_t *dim;
} cl_blur_slab;

static void cl_blur_rows(void *context, int z_low, int z_high) {
	const cl_blur_slab *slab = (const cl_blur_slab *)context;
	const size_t nx = slab->dim[0], ny = slab->dim[1], nz = slab->dim[2];
	const size_t plane = nx * ny;
	for (int z = z_low; z <= z_high; z++)
		for (size_t y = 0; y < ny; y++)
			for (size_t x = 0; x < nx; x++) {
				double weighted = 0.0, total = 0.0;
				for (int offset = -slab->radius; offset <= slab->radius; offset++) {
					double w = slab->weight[offset + slab->radius];
					total += w;
					int64_t sx = (int64_t)x, sy = (int64_t)y, sz = z;
					if (slab->axis == 0) sx += offset;
					else if (slab->axis == 1) sy += offset;
					else sz += offset;
					if (sx < 0 || sy < 0 || sz < 0 || sx >= (int64_t)nx ||
							sy >= (int64_t)ny || sz >= (int64_t)nz)
						continue;
					weighted += w * slab->source[(size_t)sx + (size_t)sy * nx +
						(size_t)sz * plane];
				}
				slab->destination[x + y * nx + (size_t)z * plane] =
					(float)(weighted / total);
			}
}

static int cl_blur(float *volume, float *scratch, const size_t dim[3]) {
	const double sigma = CL_PARTIAL_VOLUME_SIGMA;
	int radius = (int)(3.0 * sigma + 0.5);
	if (radius < 1) radius = 1;
	double *weight = (double *)malloc((size_t)(2 * radius + 1) * sizeof(*weight));
	if (!weight) return 1;
	for (int i = -radius; i <= radius; i++)
		weight[i + radius] = exp(-0.5 * (double)i * i / (sigma * sigma));
	float *source = volume, *destination = scratch;
	for (int axis = 0; axis < 3; axis++) {
		cl_blur_slab slab = { .source = source, .destination = destination,
			.weight = weight, .radius = radius, .axis = axis, .dim = dim };
		bc_parallel_for(0, (int)dim[2] - 1, 0, cl_blur_rows, &slab);
		float *swap = source; source = destination; destination = swap;
	}
	if (source != volume) memcpy(volume, source, dim[0] * dim[1] * dim[2] * sizeof(*volume));
	free(weight);
	return 0;
}

/* Tissue means and sigmas, weighted by each tissue's share of the support. */
typedef struct { int applied, refusal; double mean[3], sigma[3]; } cl_stats;

/* BRAINCHOPC_PVE_DEBUG: the fitted tissue model, for comparing against the
   reference's own printout when the two implementations disagree. */
static void cl_debug_stats(const char *label, const cl_stats *stats) {
	if (!getenv("BRAINCHOPC_PVE_DEBUG")) return;
	fprintf(stderr, "[c] %s means csf=%.17g gray=%.17g white=%.17g\n"
		"[c] %s sigmas csf=%.17g gray=%.17g white=%.17g\n",
		label, stats->mean[0], stats->mean[1], stats->mean[2],
		label, stats->sigma[0], stats->sigma[1], stats->sigma[2]);
}

/* Supported voxels retain raster order, so cached reductions have exactly the
   same addition order. The 192 MiB cap includes indices, corrected values and
   optional weights; dense support recomputes weights instead. */
typedef struct {
	size_t count;
	uint32_t *indices;
	double *values, *weights;
	const float *gm, *wm, *csf;
} cl_fit_data;

static void cl_fit_free(cl_fit_data *fit) {
	free(fit->indices); free(fit->values); free(fit->weights);
	memset(fit, 0, sizeof(*fit));
}

static void cl_weights(const cl_fit_data *fit, size_t i, double w[3]) {
	double g = fit->gm[i] > 0 ? fit->gm[i] : 0;
	double v = fit->wm[i] > 0 ? fit->wm[i] : 0;
	double c = fit->csf[i] > 0 ? fit->csf[i] : 0;
	double raw = g + v + c, support = raw < 1.0 ? raw : 1.0;
	double pc = c / raw, pg = g / raw, pv = v / raw;
	/* Exclude 50/50 mixed priors (myelinated subcorticals {7,10,17} and PVE
	   borders) from global tissue mean/sigma estimation so they cannot skew
	   cortical GM / WM statistics. */
	w[0] = pc > CL_STAT_PURITY ? support * pow(pc, CL_PRIOR_POWER) : 0.0;
	w[1] = pg > CL_STAT_PURITY ? support * pow(pg, CL_PRIOR_POWER) : 0.0;
	w[2] = pv > CL_STAT_PURITY ? support * pow(pv, CL_PRIOR_POWER) : 0.0;
}

/* Positive comparisons preserve the reference's NaN-to-zero behavior here.
   Unlike fmaxf under GCC with -fno-finite-math-only, they do not call libm
   three times per voxel in each full-volume support scan. */
static int cl_fit_make(cl_fit_data *fit, const float *gm, const float *wm,
		const float *csf, const float *intensity, size_t count) {
	*fit = (cl_fit_data){.gm=gm, .wm=wm, .csf=csf};
	if (count > UINT32_MAX) return 1;
	for (size_t i=0; i<count; i++) {
		double raw = (double)(gm[i]>0?gm[i]:0) + (wm[i]>0?wm[i]:0) + (csf[i]>0?csf[i]:0);
		if (raw >= CL_MIN_SUPPORT && isfinite(intensity[i])) fit->count++;
	}
	size_t n = fit->count ? fit->count : 1;
	fit->indices = malloc(n * sizeof(*fit->indices));
	fit->values = malloc(n * sizeof(*fit->values));
	if (n <= (192u * 1024u * 1024u) / 36u)
		fit->weights = malloc(n * 3 * sizeof(*fit->weights));
	if (!fit->indices || !fit->values) { cl_fit_free(fit); return 1; }
	size_t k=0;
	for (size_t i=0; i<count; i++) {
		double raw = (double)(gm[i]>0?gm[i]:0) + (wm[i]>0?wm[i]:0) + (csf[i]>0?csf[i]:0);
		if (!(raw >= CL_MIN_SUPPORT) || !isfinite(intensity[i])) continue;
		fit->indices[k] = (uint32_t)i;
		fit->values[k] = intensity[i];
		if (fit->weights) cl_weights(fit, i, fit->weights + 3*k);
		k++;
	}
	return 0;
}

static cl_stats cl_estimate(const cl_fit_data *fit, int means_only) {
	cl_stats stats; memset(&stats, 0, sizeof(stats));
	double sums[3] = {0}, weights[3] = {0};
	for (size_t k=0; k<fit->count; k++) {
		double local[3];
		const double *w = fit->weights ? fit->weights+3*k : local;
		if (!fit->weights) cl_weights(fit, fit->indices[k], local);
		for (int t=0; t<3; t++) {
			sums[t] += w[t] * fit->values[k]; weights[t] += w[t];
		}
	}
	for (int t=0; t<3; t++) {
		if (weights[t] < 100.0) { stats.refusal=1; return stats; }
		stats.mean[t] = sums[t] / weights[t];
	}
	if (!(stats.mean[0] + CL_MIN_SEPARATION < stats.mean[1] &&
			stats.mean[1] + CL_MIN_SEPARATION < stats.mean[2])) { stats.refusal=2; return stats; }
	stats.applied = 1;
	if (means_only) return stats;
	double squares[3] = {0};
	for (size_t k=0; k<fit->count; k++) {
		double local[3];
		const double *w = fit->weights ? fit->weights+3*k : local;
		if (!fit->weights) cl_weights(fit, fit->indices[k], local);
		for (int t=0; t<3; t++) {
			double d = fit->values[k] - stats.mean[t];
			squares[t] += w[t] * d * d;
		}
	}
	for (int t=0; t<3; t++) {
		double variance = squares[t] / weights[t];
		stats.sigma[t] = sqrt(fmax(CL_SIGMA_FLOOR * CL_SIGMA_FLOOR, variance));
	}
	return stats;
}

/*
 * Coarse multiplicative-free bias estimate on a block grid, smoothed, then
 * trilinearly interpolated back. Nearest-cell lookup would step every block
 * and those steps become contours once the likelihoods are evaluated.
 */
static int cl_bias_values(const float *gm, const float *wm, const float *csf,
		const float *intensity, const double mean[3], const size_t dim[3], cl_fit_data *fit) {
	const size_t nx = dim[0], ny = dim[1], nz = dim[2], plane = nx * ny;
	const size_t gx = (nx + CL_BIAS_BLOCK - 1) / CL_BIAS_BLOCK;
	const size_t gy = (ny + CL_BIAS_BLOCK - 1) / CL_BIAS_BLOCK;
	const size_t gz = (nz + CL_BIAS_BLOCK - 1) / CL_BIAS_BLOCK;
	const size_t cells = gx * gy * gz;
	double *sums = (double *)calloc(cells, sizeof(*sums));
	double *weights = (double *)calloc(cells, sizeof(*weights));
	float *values = (float *)calloc(cells, sizeof(*values));
	float *next = (float *)calloc(cells, sizeof(*next));
	unsigned char *valid = (unsigned char *)calloc(cells, 1);
	unsigned char *next_valid = (unsigned char *)calloc(cells, 1);
	if (!sums || !weights || !values || !next || !valid || !next_valid) {
		free(sums); free(weights); free(values); free(next);
		free(valid); free(next_valid);
		return 1;
	}

	for (size_t z = 0; z < nz; z++)
		for (size_t y = 0; y < ny; y++)
			for (size_t x = 0; x < nx; x++) {
				size_t i = x + y * nx + z * plane;
				double g = gm[i] > 0 ? gm[i] : 0, w = wm[i] > 0 ? wm[i] : 0;
				double c = csf[i] > 0 ? csf[i] : 0;
				double raw = g + w + c, support = raw < 1.0 ? raw : 1.0;
				if (!(support >= CL_MIN_SUPPORT) || !isfinite(intensity[i])) continue;
				double pg = g / raw, pw = w / raw, pc = c / raw;
				double expected = pc * mean[0] + pg * mean[1] + pw * mean[2];
				double confidence = support * (pc * pc + pg * pg + pw * pw);
				size_t cell = (z / CL_BIAS_BLOCK * gy + y / CL_BIAS_BLOCK) * gx +
					x / CL_BIAS_BLOCK;
				sums[cell] += confidence * (intensity[i] - expected);
				weights[cell] += confidence;
			}
	for (size_t i = 0; i < cells; i++)
		if (weights[i] > 1.0) { values[i] = (float)(sums[i] / weights[i]); valid[i] = 1; }

	for (int pass = 0; pass < CL_BIAS_PASSES; pass++) {
		memset(next, 0, cells * sizeof(*next));
		memset(next_valid, 0, cells);
		for (size_t z = 0; z < gz; z++)
			for (size_t y = 0; y < gy; y++)
				for (size_t x = 0; x < gx; x++) {
					double sum = 0.0; int n = 0;
					for (int dz = -1; dz <= 1; dz++)
					for (int dy = -1; dy <= 1; dy++)
					for (int dx = -1; dx <= 1; dx++) {
						int64_t zz = (int64_t)z + dz, yy = (int64_t)y + dy;
						int64_t xx = (int64_t)x + dx;
						if (zz < 0 || yy < 0 || xx < 0 || zz >= (int64_t)gz ||
								yy >= (int64_t)gy || xx >= (int64_t)gx) continue;
						size_t nb = ((size_t)zz * gy + (size_t)yy) * gx + (size_t)xx;
						if (!valid[nb]) continue;
						sum += values[nb]; n++;
					}
					size_t cell = (z * gy + y) * gx + x;
					if (n > 0) { next[cell] = (float)(sum / n); next_valid[cell] = 1; }
				}
		memcpy(values, next, cells * sizeof(*values));
		memcpy(valid, next_valid, cells);
	}

	/* Per-axis trilinear tables, exactly as the reference builds them. */
	size_t max_axis = nx > ny ? (nx > nz ? nx : nz) : (ny > nz ? ny : nz);
	int64_t *low = (int64_t *)malloc(3 * max_axis * sizeof(*low));
	int64_t *high = (int64_t *)malloc(3 * max_axis * sizeof(*high));
	double *frac = (double *)malloc(3 * max_axis * sizeof(*frac));
	if (!low || !high || !frac) {
		free(sums); free(weights); free(values); free(next); free(valid);
		free(next_valid); free(low); free(high); free(frac);
		return 1;
	}
	const size_t extent[3] = { nx, ny, nz }, grid[3] = { gx, gy, gz };
	for (int axis = 0; axis < 3; axis++)
		for (size_t v = 0; v < extent[axis]; v++) {
			double g = cl_clamp(((double)v + 0.5) / CL_BIAS_BLOCK - 0.5, 0.0,
				(double)grid[axis] - 1.0);
			int64_t base = (int64_t)floor(g);
			size_t slot = (size_t)axis * max_axis + v;
			low[slot] = base;
			high[slot] = base + 1 < (int64_t)grid[axis] ? base + 1 : (int64_t)grid[axis] - 1;
			frac[slot] = g - (double)base;
		}

	for (size_t k=0; k<fit->count; k++) {
		size_t i=fit->indices[k], z=i/plane, y=(i%plane)/nx, x=i%nx;
		size_t zs=2*max_axis+z, ys=max_axis+y;
		size_t r00=((size_t)low[zs]*gy+(size_t)low[ys])*gx;
		size_t r01=((size_t)low[zs]*gy+(size_t)high[ys])*gx;
		size_t r10=((size_t)high[zs]*gy+(size_t)low[ys])*gx;
		size_t r11=((size_t)high[zs]*gy+(size_t)high[ys])*gx;
		size_t x0=(size_t)low[x], x1=(size_t)high[x];
		double tx=frac[x], ty=frac[ys], tz=frac[zs];
		double a=values[r00+x0]+(values[r00+x1]-values[r00+x0])*tx;
		double b=values[r01+x0]+(values[r01+x1]-values[r01+x0])*tx;
		double c=values[r10+x0]+(values[r10+x1]-values[r10+x0])*tx;
		double d=values[r11+x0]+(values[r11+x1]-values[r11+x0])*tx;
		double z0=a+(b-a)*ty, z1=c+(d-c)*ty;
		/* Keep the C reference's float32 bias boundary, then subtract in
		   double once for both statistics and the likelihood evaluation. */
		float bias=(float)(z0+(z1-z0)*tz);
		fit->values[k]=(double)intensity[i]-bias;
	}

	free(sums); free(weights); free(values); free(next); free(valid);
	free(next_valid); free(low); free(high); free(frac);
	return 0;
}

static void cl_neighbour_prior(const float *gm, const float *wm, const float *csf,
		size_t j, double *sums) {
	double g = gm[j] > 0 ? gm[j] : 0, w = wm[j] > 0 ? wm[j] : 0;
	double c = csf[j] > 0 ? csf[j] : 0;
	double raw = g + w + c;
	if (!(raw > CL_EPSILON)) return;
	sums[0] += g / raw; sums[1] += w / raw; sums[2] += c / raw; sums[3] += 1.0;
}

static double cl_log_likelihood(double value, double mean, double sigma, double log_sigma) {
	double z = (value - mean) / sigma;
	return -0.5 * z * z - log_sigma;
}

typedef struct {
	const float *gm, *wm, *csf;
	const cl_fit_data *fit;
	float *out_gm, *out_wm, *out_csf;
	const cl_stats *stats;
	const size_t *dim;
} cl_fit_slab;

static void cl_fit_rows(void *context, int k_low, int k_high) {
	const cl_fit_slab *s = (const cl_fit_slab *)context;
	const float *gm = s->gm, *wm = s->wm, *csf = s->csf;
	const size_t nx = s->dim[0], ny = s->dim[1], nz = s->dim[2], plane = nx * ny;
	const double mu_csf = s->stats->mean[0], mu_gm = s->stats->mean[1];
	const double mu_wm = s->stats->mean[2];
	const double sd_csf = s->stats->sigma[0], sd_gm = s->stats->sigma[1];
	const double sd_wm = s->stats->sigma[2];
	const double mix_gc_mean = 0.5 * (mu_csf + mu_gm);
	const double mix_gw_mean = 0.5 * (mu_gm + mu_wm);
	const double mix_cw_mean = 0.5 * (mu_csf + mu_wm);
	const double mix_gc_sigma = sqrt((mu_gm - mu_csf) * (mu_gm - mu_csf) / 12.0 +
		0.5 * (sd_csf * sd_csf + sd_gm * sd_gm));
	const double mix_gw_sigma = sqrt((mu_wm - mu_gm) * (mu_wm - mu_gm) / 12.0 +
		0.5 * (sd_gm * sd_gm + sd_wm * sd_wm));
	const double mix_cw_sigma = sqrt((mu_wm - mu_csf) * (mu_wm - mu_csf) / 12.0 +
		0.5 * (sd_csf * sd_csf + sd_wm * sd_wm));

	const double log_sd_csf=log(sd_csf), log_sd_gm=log(sd_gm), log_sd_wm=log(sd_wm);
	const double log_gc=log(mix_gc_sigma), log_gw=log(mix_gw_sigma), log_cw=log(mix_cw_sigma);
	for (int k=k_low; k<=k_high; k++) {
		size_t i=s->fit->indices[k], z=i/plane, y=(i%plane)/nx, x=i%nx;
		double g=gm[i]>0?gm[i]:0, w=wm[i]>0?wm[i]:0, c=csf[i]>0?csf[i]:0;
		double raw=g+w+c, support=raw<1.0?raw:1.0;

		double pg = g / raw, pw = w / raw, pc = c / raw;
		double sums[4] = {0, 0, 0, 0};
		if (x > 0) cl_neighbour_prior(gm, wm, csf, i - 1, sums);
		if (x + 1 < nx) cl_neighbour_prior(gm, wm, csf, i + 1, sums);
		if (y > 0) cl_neighbour_prior(gm, wm, csf, i - nx, sums);
		if (y + 1 < ny) cl_neighbour_prior(gm, wm, csf, i + nx, sums);
		if (z > 0) cl_neighbour_prior(gm, wm, csf, i - plane, sums);
		if ((size_t)z + 1 < nz) cl_neighbour_prior(gm, wm, csf, i + plane, sums);
		if (sums[3] != 0.0) {
			double keep = 1.0 - CL_SPATIAL_WEIGHT;
			pg = keep * pg + CL_SPATIAL_WEIGHT * sums[0] / sums[3];
			pw = keep * pw + CL_SPATIAL_WEIGHT * sums[1] / sums[3];
			pc = keep * pc + CL_SPATIAL_WEIGHT * sums[2] / sums[3];
		}
		double value = s->fit->values[k];

		double l_csf = CL_PRIOR_STRENGTH * log(pc + CL_EPSILON) +
			CL_INTENSITY_STRENGTH * cl_log_likelihood(value, mu_csf, sd_csf, log_sd_csf);
		double l_gm = CL_PRIOR_STRENGTH * log(pg + CL_EPSILON) +
			CL_INTENSITY_STRENGTH * cl_log_likelihood(value, mu_gm, sd_gm, log_sd_gm);
		double l_wm = CL_PRIOR_STRENGTH * log(pw + CL_EPSILON) +
			CL_INTENSITY_STRENGTH * cl_log_likelihood(value, mu_wm, sd_wm, log_sd_wm);
		double l_gc = CL_PRIOR_STRENGTH *
			log(CL_MIXEL_PRIOR * 2.0 * sqrt(pc * pg) + CL_EPSILON) +
			CL_INTENSITY_STRENGTH *
			cl_log_likelihood(value, mix_gc_mean, mix_gc_sigma, log_gc);
		double l_gw = CL_PRIOR_STRENGTH *
			log(CL_MIXEL_PRIOR * 2.0 * sqrt(pg * pw) + CL_EPSILON) +
			CL_INTENSITY_STRENGTH *
			cl_log_likelihood(value, mix_gw_mean, mix_gw_sigma, log_gw);
		/*
		 * CAT's five-class model has no CSF-WM mixel, but the two meet
		 * at the ventricles. Without this zero-GM nuisance class the
		 * uncertainty there can only be explained by the GM-containing
		 * mixels and shows up as a spurious gray rim.
		 */
		double l_cw = CL_PRIOR_STRENGTH *
			log(CL_CSF_WM_MIXEL_PRIOR * 2.0 * sqrt(pc * pw) *
				(1.0 - pg) * (1.0 - pg) + CL_EPSILON) +
			CL_INTENSITY_STRENGTH *
			cl_log_likelihood(value, mix_cw_mean, mix_cw_sigma, log_cw);

		double maximum = l_csf;
		if (l_gm > maximum) maximum = l_gm;
		if (l_wm > maximum) maximum = l_wm;
		if (l_gc > maximum) maximum = l_gc;
		if (l_gw > maximum) maximum = l_gw;
		if (l_cw > maximum) maximum = l_cw;
		double pure_csf = exp(l_csf - maximum), pure_gm = exp(l_gm - maximum);
		double pure_wm = exp(l_wm - maximum), mix_gc = exp(l_gc - maximum);
		double mix_gw = exp(l_gw - maximum), mix_cw = exp(l_cw - maximum);
		double total = pure_csf + pure_gm + pure_wm + mix_gc + mix_gw + mix_cw;

		double gm_in_gc = cl_clamp((value - mu_csf) / (mu_gm - mu_csf), 0, 1);
		double gm_in_gw = cl_clamp((mu_wm - value) / (mu_wm - mu_gm), 0, 1);
		double wm_in_cw = cl_clamp((value - mu_csf) / (mu_wm - mu_csf), 0, 1);
		double f_gm = (pure_gm + mix_gc * gm_in_gc + mix_gw * gm_in_gw) / total;
		double f_wm = (pure_wm + mix_gw * (1.0 - gm_in_gw) +
			mix_cw * wm_in_cw) / total;
		double f_csf = (pure_csf + mix_gc * (1.0 - gm_in_gc) +
			mix_cw * (1.0 - wm_in_cw)) / total;

		/* Where CSF and WM both dominate, their direct contact must not
		   leave a one-voxel GM contour -- but not where GM is at least
		   as plausible, which is the normal cortical configuration. */
		double minimum = pc < pw ? pc : pw;
		double dominance = cl_clamp((minimum - pg) / (minimum + CL_EPSILON), 0, 1);
		double contact = cl_clamp(4.0 * pc * pw, 0, 1);
		double gate = (1.0 - contact * dominance) * (1.0 - contact * dominance);
		s->out_gm[i] = (float)cl_clamp(f_gm * support * gate, 0, 1);
		/* GM the gate removed goes to WM/CSF so the three still sum to support. */
		double rest = f_wm + f_csf;
		double rescale = (1.0 - f_gm * gate) / (rest > CL_EPSILON ? rest : CL_EPSILON);
		s->out_wm[i] = (float)cl_clamp(f_wm * rescale * support, 0, 1);
		s->out_csf[i] = (float)cl_clamp(f_csf * rescale * support, 0, 1);
	}
}

int bc_cat_lite_priors(float *planes, const size_t dim[3]) {
	const size_t count = dim[0] * dim[1] * dim[2];
	float *gm = planes + (size_t)BC_PVE_GM * count;
	float *wm = planes + (size_t)BC_PVE_WM * count;
	float *csf = planes + (size_t)BC_PVE_CSF * count;
	const float *support = planes + (size_t)BC_PVE_SUPPORT * count;
	float *scratch = (float *)malloc(count * sizeof(*scratch));
	if (!scratch) return 1;
	/* Blur each prior, then apply the (unblurred) support, matching the
	   reference's pass order: three separable blurs, then the support gate. */
	if (cl_blur(gm, scratch, dim) || cl_blur(wm, scratch, dim) ||
			cl_blur(csf, scratch, dim)) {
		free(scratch);
		return 1;
	}
	for (size_t i = 0; i < count; i++) {
		double s = cl_clamp(support[i], 0.0, 1.0);
		gm[i] = (float)(gm[i] * s); wm[i] = (float)(wm[i] * s);
		csf[i] = (float)(csf[i] * s);
	}
	free(scratch);
	return 0;
}

/* Queue spans partition the support, so retaining the winner needs neither
   a full uint32 label map nor a second flood fill. Below-threshold priors are
   untouched: they still contribute to neighboring spatial priors. */
static int cl_retain_support(float *gm, float *wm, float *csf, const size_t dim[3],
		bc_cat_lite_report *report) {
	size_t nx=dim[0], ny=dim[1], xy=nx*ny, n=xy*dim[2], supported=0;
	if (n > UINT32_MAX) return 1;
	unsigned char *mask=calloc(n,1);
	if (!mask) return 1;
	for (size_t i=0; i<n; i++) {
		double raw=(double)(gm[i]>0?gm[i]:0)+(wm[i]>0?wm[i]:0)+(csf[i]>0?csf[i]:0);
		if (raw>=CL_MIN_SUPPORT) {mask[i]=1; supported++;}
	}
	uint32_t *queue=malloc((supported?supported:1)*sizeof(*queue));
	if (!queue) {free(mask); return 1;}
	size_t tail=0, best_start=0, best_end=0;
	for (size_t seed=0; seed<n; seed++) {
		if (mask[seed]!=1) continue;
		size_t start=tail;
		if (report) report->component_count++;
		queue[tail++]=(uint32_t)seed; mask[seed]=2;
		for (size_t head=start; head<tail; head++) {
			size_t i=queue[head], x=i%nx, y=(i%xy)/nx;
			size_t neighbors[6]; int count=0;
			if (x) neighbors[count++]=i-1;
			if (x+1<nx) neighbors[count++]=i+1;
			if (y) neighbors[count++]=i-nx;
			if (y+1<ny) neighbors[count++]=i+nx;
			if (i>=xy) neighbors[count++]=i-xy;
			if (i+xy<n) neighbors[count++]=i+xy;
			for (int j=0; j<count; j++) {
				size_t q=neighbors[j];
				if (mask[q]==1) {mask[q]=2; queue[tail++]=(uint32_t)q;}
			}
		}
		/* Browser support cleanup retains the last component on size ties. */
		if (tail-start>=best_end-best_start) {best_start=start; best_end=tail;}
	}
	for (size_t k=0; k<tail; k++) if (k<best_start || k>=best_end)
		gm[queue[k]]=wm[queue[k]]=csf[queue[k]]=0;
	if (report) report->removed_voxels=(unsigned int)(tail-(best_end-best_start));
	free(mask); free(queue); return 0;
}

static void cl_report(const cl_stats *stats, bc_cat_lite_report *report) {
    if (!report) return;
    memcpy(report->mean,stats->mean,sizeof(report->mean));
    memcpy(report->sigma,stats->sigma,sizeof(report->sigma));
    report->applied=stats->applied; report->refusal=stats->refusal;
}

int bc_cat_lite_fit_report(float *planes, const float *intensity, const size_t dim[3],
        bc_cat_lite_report *report) {
    if (report) memset(report,0,sizeof(*report));
	const size_t count = dim[0] * dim[1] * dim[2];
	float *gm = planes + (size_t)BC_PVE_GM * count;
	float *wm = planes + (size_t)BC_PVE_WM * count;
	float *csf = planes + (size_t)BC_PVE_CSF * count;
	float *support = planes + (size_t)BC_PVE_SUPPORT * count;
	if (cl_retain_support(gm,wm,csf,dim,report)) return 1;
	cl_fit_data fit;
	if (cl_fit_make(&fit,gm,wm,csf,intensity,count)) return 1;
	cl_stats initial=cl_estimate(&fit,1);
	cl_report(&initial,report);
	if (!initial.applied) { cl_fit_free(&fit); return 0; }
	cl_debug_stats("initial", &initial);
	if (cl_bias_values(gm,wm,csf,intensity,initial.mean,dim,&fit)) {
		cl_fit_free(&fit); return 1;
	}
	cl_stats stats=cl_estimate(&fit,0);
	cl_report(&stats,report);
	if (!stats.applied) { cl_fit_free(&fit); return 0; }
	cl_debug_stats("fitted", &stats);
	free(fit.weights); fit.weights=NULL;

	/* The fit reads all three priors at neighbouring voxels, so it cannot write
	   them in place; support doubles as the third output buffer. */
	float *out_gm = calloc(count,sizeof(float)), *out_wm = support;
	float *out_csf = (float *)calloc(count, sizeof(*out_csf));
	if (!out_csf || !out_gm) { free(out_csf); free(out_gm); cl_fit_free(&fit); return 1; }
	memset(out_wm, 0, count * sizeof(*out_wm));
	cl_fit_slab slab = { .gm = gm, .wm = wm, .csf = csf, .fit = &fit, .out_gm = out_gm, .out_wm = out_wm, .out_csf = out_csf,
		.stats = &stats, .dim = dim };
	bc_parallel_for(0, (int)fit.count - 1, 0, cl_fit_rows, &slab);

	memcpy(gm, out_gm, count * sizeof(*gm));
	memcpy(wm, out_wm, count * sizeof(*wm));
	memcpy(csf, out_csf, count * sizeof(*csf));
	free(out_gm); free(out_csf); cl_fit_free(&fit);
	return 0;
}

int bc_cat_lite_fit(float *planes, const float *intensity, const size_t dim[3]) {
    return bc_cat_lite_fit_report(planes,intensity,dim,NULL);
}

#endif /* BC_HAS_PVE */
