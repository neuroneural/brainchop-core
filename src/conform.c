#include "brainchopc.h"
#include "bc_threads.h"
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	float *img;
	size_t n;
	float lo, hi, src_min, scale;
	int nchunks;
} scalecrop_ctx;

static void scalecrop_chunk(void *ctx, int c0, int c1) {
	const scalecrop_ctx *c = ctx;
	float *img = c->img;
	const float lo = c->lo, hi = c->hi, src_min = c->src_min, scale = c->scale;
	for (int ch = c0; ch <= c1; ch++) {
		size_t i0 = (size_t)ch * c->n / (size_t)c->nchunks;
		size_t i1 = (size_t)(ch + 1) * c->n / (size_t)c->nchunks;
		for (size_t i = i0; i < i1; i++)
			img[i] = fminf(fmaxf(lo + scale * (img[i] - src_min), lo), hi);
	}
}

static void scalecrop(float *img, size_t n, float lo, float hi, float src_min, float scale) {
	if (!n) return;
	if (n < 65536) {
		for (size_t i = 0; i < n; i++)
			img[i] = fminf(fmaxf(lo + scale * (img[i] - src_min), lo), hi);
		return;
	}
	scalecrop_ctx ctx = { .img = img, .n = n, .lo = lo, .hi = hi,
		.src_min = src_min, .scale = scale, .nchunks = 64 };
	bc_parallel_for(0, ctx.nchunks - 1, 1, scalecrop_chunk, &ctx);
}

#define CONF_HIST_CHUNKS 32

typedef struct {
	float *img;
	size_t n;
	int nchunks;
	float cmin[CONF_HIST_CHUNKS];
	float cmax[CONF_HIST_CHUNKS];
	size_t cnz[CONF_HIST_CHUNKS];
	size_t cnltz[CONF_HIST_CHUNKS];
} scale_scan_ctx;

static void scale_scan_chunk(void *ctx, int c0, int c1) {
	scale_scan_ctx *c = ctx;
	float *img = c->img;
	for (int ch = c0; ch <= c1; ch++) {
		size_t i0 = (size_t)ch * c->n / (size_t)c->nchunks;
		size_t i1 = (size_t)(ch + 1) * c->n / (size_t)c->nchunks;
		float mn = INFINITY, mx = -INFINITY;
		size_t nz = 0, nltz = 0;
		for (size_t i = i0; i < i1; i++) {
			float v = img[i];
			if (!isfinite(v)) {
				v = 0.0f;
				img[i] = 0.0f;
			}
			if (v < mn) mn = v;
			if (v > mx) mx = v;
			nz += fabsf(v) >= 1e-15f;
			nltz += v < 0.0f;
		}
		c->cmin[ch] = mn;
		c->cmax[ch] = mx;
		c->cnz[ch] = nz;
		c->cnltz[ch] = nltz;
	}
}

typedef struct {
	const float *img;
	size_t n;
	int nchunks;
	float src_min, bin_size;
	int *hists; /* nchunks * 1000 */
} scale_hist_ctx;

static void scale_hist_chunk(void *ctx, int c0, int c1) {
	const scale_hist_ctx *c = ctx;
	const float *img = c->img;
	const float src_min = c->src_min, bin_size = c->bin_size;
	for (int ch = c0; ch <= c1; ch++) {
		size_t i0 = (size_t)ch * c->n / (size_t)c->nchunks;
		size_t i1 = (size_t)(ch + 1) * c->n / (size_t)c->nchunks;
		int *h = c->hists + (size_t)ch * 1000;
		for (size_t i = i0; i < i1; i++) {
			float raw = (img[i] - src_min) / bin_size;
			int bin = !(raw > 0.0f) ? 0 : !(raw < 999.0f) ? 999 : (int)raw;
			h[bin]++;
		}
	}
}

/* This is niimath conform's training-intensity transform, reduced to the
 * fixed 0..255, f_low=0, f_high=.98 case. Retaining f_high=.98 (clipping the
 * top 2% of non-zero voxels) preserves MindGrab bit-exactness, matches the
 * v3.5/v4.0 training pipeline, and prevents 1-1.5% bright clinical fat/flare
 * outliers from exploding unclamped 5/95 quantile normalization.
 * BSD-2-Clause, Chris Rorden's Lab -- see THIRD_PARTY_NOTICES.md. */
static int scale_intensity(float *img, size_t n) {
	if (!n) return 0;
	int nchunks = n >= 65536 ? CONF_HIST_CHUNKS : 1;
	scale_scan_ctx sc;
	memset(&sc, 0, sizeof(sc));
	sc.img = img; sc.n = n; sc.nchunks = nchunks;
	bc_parallel_for(0, nchunks - 1, 1, scale_scan_chunk, &sc);
	float src_min = INFINITY, src_max = -INFINITY;
	size_t nz = 0, nltz = 0;
	for (int ch = 0; ch < nchunks; ch++) {
		if (sc.cmin[ch] < src_min) src_min = sc.cmin[ch];
		if (sc.cmax[ch] > src_max) src_max = sc.cmax[ch];
		nz += sc.cnz[ch];
		nltz += sc.cnltz[ch];
	}
	if (src_min < 0.0f && 100.0 * (double)nltz / (double)n < 2.0)
		src_min = 0.0f;
	double span = (double)src_max - (double)src_min;
	float fspan = src_max - src_min;
	float bin_size = isfinite(fspan) ? fspan / 1000.0f : (float)(span / 1000.0);
	if (!isfinite(bin_size) || !(bin_size > 0.0f)) {
		scalecrop(img, n, 0.0f, 255.0f, 0.0f, 1.0f);
		return 0;
	}
	int *hists = calloc((size_t)nchunks * 1000, sizeof(*hists));
	if (!hists) return 1;
	scale_hist_ctx hc = { .img = img, .n = n, .nchunks = nchunks,
		.src_min = src_min, .bin_size = bin_size, .hists = hists };
	bc_parallel_for(0, nchunks - 1, 1, scale_hist_chunk, &hc);
	int hist[1000] = {0};
	for (int ch = 0; ch < nchunks; ch++) {
		const int *h = hists + (size_t)ch * 1000;
		for (int b = 0; b < 1000; b++) hist[b] += h[b];
	}
	free(hists);
	size_t nth = n - (size_t)(0.02 * (double)nz);
	size_t cumulative = 0, idx = 0;
	for (; idx < 999; idx++) {
		cumulative += (size_t)hist[idx];
		if (cumulative + (size_t)hist[idx + 1] >= nth) break;
	}
	src_max = idx * bin_size + src_min;
	float scale = src_min != src_max ? 255.0f / (src_max - src_min) : 1.0f;
	scalecrop(img, n, 0.0f, 255.0f, src_min, scale);
	return 0;
}

/* niimath `-comply ... f_high=1`: subtract the accepted minimum and clamp. */
static void scale_intensity_full_range(float *img, size_t n) {
	if (!n) return;
	int nchunks = n >= 65536 ? CONF_HIST_CHUNKS : 1;
	scale_scan_ctx sc;
	memset(&sc, 0, sizeof(sc));
	sc.img = img; sc.n = n; sc.nchunks = nchunks;
	bc_parallel_for(0, nchunks - 1, 1, scale_scan_chunk, &sc);
	float src_min = INFINITY;
	size_t nltz = 0;
	for (int ch = 0; ch < nchunks; ch++) {
		if (sc.cmin[ch] < src_min) src_min = sc.cmin[ch];
		nltz += sc.cnltz[ch];
	}
	if (src_min < 0.0f && 100.0 * (double)nltz / (double)n < 2.0)
		src_min = 0.0f;
	scalecrop(img, n, 0.0f, 255.0f, src_min, 1.0f);
}

static mat44 dmat_to_mat(const nifti_dmat44 *d) {
	mat44 f;
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
			f.m[r][c] = (float)d->m[r][c];
	return f;
}

static int valid_mat(mat44 m) {
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
			if (!isfinite(m.m[r][c])) return 0;
	float det =
		m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) -
		m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]) +
		m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
	/* Absolute threshold, matching the niimath conform this file reproduces.
	   It is scale-dependent -- a 5 um isotropic grid gives det 1.25e-7 and is
	   rejected -- but the reference behaves the same way, and parity with it
	   matters more here than covering grids these models never see. */
	return isfinite(det) && fabsf(det) >= 1e-6f;
}

static void conform_xform(const int in_dim[3], mat44 in_affine, int ras,
		mat44 *out_affine) {
	vec4 half = setVec4(in_dim[0] / 2.0f, in_dim[1] / 2.0f, in_dim[2] / 2.0f);
	vec4 center = nifti_vect44mat44_mul(half, in_affine);
	if (ras)
		LOAD_MAT44(*out_affine, 1,0,0,0, 0,1,0,0, 0,0,1,0);
	else
		LOAD_MAT44(*out_affine, -1,0,0,0, 0,0,1,0, 0,-1,0,0);
	vec4 out_center = nifti_vect44mat44_mul(setVec4(256, 256, 256), *out_affine);
	for (int i = 0; i < 3; i++)
		out_affine->m[i][3] = center.v[i] - 0.5f * out_center.v[i];
}

/*
 * The reslice gather, hoisted into its own function so the pool can hand out z
 * slabs of it. Everything the loop reads is captured here; the loop body below
 * is unchanged from when it carried a `#pragma omp parallel for`.
 */
enum { BC_LABEL_MAX_PROBES = 512 };

typedef struct {
	const float *in;
	float *out;
	mat44 out_to_in;
	int out_dim[3];
	int nx, ny, nz, nxy;
	size_t out_xy;
	int linear;
	int label_probes;
	double label_origin[3], label_step[3][3], label_delta[BC_LABEL_MAX_PROBES][3];
} reslice_slab;

/* Adaptive center-outward midpoint offsets per destination axis, matching
   brainchop-test's reslice-volume.js bit-for-bit.
   - L <= 1.5001: 1 probe at {0.0} (fast nearest path, zero blur on 1-1.5 mm).
   - L <= 3.5001: 4 midpoint probes at +/-{1/8, 3/8} (covers 2-3 mm with double
     weight on center at integer centers and zero half-integer hits at BOTH
     integer and half-integer centers, fixing both the 2 mm +/-0.25 bias and
     odd-dimension 3 mm half-integer center bias).
   - L <= 5.5001: 6 midpoint probes at +/-{1/12, 3/12, 5/12} (full 5 mm slab
     coverage with zero half-integer hits at both integer and half-integer
     centers, and double weight on center at integer centers).
   - L >  5.5001: 8 midpoint probes at +/-{1/16, 3/16, 5/16, 7/16}.
   Placing the innermost pair +/-0.5/n first ensures center-closest tie-break. */
static int label_axis_offsets(const double step[3], double offs[8]) {
	double len = sqrt(step[0]*step[0] + step[1]*step[1] + step[2]*step[2]);
	if (len <= 1.5001) {
		offs[0] = 0.0;
		return 1;
	}
	int n = len <= 3.5001 ? 4 : len <= 5.5001 ? 6 : 8;
	for (int m = 0; m < n / 2; m++) {
		double d = ((double)m + 0.5) / (double)n;
		offs[2*m]     = -d;
		offs[2*m + 1] =  d;
	}
	return n;
}

/* Browser categorical export: adaptive center-outward midpoint majority vote.
   Count background too; equal tallies retain the FIRST sample label (closest
   to the destination voxel center).
   Use double coordinates like JS, and floor(x+0.5) rather than roundf: JS
   rounds negative half coordinates towards +infinity. Check before casting. */
static void label_reslice_rows(void *context, int z_low, int z_high) {
	const reslice_slab *s = context;
	int np = s->label_probes;
	for (int z = z_low; z <= z_high; z++)
	for (int y = 0; y < s->out_dim[1]; y++) {
		double base[3];
		for (int a = 0; a < 3; a++)
			base[a] = s->label_origin[a] + y*s->label_step[1][a] + z*s->label_step[2][a];
		if (np == 1) {
			for (int x = 0; x < s->out_dim[0]; x++) {
				double ix = floor(base[0] + 0.5);
				double iy = floor(base[1] + 0.5);
				double iz = floor(base[2] + 0.5);
				float v = 0;
				if (ix >= 0 && ix < s->nx && iy >= 0 && iy < s->ny && iz >= 0 && iz < s->nz)
					v = s->in[(size_t)ix + (size_t)iy*s->nx + (size_t)iz*s->nxy];
				s->out[(size_t)x + (size_t)y*s->out_dim[0] + (size_t)z*s->out_xy] = v;
				for (int a = 0; a < 3; a++) base[a] += s->label_step[0][a];
			}
			continue;
		}
		for (int x = 0; x < s->out_dim[0]; x++) {
			float values[BC_LABEL_MAX_PROBES];
			unsigned counts[BC_LABEL_MAX_PROBES];
			int used = 0;
			for (int k = 0; k < np; k++) {
				double ix = floor(base[0] + s->label_delta[k][0] + 0.5);
				double iy = floor(base[1] + s->label_delta[k][1] + 0.5);
				double iz = floor(base[2] + s->label_delta[k][2] + 0.5);
				float v = 0;
				if (ix >= 0 && ix < s->nx && iy >= 0 && iy < s->ny && iz >= 0 && iz < s->nz)
					v = s->in[(size_t)ix + (size_t)iy*s->nx + (size_t)iz*s->nxy];
				int q = 0;
				while (q < used && values[q] != v) q++;
				if (q == used) {
					values[used] = v;
					counts[used] = 1;
					used++;
				} else {
					counts[q]++;
				}
			}
			int best = 0;
			for (int q = 1; q < used; q++) if (counts[q] > counts[best]) best = q;
			s->out[(size_t)x + (size_t)y*s->out_dim[0] + (size_t)z*s->out_xy] = values[best];
			for (int a = 0; a < 3; a++) base[a] += s->label_step[0][a];
		}
	}
}

static void reslice_rows(void *context, int z_low, int z_high) {
	const reslice_slab *s = (const reslice_slab *)context;
	const float *in = s->in;
	float *out = s->out;
	mat44 out_to_in = s->out_to_in;
	const int *out_dim = s->out_dim;
	int nx = s->nx, ny = s->ny, nz = s->nz, nxy = s->nxy;
	size_t out_xy = s->out_xy;
	int linear = s->linear;
	for (int z = z_low; z <= z_high; z++)
	for (int y = 0; y < out_dim[1]; y++) {
		float bx = y*out_to_in.m[0][1] + z*out_to_in.m[0][2] + out_to_in.m[0][3];
		float by = y*out_to_in.m[1][1] + z*out_to_in.m[1][2] + out_to_in.m[1][3];
		float bz = y*out_to_in.m[2][1] + z*out_to_in.m[2][2] + out_to_in.m[2][3];
		for (int x = 0; x < out_dim[0]; x++) {
			float fxp = x*out_to_in.m[0][0] + bx;
			float fyp = x*out_to_in.m[1][0] + by;
			float fzp = x*out_to_in.m[2][0] + bz;
			size_t oi = (size_t)x + (size_t)y*out_dim[0] + (size_t)z*out_xy;
			if (!linear) {
				int ix = (int)roundf(fxp), iy = (int)roundf(fyp), iz = (int)roundf(fzp);
				if (ix >= 0 && iy >= 0 && iz >= 0 && ix < nx && iy < ny && iz < nz)
					out[oi] = in[ix + iy*nx + iz*nxy];
				continue;
			}
			int ix = (int)floorf(fxp), iy = (int)floorf(fyp), iz = (int)floorf(fzp);
			if (ix < 0 || iy < 0 || iz < 0 || ix+1 >= nx || iy+1 >= ny || iz+1 >= nz) continue;
			float dx=fxp-ix, dy=fyp-iy, dz=fzp-iz;
			float wx0=1-dx, wy0=1-dy, wz0=1-dz;
			int p = ix + iy*nx + iz*nxy;
			float v = 0;
			v += in[p]             * wx0 * wy0 * wz0;
			v += in[p+nxy]         * wx0 * wy0 * dz;
			v += in[p+nx]          * wx0 * dy  * wz0;
			v += in[p+nx+nxy]      * wx0 * dy  * dz;
			v += in[p+1]           * dx  * wy0 * wz0;
			v += in[p+1+nxy]       * dx  * wy0 * dz;
			v += in[p+1+nx]        * dx  * dy  * wz0;
			v += in[p+1+nx+nxy]    * dx  * dy  * dz;
			out[oi] = v;
		}
	}
}

static int do_reslice(nifti_image *nim, const int out_dim[3], const float out_mm[3],
                      mat44 out_affine, mat44 out_to_in, int linear) {
	if (nim->datatype != DT_FLOAT32 || nim->nx < 1 || nim->ny < 1 || nim->nz < 1)
		return 1;
	size_t in_xy, in_n, out_xy, out_n;
	if (nii_mul_size((size_t)nim->nx, (size_t)nim->ny, &in_xy) ||
	    nii_mul_size(in_xy, (size_t)nim->nz, &in_n) ||
	    nii_mul_size((size_t)out_dim[0], (size_t)out_dim[1], &out_xy) ||
	    nii_mul_size(out_xy, (size_t)out_dim[2], &out_n) ||
	    in_n > INT_MAX || out_n > INT_MAX) return 1;
	/* A pre-conformed input, and the reslice back onto it, map every voxel onto
	   itself up to float rounding (~1e-5 voxel): skip the copy (~0.4 s each way
	   in wasm). Bounded by the farthest voxel's displacement, not per element,
	   so a real 0.05 degree tilt still resamples. The trilinear gather would
	   fill some edge faces with the minimum (which ones depends on the roundoff
	   sign); the copy keeps them. t1_crop's are background: fixtures unchanged. */
	int same_grid = out_dim[0] == nim->nx && out_dim[1] == nim->ny && out_dim[2] == nim->nz;
	for (int r = 0; r < 3 && same_grid; r++) {
		float shift = fabsf(out_to_in.m[r][3]);
		for (int c = 0; c < 3; c++)
			shift += fabsf(out_to_in.m[r][c] - (r == c)) * (out_dim[c] - 1);
		same_grid = shift < 1e-3f;
	}
	if (same_grid) goto header;
	float *in = nim->data;
	float *out = nii_malloc(out_n, sizeof(*out));
	if (!out) return 1;
	if (linear != BC_RESAMPLE_LABELS) {
		float mn = in[0];
		for (size_t i = 1; i < in_n; i++) mn = MIN(mn, in[i]);
		for (size_t i = 0; i < out_n; i++) out[i] = mn;
	}
	int nx = (int)nim->nx, ny = (int)nim->ny, nz = (int)nim->nz;
	int nxy = nx * ny;
	/*
	 * Parallel over z, and bit-exact for it: every output voxel is a pure
	 * gather from a read-only source with no cross-iteration dependency, so
	 * splitting the range cannot change a value. `make test` asserts that
	 * directly -- the CPU fixtures are bit-exact and this function is on the
	 * CPU path too.
	 *
	 * Worth doing for the FORWARD conform reslice, which is most of the
	 * `conform` stage: 128 -> 63 ms of a ~1.05 s CUDA run where the GPU itself
	 * is only ~350 ms. It does NOT move the inverse reslice in the
	 * `reslice + mask` stage (105 ms at one thread, 105 at 32) -- that stage is
	 * dominated by the two serial mask loops over the input volume, not by this
	 * function. Do not read a gain there. benchmarking records the reslice as
	 * "~0.1%", which was measured against a 140 s CPU run.
	 */
	reslice_slab slab = { .in = in, .out = out, .out_to_in = out_to_in,
		.out_dim = { out_dim[0], out_dim[1], out_dim[2] },
		.nx = nx, .ny = ny, .nz = nz, .nxy = nxy,
		.out_xy = out_xy, .linear = linear };
	if (linear == BC_RESAMPLE_LABELS) {
		mat44 inverse = nifti_mat44_inverse(dmat_to_mat(&nim->sto_xyz));
		/* Probe origin and unit steps as upstream does, retaining its float32
		   inverse and double coordinate arithmetic. Do not reuse the legacy
		   float matrix product: cancellation at native boundaries can change votes. */
		for (int a = 0; a < 3; a++) {
			double origin = inverse.m[a][3];
			for (int b = 0; b < 3; b++) origin += (double)inverse.m[a][b] * out_affine.m[b][3];
			slab.label_origin[a] = origin;
			for (int axis = 0; axis < 3; axis++) {
				double point = inverse.m[a][3];
				for (int b = 0; b < 3; b++)
					point += (double)inverse.m[a][b] * ((double)out_affine.m[b][axis] + out_affine.m[b][3]);
				slab.label_step[axis][a] = point - origin;
			}
		}
		double ox[8], oy[8], oz[8];
		int nx_p = label_axis_offsets(slab.label_step[0], ox);
		int ny_p = label_axis_offsets(slab.label_step[1], oy);
		int nz_p = label_axis_offsets(slab.label_step[2], oz);
		int k = 0;
		for (int ix = 0; ix < nx_p; ix++)
		for (int iy = 0; iy < ny_p; iy++)
		for (int iz = 0; iz < nz_p; iz++, k++)
			for (int a = 0; a < 3; a++) slab.label_delta[k][a] =
				ox[ix]*slab.label_step[0][a] + oy[iy]*slab.label_step[1][a] + oz[iz]*slab.label_step[2][a];
		slab.label_probes = k;
		bc_parallel_for(0, out_dim[2] - 1, 0, label_reslice_rows, &slab);
	} else bc_parallel_for(0, out_dim[2] - 1, 0, reslice_rows, &slab);
	free(nim->data);
	nim->data = out;
header:
	nim->nx=nim->dim[1]=out_dim[0]; nim->ny=nim->dim[2]=out_dim[1]; nim->nz=nim->dim[3]=out_dim[2];
	nim->dx=nim->pixdim[1]=out_mm[0]; nim->dy=nim->pixdim[2]=out_mm[1]; nim->dz=nim->pixdim[3]=out_mm[2];
	nim->nvox = (int64_t)out_n;
	for (int r=0;r<4;r++) for(int c=0;c<4;c++)
		nim->sto_xyz.m[r][c]=nim->qto_xyz.m[r][c]=out_affine.m[r][c];
	nifti_dmat44_to_quatern(nim->sto_xyz, &nim->quatern_b,&nim->quatern_c,&nim->quatern_d,
		&nim->qoffset_x,&nim->qoffset_y,&nim->qoffset_z,&nim->dx,&nim->dy,&nim->dz,&nim->qfac);
	nim->scl_slope=1; nim->scl_inter=0; nim->cal_min=0; nim->cal_max=0;
	int code = MAX(nim->qform_code, nim->sform_code);
	nim->qform_code=nim->sform_code=code;
	return 0;
}

static int conform_reslice(nifti_image *nim, int ras) {
	mat44 primary = dmat_to_mat(nim->qform_code > nim->sform_code ? &nim->qto_xyz : &nim->sto_xyz);
	mat44 secondary = dmat_to_mat(nim->qform_code > nim->sform_code ? &nim->sto_xyz : &nim->qto_xyz);
	mat44 in_affine = valid_mat(primary) ? primary : secondary;
	if (!valid_mat(in_affine)) return 1;
	int in_dim[3]={(int)nim->nx,(int)nim->ny,(int)nim->nz}, out_dim[3]={256,256,256};
	float out_mm[3]={1,1,1};
	mat44 out_affine;
	conform_xform(in_dim, in_affine, ras, &out_affine);
	mat44 in_to_out = nifti_mat44_mul(nifti_mat44_inverse(out_affine), in_affine);
	mat44 out_to_in = nifti_mat44_inverse(in_to_out);
	return do_reslice(nim, out_dim, out_mm, out_affine, out_to_in, 1);
}

int brainchopc_comply(nifti_image *nim) {
	if (!nim || nim->datatype != DT_FLOAT32) return 1;
	scale_intensity_full_range(nim->data, (size_t)nim->nvox);
	return conform_reslice(nim, 1);
}

int brainchopc_conform(nifti_image *nim) {
	if (!nim || nim->datatype != DT_FLOAT32) return 1;
	if (scale_intensity(nim->data, (size_t)nim->nvox)) return 1;
	return conform_reslice(nim, 0);
}

int brainchopc_reslice(nifti_image *source, const nifti_image *target, int linear) {
	if (!source || !target || !source->data || linear < BC_RESAMPLE_NEAREST ||
			linear > BC_RESAMPLE_LABELS || target->nx < 1 || target->ny < 1 || target->nz < 1 ||
			target->nx > INT_MAX || target->ny > INT_MAX || target->nz > INT_MAX) return 1;
	/*
	 * Deliberately the sform for both images, even when the target's qform_code
	 * is higher. This matches the niimath reslice the Python reference calls,
	 * and was verified against it: substituting the higher-coded qform here (as
	 * conform_reslice does for its own input) diverges from the reference by
	 * ~7% of voxels. Do not "fix" this to agree with conform_reslice.
	 *
	 * The matrices are validated, though: unvalidated, a NaN sform reaches
	 * do_reslice's float-to-int voxel indexing as undefined behaviour, and a
	 * singular one inverts to the zero matrix so every sample reads in[0] --
	 * both of which produced a blank image and exit status 0.
	 */
	mat44 in_affine=dmat_to_mat(&source->sto_xyz), out_affine=dmat_to_mat(&target->sto_xyz);
	if (!valid_mat(in_affine) || !valid_mat(out_affine)) return 1;
	mat44 in_to_out=nifti_mat44_mul(nifti_mat44_inverse(out_affine),in_affine);
	mat44 out_to_in=nifti_mat44_inverse(in_to_out);
	int dim[3]={(int)target->nx,(int)target->ny,(int)target->nz};
	float mm[3]={(float)target->dx,(float)target->dy,(float)target->dz};
	int status=do_reslice(source,dim,mm,out_affine,out_to_in,linear);
	if(!status) {
		/* niimath's reference-grid reslice preserves the target header's stored
		   voxel sizes instead of round-tripping them through the affine. */
		source->dx=source->pixdim[1]=target->dx;
		source->dy=source->pixdim[2]=target->dy;
		source->dz=source->pixdim[3]=target->dz;
	}
	return status;
}
