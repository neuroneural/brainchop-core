/*
 * Regression test for Stage 1 accuracy guards:
 *   1. Small (< 838.9 mL, q95 == 0) skull-stripped / infant normalization guard
 *      in both CPU (mn_normalize) and accelerator (bc_accel_normalize_input).
 *   2. Bit-identical preservation on normal (>= 839 mL) skull-stripped and
 *      whole-head inputs.
 *   3. PVE prior mask coverage: Brain-Stem (13) in WM_MASK, myelinated
 *      subcorticals {7, 10, 17} in MIXED_GWM_MASK, zero anatomical holes across
 *      classes 1..17.
 */
#include "backend_accel.h"
#include "meshnet_cpu.c"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_brain(unsigned char *vol, size_t nvox, size_t fg_count, int constant_fg) {
	memset(vol, 0, nvox);
	for (size_t i = 0; i < fg_count; i++) {
		if (constant_fg) {
			vol[i] = 180;
		} else {
			/* Uniform distribution across [40..220] (181 bins). */
			vol[i] = (unsigned char)(40 + (i % 181));
		}
	}
}

static void check_identical_cpu_accel(const unsigned char *vol, float *cpu_out) {
	mn_region full = { 0, MN_DIM - 1, 0, MN_DIM - 1, 0, MN_DIM - 1, MN_NVOX };
	mn_normalize(vol, cpu_out, &full);
	float *acc_out = bc_accel_normalize_input(vol);
	assert(acc_out != NULL);
	for (size_t i = 0; i < MN_NVOX; i++) {
		assert(memcmp(&cpu_out[i], &acc_out[i], sizeof(float)) == 0);
	}
	free(acc_out);
}

int main(void) {
	unsigned char *vol = (unsigned char *)malloc(MN_NVOX);
	float *cpu_out = (float *)malloc(MN_NVOX * sizeof(float));
	assert(vol != NULL && cpu_out != NULL);

	/* 1. ~500 mL skull-stripped brain (500,000 nonzero voxels = 2.98% of 256^3,
	      so >97% zeros -> whole-volume q95 == 0 and q98 > 0 unless < 2%).
	      Use 300,000 voxels (1.79%, ~300 mL infant brain) so BOTH q95 (5%)
	      AND q98 (2%) hit 0 on the whole volume, exercising MindGrab & MindMap! */
	const size_t infant_vox = 300000;
	fill_brain(vol, MN_NVOX, infant_vox, 0);
	check_identical_cpu_accel(vol, cpu_out);

	float min_fg = 1e30f, max_fg = -1e30f;
	for (size_t i = 0; i < infant_vox; i++) {
		assert(isfinite(cpu_out[i]));
		if (cpu_out[i] < min_fg) min_fg = cpu_out[i];
		if (cpu_out[i] > max_fg) max_fg = cpu_out[i];
	}
	/* Without the guard, max_fg was 220.0f (255x unscaled!). With the guard,
	   brain voxels normalize cleanly into O(1) range. */
	assert(max_fg <= 1.25f);
	assert(max_fg >= 0.95f);
	assert(min_fg >= -0.25f);
	assert(min_fg <= 0.05f);
	printf("PASS [%s]: ~300 mL skull-stripped guard (fg range [%.4f, %.4f], CPU==accel bit-exact)\n",
		BC_MODEL_NAME, min_fg, max_fg);

	/* 2. ~500 mL skull-stripped brain (500,000 nonzero voxels = 2.98%):
	      triggers q95 == 0 on 5%/95% models (MindMap / MindMap-Fast / MindSnap). */
	const size_t small_vox = 500000;
	fill_brain(vol, MN_NVOX, small_vox, 0);
	check_identical_cpu_accel(vol, cpu_out);
	max_fg = -1e30f;
	for (size_t i = 0; i < small_vox; i++) {
		if (cpu_out[i] > max_fg) max_fg = cpu_out[i];
	}
	assert(max_fg <= 2.0f && max_fg >= 0.95f);
	printf("PASS [%s]: ~500 mL skull-stripped guard (max_fg=%.4f, CPU==accel bit-exact)\n",
		BC_MODEL_NAME, max_fg);

	/* 3. Constant nonzero foreground in sparse mask (300,000 voxels == 180):
	      background -> 0.0f, foreground -> ~1.0f. */
	fill_brain(vol, MN_NVOX, infant_vox, 1);
	check_identical_cpu_accel(vol, cpu_out);
	assert(fabsf(cpu_out[0] - 1.0f) < 1e-4f);
	assert(fabsf(cpu_out[infant_vox] - 0.0f) < 1e-6f);
	printf("PASS [%s]: sparse constant-foreground guard (bg=%.4f, fg=%.4f)\n",
		BC_MODEL_NAME, cpu_out[infant_vox], cpu_out[0]);

	/* 4. Normal >= 839 mL brain (1,200,000 nonzero voxels = 7.15%):
	      high > low on the whole volume, so output is bit-identical to legacy. */
	const size_t normal_vox = 1200000;
	fill_brain(vol, MN_NVOX, normal_vox, 0);
	check_identical_cpu_accel(vol, cpu_out);
	int64_t cum[256] = {0};
	mn_region full = { 0, MN_DIM - 1, 0, MN_DIM - 1, 0, MN_DIM - 1, MN_NVOX };
	mn_histogram(vol, cum, &full);
	float legacy_lo = (float)mn_quantile(cum, MN_NVOX, BC_MODEL_NORM_Q_LOW, 0);
	float legacy_hi = (float)mn_quantile(cum, MN_NVOX, BC_MODEL_NORM_Q_HIGH, 1);
	assert(legacy_hi > legacy_lo);
	float legacy_den = legacy_hi - legacy_lo + BC_MODEL_NORM_DENOM_EPS;
	for (size_t i = 0; i < 4096; i++) {
		size_t idx = (i * 4099) % MN_NVOX;
		float want = ((float)vol[idx] - legacy_lo) / legacy_den;
#if BC_MODEL_NORM_CLAMP
		if (!(want > 0.0f)) want = 0.0f;
		if (want > 1.0f) want = 1.0f;
#endif
		assert(memcmp(&cpu_out[idx], &want, sizeof(float)) == 0);
	}
	printf("PASS [%s]: normal >=839 mL brain bit-identical to legacy quantiles (lo=%.2f, hi=%.2f)\n",
		BC_MODEL_NAME, legacy_lo, legacy_hi);

#if BC_HAS_PVE
	/* 5. PVE prior mask coverage: Brain-Stem (13) in WM_MASK, myelinated {7,10,17}
	      in MIXED_GWM_MASK, pairwise disjoint, complete coverage of classes 1..17. */
	assert((BC_PRIOR_WM_MASK & (1u << 13)) != 0);
	assert((BC_PRIOR_MIXED_GWM_MASK & (1u << 7)) != 0);
	assert((BC_PRIOR_MIXED_GWM_MASK & (1u << 10)) != 0);
	assert((BC_PRIOR_MIXED_GWM_MASK & (1u << 17)) != 0);
	assert((BC_PRIOR_GM_MASK & BC_PRIOR_MIXED_GWM_MASK) == 0);
	assert((BC_PRIOR_GM_MASK & BC_PRIOR_WM_MASK) == 0);
	assert((BC_PRIOR_WM_MASK & BC_PRIOR_MIXED_GWM_MASK) == 0);
	assert((BC_PRIOR_CSF_MASK & (BC_PRIOR_GM_MASK | BC_PRIOR_WM_MASK | BC_PRIOR_MIXED_GWM_MASK)) == 0);
	unsigned covered = BC_PRIOR_GM_MASK | BC_PRIOR_WM_MASK | BC_PRIOR_MIXED_GWM_MASK | BC_PRIOR_CSF_MASK;
	assert(covered == ((1u << 18) - 2u)); /* bits 1..17 all set, bit 0 clear */
	printf("PASS [%s]: PVE masks cover all 17 brain classes with Brain-Stem in WM and {7,10,17} mixed\n",
		BC_MODEL_NAME);
#endif

	free(vol);
	free(cpu_out);
	return 0;
}
