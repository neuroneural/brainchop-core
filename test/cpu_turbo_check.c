/* CPU turbo: (1) integer kernels (i8mm, ARMv8.2 SDOT, AVX-512 VNNI, AVX-VNNI,
   AVX2) equal the scalar reference bit for bit (integer accumulation, same float
   combine); (2) y8 64-voxel block epilogue + dequantized residual quantization
   match the exact GELU + residual bounds; (3) the row driver, taps, zero-point
   correction and padding match the exact convolution on dequantized codes at
   border slices; (4) SME fp16 matches its fp32 oracle. Build: make test-cpu-turbo. */
#define BC_CPU_TURBO_CHECK 1
#include "../src/meshnet_cpu.c"
#include <stdio.h>

static int check_kernel_exact(const char *name, mt_block_fn block2, mt_block_fn block1) {
	enum { VOX = 4096 };
	static uint8_t src[VOX * MT_AS + MT_ACT_PAD];
	static float kernel[27 * BC_MODEL_CHAN * MN_CS];
	static int8_t wq[27 * MT_TAP];
	int32_t tap_sum[27 * 2 * MN_CS];
	float scale[2 * MN_CS], qs[MN_CS], fast[2 * MN_CS], ref[2 * MN_CS];
	unsigned seed = 12345;
#define CK_RAND() (seed = seed * 1103515245u + 12345u, seed >> 8)
	int mismatches = 0;
	for (int trial = 0; trial < 200; trial++) {
		for (size_t i = 0; i < sizeof src; i++)
			src[i] = (uint8_t)(CK_RAND() & (i % MT_AS < MN_CS ? 255 : 127));
		for (size_t i = 0; i < sizeof kernel / sizeof *kernel; i++)
			kernel[i] = ((float)(CK_RAND() % 20001) - 10000.0f) * 1e-5f;
		for (int c = 0; c < MN_CS; c++) qs[c] = 0.01f + (float)(CK_RAND() % 1000) * 1e-4f;
		mt_prepare(kernel, qs, wq, tap_sum, scale);
		ptrdiff_t tap_voxel[27];
		const int8_t *tap_w[27];
		int32_t corr[2 * MN_CS] = { 0 };
		int ntap = 1 + (int)(CK_RAND() % 27);
		for (int t = 0; t < ntap; t++) {
			int tap = (int)(CK_RAND() % 27);
			tap_voxel[t] = (ptrdiff_t)(CK_RAND() % (VOX / 2));
			tap_w[t] = wq + (size_t)tap * MT_TAP;
			for (int k = 0; k < 2 * MN_CS; k++) corr[k] += tap_sum[tap * 2 * MN_CS + k];
		}
		size_t x = CK_RAND() % (VOX / 2 - 2);
		block2(src, x, tap_voxel, tap_w, ntap, corr, scale, fast);
		mt_scalar_block2(src, x, tap_voxel, tap_w, ntap, corr, scale, ref);
		mismatches += memcmp(fast, ref, sizeof fast) != 0;
		block1(src, x, tap_voxel, tap_w, ntap, corr, scale, fast);
		mt_scalar_block1(src, x, tap_voxel, tap_w, ntap, corr, scale, ref);
		mismatches += memcmp(fast, ref, MN_CS * sizeof *fast) != 0;
	}
#undef CK_RAND
	printf("cpu_turbo_check: %s kernel %s (%d mismatching blocks of 400)\n",
		name, mismatches ? "FAIL" : "ok", mismatches);
	return mismatches;
}

int main(void) {
	mt_block_fn block2, block1;
	int mode = MT_MODE_INT8;
	const char *kname = mt_select_ex(&block2, &block1, &mode);
	if (!kname) {
		printf("cpu_turbo_check: SKIP, this CPU has no i8mm/DotProd/AVX2 kernel\n");
		return 0;
	}
	int mismatches = check_kernel_exact(kname, block2, block1);
	/* Also test fallback SIMD kernels directly whenever the host CPU supports them,
	   so CI on i8mm or VNNI machines still verifies SDOT and AVX2 bit-exactness. */
#if defined(__aarch64__)
	if (strcmp(kname, "sdot") != 0) {
		int dotprod = 0;
#if defined(__APPLE__)
		size_t sz = sizeof(dotprod);
		if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &dotprod, &sz, NULL, 0)) dotprod = 0;
#elif defined(__linux__)
		dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#endif
		if (dotprod) mismatches += check_kernel_exact("sdot", mt_sdot_block2, mt_sdot_block1);
	}
#elif (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
	if (strcmp(kname, "avx2") != 0 && strcmp(kname, "avx512f") != 0 &&
			__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
		mismatches += check_kernel_exact("avx2", mt_avx2_block2, mt_avx2_block1);
#endif

	/* (2) Allocations are virtual; only the tested slices are touched. */
	static float kernel[27 * BC_MODEL_CHAN * MN_CS];
	static int8_t wq[27 * MT_TAP];
	int32_t tap_sum[27 * 2 * MN_CS];
	float scale[2 * MN_CS], qs[MN_CS];
	unsigned seed = 54321;
#define RAND() (seed = seed * 1103515245u + 12345u, seed >> 8)
	const size_t slice = (size_t)MN_DIM * MN_DIM;
	uint8_t *codes = calloc(MN_NVOX * MT_AS + MT_ACT_PAD, 1);
	uint8_t *y8 = calloc(MN_NVOX * MN_CS, 1);
	float *ymom = calloc(MT_YBLK_TOTAL * 2 * MN_CS, sizeof(float));
	float *exact_in = calloc(MN_NVOX * MN_CS, sizeof(float));
	float *got = malloc(MN_NVOX * MN_CS * sizeof(float));
	float *want = malloc(MN_NVOX * MN_CS * sizeof(float));
	uint32_t *row_list = malloc(BC_TC_ROWS * sizeof(uint32_t));
	float *row_wt = malloc(BC_TC_ROWS * sizeof(float));
	double *partial = malloc(MT_STAT_BINS * 2 * MN_CS * sizeof(double));
	float *range = malloc(MT_STAT_BINS * 2 * MN_CS * sizeof(float));
	if (!codes || !y8 || !ymom || !exact_in || !got || !want || !row_list || !row_wt || !partial || !range)
		return 1;

	float mul[MN_CS], add[MN_CS], inv_qs[MN_CS];
	for (int c = 0; c < MN_CS; c++) {
		mul[c] = 1.0f; add[c] = 0.0f;
		qs[c] = 0.025f + 0.01f * c; inv_qs[c] = 1.0f / qs[c];
	}
	for (size_t i = 0; i < slice * MN_CS; i++)
		got[i] = ((int)(i % 12001) - 6000) * 0.001f;
	/* Verify y8 64-voxel epilogue + moments + mt_quant_y8_bins round-trip on slice 0. */
	double s_sum[MN_CS] = { 0 }, s_sq[MN_CS] = { 0 };
	double r_sum[MN_CS] = { 0 }, r_sq[MN_CS] = { 0 };
	float s_lo[MN_CS], s_hi[MN_CS], r_lo[MN_CS], r_hi[MN_CS];
	for (int c = 0; c < MN_CS; c++) {
		s_lo[c] = r_lo[c] = FLT_MAX;
		s_hi[c] = r_hi[c] = -FLT_MAX;
	}
	for (uint32_t y = 0; y < MN_DIM; y++) {
		row_list[y] = y;
		row_wt[y] = 1.0f;
		const float *row_in = got + (size_t)y * MN_DIM * MN_CS;
		mt_row_epilogue(row_in, y, 1.0f, y8, ymom, s_sum, s_sq, s_lo, s_hi);
		for (int x = 0; x < MN_DIM; x++)
			for (int c = 0; c < MN_CS; c++) {
				float val = row_in[x * MN_CS + c];
				r_sum[c] += val;
				r_sq[c] += (double)val * val;
				if (val < r_lo[c]) r_lo[c] = val;
				if (val > r_hi[c]) r_hi[c] = val;
			}
	}
	mt_quant_y8_slab quant = { y8, ymom, codes, mul, add, inv_qs, row_list, 0, MN_DIM };
	mt_quant_y8_bins(&quant, 0, MT_STAT_BINS - 1);
	int quant_bad = 0, sme_bad = 0;
	for (int c = 0; c < MN_CS; c++) {
		quant_bad += fabs(s_sum[c] - r_sum[c]) > 1e-11 * fmax(1.0, fabs(r_sum[c]));
		quant_bad += fabs(s_sq[c] - r_sq[c]) > 1e-11 * fmax(1.0, fabs(r_sq[c]));
		quant_bad += s_lo[c] != r_lo[c] || s_hi[c] != r_hi[c];
	}
	/* Constant high-DC row must stay double-exact (float32 sums yield ~9e-5 apparent variance). */
	{
		float dc_row[MN_DIM * MN_CS];
		uint8_t dc_y8[MN_DIM * MN_CS];
		float dc_mom[MT_YBLKS_PER_ROW * 2 * MN_CS];
		double dc_sum[MN_CS] = { 0 }, dc_sq[MN_CS] = { 0 };
		float dc_lo[MN_CS], dc_hi[MN_CS];
		for (int c = 0; c < MN_CS; c++) { dc_lo[c] = FLT_MAX; dc_hi[c] = -FLT_MAX; }
		for (int i = 0; i < MN_DIM * MN_CS; i++) dc_row[i] = 10.3f;
		mt_row_epilogue(dc_row, 0, 1.0f, dc_y8, dc_mom, dc_sum, dc_sq, dc_lo, dc_hi);
		for (int c = 0; c < MN_CS; c++) {
			double mean = dc_sum[c] / MN_DIM;
			double var = dc_sq[c] / MN_DIM - mean * mean;
			quant_bad += fabs(var) > 1e-11;
		}
	}
	for (size_t v = 0; v < slice; v++) {
		size_t blk = v / MT_YBLK;
		const float *bmin = ymom + blk * 2 * MN_CS;
		const float *bscale = bmin + MN_CS;
		for (int c = 0; c < MN_CS; c++) {
			unsigned hi = codes[v * MT_AS + c], lo = codes[v * MT_AS + MN_CS + c];
			float decoded = ((float)hi - MT_ZP + ((float)lo - MT_RES_ZP) / 128.0f) * qs[c];
			float pre_q = (float)y8[v * MN_CS + c] * bscale[c] + bmin[c];
			float y8_err = fabsf(pre_q - got[v * MN_CS + c]);
			quant_bad += y8_err > 0.5f * bscale[c] + 1e-4f;
			quant_bad += lo > 127 || fabsf(decoded - mn_gelu(pre_q)) > qs[c] / 256.0f + 2e-6f;
		}
	}
	printf("cpu_turbo_check: y8 epilogue + residual quantization %s\n", quant_bad ? "FAIL" : "ok");

#if MN_SME
	if (mt_sme_available()) {
		for (int c = 0; c < MN_CS; c++) { mul[c] = 0.25f + 0.1f * c; add[c] = -0.3f * c; }
		mt_quant_sme_y8_bins(&quant, 0, MT_STAT_BINS - 1);
		const float16_t *half = (const float16_t *)codes;
		for (int y = 0; y < MN_DIM; y++) for (int x = 0; x < MN_DIM; x++) {
			size_t v = (size_t)y * MN_DIM + x;
			size_t blk = v / MT_YBLK;
			const float *bmin = ymom + blk * 2 * MN_CS;
			const float *bscale = bmin + MN_CS;
			for (int c = 0; c < MN_CS; c++) {
				float pre_q = (float)y8[v * MN_CS + c] * bscale[c] + bmin[c];
				float expected = mn_gelu(pre_q * mul[c] + add[c]);
				float actual = half[(size_t)y * MN_DIM * MN_CS + (c / 2) * MN_DIM * 2 + x * 2 + c % 2];
				sme_bad += fabsf(actual - expected) > 0.0005f * fabsf(expected) + 2e-6f;
			}
		}
		printf("cpu_turbo_check: SME quantization %s\n", sme_bad ? "FAIL" : "ok");
	}
#endif

	for (size_t i = 0; i < sizeof kernel / sizeof *kernel; i++)
		kernel[i] = ((float)(RAND() % 20001) - 10000.0f) * 1e-5f;
	for (int c = 0; c < MN_CS; c++) qs[c] = c < BC_MODEL_CHAN ? (0.025f + 0.01f * c) : 1.0f;
	mt_prepare(kernel, qs, wq, tap_sum, scale);
	const int slab_low[] = { 0, 123, 224 }, slab_high[] = { 34, 133, 255 };
	for (int k = 0; k < 3; k++)
		for (size_t v = slab_low[k] * slice; v < (slab_high[k] + 1) * slice; v++)
			for (int c = 0; c < MN_CS; c++) {
				uint8_t q = c < BC_MODEL_CHAN ? (uint8_t)RAND() : MT_ZP;
				uint8_t r = c < BC_MODEL_CHAN ? (uint8_t)(RAND() & 127) : MT_RES_ZP;
				codes[v * MT_AS + c] = q;
				codes[v * MT_AS + MN_CS + c] = r;
				exact_in[v * MN_CS + c] = ((float)q - MT_ZP + ((float)r - MT_RES_ZP) / 128.0f) * qs[c];
			}
	mn_region region;
	mn_make_region(NULL, &region);
	const struct { int d, z; } cases[] = { {31, 0}, {31, 255}, {19, 3}, {5, 128}, {1, 0}, {1, 255} };
	int bad = 0;
	for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) {
		int z = cases[k].z;
		for (int y = 0; y < MN_DIM; y++) {
			row_list[y] = (uint32_t)(z * MN_DIM + y);
			row_wt[y] = 1.0f;
			mt_conv_row_compute(codes, z, y, cases[k].d, wq, kernel, tap_sum, scale, block2, block1, MT_MODE_INT8,
				got + ((size_t)z * MN_DIM + (size_t)y) * MN_DIM * MN_CS);
		}
		mt_conv_slab conv = { codes, y8, ymom, wq, kernel, tap_sum, scale, block2, block1, cases[k].d,
			row_list, row_wt, 0, MN_DIM, partial, range, MT_MODE_INT8 };
		mt_conv_bins(&conv, 0, MT_STAT_BINS - 1);
		mn_conv_hidden_slab exact = { exact_in, want, kernel, cases[k].d, &region };
		mn_conv_hidden_rows(&exact, z, z);
		double err = 0, err_y8 = 0, peak = 0;
		for (size_t v = (size_t)z * slice; v < (size_t)(z + 1) * slice; v++) {
			size_t blk = v / MT_YBLK;
			const float *bmin = ymom + blk * 2 * MN_CS;
			const float *bscale = bmin + MN_CS;
			for (int c = 0; c < MN_CS; c++) {
				size_t i = v * MN_CS + c;
				err = fmax(err, fabs((double)got[i] - want[i]));
				double deq = (double)y8[i] * bscale[c] + bmin[c];
				err_y8 = fmax(err_y8, fabs(deq - want[i]));
				peak = fmax(peak, fabs((double)want[i]));
			}
		}
		/* Raw int8 conv matches exact within 0.1%; y8 block quantization within 0.5%. */
		bad += !(err <= 1e-3 * peak) || !(err_y8 <= 5e-3 * peak);
		printf("cpu_turbo_check: conv d=%2d z=%3d rel_err=%.2g y8_err=%.2g\n",
			cases[k].d, z, err / peak, err_y8 / peak);
	}
	printf("cpu_turbo_check: conv %s\n", bad ? "FAIL" : "ok");

#if defined(__x86_64__) || defined(_M_X64)
	if (mode == MT_MODE_X86_F16_AVX512 || mode == MT_MODE_X86_F16_AVX2) {
		uint16_t *hcodes = (uint16_t *)codes;
		for (int k = 0; k < 3; k++)
			for (size_t v = slab_low[k] * slice; v < (slab_high[k] + 1) * slice; v++) {
				float tmp[MN_CS];
				for (int c = 0; c < MN_CS; c++)
					tmp[c] = c < BC_MODEL_CHAN ? (((int)(RAND() % 8001) - 4000) * 0.001f) : 0.0f;
				mt_store_f16_x86(hcodes + v * MN_CS, tmp);
				mt_load_f16_x86(exact_in + v * MN_CS, hcodes + v * MN_CS);
			}
		int f16_bad = 0;
		int test_modes[2] = { MT_MODE_X86_F16_AVX2, mode == MT_MODE_X86_F16_AVX512 ? MT_MODE_X86_F16_AVX512 : 0 };
		for (int mi = 0; mi < 2; mi++) {
			int tm = test_modes[mi];
			if (!tm) continue;
			for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) {
				int z = cases[k].z;
				for (int y = 0; y < MN_DIM; y++)
					mt_conv_row_compute(codes, z, y, cases[k].d, wq, kernel, tap_sum, scale,
						block2, block1, tm, got + ((size_t)z * MN_DIM + (size_t)y) * MN_DIM * MN_CS);
				mn_conv_hidden_slab exact = { exact_in, want, kernel, cases[k].d, &region };
				mn_conv_hidden_rows(&exact, z, z);
				double err = 0, peak = 0;
				for (size_t i = (size_t)z * slice * MN_CS; i < (size_t)(z + 1) * slice * MN_CS; i++) {
					err = fmax(err, fabs((double)got[i] - want[i]));
					peak = fmax(peak, fabs((double)want[i]));
				}
				f16_bad += !(err <= 2e-5 * peak);
				printf("cpu_turbo_check: x86-f16(%s) d=%2d z=%3d rel_err=%.2g\n",
					tm == MT_MODE_X86_F16_AVX512 ? "avx512f" : "avx2", cases[k].d, z, err / peak);
			}
		}
		printf("cpu_turbo_check: x86-f16 %s\n", f16_bad ? "FAIL" : "ok");
		bad += f16_bad;
	}
#endif
#if MN_SME
	if (mt_sme_available()) {
		float16_t *half = (float16_t *)codes;
		memset(wq, 0, sizeof wq);
		for (size_t i = 0; i < sizeof kernel / sizeof *kernel; i++) kernel[i] = (float16_t)kernel[i];
		mt_prepare_sme(kernel, wq);
		for (int k = 0; k < 3; k++)
			for (size_t row = (size_t)slab_low[k] * MN_DIM; row < (size_t)(slab_high[k] + 1) * MN_DIM; row++)
				for (int x = 0; x < MN_DIM; x++) for (int c = 0; c < MN_CS; c++) {
					float16_t value = c < BC_MODEL_CHAN ? (float16_t)(((int)(RAND() % 8001) - 4000) * 0.001f) : 0;
					half[row * MN_DIM * MN_CS + (c / 2) * MN_DIM * 2 + x * 2 + c % 2] = value;
					exact_in[(row * MN_DIM + x) * MN_CS + c] = value;
				}
		for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) {
			int z = cases[k].z;
			for (int y = 0; y < MN_DIM; y++)
				mt_conv_row_compute(codes, z, y, cases[k].d, wq, kernel, tap_sum, scale, block2, block1, MT_MODE_SME,
					got + ((size_t)z * MN_DIM + (size_t)y) * MN_DIM * MN_CS);
			mn_conv_hidden_slab exact = { exact_in, want, kernel, cases[k].d, &region };
			mn_conv_hidden_rows(&exact, z, z);
			double err = 0, peak = 0;
			for (size_t i = (size_t)z * slice * MN_CS; i < (size_t)(z + 1) * slice * MN_CS; i++) {
				err = fmax(err, fabs((double)got[i] - want[i]));
				peak = fmax(peak, fabs((double)want[i]));
			}
			sme_bad += !(err <= 2e-5 * peak);
			printf("cpu_turbo_check: SME d=%2d z=%3d relative error %.2g\n", cases[k].d, z, err / peak);
		}
		printf("cpu_turbo_check: SME %s\n", sme_bad ? "FAIL" : "ok");
	} else printf("cpu_turbo_check: SME skipped (requires 512-bit Apple SME fp16)\n");
#endif
	free(codes); free(y8); free(ymom); free(exact_in); free(got); free(want);
	free(row_list); free(row_wt); free(partial); free(range);
	return mismatches != 0 || quant_bad != 0 || bad != 0 || sme_bad != 0;
}
