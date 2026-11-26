/*
 * CPU turbo (default; --no-turbo runs exact): compact hidden convolutions.
 * Included only by meshnet_cpu.c, after the exact engine, whose normalization
 * and exact path stay the fixture-gated reference.
 *
 * Activations and weights use hi + lo/128. Three byte dot products compute
 * hi*hi and hi*lo + lo*hi; omitting lo*lo/16384 keeps two accumulators.
 * GroupNorm/GELU run in float; intermediate pre-norm outputs are quantized
 * per 64-voxel block into uint8_t (`y8`) in L1 cache while fusing moments,
 * eliminating the 1.61 GB float `y` buffer (~2.42 GiB -> ~1.22 GiB at 24ch,
 * ~0.84 GiB at 16ch). Row work follows turbo_cone.h (pure background copies
 * and backward-cone exterior sampling).
 *
 * Runs on arm64 i8mm USDOT or ARMv8.2 DotProd SDOT (Apple M1+, Cortex-A76+),
 * and x86-64 AVX-512 VNNI, AVX-VNNI, or AVX2+FMA, with every integer kernel
 * bit-identical to the scalar reference (test/cpu_turbo_check.c). Apple SME
 * instead uses fp16 operands and fp32 sums, with the same two-byte storage.
 */
#ifndef BRAINCHOPC_MESHNET_TURBO_H
#define BRAINCHOPC_MESHNET_TURBO_H

#include "turbo_cone.h"
#include <float.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef AT_HWCAP2       /* glibc < 2.33 lacks the names, not the bits */
#define AT_HWCAP2 26
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#endif
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

/* GELU's minimum is -0.17. The residual byte is centered at 64. */
#define MT_ZP 7
#define MT_G (MN_CS / 4)
#define MT_AS (2 * MN_CS) /* per voxel: coarse channels, then residuals */
#define MT_RES_ZP 64
/* A u8 activation volume is read 16 bytes at a time past its last voxel. */
#define MT_ACT_PAD 16
/* 64-voxel block granule for compact pre-norm y8 storage (4 blocks/row). */
#define MT_YBLK 64
#define MT_YBLKS_PER_ROW (MN_DIM / MT_YBLK)
#define MT_YBLK_TOTAL ((size_t)BC_TC_ROWS * MT_YBLKS_PER_ROW)
#define MT_STAT_BINS 256

_Static_assert(MT_ZP != 0, "MT_ZP must be nonzero for sdot bias recovery");
_Static_assert(MN_DIM % MT_YBLK == 0, "MN_DIM must divide by MT_YBLK");

/*
 * Weights are split per output channel: w = (hi + lo / 128) * s, hi and lo s8,
 * ~15 bits. Plain s8 weights cost 1.5-3 points of agreement (docs/audit-history.md).
 * Per tap the hi block then the lo block, each [ic/4][oc][4]: four input
 * channels of one output channel adjacent, as USDOT/SDOT and VPDPBUSD/AVX2
 * consume them. corr includes both activation zero points. The low accumulator
 * contains both cross products; scale is [s oc..., s/128 oc...]. Every kernel
 * combines as fma(lo, s_lo, hi * s_hi), so all are bit-identical to scalar.
 */
#define MT_TAP (2 * MN_CS * MN_CS)
typedef void (*mt_block_fn)(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel,
	const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out);

static inline int mt_widx(int ic, int oc) { return ((ic >> 2) * MN_CS + oc) * 4 + (ic & 3); }

/* The reference every kernel must match bit for bit (test/cpu_turbo_check.c). */
#ifdef BC_CPU_TURBO_CHECK
#define MT_DEFINE_SCALAR(NB) \
static void mt_scalar_block##NB(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel, \
		const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out) { \
	for (int b = 0; b < NB; b++) \
		for (int oc = 0; oc < MN_CS; oc++) { \
			int32_t hi = 0, lo = 0; \
			for (int t = 0; t < ntap; t++) { \
				const uint8_t *in = src + (x + b + tap_voxel[t]) * MT_AS; \
				for (int ic = 0; ic < MN_CS; ic++) { \
					hi += tap_w[t][mt_widx(ic, oc)] * in[ic]; \
					lo += tap_w[t][MN_CS * MN_CS + mt_widx(ic, oc)] * in[ic]; \
					lo += tap_w[t][mt_widx(ic, oc)] * in[MN_CS + ic]; \
				} \
			} \
			out[b * MN_CS + oc] = fmaf((float)(lo - corr[MN_CS + oc]), scale[MN_CS + oc], \
				(float)(hi - corr[oc]) * scale[oc]); \
		} \
}
MT_DEFINE_SCALAR(1)
MT_DEFINE_SCALAR(2)
#endif

#if defined(__aarch64__)
#define MT_NEON_STEP(NB, g, lane) \
	for (int o = 0; o < MT_G; o++) { \
		int8x16_t wh = vld1q_s8(tw + ((g) * MN_CS + o * 4) * 4); \
		int8x16_t wl = vld1q_s8(tw + MN_CS * MN_CS + ((g) * MN_CS + o * 4) * 4); \
		for (int b = 0; b < NB; b++) { \
			hi[b][o] = vsudotq_laneq_s32(hi[b][o], wh, a[b], lane); \
			lo[b][o] = vsudotq_laneq_s32(lo[b][o], wl, a[b], lane); \
			lo[b][o] = vsudotq_laneq_s32(lo[b][o], wh, r[b], lane); \
		} \
	}
#if MN_CS == 16
#define MT_NEON_TAIL(NB)
#elif MN_CS == 24
/* Keep tail loads after the first 16 channels so the accumulators stay in registers. */
#define MT_NEON_TAIL(NB) \
	__asm__ volatile("" ::: "memory"); \
	for (int b = 0; b < NB; b++) { \
		a[b] = vld1q_u8(in + b * MT_AS + 16); \
		r[b] = vld1q_u8(in + b * MT_AS + MN_CS + 16); \
	} \
	MT_NEON_STEP(NB, 4, 0) MT_NEON_STEP(NB, 5, 1)
#endif
#define MT_DEFINE_NEON(NB) \
__attribute__((target("+i8mm"))) /* GCC needs the "+" */ \
static void mt_neon_block##NB(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel, \
		const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out) { \
	int32x4_t hi[NB][MT_G], lo[NB][MT_G]; \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_G; o++) hi[b][o] = lo[b][o] = vdupq_n_s32(0); \
	for (int t = 0; t < ntap; t++) { \
		const uint8_t *in = src + (x + tap_voxel[t]) * MT_AS; \
		const int8_t *tw = tap_w[t]; \
		uint8x16_t a[NB], r[NB]; \
		for (int b = 0; b < NB; b++) { \
			a[b] = vld1q_u8(in + b * MT_AS); \
			r[b] = vld1q_u8(in + b * MT_AS + MN_CS); \
		} \
		MT_NEON_STEP(NB, 0, 0) MT_NEON_STEP(NB, 1, 1) MT_NEON_STEP(NB, 2, 2) MT_NEON_STEP(NB, 3, 3) \
		MT_NEON_TAIL(NB) \
	} \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_G; o++) { \
		float32x4_t h = vcvtq_f32_s32(vsubq_s32(hi[b][o], vld1q_s32(corr + 4 * o))); \
		float32x4_t l = vcvtq_f32_s32(vsubq_s32(lo[b][o], vld1q_s32(corr + MN_CS + 4 * o))); \
		vst1q_f32(out + b * MN_CS + 4 * o, vfmaq_f32(vmulq_f32(h, vld1q_f32(scale + 4 * o)), \
			l, vld1q_f32(scale + MN_CS + 4 * o))); \
	} \
}
MT_DEFINE_NEON(1)
MT_DEFINE_NEON(2)

/*
 * ARMv8.2 DotProd (SDOT) kernel for CPUs without I8MM (Apple M1, Cortex-A76,
 * Graviton2). Flipping bit 7 of unsigned coarse byte `a` maps [0..255] exactly
 * to signed [-128..127] = (a - 128), while residual byte `r` is already in
 * [0..127]. Recovering sum_hi = corr_hi / MT_ZP and sum_lo = (corr_lo -
 * MT_RES_ZP * sum_hi) / MT_ZP folds the +128*sum_w compensation into the
 * epilogue with zero inner-loop overhead and bit-exact equality to scalar.
 */
#define MT_SDOT_STEP(NB, g, lane) \
	for (int o = 0; o < MT_G; o++) { \
		int8x16_t wh = vld1q_s8(tw + ((g) * MN_CS + o * 4) * 4); \
		int8x16_t wl = vld1q_s8(tw + MN_CS * MN_CS + ((g) * MN_CS + o * 4) * 4); \
		for (int b = 0; b < NB; b++) { \
			hi[b][o] = vdotq_laneq_s32(hi[b][o], wh, a[b], lane); \
			lo[b][o] = vdotq_laneq_s32(lo[b][o], wl, a[b], lane); \
			lo[b][o] = vdotq_laneq_s32(lo[b][o], wh, r[b], lane); \
		} \
	}
#if MN_CS == 16
#define MT_SDOT_TAIL(NB)
#elif MN_CS == 24
#define MT_SDOT_TAIL(NB) \
	__asm__ volatile("" ::: "memory"); \
	for (int b = 0; b < NB; b++) { \
		a[b] = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(in + b * MT_AS + 16), bias80)); \
		r[b] = vreinterpretq_s8_u8(vld1q_u8(in + b * MT_AS + MN_CS + 16)); \
	} \
	MT_SDOT_STEP(NB, 4, 0) MT_SDOT_STEP(NB, 5, 1)
#endif
#define MT_DEFINE_SDOT(NB) \
__attribute__((target("+dotprod"))) \
static void mt_sdot_block##NB(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel, \
		const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out) { \
	int32x4_t hi[NB][MT_G], lo[NB][MT_G]; \
	const uint8x16_t bias80 = vdupq_n_u8(0x80); \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_G; o++) hi[b][o] = lo[b][o] = vdupq_n_s32(0); \
	for (int t = 0; t < ntap; t++) { \
		const uint8_t *in = src + (x + tap_voxel[t]) * MT_AS; \
		const int8_t *tw = tap_w[t]; \
		int8x16_t a[NB], r[NB]; \
		for (int b = 0; b < NB; b++) { \
			a[b] = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(in + b * MT_AS), bias80)); \
			r[b] = vreinterpretq_s8_u8(vld1q_u8(in + b * MT_AS + MN_CS)); \
		} \
		MT_SDOT_STEP(NB, 0, 0) MT_SDOT_STEP(NB, 1, 1) MT_SDOT_STEP(NB, 2, 2) MT_SDOT_STEP(NB, 3, 3) \
		MT_SDOT_TAIL(NB) \
	} \
	int32_t adj_hi[MN_CS], adj_lo[MN_CS]; \
	for (int oc = 0; oc < MN_CS; oc++) { \
		int32_t sh = corr[oc] / MT_ZP; \
		int32_t sl = (corr[MN_CS + oc] - MT_RES_ZP * sh) / MT_ZP; \
		adj_hi[oc] = corr[oc] - 128 * sh; \
		adj_lo[oc] = corr[MN_CS + oc] - 128 * sl; \
	} \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_G; o++) { \
		float32x4_t h = vcvtq_f32_s32(vsubq_s32(hi[b][o], vld1q_s32(adj_hi + 4 * o))); \
		float32x4_t l = vcvtq_f32_s32(vsubq_s32(lo[b][o], vld1q_s32(adj_lo + 4 * o))); \
		vst1q_f32(out + b * MN_CS + 4 * o, vfmaq_f32(vmulq_f32(h, vld1q_f32(scale + 4 * o)), \
			l, vld1q_f32(scale + MN_CS + 4 * o))); \
	} \
}
MT_DEFINE_SDOT(1)
MT_DEFINE_SDOT(2)
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define MT_O8 (MN_CS / 8)
/* VPDPBUSD: the 4 input bytes of each voxel broadcast against 8 output
   channels x 4 inputs of weights. EVEX (AVX-512 VNNI + VL) or VEX (AVX-VNNI). */
#define MT_DEFINE_X86(NAME, TARGET, DPBUSD, NB) \
__attribute__((target(TARGET))) \
static void NAME##NB(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel, \
		const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out) { \
	__m256i hi[NB][MT_O8], lo[NB][MT_O8]; \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_O8; o++) hi[b][o] = lo[b][o] = _mm256_setzero_si256(); \
	for (int t = 0; t < ntap; t++) { \
		const uint8_t *in = src + (x + tap_voxel[t]) * MT_AS; \
		const int8_t *tw = tap_w[t]; \
		for (int g = 0; g < MT_G; g++) { \
			__m256i a[NB], r[NB]; \
			for (int b = 0; b < NB; b++) { \
				int32_t four; memcpy(&four, in + b * MT_AS + 4 * g, 4); \
				a[b] = _mm256_set1_epi32(four); \
				memcpy(&four, in + b * MT_AS + MN_CS + 4 * g, 4); \
				r[b] = _mm256_set1_epi32(four); \
			} \
			for (int o = 0; o < MT_O8; o++) { \
				const int8_t *w = tw + (g * MN_CS + 8 * o) * 4; \
				__m256i wh = _mm256_loadu_si256((const __m256i *)w); \
				__m256i wl = _mm256_loadu_si256((const __m256i *)(w + MN_CS * MN_CS)); \
				for (int b = 0; b < NB; b++) { \
					hi[b][o] = DPBUSD(hi[b][o], a[b], wh); \
					lo[b][o] = DPBUSD(lo[b][o], a[b], wl); \
					lo[b][o] = DPBUSD(lo[b][o], r[b], wh); \
				} \
			} \
		} \
	} \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_O8; o++) { \
		__m256 h = _mm256_cvtepi32_ps(_mm256_sub_epi32(hi[b][o], \
			_mm256_loadu_si256((const __m256i *)(corr + 8 * o)))); \
		__m256 l = _mm256_cvtepi32_ps(_mm256_sub_epi32(lo[b][o], \
			_mm256_loadu_si256((const __m256i *)(corr + MN_CS + 8 * o)))); \
		_mm256_storeu_ps(out + b * MN_CS + 8 * o, _mm256_fmadd_ps(l, \
			_mm256_loadu_ps(scale + MN_CS + 8 * o), _mm256_mul_ps(h, _mm256_loadu_ps(scale + 8 * o)))); \
	} \
}
MT_DEFINE_X86(mt_vnni512_block, "avx2,fma,avx512f,avx512vl,avx512vnni", _mm256_dpbusd_epi32, 1)
MT_DEFINE_X86(mt_vnni512_block, "avx2,fma,avx512f,avx512vl,avx512vnni", _mm256_dpbusd_epi32, 2)
MT_DEFINE_X86(mt_avxvnni_block, "avx2,fma,avxvnni", _mm256_dpbusd_avx_epi32, 1)
MT_DEFINE_X86(mt_avxvnni_block, "avx2,fma,avxvnni", _mm256_dpbusd_avx_epi32, 2)

/*
 * Overflow-free AVX2+FMA kernel for x86-64 CPUs without VNNI (Haswell through
 * Skylake-SP, Zen 1-3). vpmaddubsw saturates at int16 (+-32767), so raw a*wh
 * could saturate (255*127*2 = 64770). Splitting a = 2*(a>>1) + (a&1) keeps
 * every pair sum strictly in bounds:
 *   (a>>1)*wh <= 127*127*2 = 32258,  (a&1)*wh <= 1*127*2 = 254,
 *   a*wl      <= 255*64*2  = 32640,  r*wh     <= 127*127*2 = 32258.
 * Processing wh then wl keeps live YMM registers <= 16 at NB=2, CS=24.
 */
#define MT_DEFINE_AVX2(NB) \
__attribute__((target("avx2,fma"))) \
static void mt_avx2_block##NB(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel, \
		const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out) { \
	__m256i hi[NB][MT_O8], lo[NB][MT_O8]; \
	const __m256i ones16 = _mm256_set1_epi16(1); \
	const __m256i twos16 = _mm256_set1_epi16(2); \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_O8; o++) \
		hi[b][o] = lo[b][o] = _mm256_setzero_si256(); \
	for (int t = 0; t < ntap; t++) { \
		const uint8_t *in = src + (x + tap_voxel[t]) * MT_AS; \
		const int8_t *tw = tap_w[t]; \
		for (int g = 0; g < MT_G; g++) { \
			uint32_t af[NB], rf[NB]; \
			for (int b = 0; b < NB; b++) { \
				memcpy(&af[b], in + b * MT_AS + 4 * g, 4); \
				memcpy(&rf[b], in + b * MT_AS + MN_CS + 4 * g, 4); \
			} \
			for (int o = 0; o < MT_O8; o++) { \
				const int8_t *w = tw + (g * MN_CS + 8 * o) * 4; \
				__m256i wh = _mm256_loadu_si256((const __m256i *)w); \
				for (int b = 0; b < NB; b++) { \
					__m256i ah = _mm256_set1_epi32((int32_t)((af[b] >> 1) & 0x7f7f7f7fu)); \
					__m256i al = _mm256_set1_epi32((int32_t)(af[b] & 0x01010101u)); \
					__m256i r  = _mm256_set1_epi32((int32_t)rf[b]); \
					hi[b][o] = _mm256_add_epi32(hi[b][o], _mm256_madd_epi16(_mm256_maddubs_epi16(ah, wh), twos16)); \
					hi[b][o] = _mm256_add_epi32(hi[b][o], _mm256_madd_epi16(_mm256_maddubs_epi16(al, wh), ones16)); \
					lo[b][o] = _mm256_add_epi32(lo[b][o], _mm256_madd_epi16(_mm256_maddubs_epi16(r, wh), ones16)); \
				} \
				__m256i wl = _mm256_loadu_si256((const __m256i *)(w + MN_CS * MN_CS)); \
				for (int b = 0; b < NB; b++) { \
					__m256i a  = _mm256_set1_epi32((int32_t)af[b]); \
					lo[b][o] = _mm256_add_epi32(lo[b][o], _mm256_madd_epi16(_mm256_maddubs_epi16(a, wl), ones16)); \
				} \
			} \
		} \
	} \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_O8; o++) { \
		__m256 h = _mm256_cvtepi32_ps(_mm256_sub_epi32(hi[b][o], \
			_mm256_loadu_si256((const __m256i *)(corr + 8 * o)))); \
		__m256 l = _mm256_cvtepi32_ps(_mm256_sub_epi32(lo[b][o], \
			_mm256_loadu_si256((const __m256i *)(corr + MN_CS + 8 * o)))); \
		_mm256_storeu_ps(out + b * MN_CS + 8 * o, _mm256_fmadd_ps(l, \
			_mm256_loadu_ps(scale + MN_CS + 8 * o), _mm256_mul_ps(h, _mm256_loadu_ps(scale + 8 * o)))); \
	} \
}
MT_DEFINE_AVX2(1)
MT_DEFINE_AVX2(2)
#endif

/* The fast kernel this CPU runs (its name), or NULL: turbo then runs the
   exact engine. */
static const char *mt_select(mt_block_fn *block2, mt_block_fn *block1) {
#if defined(__aarch64__)
	int i8mm = 0, dotprod = 0;
#if defined(__APPLE__)
	size_t size = sizeof(i8mm);
	if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &i8mm, &size, NULL, 0)) i8mm = 0;
	size = sizeof(dotprod);
	if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &dotprod, &size, NULL, 0)) dotprod = 0;
#elif defined(__linux__)
	i8mm = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
	dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#endif
	if (i8mm) { *block2 = mt_neon_block2; *block1 = mt_neon_block1; return "i8mm"; }
	if (dotprod) { *block2 = mt_sdot_block2; *block1 = mt_sdot_block1; return "sdot"; }
#elif (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
	if (__builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl")) {
		*block2 = mt_vnni512_block2; *block1 = mt_vnni512_block1; return "avx512vnni";
	}
	/* Clang < 19 lacks the "avxvnni" builtin string: CPUID.(7,1):EAX[4].
	   The AVX2 test covers the OS's YMM state. */
	unsigned a, b, c, d;
	if (__builtin_cpu_supports("avx2") && __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u << 4))) {
		*block2 = mt_avxvnni_block2; *block1 = mt_avxvnni_block1; return "avxvnni";
	}
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		*block2 = mt_avx2_block2; *block1 = mt_avx2_block1; return "avx2";
	}
#endif
	(void)block2; (void)block1;
	return NULL;
}

static int mt_sme_available(void) {
	return mn_sme_available("hw.optional.arm.SME_F16F32");
}

#if MN_SME
/* 512-bit SME: 32 voxels, fp16 operands, fp32 sums. Channel pairs are
   contiguous across x so each matrix operand loads without a gather. */
__attribute__((target("sme"))) __arm_locally_streaming __arm_new("za")
static void mt_sme_segment(const uint8_t *src, int x0, int x1, const ptrdiff_t *tap_voxel,
		const int8_t *const *tap_w, int ntap, float *out) {
	svbool_t all = svptrue_b16(), floats = svptrue_b32();
#if MN_CS == 24
	svbool_t tail = svwhilelt_b16((uint64_t)0, (uint64_t)16);
	svbool_t ftail = svwhilelt_b32((uint64_t)0, (uint64_t)8);
#endif
	for (int x = x0; x < x1; x += 32) {
		int n = x1 - x < 32 ? x1 - x : 32;
		int second = n > 16 ? 32 : 0; /* Keep even inactive load addresses in bounds. */
		svbool_t p0 = svwhilelt_b16((uint64_t)0, (uint64_t)(n < 16 ? n : 16) * 2);
		svbool_t p1 = svwhilelt_b16((uint64_t)0, (uint64_t)(n > 16 ? n - 16 : 0) * 2);
		svzero_za();
		for (int t = 0; t < ntap; t++) {
			size_t v = (size_t)(tap_voxel[t] + x);
			const float16_t *in = (const float16_t *)src
				+ (v / MN_DIM) * MN_DIM * MN_CS + (v % MN_DIM) * 2;
			const float16_t *tw = (const float16_t *)tap_w[t];
			for (int g = 0; g < MN_CS / 2; g++) {
				svfloat16_t a = svld1_f16(p0, in + g * MN_DIM * 2);
				svfloat16_t b = svld1_f16(p1, in + g * MN_DIM * 2 + second);
				svfloat16_t w = svld1_f16(all, tw + g * MN_CS * 2);
				svmopa_za32_f16_m(0, p0, all, a, w);
				svmopa_za32_f16_m(2, p1, all, b, w);
#if MN_CS == 24
				w = svld1_f16(tail, tw + g * MN_CS * 2 + 32);
				svmopa_za32_f16_m(1, p0, tail, a, w);
				svmopa_za32_f16_m(3, p1, tail, b, w);
#endif
			}
		}
		for (int b = 0; b < (n < 16 ? n : 16); b++) {
			svst1_hor_za32(0, b, floats, out + (x + b) * MN_CS);
#if MN_CS == 24
			svst1_hor_za32(1, b, ftail, out + (x + b) * MN_CS + 16);
#endif
		}
		for (int b = 16; b < n; b++) {
			svst1_hor_za32(2, b - 16, floats, out + (x + b) * MN_CS);
#if MN_CS == 24
			svst1_hor_za32(3, b - 16, ftail, out + (x + b) * MN_CS + 16);
#endif
		}
	}
}

static void mt_prepare_sme(const float *kernel, int8_t *packed) {
	float16_t *w = (float16_t *)packed;
	/* Same byte count as hi/lo s8 weights; padded channels remain calloc zeros. */
	for (int t = 0; t < 27; t++)
		for (int ic = 0; ic < BC_MODEL_CHAN; ic++)
			for (int oc = 0; oc < MN_CS; oc++)
				w[t * MN_CS * MN_CS + ((ic / 2) * MN_CS + oc) * 2 + ic % 2]
					= (float16_t)kernel[((size_t)t * BC_MODEL_CHAN + ic) * MN_CS + oc];
}
#endif

/* ------------------------------------------- L1 epilogue: moments + y8 pack */

#define MT_STATS_BEGIN \
	double sum[MN_CS] = { 0 }, square[MN_CS] = { 0 }; \
	float low[MN_CS], high[MN_CS]; \
	for (int c = 0; c < MN_CS; c++) { low[c] = FLT_MAX; high[c] = -FLT_MAX; }
#define MT_STATS_END(partial, range, bin) \
	for (int c = 0; c < MN_CS; c++) { \
		(partial)[(size_t)(bin) * 2 * MN_CS + c] = sum[c]; \
		(partial)[(size_t)(bin) * 2 * MN_CS + MN_CS + c] = square[c]; \
		(range)[(size_t)(bin) * 2 * MN_CS + c] = low[c]; \
		(range)[(size_t)(bin) * 2 * MN_CS + MN_CS + c] = high[c]; \
	}

/* Quantize one L1-resident row [256][CS] into 64-voxel u8 blocks + [min,scale],
   while accumulating weighted GroupNorm moments and global [low,high]. */
static void mt_row_epilogue(const float *restrict row_buf, uint32_t row, float rw,
		uint8_t *restrict y8, float *restrict ymom,
		double *sum, double *square, float *low, float *high) {
	uint8_t *dst_row = y8 + (size_t)row * MN_DIM * MN_CS;
	float *mom_row = ymom + (size_t)row * MT_YBLKS_PER_ROW * 2 * MN_CS;
	for (int b = 0; b < MT_YBLKS_PER_ROW; b++) {
		const float *blk_in = row_buf + (size_t)b * MT_YBLK * MN_CS;
		uint8_t *blk_out = dst_row + (size_t)b * MT_YBLK * MN_CS;
		float *bmin = mom_row + (size_t)b * 2 * MN_CS;
		float *bscale = bmin + MN_CS;
		float bmax[MN_CS];
		double bs[MN_CS] = { 0 }, bq[MN_CS] = { 0 };
		for (int c = 0; c < MN_CS; c++) { bmin[c] = FLT_MAX; bmax[c] = -FLT_MAX; }
		for (int v = 0; v < MT_YBLK; v++) {
			const float *p = blk_in + v * MN_CS;
			for (int c = 0; c < MN_CS; c++) {
				float val = p[c];
				bs[c] += val;
				bq[c] += (double)val * val;
				if (val < bmin[c]) bmin[c] = val;
				if (val > bmax[c]) bmax[c] = val;
			}
		}
		float inv[MN_CS];
		for (int c = 0; c < MN_CS; c++) {
			if (rw > 0.0f) {
				sum[c] += (double)rw * bs[c];
				square[c] += (double)rw * bq[c];
			}
			if (bmin[c] < low[c]) low[c] = bmin[c];
			if (bmax[c] > high[c]) high[c] = bmax[c];
			float span = bmax[c] - bmin[c];
			if (span > 1e-30f) {
				bscale[c] = span * (1.0f / 255.0f);
				inv[c] = 255.0f / span;
			} else {
				bscale[c] = 0.0f;
				inv[c] = 0.0f;
			}
		}
		for (int v = 0; v < MT_YBLK; v++) {
			const float *p = blk_in + v * MN_CS;
			uint8_t *q = blk_out + v * MN_CS;
			for (int c = 0; c < MN_CS; c++) {
				int qi = (int)((p[c] - bmin[c]) * inv[c] + 0.5f);
				if (qi < 0) qi = 0;
				if (qi > 255) qi = 255;
				q[c] = (uint8_t)qi;
			}
		}
	}
}

/* ----------------------------------------------------- layer-0 convolution */

typedef struct {
	const float *source;
	uint8_t *y8;
	float *ymom;
	const float *weights;
	int dilation;
	const uint32_t *list;
	const float *rw;
	uint32_t list_off, count;
	double *partial;
	float *range;
} mt_conv0_slab;

static void mt_conv0_bins(void *context, int bin_low, int bin_high) {
	const mt_conv0_slab *s = (const mt_conv0_slab *)context;
	const float *restrict source = s->source;
	const float *restrict weights = s->weights;
	const int d = s->dilation;
	float row_buf[MN_DIM * MN_CS];
	for (int bin = bin_low; bin <= bin_high; bin++) {
		uint32_t i0 = (uint32_t)(((size_t)bin * s->count) / MT_STAT_BINS);
		uint32_t i1 = (uint32_t)(((size_t)(bin + 1) * s->count) / MT_STAT_BINS);
		MT_STATS_BEGIN
		for (uint32_t idx = i0; idx < i1; idx++) {
			uint32_t row = s->list[s->list_off + idx];
			int z = (int)(row >> 8), y = (int)(row & 255);
			ptrdiff_t row_offset[9];
			int row_tap[9], rows = 0;
			for (int tz = 0; tz < 3; tz++) {
				int zz = z + (tz - 1) * d;
				if (zz < 0 || zz >= MN_DIM) continue;
				for (int ty = 0; ty < 3; ty++) {
					int yy = y + (ty - 1) * d;
					if (yy < 0 || yy >= MN_DIM) continue;
					row_offset[rows] = ((ptrdiff_t)zz * MN_DIM + yy) * MN_DIM;
					row_tap[rows++] = (tz * 3 + ty) * 3;
				}
			}
			const int segment_x[4] = { 0, d, MN_DIM - d, MN_DIM };
			for (int segment = 0; segment < 3; segment++) {
				ptrdiff_t tap_voxel[27];
				const float *tap_w[27];
				int ntap = 0;
				for (int r = 0; r < rows; r++)
					for (int tx = segment == 0; tx <= 2 - (segment == 2); tx++) {
						tap_voxel[ntap] = row_offset[r] + (tx - 1) * d;
						tap_w[ntap++] = weights + (size_t)(row_tap[r] + tx) * MN_CS;
					}
				int x = segment_x[segment];
				const int x_end = segment_x[segment + 1];
				for (; x + MN_BLOCK <= x_end; x += MN_BLOCK) {
					float acc[MN_BLOCK][MN_CS];
					for (int b = 0; b < MN_BLOCK; b++)
						for (int c = 0; c < MN_CS; c++) acc[b][c] = 0.0f;
					for (int t = 0; t < ntap; t++) {
						const float *in = source + (size_t)(tap_voxel[t] + x);
						const float *wt = tap_w[t];
						float iv[MN_BLOCK];
						for (int b = 0; b < MN_BLOCK; b++) iv[b] = in[b];
						for (int c = 0; c < MN_CS; c++) {
							float wv = wt[c];
							for (int b = 0; b < MN_BLOCK; b++) acc[b][c] += iv[b] * wv;
						}
					}
					float *out = row_buf + (size_t)x * MN_CS;
					for (int b = 0; b < MN_BLOCK; b++)
						for (int c = 0; c < MN_CS; c++) out[b * MN_CS + c] = acc[b][c];
				}
				for (; x < x_end; x++) {
					float acc[MN_CS];
					for (int c = 0; c < MN_CS; c++) acc[c] = 0.0f;
					for (int t = 0; t < ntap; t++) {
						float iv = source[(size_t)(tap_voxel[t] + x)];
						const float *wt = tap_w[t];
						for (int c = 0; c < MN_CS; c++) acc[c] += iv * wt[c];
					}
					float *out = row_buf + (size_t)x * MN_CS;
					for (int c = 0; c < MN_CS; c++) out[c] = acc[c];
				}
			}
			mt_row_epilogue(row_buf, row, s->rw[s->list_off + idx], s->y8, s->ymom,
				sum, square, low, high);
		}
		MT_STATS_END(s->partial, s->range, bin)
	}
}

/* ----------------------------------------------------- hidden convolutions */

static void mt_conv_row_compute(const uint8_t *source, int z, int y, int d,
		const int8_t *weights, const int32_t *tap_sum, const float *scale,
		mt_block_fn block2, mt_block_fn block1, int sme, float *dst_row) {
	ptrdiff_t row_offset[9];
	int row_tap[9], rows = 0;
	for (int tz = 0; tz < 3; tz++) {
		int zz = z + (tz - 1) * d;
		if (zz < 0 || zz >= MN_DIM) continue;
		for (int ty = 0; ty < 3; ty++) {
			int yy = y + (ty - 1) * d;
			if (yy < 0 || yy >= MN_DIM) continue;
			row_offset[rows] = ((ptrdiff_t)zz * MN_DIM + yy) * MN_DIM;
			row_tap[rows++] = (tz * 3 + ty) * 3;
		}
	}
	const int segment_x[4] = { 0, d, MN_DIM - d, MN_DIM };
	for (int segment = 0; segment < 3; segment++) {
		ptrdiff_t tap_voxel[27];
		const int8_t *tap_w[27];
		int32_t corr[2 * MN_CS] = { 0 };
		int ntap = 0;
		for (int r = 0; r < rows; r++)
			for (int tx = segment == 0; tx <= 2 - (segment == 2); tx++) {
				int tap = row_tap[r] + tx;
				tap_voxel[ntap] = row_offset[r] + (tx - 1) * d;
				tap_w[ntap++] = weights + (size_t)tap * MT_TAP;
				if (!sme) /* SME has no zero points */
					for (int k = 0; k < 2 * MN_CS; k++) corr[k] += tap_sum[tap * 2 * MN_CS + k];
			}
#if MN_SME
		if (sme) {
			mt_sme_segment(source, segment_x[segment], segment_x[segment + 1],
				tap_voxel, tap_w, ntap, dst_row);
			continue;
		}
#else
		(void)sme;
#endif
		int x = segment_x[segment];
		for (; x + 2 <= segment_x[segment + 1]; x += 2)
			block2(source, (size_t)x, tap_voxel, tap_w, ntap, corr, scale, dst_row + (size_t)x * MN_CS);
		for (; x < segment_x[segment + 1]; x++)
			block1(source, (size_t)x, tap_voxel, tap_w, ntap, corr, scale, dst_row + (size_t)x * MN_CS);
	}
}

typedef struct {
	const uint8_t *source;
	uint8_t *y8;
	float *ymom;
	const int8_t *weights;   /* int8 planes, or SME fp16 [tap][ic/2][oc][2] */
	const int32_t *tap_sum;  /* [tap][hi oc, lo oc]: MT_ZP * sum of the tap's weights */
	const float *scale;      /* [hi oc, lo oc] */
	mt_block_fn block2, block1;
	int dilation;
	const uint32_t *list;
	const float *rw;
	uint32_t list_off, count;
	double *partial;
	float *range;
	int sme;
} mt_conv_slab;

static void mt_conv_bins(void *context, int bin_low, int bin_high) {
	const mt_conv_slab *s = (const mt_conv_slab *)context;
	float row_buf[MN_DIM * MN_CS];
	for (int bin = bin_low; bin <= bin_high; bin++) {
		uint32_t i0 = (uint32_t)(((size_t)bin * s->count) / MT_STAT_BINS);
		uint32_t i1 = (uint32_t)(((size_t)(bin + 1) * s->count) / MT_STAT_BINS);
		MT_STATS_BEGIN
		for (uint32_t idx = i0; idx < i1; idx++) {
			uint32_t row = s->list[s->list_off + idx];
			mt_conv_row_compute(s->source, (int)(row >> 8), (int)(row & 255), s->dilation,
				s->weights, s->tap_sum, s->scale, s->block2, s->block1, s->sme, row_buf);
			mt_row_epilogue(row_buf, row, s->rw[s->list_off + idx], s->y8, s->ymom,
				sum, square, low, high);
		}
		MT_STATS_END(s->partial, s->range, bin)
	}
}
_Static_assert(MN_DIM > 2 * 31, "x segments assume every dilation is under DIM/2");

/* Next layer's weights: float kernel times its input's qs, split per output
   channel into hi (+-127) and lo (+-64, the hi residual in 1/128 steps). */
static void mt_prepare(const float *kernel, const float *qs, int8_t *wq, int32_t *tap_sum,
		float *scale) {
	for (int oc = 0; oc < MN_CS; oc++) {
		float peak = 0.0f;
		for (int tap = 0; tap < 27; tap++)
			for (int ic = 0; ic < BC_MODEL_CHAN; ic++)
				peak = fmaxf(peak, fabsf(kernel[((size_t)tap * BC_MODEL_CHAN + ic) * MN_CS + oc] * qs[ic]));
		scale[oc] = peak > 0.0f ? peak / 127.0f : 1.0f;
		scale[MN_CS + oc] = scale[oc] / 128.0f;
	}
	/* Pad input channels stay zero from the calloc in mt_infer. */
	for (int tap = 0; tap < 27; tap++)
		for (int oc = 0; oc < MN_CS; oc++) {
			int32_t sum_hi = 0, sum_lo = 0;
			for (int ic = 0; ic < BC_MODEL_CHAN; ic++) {
				float w = kernel[((size_t)tap * BC_MODEL_CHAN + ic) * MN_CS + oc] * qs[ic] / scale[oc];
				int hi = (int)lrintf(w), lo = (int)lrintf((w - (float)hi) * 128.0f);
				if (hi < -127) hi = -127; else if (hi > 127) hi = 127;
				if (lo < -64) lo = -64; else if (lo > 64) lo = 64;
				int8_t *t = wq + (size_t)tap * MT_TAP;
				t[mt_widx(ic, oc)] = (int8_t)hi;
				t[MN_CS * MN_CS + mt_widx(ic, oc)] = (int8_t)lo;
				sum_hi += hi;
				sum_lo += lo;
			}
			tap_sum[tap * 2 * MN_CS + oc] = MT_ZP * sum_hi;
			tap_sum[tap * 2 * MN_CS + MN_CS + oc] = MT_ZP * sum_lo + MT_RES_ZP * sum_hi;
		}
}

/* ------------------------------------------- GroupNorm, GELU, quantize (y8) */

typedef struct {
	const uint8_t *y8;
	const float *ymom;
	uint8_t *codes;
	const float *mul, *add, *inv_qs;
	const uint32_t *list;
	uint32_t list_off, count;
} mt_quant_y8_slab;

static void mt_quant_y8_bins(void *context, int bin_low, int bin_high) {
	const mt_quant_y8_slab *s = (const mt_quant_y8_slab *)context;
	const uint8_t *restrict y8 = s->y8;
	const float *restrict ymom = s->ymom;
	uint8_t *restrict codes = s->codes;
	const float *restrict mul = s->mul, *restrict add = s->add, *restrict inv_qs = s->inv_qs;
	for (int bin = bin_low; bin <= bin_high; bin++) {
		uint32_t i0 = (uint32_t)(((size_t)bin * s->count) / MT_STAT_BINS);
		uint32_t i1 = (uint32_t)(((size_t)(bin + 1) * s->count) / MT_STAT_BINS);
		for (uint32_t idx = i0; idx < i1; idx++) {
			uint32_t row = s->list[s->list_off + idx];
			const uint8_t *src_row = y8 + (size_t)row * MN_DIM * MN_CS;
			const float *mom_row = ymom + (size_t)row * MT_YBLKS_PER_ROW * 2 * MN_CS;
			uint8_t *dst_row = codes + (size_t)row * MN_DIM * MT_AS;
			for (int b = 0; b < MT_YBLKS_PER_ROW; b++) {
				const float *bmin = mom_row + (size_t)b * 2 * MN_CS;
				const float *bscale = bmin + MN_CS;
				float bmul[MN_CS], badd[MN_CS];
				for (int c = 0; c < MN_CS; c++) {
					bmul[c] = bscale[c] * mul[c];
					badd[c] = bmin[c] * mul[c] + add[c];
				}
				const uint8_t *src_blk = src_row + (size_t)b * MT_YBLK * MN_CS;
				uint8_t *dst_blk = dst_row + (size_t)b * MT_YBLK * MT_AS;
				for (int v = 0; v < MT_YBLK; v++) {
					const uint8_t *sv = src_blk + v * MN_CS;
					uint8_t *dv = dst_blk + v * MT_AS;
					for (int c = 0; c < MN_CS; c++) {
						float code = mn_gelu((float)sv[c] * bmul[c] + badd[c]) * inv_qs[c] * 128.0f
							+ MT_ZP * 128.0f + 0.5f;
						unsigned q = (unsigned)(code < 32703.0f ? code : 32703.0f);
						dv[c] = (uint8_t)((q + MT_RES_ZP) >> 7);
						dv[c + MN_CS] = (uint8_t)((q + MT_RES_ZP) & 127);
					}
				}
			}
		}
	}
}

#if MN_SME
static void mt_quant_sme_y8_bins(void *context, int bin_low, int bin_high) {
	const mt_quant_y8_slab *s = (const mt_quant_y8_slab *)context;
	const uint8_t *restrict y8 = s->y8;
	const float *restrict ymom = s->ymom;
	float16_t *restrict codes = (float16_t *)s->codes;
	const float *restrict mul = s->mul, *restrict add = s->add;
	float act_row[MN_DIM * MN_CS];
	for (int bin = bin_low; bin <= bin_high; bin++) {
		uint32_t i0 = (uint32_t)(((size_t)bin * s->count) / MT_STAT_BINS);
		uint32_t i1 = (uint32_t)(((size_t)(bin + 1) * s->count) / MT_STAT_BINS);
		for (uint32_t idx = i0; idx < i1; idx++) {
			uint32_t row = s->list[s->list_off + idx];
			const uint8_t *src_row = y8 + (size_t)row * MN_DIM * MN_CS;
			const float *mom_row = ymom + (size_t)row * MT_YBLKS_PER_ROW * 2 * MN_CS;
			for (int b = 0; b < MT_YBLKS_PER_ROW; b++) {
				const float *bmin = mom_row + (size_t)b * 2 * MN_CS;
				const float *bscale = bmin + MN_CS;
				float bmul[MN_CS], badd[MN_CS];
				for (int c = 0; c < MN_CS; c++) {
					bmul[c] = bscale[c] * mul[c];
					badd[c] = bmin[c] * mul[c] + add[c];
				}
				const uint8_t *src_blk = src_row + (size_t)b * MT_YBLK * MN_CS;
				float *dst_blk = act_row + (size_t)b * MT_YBLK * MN_CS;
				for (int v = 0; v < MT_YBLK; v++) {
					const uint8_t *sv = src_blk + v * MN_CS;
					float *dv = dst_blk + v * MN_CS;
					for (int c = 0; c < MN_CS; c++)
						dv[c] = mn_gelu((float)sv[c] * bmul[c] + badd[c]);
				}
			}
			float16_t *out = codes + (size_t)row * MN_DIM * MN_CS;
			for (int g = 0; g < MN_CS / 2; g++)
				for (int x = 0; x < MN_DIM; x++) {
					out[g * MN_DIM * 2 + x * 2]     = (float16_t)act_row[x * MN_CS + g * 2];
					out[g * MN_DIM * 2 + x * 2 + 1] = (float16_t)act_row[x * MN_CS + g * 2 + 1];
				}
		}
	}
}
#endif

typedef struct {
	uint8_t *buf;
	size_t row_bytes;
	const uint32_t *list;
	uint32_t pure_off, npure, rep;
} mt_fill_rows_ctx;

static void mt_fill_rows(void *context, int lo, int hi) {
	const mt_fill_rows_ctx *f = (const mt_fill_rows_ctx *)context;
	const uint8_t *src = f->buf + (size_t)f->rep * f->row_bytes;
	for (int i = lo; i <= hi; i++) {
		uint32_t dst_row = f->list[f->pure_off + (uint32_t)i];
		memcpy(f->buf + (size_t)dst_row * f->row_bytes, src, f->row_bytes);
	}
}

static void mt_fold_stats(const double *partial, const float *range,
		const meshnet_cpu_plan *plan, int layer, float *mul, float *add,
		float *qs, float *inv_qs) {
	for (int c = 0; c < MN_CS; c++) {
		double sum = 0.0, square = 0.0;
		float low = FLT_MAX, high = -FLT_MAX;
		for (int bin = 0; bin < MT_STAT_BINS; bin++) {
			sum += partial[(size_t)bin * 2 * MN_CS + c];
			square += partial[(size_t)bin * 2 * MN_CS + MN_CS + c];
			low = fminf(low, range[(size_t)bin * 2 * MN_CS + c]);
			high = fmaxf(high, range[(size_t)bin * 2 * MN_CS + MN_CS + c]);
		}
		double mean = sum / (double)MN_NVOX, variance = square / (double)MN_NVOX - mean * mean;
		if (!(variance > 0.0)) variance = 0.0;
		float invstd = (float)(1.0 / sqrt(variance + MN_EPS)), gamma = 1.0f, beta = 0.0f;
#if BC_MODEL_HAS_AFFINE
		gamma = plan->scale[layer][c];
		beta = c < BC_MODEL_CHAN ? plan->bias[layer][c] : 0.0f;
#else
		(void)plan; (void)layer;
#endif
		mul[c] = invstd * gamma;
		add[c] = beta - (float)mean * mul[c];
		if (qs && inv_qs) {
			float vmax = fmaxf(0.25f, fmaxf(mn_gelu(low * mul[c] + add[c]), mn_gelu(high * mul[c] + add[c])));
			qs[c] = c < BC_MODEL_CHAN ? fmaxf(vmax / (255.0f - MT_ZP), 0.1705f / MT_ZP) : 1.0f;
			inv_qs[c] = 1.0f / qs[c];
		}
	}
}

static void mt_norm_quant_y8(const uint8_t *y8, const float *ymom, uint8_t *codes,
		const double *partial, const float *range, const meshnet_cpu_plan *plan, int layer,
		const bc_tc_plan *tc, float *qs, int sme) {
	float mul[MN_CS], add[MN_CS], inv_qs[MN_CS];
	mt_fold_stats(partial, range, plan, layer, mul, add, qs, inv_qs);
	const bc_tc_layer *t = &tc->layer[layer];
	mt_quant_y8_slab quant = { y8, ymom, codes, mul, add, inv_qs, tc->list, t->list_off, t->count };
#if MN_SME
	if (sme)
		bc_parallel_for(0, MT_STAT_BINS - 1, 1, mt_quant_sme_y8_bins, &quant);
	else
#else
	(void)sme;
#endif
		bc_parallel_for(0, MT_STAT_BINS - 1, 1, mt_quant_y8_bins, &quant);
	if (t->npure) {
		/* Both int8 (2*CS u8/vox) and SME (CS fp16/vox) use MN_DIM * MT_AS bytes per row. */
		mt_fill_rows_ctx fill = { codes, (size_t)MN_DIM * MT_AS, tc->list, t->pure_off, t->npure, t->rep };
		bc_parallel_for(0, (int)t->npure - 1, 0, mt_fill_rows, &fill);
	}
}

/* ------------------------------------------- fused last GroupNorm + classify */

typedef struct {
	const uint8_t *y8;
	const float *ymom;
	float *labels;
	const float *mul, *add;
	const float *weights;
#if BC_MODEL_CLASSIFIER_BIAS
	const float *bias;
#endif
	const uint32_t *list;
	uint32_t cls_off, ncls;
} mt_classify_y8_slab;

static void mt_classify_y8_bins(void *context, int bin_low, int bin_high) {
	const mt_classify_y8_slab *s = (const mt_classify_y8_slab *)context;
	const uint8_t *restrict y8 = s->y8;
	const float *restrict ymom = s->ymom;
	float *restrict labels = s->labels;
	const float *restrict mul = s->mul, *restrict add = s->add;
	const float *restrict weights = s->weights;
#if BC_MODEL_CLASSIFIER_BIAS
	const float *restrict bias = s->bias;
#endif
	for (int bin = bin_low; bin <= bin_high; bin++) {
		uint32_t i0 = (uint32_t)(((size_t)bin * s->ncls) / MT_STAT_BINS);
		uint32_t i1 = (uint32_t)(((size_t)(bin + 1) * s->ncls) / MT_STAT_BINS);
		for (uint32_t idx = i0; idx < i1; idx++) {
			uint32_t row = s->list[s->cls_off + idx];
			const uint8_t *src_row = y8 + (size_t)row * MN_DIM * MN_CS;
			const float *mom_row = ymom + (size_t)row * MT_YBLKS_PER_ROW * 2 * MN_CS;
			float *lbl_row = labels + (size_t)row * MN_DIM;
			for (int b = 0; b < MT_YBLKS_PER_ROW; b++) {
				const float *bmin = mom_row + (size_t)b * 2 * MN_CS;
				const float *bscale = bmin + MN_CS;
				float bmul[MN_CS], badd[MN_CS];
				for (int c = 0; c < MN_CS; c++) {
					bmul[c] = bscale[c] * mul[c];
					badd[c] = bmin[c] * mul[c] + add[c];
				}
				const uint8_t *src_blk = src_row + (size_t)b * MT_YBLK * MN_CS;
				float *lbl_blk = lbl_row + (size_t)b * MT_YBLK;
				for (int v = 0; v < MT_YBLK; v++) {
					const uint8_t *sv = src_blk + v * MN_CS;
					float val[MN_CS];
					for (int c = 0; c < MN_CS; c++)
						val[c] = mn_gelu((float)sv[c] * bmul[c] + badd[c]);
					float logit[BC_MODEL_NCLASS];
					for (int k = 0; k < BC_MODEL_NCLASS; k++)
#if BC_MODEL_CLASSIFIER_BIAS
						logit[k] = bias[k];
#else
						logit[k] = 0.0f;
#endif
					for (int c = 0; c < BC_MODEL_CHAN; c++)
						for (int k = 0; k < BC_MODEL_NCLASS; k++)
							logit[k] += val[c] * weights[c * BC_MODEL_NCLASS + k];
					int best = 0;
					float bv = -INFINITY;
					for (int k = 0; k < BC_MODEL_NCLASS; k++)
						if (logit[k] > bv) { bv = logit[k]; best = k; }
					lbl_blk[v] = (float)best;
				}
			}
		}
	}
}

/* ---------------------------------------------------------------- the run */

static int mt_infer(const unsigned char *conformed, float *labels, const meshnet_cpu_plan *plan,
		mt_block_fn block2, mt_block_fn block1, const char *kernel) {
	const int sme = mt_sme_available();
	printfx("%s: CPU turbo (%s, %s)\n", BC_MODEL_NAME, sme ? "fp16" : "int8", sme ? "SME" : kernel);
	bc_tc_plan tc;
	if (bc_tc_make(conformed, plan->dilation, &tc)) return 1;
	mn_region region;
	mn_make_region(NULL, &region);
	/* Compact y8 + 64-voxel block moments replace the 1.61 GB float y buffer. */
	uint8_t *y8 = (uint8_t *)malloc(MN_NVOX * MN_CS);
	float *ymom = (float *)malloc(MT_YBLK_TOTAL * 2 * MN_CS * sizeof(*ymom));
	/* The codes buffer first holds the normalized fp32 input (CS >= 4 bytes). */
	uint8_t *codes = (uint8_t *)malloc(MN_NVOX * MT_AS + MT_ACT_PAD);
	double *partial = (double *)malloc((size_t)MT_STAT_BINS * 2 * MN_CS * sizeof(*partial));
	float *range = (float *)malloc((size_t)MT_STAT_BINS * 2 * MN_CS * sizeof(*range));
	int8_t *wq = (int8_t *)calloc((size_t)27 * MT_TAP, 1);
	int32_t tap_sum[27 * 2 * MN_CS] = { 0 };
	float scale[2 * MN_CS], qs[MN_CS];
	if (!y8 || !ymom || !codes || !partial || !range || !wq) {
		printfx("%s: out of memory (turbo buffers need about %.1f GB)\n", BC_MODEL_NAME,
			((double)MN_NVOX * MN_CS * 3.0 + (double)MT_YBLK_TOTAL * 2 * MN_CS * sizeof(float)) / 1e9);
		free(y8); free(ymom); free(codes); free(partial); free(range); free(wq);
		bc_tc_free(&tc);
		return 1;
	}
	memset(codes + MN_NVOX * MT_AS, 0, MT_ACT_PAD);
	memset(labels, 0, MN_NVOX * sizeof(*labels));
	mn_normalize(conformed, (float *)codes, &region);
	{
		const bc_tc_layer *t0 = &tc.layer[0];
		mt_conv0_slab conv0 = { (const float *)codes, y8, ymom, plan->kernel[0], plan->dilation[0],
			tc.list, tc.weight, t0->list_off, t0->count, partial, range };
		bc_parallel_for(0, MT_STAT_BINS - 1, 1, mt_conv0_bins, &conv0);
	}
	mt_norm_quant_y8(y8, ymom, codes, partial, range, plan, 0, &tc, qs, sme);
	for (int layer = 1; layer < plan->layer_count; layer++) {
#if MN_SME
		if (sme) mt_prepare_sme(plan->kernel[layer], wq);
		else
#endif
			mt_prepare(plan->kernel[layer], qs, wq, tap_sum, scale);
		const bc_tc_layer *tl = &tc.layer[layer];
		mt_conv_slab conv = { codes, y8, ymom, wq, tap_sum, scale, block2, block1,
			plan->dilation[layer], tc.list, tc.weight, tl->list_off, tl->count,
			partial, range, sme };
		bc_parallel_for(0, MT_STAT_BINS - 1, 1, mt_conv_bins, &conv);
		if (layer + 1 < plan->layer_count)
			mt_norm_quant_y8(y8, ymom, codes, partial, range, plan, layer, &tc, qs, sme);
	}
	const int last = plan->layer_count - 1;
	float mul[MN_CS], add[MN_CS];
	mt_fold_stats(partial, range, plan, last, mul, add, NULL, NULL);
	mt_classify_y8_slab cls = { y8, ymom, labels, mul, add, plan->classifier_weights,
#if BC_MODEL_CLASSIFIER_BIAS
		plan->classifier_bias,
#endif
		tc.list, tc.cls_off, tc.ncls };
	bc_parallel_for(0, MT_STAT_BINS - 1, 1, mt_classify_y8_bins, &cls);
	if (tc.layer[last].npure) {
		mt_fill_rows_ctx fill = { (uint8_t *)labels, (size_t)MN_DIM * sizeof(float),
			tc.list, tc.layer[last].pure_off, tc.layer[last].npure, tc.layer[last].rep };
		bc_parallel_for(0, (int)tc.layer[last].npure - 1, 0, mt_fill_rows, &fill);
	}
	free(y8); free(ymom); free(codes); free(partial); free(range); free(wq);
	bc_tc_free(&tc);
	return 0;
}

#endif
