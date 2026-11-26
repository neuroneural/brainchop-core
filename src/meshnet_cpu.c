/* Shared, fixed-shape CPU inference kernels for the supported MeshNet models. */

#include "cat_lite.h"
#include "meshnet_cpu.h"
#include "bc_threads.h"
#include "print.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MN_DIM BRAINCHOP_MODEL_DIM
#define MN_NVOX ((size_t)MN_DIM * MN_DIM * MN_DIM)
#define MN_CS MESHNET_CPU_CHANNEL_STRIDE
#define MN_BLOCK 4
#define MN_EPS 1e-5

/* The packed kernel layout zero-pads each model's channels up to MN_CS, so both
   must fit. This replaces a runtime guard in meshnet_cpu_pack_kernel. */
_Static_assert(BC_MODEL_CHAN >= 1 && BC_MODEL_CHAN <= MN_CS,
	"this model's channels must fit the packed channel stride");
_Static_assert(BC_MODEL_NHIDDEN <= MESHNET_CPU_MAX_LAYERS,
	"layer count must fit the plan's kernel array");
_Static_assert(BC_MODEL_NCLASS >= 2, "a classifier needs at least two classes");

typedef struct {
	int x_low, x_high;
	int y_low, y_high;
	int z_low, z_high;
	size_t count;
} mn_region;

/*
 * Dynamic scheduling: the z slices are equal-cost, but hybrid CPUs mix fast and
 * slow cores, so a static split leaves the efficiency cores as stragglers.
 * Handing out small chunks on demand costs nothing at this granularity and
 * removes the tail. Every loop below writes only its own z slice, so the
 * schedule never affects results.
 */
#define MN_GRAIN 4

/*
 * Each loop below is split into a `_rows` function taking an inclusive z range
 * and a wrapper that hands that range to the pool. That shape is not a style
 * choice: the pool has to call the body from several threads, so the body must
 * be a function. It is the same outlining the OpenMP version did behind the
 * pragma -- clang emitted `.omp_outlined.` companions, which is what the
 * ThreadSanitizer traces in the build notes name -- so the code generation this file
 * is tuned around is unchanged. Verify that with instruction counts, per the
 * rule at the top of this project, not with a clock.
 */

float *meshnet_cpu_pack_kernel(const float *source, int input_channels,
		int logical_channels, int transpose_spatial) {
	size_t count = (size_t)27 * input_channels * MN_CS;
	float *packed = (float *)calloc(count, sizeof(*packed));
	if (!packed) return NULL;
	for (int dz = -1; dz <= 1; dz++)
		for (int dy = -1; dy <= 1; dy++)
			for (int dx = -1; dx <= 1; dx++) {
				int dst_tap = (dz + 1) * 9 + (dy + 1) * 3 + dx + 1;
				int src_tap = transpose_spatial
					? (dx + 1) * 9 + (dy + 1) * 3 + dz + 1
					: dst_tap;
				for (int ic = 0; ic < input_channels; ic++) {
					const float *src = source +
						((size_t)src_tap * input_channels + ic) * logical_channels;
					float *dst = packed +
						((size_t)dst_tap * input_channels + ic) * MN_CS;
					memcpy(dst, src, (size_t)logical_channels * sizeof(*dst));
				}
			}
	return packed;
}

void meshnet_cpu_destroy_plan(meshnet_cpu_plan *plan) {
	if (!plan) return;
	for (int layer = 0; layer < MESHNET_CPU_MAX_LAYERS; layer++) {
		free(plan->kernel[layer]);
		plan->kernel[layer] = NULL;
	}
}

/* -------------------------------------------------------- input normalization */

static void mn_histogram(const unsigned char *source, int64_t *cum,
		const mn_region *region) {
	int64_t hist[256] = {0};
	for (int z = region->z_low; z <= region->z_high; z++)
		for (int y = region->y_low; y <= region->y_high; y++)
			for (int x = region->x_low; x <= region->x_high; x++) {
				size_t i = (size_t)x + (size_t)y * MN_DIM +
					(size_t)z * MN_DIM * MN_DIM;
				hist[source[i]]++;
			}
	int64_t run = 0;
	for (int i = 0; i < 256; i++) {
		run += hist[i];
		cum[i] = run;
	}
}

#if BC_MODEL_QUANTILE_MODE == 0
/* numpy's default linear quantile, including its numerically stable lerp
   branch. Selected by quantile_mode in model_meta.json. */
static double mn_quantile(const int64_t *cum, size_t count, double q, int upper) {
	(void)upper;
	double rank = q * (double)(count - 1);
	double base = floor(rank);
	double t = rank - base;
	size_t i = (size_t)base;
	int a = 0;
	while (a < 255 && cum[a] <= (int64_t)i) a++;
	int b = a;
	while (b < 255 && cum[b] <= (int64_t)(i + 1)) b++;
	if (t < 0.5) return (double)a + (double)(b - a) * t;
	return (double)b - (double)(b - a) * (1.0 - t);
}
#else
/* Plain rank selection. `upper` says which end this is, rather than inferring
   it from the value: the upper rank uses ceil()-1 and the lower a plain
   truncation, and guessing from `q > 0.5` would silently pick the wrong form
   for a model whose quantile pair sits entirely above or below the midpoint. */
static double mn_quantile(const int64_t *cum, size_t count, double q, int upper) {
	double raw = upper ? ceil((double)count * q) : (double)count * q;
	size_t rank = upper ? (raw > 0.0 ? (size_t)raw - 1 : 0) : (size_t)raw;
	if (rank >= count) rank = count - 1;
	int bin = 0;
	while (bin < 255 && cum[bin] <= (int64_t)rank) bin++;
	return (double)bin;
}
#endif

/*
 * One normalization for every model. The quantiles, the denominator guard and
 * whether the result is clamped all come from the descriptor, so a model that
 * normalizes differently is a metadata change rather than a code change.
 */
typedef struct {
	const unsigned char *source;
	float *destination;
	float low, denominator;
	const mn_region *region;
} mn_normalize_slab;

static void mn_normalize_rows(void *context, int z_low, int z_high) {
	const mn_normalize_slab *slab = (const mn_normalize_slab *)context;
	const unsigned char *source = slab->source;
	float *destination = slab->destination;
	float low = slab->low, denominator = slab->denominator;
	const mn_region *region = slab->region;
	for (int z = z_low; z <= z_high; z++) {
		for (int y = region->y_low; y <= region->y_high; y++)
		for (int x = region->x_low; x <= region->x_high; x++) {
			size_t i = (size_t)x + (size_t)y * MN_DIM +
				(size_t)z * MN_DIM * MN_DIM;
			float value = ((float)source[i] - low) / denominator;
#if BC_MODEL_NORM_CLAMP
			/* Written as a negated comparison so a NaN lands in range too. */
			if (!(value > 0.0f)) value = 0.0f;
			if (value > 1.0f) value = 1.0f;
#endif
			destination[i] = value;
		}
	}
}

static void mn_normalize(const unsigned char *source, float *destination,
		const mn_region *region) {
	int64_t cum[256];
	mn_histogram(source, cum, region);
	float low = (float)mn_quantile(cum, region->count, BC_MODEL_NORM_Q_LOW, 0);
	float high = (float)mn_quantile(cum, region->count, BC_MODEL_NORM_Q_HIGH, 1);
	if (high <= low && cum[0] < (int64_t)region->count) {
		int64_t fg_cum[256];
		fg_cum[0] = 0;
		for (int i = 1; i < 256; i++) fg_cum[i] = cum[i] - cum[0];
		size_t fg_n = region->count - (size_t)cum[0];
		low = (float)mn_quantile(fg_cum, fg_n, BC_MODEL_NORM_Q_LOW, 0);
		high = (float)mn_quantile(fg_cum, fg_n, BC_MODEL_NORM_Q_HIGH, 1);
		if (high <= low && high > 0.0f) low = 0.0f;
	}
	float denominator = high - low + BC_MODEL_NORM_DENOM_EPS;
	if (!(denominator > 0.0f)) denominator = 1.0f;
	mn_normalize_slab slab = { .source = source, .destination = destination,
		.low = low, .denominator = denominator, .region = region };
	bc_parallel_for(region->z_low, region->z_high, MN_GRAIN,
		mn_normalize_rows, &slab);
}

/* ------------------------------------------------------------ 3-D convolution */

/*
 * Generate only the three shapes the models use. Keeping the channel counts
 * literal lets Clang retain the four 128-bit output vectors in registers. Each
 * weight vector is shared by four adjacent x voxels. Boundary segments remove
 * all bounds tests from the accumulation loop.
 */
#define MN_DEFINE_CONV(NAME, SOURCE_STRIDE, INPUT_CHANNELS) \
typedef struct { \
	const float *source; \
	float *destination; \
	const float *weights; \
	int dilation; \
	const mn_region *region; \
} NAME##_slab; \
static void NAME##_rows(void *context, int z_low, int z_high) { \
	const NAME##_slab *slab = (const NAME##_slab *)context; \
	const float *restrict source = slab->source; \
	float *restrict destination = slab->destination; \
	const float *restrict weights = slab->weights; \
	int dilation = slab->dilation; \
	const mn_region *region = slab->region; \
	const size_t weight_layer = (size_t)(INPUT_CHANNELS) * MN_CS; \
	for (int z = z_low; z <= z_high; z++) { \
		for (int y = region->y_low; y <= region->y_high; y++) { \
			ptrdiff_t row_offset[9]; \
			int row_tap[9]; \
			int row_count = 0; \
			for (int tz = 0; tz < 3; tz++) { \
				int zz = z + (tz - 1) * dilation; \
				if (zz < region->z_low || zz > region->z_high) continue; \
				for (int ty = 0; ty < 3; ty++) { \
					int yy = y + (ty - 1) * dilation; \
					if (yy < region->y_low || yy > region->y_high) continue; \
					row_offset[row_count] = ((ptrdiff_t)zz * MN_DIM + yy) * MN_DIM; \
					row_tap[row_count] = (tz * 3 + ty) * 3; \
					row_count++; \
				} \
			} \
			size_t destination_row = ((size_t)z * MN_DIM + y) * MN_DIM; \
			if (region->x_high - region->x_low + 1 <= 2 * dilation) { \
				for (int x = region->x_low; x <= region->x_high; x++) { \
					float accumulator[MN_CS]; \
					for (int channel = 0; channel < MN_CS; channel++) \
						accumulator[channel] = 0.0f; \
					for (int row = 0; row < row_count; row++) \
					for (int tx = 0; tx < 3; tx++) { \
						int xx = x + (tx - 1) * dilation; \
						if (xx < region->x_low || xx > region->x_high) continue; \
						const float *input = source + \
							(size_t)(row_offset[row] + xx) * (SOURCE_STRIDE); \
						const float *tap_weights = weights + \
							(size_t)(row_tap[row] + tx) * weight_layer; \
						for (int input_channel = 0; input_channel < (INPUT_CHANNELS); \
								input_channel++) { \
							float value = input[input_channel]; \
							for (int channel = 0; channel < MN_CS; channel++) \
								accumulator[channel] += value * \
									tap_weights[(size_t)input_channel * MN_CS + channel]; \
						} \
					} \
					float *output = destination + (destination_row + x) * MN_CS; \
					for (int channel = 0; channel < MN_CS; channel++) \
						output[channel] = accumulator[channel]; \
				} \
				continue; \
			} \
			int segment_x[4] = {region->x_low, region->x_low + dilation, \
				region->x_high + 1 - dilation, region->x_high + 1}; \
			for (int segment = 0; segment < 3; segment++) { \
				int tap_x_low = segment == 0 ? 1 : 0; \
				int tap_x_high = segment == 2 ? 1 : 2; \
				ptrdiff_t tap_voxel[27]; \
				const float *tap_weight[27]; \
				int tap_count = 0; \
				for (int row = 0; row < row_count; row++) \
					for (int tx = tap_x_low; tx <= tap_x_high; tx++) { \
						tap_voxel[tap_count] = row_offset[row] + (tx - 1) * dilation; \
						tap_weight[tap_count] = weights + \
							(size_t)(row_tap[row] + tx) * weight_layer; \
						tap_count++; \
					} \
				int x = segment_x[segment]; \
				const int x_end = segment_x[segment + 1]; \
				for (; x + MN_BLOCK <= x_end; x += MN_BLOCK) { \
					float accumulator[MN_BLOCK][MN_CS]; \
					for (int block = 0; block < MN_BLOCK; block++) \
						for (int channel = 0; channel < MN_CS; channel++) \
							accumulator[block][channel] = 0.0f; \
					for (int tap = 0; tap < tap_count; tap++) { \
						const float *input = source + \
							(size_t)(tap_voxel[tap] + x) * (SOURCE_STRIDE); \
						const float *tap_weights = tap_weight[tap]; \
						for (int input_channel = 0; input_channel < (INPUT_CHANNELS); \
								input_channel++) { \
							const float *weight_vector = tap_weights + \
								(size_t)input_channel * MN_CS; \
							float input_value[MN_BLOCK]; \
							for (int block = 0; block < MN_BLOCK; block++) \
								input_value[block] = input[block * (SOURCE_STRIDE) + input_channel]; \
							for (int channel = 0; channel < MN_CS; channel++) { \
								float weight = weight_vector[channel]; \
								for (int block = 0; block < MN_BLOCK; block++) \
									accumulator[block][channel] += input_value[block] * weight; \
							} \
						} \
					} \
					float *output = destination + (destination_row + x) * MN_CS; \
					for (int block = 0; block < MN_BLOCK; block++) \
						for (int channel = 0; channel < MN_CS; channel++) \
							output[block * MN_CS + channel] = accumulator[block][channel]; \
				} \
				for (; x < x_end; x++) { \
					float accumulator[MN_CS]; \
					for (int channel = 0; channel < MN_CS; channel++) \
						accumulator[channel] = 0.0f; \
					for (int tap = 0; tap < tap_count; tap++) { \
						const float *input = source + \
							(size_t)(tap_voxel[tap] + x) * (SOURCE_STRIDE); \
						const float *tap_weights = tap_weight[tap]; \
						for (int input_channel = 0; input_channel < (INPUT_CHANNELS); \
								input_channel++) { \
							float value = input[input_channel]; \
							for (int channel = 0; channel < MN_CS; channel++) \
								accumulator[channel] += value * \
									tap_weights[(size_t)input_channel * MN_CS + channel]; \
						} \
					} \
					float *output = destination + (destination_row + x) * MN_CS; \
					for (int channel = 0; channel < MN_CS; channel++) \
						output[channel] = accumulator[channel]; \
				} \
			} \
		} \
	} \
} \
static void NAME(const float *restrict source, float *restrict destination, \
		const float *restrict weights, int dilation, const mn_region *region) { \
	NAME##_slab slab = { .source = source, .destination = destination, \
		.weights = weights, .dilation = dilation, .region = region }; \
	bc_parallel_for(region->z_low, region->z_high, MN_GRAIN, \
		NAME##_rows, &slab); \
}

/*
 * The macro body above is frozen (frozen for performance): it is at 98% of this machine's
 * NEON fp32 FMA roofline and the literal channel counts are what keep 16
 * accumulators in registers. Only the INSTANTIATION LIST is conditional here.
 * A single-model build emits two shapes rather than three -- mn_conv_first plus
 * its own hidden-layer shape -- with byte-identical code generation for each,
 * since every expansion is independent.
 */
MN_DEFINE_CONV(mn_conv_first, 1, 1)
MN_DEFINE_CONV(mn_conv_hidden, MN_CS, BC_MODEL_CHAN)

/* ------------------------------------------------ normalization and activation */

#define MN_DEFINE_MOMENTS(NAME, CHANNELS) \
typedef struct { \
	const float *buffer; \
	double *partial; \
	const mn_region *region; \
} NAME##_slab; \
static void NAME##_rows(void *context, int z_low, int z_high) { \
	const NAME##_slab *slab = (const NAME##_slab *)context; \
	const float *buffer = slab->buffer; \
	double *partial = slab->partial; \
	const mn_region *region = slab->region; \
	for (int z = z_low; z <= z_high; z++) { \
		double sum[CHANNELS], square[CHANNELS]; \
		for (int channel = 0; channel < (CHANNELS); channel++) \
			sum[channel] = square[channel] = 0.0; \
		const float *slice = buffer + (size_t)z * MN_DIM * MN_DIM * MN_CS; \
		for (int y = region->y_low; y <= region->y_high; y++) \
		for (int x = region->x_low; x <= region->x_high; x++) \
			for (int channel = 0; channel < (CHANNELS); channel++) { \
				double value = slice[((size_t)y * MN_DIM + x) * MN_CS + channel]; \
				sum[channel] += value; \
				square[channel] += value * value; \
			} \
		for (int channel = 0; channel < (CHANNELS); channel++) { \
			partial[(size_t)z * 2 * (CHANNELS) + channel] = sum[channel]; \
			partial[(size_t)z * 2 * (CHANNELS) + (CHANNELS) + channel] = square[channel]; \
		} \
	} \
} \
static void NAME(const float *buffer, double *partial, float *mean, float *invstd, \
		const mn_region *region) { \
	NAME##_slab slab = { .buffer = buffer, .partial = partial, \
		.region = region }; \
	bc_parallel_for(region->z_low, region->z_high, MN_GRAIN, \
		NAME##_rows, &slab); \
	/* Serial, and outside the parallel range on purpose: this fold is what \
	   makes the result independent of how the z range was split, and so of \
	   the thread count. Moving it inside would be the one change here that \
	   could break bit-exactness. */ \
	for (int channel = 0; channel < (CHANNELS); channel++) { \
		double sum = 0.0, square = 0.0; \
		for (int z = region->z_low; z <= region->z_high; z++) { \
			sum += partial[(size_t)z * 2 * (CHANNELS) + channel]; \
			square += partial[(size_t)z * 2 * (CHANNELS) + (CHANNELS) + channel]; \
		} \
		double channel_mean = sum / (double)region->count; \
		double variance = square / (double)region->count - channel_mean * channel_mean; \
		if (!(variance > 0.0)) variance = 0.0; \
		mean[channel] = (float)channel_mean; \
		invstd[channel] = (float)(1.0 / sqrt(variance + MN_EPS)); \
	} \
}

/*
 * MindGrab is instantiated over the full MN_CS stride rather than its 15 logical
 * channels. Lane 15 is identically zero -- meshnet_cpu_pack_kernel calloc()s the
 * padding lane, so every convolution writes exactly 0.0f there -- which makes
 * the extra lane numerically inert: its mean is 0, GELU(0) is 0, and no consumer
 * (convolution, classifier, class export) ever reads it.
 *
 * The point is the loop bound. At 15 over a 16-float stride Clang's cost model
 * rejects the activation loop in mn_norm_mindgrab and emits it entirely scalar
 * (0 vector instructions, against 99 for the otherwise identical model16 path).
 * At 16 it vectorises, which is worth ~10% of total MindGrab runtime. Verify the
 * codegen, not just the clock, before changing these bounds: this loop is
 * cost-model-sensitive and the scalar fallback is silent.
 */
/*
 * Always over the full MN_CS stride, never the model's logical channel count.
 * For a model with fewer channels the pad lanes are inert -- pack_kernel
 * calloc()s them, so they stay zero -- and running the loop to a literal 16
 * lets Clang vectorise it. benchmarking records that the alternative emitted
 * entirely scalar code and cost ~10% of MindGrab's runtime.
 */
MN_DEFINE_MOMENTS(mn_moments, MN_CS)

#if BC_MODEL_GELU_TANH_RATIONAL
#define MN_TANH_LIMIT 9.0f
static inline float mn_mindgrab_tanh(float x) {
	if (x > MN_TANH_LIMIT) x = MN_TANH_LIMIT;
	if (x < -MN_TANH_LIMIT) x = -MN_TANH_LIMIT;
	float u = x * x;
	float p = -8.29118133e-14f;
	p = p * u + 5.19263868e-11f;
	p = p * u - 2.00294448e-08f;
	p = p * u + 1.11017944e-05f;
	p = p * u + 0.00309865153f;
	p = p * u + 0.130791619f;
	p = p * u + 0.99999994f;
	float q = 0.000253859733f;
	q = q * u + 0.024473751f;
	q = q * u + 0.464124829f;
	q = q * u + 1.0f;
	return x * p / q;
}

static inline float mn_mindgrab_gelu(float x) {
	float u = 0.797884583f * (x + 0.044715f * (x * x * x));
	return (0.5f * x) * (1.0f + mn_mindgrab_tanh(u));
}
#define mn_gelu mn_mindgrab_gelu
#endif /* BC_MODEL_GELU_TANH_RATIONAL */

#if BC_MODEL_GELU_EXP2_SIGMOID
/*
 * libm's exp2f is a call, so it blocks vectorization of the whole activation
 * loop: Clang emits four scalar calls per vector. This inline form splits the
 * exponent off with floorf and evaluates 2^fraction with a degree-6 minimax
 * polynomial, which autovectorizes. Maximum relative error is 1.2e-7, one ulp
 * of float, and the polynomial is exact at r = 0 so integral arguments are.
 * Clamping keeps the reconstructed exponent inside the finite float range;
 * exp2f would return infinity there, but the only caller divides by the
 * result, so both spellings collapse to the same value. The low clamp is
 * written negated so that a NaN argument lands in range too, because the
 * conversion to int below is undefined for NaN.
 */
static inline float mn_fast_exp2(float x) {
	if (!(x > -126.0f)) x = -126.0f;
	if (x > 126.0f) x = 126.0f;
	float f = floorf(x);
	float r = x - f;
	float p = 0.000216128448f;
	p = p * r + 0.00124678648f;
	p = p * r + 0.0096754498f;
	p = p * r + 0.0554852814f;
	p = p * r + 0.240229305f;
	p = p * r + 0.693147044f;
	p = p * r + 1.0f;
	union { float f; uint32_t u; } scale;
	scale.u = (uint32_t)(((int)f + 127) << 23);
	return p * scale.f;
}

static inline float mn_model16_gelu(float x) {
	float u = x + 0.044715f * x * x * x;
	return x / (1.0f + mn_fast_exp2(-2.302208198144325f * u));
}
#define mn_gelu mn_model16_gelu
#endif /* BC_MODEL_GELU_EXP2_SIGMOID */

#ifndef mn_gelu
#error "model_meta.json named a GELU flavour with no implementation here"
#endif

/*
 * GroupNorm, the optional per-channel affine, and the activation, in one pass.
 * Whether the affine exists is a descriptor fact rather than a branch, so a
 * model without one compiles to exactly the loop it had before.
 *
 * The channel loop runs to the literal MN_CS for the reason given above the
 * moments definition: bounding it by the logical channel count instead made
 * Clang emit the whole loop scalar.
 */
typedef struct {
	float *buffer;
	const float *mean;
	const float *invstd;
#if BC_MODEL_HAS_AFFINE
	const float *scale;
	const float *bias;
#endif
	const mn_region *region;
} mn_norm_slab;

static void mn_norm_rows(void *context, int z_low, int z_high) {
	const mn_norm_slab *slab = (const mn_norm_slab *)context;
	float *buffer = slab->buffer;
	const float *mean = slab->mean, *invstd = slab->invstd;
#if BC_MODEL_HAS_AFFINE
	const float *scale = slab->scale, *bias = slab->bias;
#endif
	const mn_region *region = slab->region;
	for (int z = z_low; z <= z_high; z++) {
		float *slice = buffer + (size_t)z * MN_DIM * MN_DIM * MN_CS;
		for (int y = region->y_low; y <= region->y_high; y++)
		for (int x = region->x_low; x <= region->x_high; x++) {
			float *value = slice + ((size_t)y * MN_DIM + x) * MN_CS;
			for (int channel = 0; channel < MN_CS; channel++) {
				float normalized =
					(value[channel] - mean[channel]) * invstd[channel];
#if BC_MODEL_HAS_AFFINE
				normalized = normalized * scale[channel] + bias[channel];
#endif
				value[channel] = mn_gelu(normalized);
			}
		}
	}
}

static void mn_norm(float *buffer, double *partial,
#if BC_MODEL_HAS_AFFINE
		const float *scale, const float *bias,
#endif
		const mn_region *region) {
	float mean[MN_CS], invstd[MN_CS];
	mn_moments(buffer, partial, mean, invstd, region);
	mn_norm_slab slab = { .buffer = buffer, .mean = mean, .invstd = invstd,
#if BC_MODEL_HAS_AFFINE
		.scale = scale, .bias = bias,
#endif
		.region = region };
	bc_parallel_for(region->z_low, region->z_high, MN_GRAIN, mn_norm_rows, &slab);
}

/* --------------------------------------------------------------- classifier */

/*
 * One argmax classifier for every model.
 *
 * MindGrab's two-class form used to be written out separately as
 * `logit1 > logit0 ? 1 : 0`. This is the same computation: the per-class inner
 * loops accumulate the same products in the same order, and a tie leaves
 * best_class at the lower index exactly as the strict `>` did, so the result is
 * bit-identical rather than merely equivalent.
 */
typedef struct {
	const float *source;
	float *labels;
	const float *weights;
#if BC_MODEL_CLASSIFIER_BIAS
	const float *bias;
#endif
	const mn_region *region;
} mn_classify_slab;

static void mn_classify_rows(void *context, int z_low, int z_high) {
	const mn_classify_slab *slab = (const mn_classify_slab *)context;
	const float *source = slab->source;
	float *labels = slab->labels;
	const float *weights = slab->weights;
#if BC_MODEL_CLASSIFIER_BIAS
	const float *bias = slab->bias;
#endif
	const mn_region *region = slab->region;
	for (int z = z_low; z <= z_high; z++) {
		for (int y = region->y_low; y <= region->y_high; y++)
		for (int x = region->x_low; x <= region->x_high; x++) {
			size_t i = (size_t)x + (size_t)y * MN_DIM +
				(size_t)z * MN_DIM * MN_DIM;
			const float *value = source + i * MN_CS;
			/* Class-inner, so the loop vectorises across classes while each
			   class still sums bias, then channels 0..CHAN-1, in order: the
			   bits match the class-outer form. Halved the 104-class model's
			   CPU gap to mindmap (8-10% -> 4-6% of segment time). */
			float logit[BC_MODEL_NCLASS];
			for (int class_index = 0; class_index < BC_MODEL_NCLASS; class_index++)
#if BC_MODEL_CLASSIFIER_BIAS
				logit[class_index] = bias[class_index];
#else
				logit[class_index] = 0.0f;
#endif
			for (int channel = 0; channel < BC_MODEL_CHAN; channel++)
				for (int class_index = 0; class_index < BC_MODEL_NCLASS; class_index++)
					logit[class_index] += value[channel] *
						weights[channel * BC_MODEL_NCLASS + class_index];
			int best_class = 0;
			/* -INFINITY is well defined here only because the build passes
			   -fno-finite-math-only; do not copy this idiom into the Metal or
			   OpenCL kernels, which are built with finite-math assumptions. */
			float best_value = -INFINITY;
			for (int class_index = 0; class_index < BC_MODEL_NCLASS; class_index++)
				if (logit[class_index] > best_value) {
					best_value = logit[class_index];
					best_class = class_index;
				}
			labels[i] = (float)best_class;
		}
	}
}

static void mn_classify(const float *source, float *labels,
		const float *weights,
#if BC_MODEL_CLASSIFIER_BIAS
		const float *bias,
#endif
		const mn_region *region) {
	mn_classify_slab slab = { .source = source, .labels = labels,
		.weights = weights,
#if BC_MODEL_CLASSIFIER_BIAS
		.bias = bias,
#endif
		.region = region };
	bc_parallel_for(region->z_low, region->z_high, MN_GRAIN,
		mn_classify_rows, &slab);
}

typedef struct {
	const float *source;
	float *map;
	const float *class_weights;
	float class_bias;
	const mn_region *region;
} mn_export_slab;

static void mn_export_rows(void *context, int z_low, int z_high) {
	const mn_export_slab *slab = (const mn_export_slab *)context;
	const float *source = slab->source;
	float *map = slab->map;
	const float *class_weights = slab->class_weights;
	float class_bias = slab->class_bias;
	const mn_region *region = slab->region;
	for (int z = z_low; z <= z_high; z++) {
		for (int y = region->y_low; y <= region->y_high; y++)
		for (int x = region->x_low; x <= region->x_high; x++) {
			size_t i = (size_t)x + (size_t)y * MN_DIM +
				(size_t)z * MN_DIM * MN_DIM;
			const float *value = source + i * MN_CS;
			float logit = class_bias;
			for (int channel = 0; channel < BC_MODEL_CHAN; channel++)
				logit += value[channel] *
					class_weights[channel * BC_MODEL_NCLASS];
			map[i] = logit;
		}
	}
}

#if BC_HAS_PVE
/*
 * Grouped-softmax tissue priors plus a brain-support gate, reproducing the
 * tail kernel in the page's probability runner.
 *
 * All 18 logits live on the stack, so nothing here materializes the 18 class
 * volumes (1.2 GB) that an export-then-reduce would need. The temperature is
 * upstream's 2.0, the support temperature 1.0.
 */
#include "pve_constants.h"

typedef struct {
	const float *source;
	float *planes;
	const float *classifier_weights;
#if BC_MODEL_CLASSIFIER_BIAS
	const float *classifier_bias;
#endif
	const mn_region *region;
} mn_pve_slab;

static void mn_pve_rows(void *context, int z_low, int z_high) {
	const mn_pve_slab *slab = (const mn_pve_slab *)context;
	const mn_region *region = slab->region;
	const unsigned masks[3] = { BC_PRIOR_GM_MASK, BC_PRIOR_WM_MASK, BC_PRIOR_CSF_MASK };
	for (int z = z_low; z <= z_high; z++)
	for (int y = region->y_low; y <= region->y_high; y++)
	for (int x = region->x_low; x <= region->x_high; x++) {
		size_t i = (size_t)x + (size_t)y * MN_DIM + (size_t)z * MN_DIM * MN_DIM;
		const float *value = slab->source + i * MN_CS;
		float logit[BC_MODEL_NCLASS];
		float maximum = -INFINITY;
		for (int c = 0; c < BC_MODEL_NCLASS; c++) {
#if BC_MODEL_CLASSIFIER_BIAS
			float sum = slab->classifier_bias[c];
#else
			float sum = 0.0f;
#endif
			for (int channel = 0; channel < BC_MODEL_CHAN; channel++)
				sum += value[channel] *
					slab->classifier_weights[channel * BC_MODEL_NCLASS + c];
			logit[c] = sum;
			if (sum > maximum) maximum = sum;
		}
		float denominator = 0.0f, support_denominator = 0.0f;
		float background = 0.0f, numerator[3] = { 0.0f, 0.0f, 0.0f };
		for (int c = 0; c < BC_MODEL_NCLASS; c++) {
			float centered = logit[c] - maximum;
			float weight = expf(centered / BC_PRIOR_TEMPERATURE);
			float support = expf(centered / BC_PRIOR_SUPPORT_TEMPERATURE);
			denominator += weight;
			support_denominator += support;
			if (c == 0) background = support;
			for (int t = 0; t < 3; t++)
				if (masks[t] & (1u << c)) numerator[t] += weight;
			if (BC_PRIOR_MIXED_GWM_MASK & (1u << c)) {
				float half = 0.5f * weight;
				numerator[0] += half;
				numerator[1] += half;
			}
		}
		for (int t = 0; t < 3; t++)
			slab->planes[(size_t)t * MN_NVOX + i] = numerator[t] / denominator;
		float support = 1.0f - background / support_denominator;
		if (!(support > 0.0f)) support = 0.0f;
		if (support > 1.0f) support = 1.0f;
		slab->planes[(size_t)BC_PVE_SUPPORT * MN_NVOX + i] = support;
	}
}

static void mn_tissue_priors(const float *source, const meshnet_cpu_plan *plan,
		const mn_region *region, float *planes) {
	mn_pve_slab slab = { .source = source, .planes = planes,
		.classifier_weights = plan->classifier_weights,
#if BC_MODEL_CLASSIFIER_BIAS
		.classifier_bias = plan->classifier_bias,
#endif
		.region = region };
	bc_parallel_for(region->z_low, region->z_high, MN_GRAIN, mn_pve_rows, &slab);
}
#endif /* BC_HAS_PVE */

static int mn_export_classes(const float *source, const meshnet_cpu_plan *plan,
		const mn_region *region, const bc_inference_options *options) {
	if (!options || !options->class_writer) return 0;
	float *map = (float *)calloc(MN_NVOX, sizeof(*map));
	if (!map) return 1;
	for (int class_index = 0; class_index < BC_MODEL_NCLASS; class_index++) {
#if BC_MODEL_CLASSIFIER_BIAS
		float class_bias = plan->classifier_bias[class_index];
#else
		float class_bias = 0.0f;
#endif
		const float *class_weights = plan->classifier_weights + class_index;
		mn_export_slab slab = { .source = source, .map = map,
			.class_weights = class_weights, .class_bias = class_bias,
			.region = region };
		bc_parallel_for(region->z_low, region->z_high, MN_GRAIN,
			mn_export_rows, &slab);
		if (options->class_writer(class_index, map, MN_NVOX,
				options->class_writer_context)) {
			free(map);
			return 1;
		}
	}
	free(map);
	return 0;
}

/* --------------------------------------------------------------- shared run */

static int mn_make_region(const bc_inference_options *options, mn_region *region) {
	region->x_low = region->y_low = region->z_low = 0;
	region->x_high = region->y_high = region->z_high = MN_DIM - 1;
	if (options && options->roi_enabled) {
		region->x_low = options->x_low;
		region->x_high = options->x_high;
		region->y_low = options->y_low;
		region->y_high = options->y_high;
		region->z_low = options->z_low;
		region->z_high = options->z_high;
	}
	if (region->x_low < 0 || region->y_low < 0 || region->z_low < 0 ||
			region->x_high >= MN_DIM || region->y_high >= MN_DIM ||
			region->z_high >= MN_DIM || region->x_low > region->x_high ||
			region->y_low > region->y_high || region->z_low > region->z_high)
		return 1;
	int width = region->x_high - region->x_low + 1;
	region->count = (size_t)width *
		(size_t)(region->y_high - region->y_low + 1) *
		(size_t)(region->z_high - region->z_low + 1);
	return region->count < 1;
}

#include "meshnet_sme.h"
#include "meshnet_turbo.h"

int meshnet_cpu_infer(const unsigned char *conformed, float *labels,
		const meshnet_cpu_plan *plan, const bc_inference_options *options) {
	/* Turbo is the default for whole-volume labels; ROI, class export and
	   tissue planes stay exact. */
	if (!(options && (options->no_turbo || options->roi_enabled || options->class_writer ||
			options->tissue_planes))) {
		mt_block_fn block2, block1;
		int mode = MT_MODE_INT8;
		const char *kernel = mt_select_ex(&block2, &block1, &mode);
		if (kernel) {
			return mt_infer(conformed, labels, plan, block2, block1, kernel, mode);
		}
#ifndef __EMSCRIPTEN__ /* wasm has no turbo kernels */
		printfx("%s: this CPU lacks i8mm/DotProd/AVX2; turbo runs the exact engine\n", BC_MODEL_NAME);
#endif
	}
	mn_region region;
	if (mn_make_region(options, &region)) return 1;
	float *first = (float *)malloc(MN_NVOX * MN_CS * sizeof(*first));
	float *second = (float *)malloc(MN_NVOX * MN_CS * sizeof(*second));
	double *partial = (double *)malloc(
		(size_t)MN_DIM * 2 * MN_CS * sizeof(*partial));
	if (!first || !second || !partial) {
		printfx("%s: out of memory (activation buffers need about %.1f GB)\n",
			BC_MODEL_NAME, 2.0 * (double)MN_NVOX * MN_CS * sizeof(float) / 1e9);
		free(first);
		free(second);
		free(partial);
		return 1;
	}

#if MN_SME
	const int sme = region.count == MN_NVOX && mn_sme_available("hw.optional.arm.SME_F32F32");
	if (sme) printfx("%s: CPU exact (fp32, SME)\n", BC_MODEL_NAME);
#define MN_NORM_FN(layer) (sme && (layer) + 1 < plan->layer_count ? mn_sme_norm : mn_norm)
#else
#define MN_NORM_FN(layer) mn_norm
#endif

	/* The affine arguments exist only for a model that has one, so the calls
	   differ in arity rather than in what they select at run time. */
#if BC_MODEL_HAS_AFFINE
#define MN_NORM(buffer, layer) \
	MN_NORM_FN(layer)((buffer), partial, plan->scale[layer], plan->bias[layer], &region)
#else
#define MN_NORM(buffer, layer) MN_NORM_FN(layer)((buffer), partial, &region)
#endif

	memset(labels, 0, MN_NVOX * sizeof(*labels));
	mn_normalize(conformed, second, &region);
	mn_conv_first(second, first, plan->kernel[0], plan->dilation[0], &region);
	MN_NORM(first, 0);

	for (int layer = 1; layer < plan->layer_count; layer++) {
		float *source = (layer & 1) ? first : second;
		float *destination = (layer & 1) ? second : first;
#if MN_SME
		if (sme) mn_sme_conv(source, destination, plan->kernel[layer], plan->dilation[layer], &region);
		else
#endif
			mn_conv_hidden(source, destination, plan->kernel[layer], plan->dilation[layer], &region);
		MN_NORM(destination, layer);
	}
#undef MN_NORM
#undef MN_NORM_FN

	const float *last = ((plan->layer_count - 1) & 1) ? second : first;
	mn_classify(last, labels, plan->classifier_weights,
#if BC_MODEL_CLASSIFIER_BIAS
		plan->classifier_bias,
#endif
		&region);
#if BC_HAS_PVE
	if (options && options->tissue_planes)
		mn_tissue_priors(last, plan, &region, options->tissue_planes);
#endif
	int export_status = mn_export_classes(last, plan, &region, options);
	free(first);
	free(second);
	free(partial);
	return export_status;
}

/* ---------------------------------------------------------- plan and driver */

/*
 * Build the plan by walking model.bin in the order its weights manifest
 * declares: for each layer a 3x3x3 kernel, then -- if this model has one -- a
 * 1x1 affine matrix and its bias; then the classifier kernel and its bias.
 *
 * This replaces the two hand-written adapters (mindgrab.c and model16.c). Their
 * layouts were never really different, only the presence of the affine, which
 * is now a descriptor fact. model_specs.c asserts these computed offsets
 * against the ones the generator derived from the manifest.
 */
int meshnet_cpu_prepare(meshnet_cpu_plan *plan, const float *weights) {
	memset(plan, 0, sizeof(*plan));
	plan->layer_count = BC_MODEL_NHIDDEN;
	plan->dilation = brainchop_model_dilation;

	size_t offset = 0;
	for (int layer = 0; layer < BC_MODEL_NHIDDEN; layer++) {
		int input_channels = layer == 0 ? 1 : BC_MODEL_CHAN;
		plan->kernel[layer] = meshnet_cpu_pack_kernel(weights + offset,
			input_channels, BC_MODEL_CHAN, !BC_MODEL_LOADER_TRANSPOSE);
		if (!plan->kernel[layer]) {
			printfx("%s: out of memory packing layer %d weights\n",
				BC_MODEL_NAME, layer);
			meshnet_cpu_destroy_plan(plan);
			return 1;
		}
		offset += (size_t)27 * input_channels * BC_MODEL_CHAN;
#if BC_MODEL_HAS_AFFINE
		/* The 1x1 affine convolution is diagonal in the shipped models, so only
		   its diagonal is kept. */
		const float *affine = weights + offset;
		offset += (size_t)BC_MODEL_CHAN * BC_MODEL_CHAN;
		plan->bias[layer] = weights + offset;
		offset += BC_MODEL_CHAN;
		for (int channel = 0; channel < BC_MODEL_CHAN; channel++)
			plan->scale[layer][channel] = affine[channel * BC_MODEL_CHAN + channel];
#endif
	}

	plan->classifier_weights = weights + offset;
	offset += (size_t)BC_MODEL_CHAN * BC_MODEL_NCLASS;
#if BC_MODEL_CLASSIFIER_BIAS
	plan->classifier_bias = weights + offset;
#endif
	/*
	 * Account for the classifier bias stored at the end of the payload.
	 */
	offset += BC_MODEL_NCLASS;
	if (offset != BC_MODEL_WEIGHT_FLOATS) {
		printfx("%s: weight layout walked %zu floats, expected %d\n",
			BC_MODEL_NAME, offset, BC_MODEL_WEIGHT_FLOATS);
		meshnet_cpu_destroy_plan(plan);
		return 1;
	}
	return 0;
}

int meshnet_cpu_segment(const unsigned char *conformed, float *result,
		const float *weights, const bc_inference_options *options) {
	meshnet_cpu_plan plan;
	if (meshnet_cpu_prepare(&plan, weights)) return 1;
	int status = meshnet_cpu_infer(conformed, result, &plan, options);
	meshnet_cpu_destroy_plan(&plan);
	return status;
}
