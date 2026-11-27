#include "pve_constants.h"
#include "turbo_cone.h"
#ifdef BC_THREADS
#include "bc_threads.h"
#endif

/* Shared inference, packing, launch selection and autotuning policy.
 * Included once by the selected CUDA adapter (runtime or Driver API).
 * gpu_* primitives below are private to that translation unit. */
static int cuda_wmma_compiled(void) {
	int *device_flag = NULL, host_flag = 0;
	if (gpu_alloc((void **)&device_flag, sizeof(int)) != GPU_SUCCESS) {
		(void)gpu_last_error();
		return 0;
	}
	gpu_probe(device_flag);
	if (gpu_sync() != GPU_SUCCESS ||
			gpu_copy(&host_flag, device_flag, sizeof(int),
				GPU_DTOH) != GPU_SUCCESS) {
		(void)gpu_last_error();
		host_flag = 0;
	}
	gpu_free(device_flag);
	return host_flag == 1;
}

static int cuda_profiling(void) {
	static int state = -1;
	if (state < 0) state = getenv("BRAINCHOPC_CUDA_PROFILE") ? 1 : 0;
	return state;
}

static double g_stage_ms[3];
static int g_timing_error;
static char g_variant_label[64] = "vpt=? threads=?";
static const char *const g_stage_name[3] = { "conv+mom", "moments", "norm+gelu" };

typedef struct { gpu_event start, stop; int active; } cuda_stage;

static void stage_begin(cuda_stage *stage) {
	stage->active = 0;
	if (!cuda_profiling()) return;
	if (gpu_event_create(&stage->start) != GPU_SUCCESS) { g_timing_error = 1; return; }
	if (gpu_event_create(&stage->stop) != GPU_SUCCESS) {
		gpu_event_destroy(stage->start);
		g_timing_error = 1;
		return;
	}
	stage->active = 1;
	if (gpu_event_record(stage->start) != GPU_SUCCESS) g_timing_error = 1;
}

static void stage_end(cuda_stage *stage, int index) {
	if (!stage->active) return;
	float ms = 0.0f;
	if (gpu_event_record(stage->stop) != GPU_SUCCESS ||
		gpu_event_sync(stage->stop) != GPU_SUCCESS ||
		gpu_event_elapsed(&ms, stage->start, stage->stop) != GPU_SUCCESS)
		g_timing_error = 1;
	else
		g_stage_ms[index] += ms;
	if (gpu_event_destroy(stage->start) != GPU_SUCCESS) g_timing_error = 1;
	if (gpu_event_destroy(stage->stop) != GPU_SUCCESS) g_timing_error = 1;
}

static void stage_report(void) {
	if (!cuda_profiling()) return;
	double total = g_stage_ms[0] + g_stage_ms[1] + g_stage_ms[2];
	fprintf(stderr, "brainchop-core: CUDA stages (%s, stripes=%d)\n",
		g_variant_label, CUDA_MOMENT_STRIPES);
	for (int i = 0; i < 3; i++)
		fprintf(stderr, "brainchop-core:   %-10s %8.1f ms  %5.1f%%\n",
			g_stage_name[i], g_stage_ms[i],
			total > 0.0 ? 100.0 * g_stage_ms[i] / total : 0.0);
	fprintf(stderr, "brainchop-core:   %-10s %8.1f ms\n", "total", total);
}

typedef void (*cuda_conv_launcher)(const bc_act *, bc_act *, const bc_act *,
	double *, const unsigned int *, const float *, int, int, int, int, int, size_t);

#define X(V, T) \
static void launch_conv_##V##_##T(const bc_act *source, bc_act *destination, \
		const bc_act *weights, double *sums, const unsigned int *row_list, \
		const float *row_weights, int row_count, int input_channels, \
		int output_channels, int input_stride, int dilation, size_t shared_bytes) { \
	gpu_conv(V, T, source, destination, weights, sums, row_list, row_weights, \
		row_count, input_channels, output_channels, input_stride, dilation, shared_bytes); \
}
BC_CUDA_SCALAR_VARIANTS(X)
#undef X

typedef struct {
	int vpt;
	int threads;
	cuda_conv_launcher launch;
} cuda_conv_variant;

typedef void (*cuda_tune_run)(const bc_act *, bc_act *, const bc_act *, int);

typedef struct {
	int first;              /* VPT, or warps */
	int second;             /* THREADS, or 0 for the tensor-core family */
	cuda_tune_run run;
} cuda_tune_candidate;

#define X(V, T) \
static void tune_run_scalar_##V##_##T(const bc_act *source, bc_act *destination, \
		const bc_act *weights, int dilation) { \
	launch_conv_##V##_##T(source, destination, weights, NULL, NULL, NULL, CUDA_ROWS, \
		CUDA_CS, CUDA_CS, CUDA_CS, dilation, (size_t)27 * CUDA_CS * CUDA_CS * sizeof(bc_act)); \
}
BC_CUDA_SCALAR_VARIANTS(X)
#undef X

static const cuda_tune_candidate cuda_scalar_candidates[] = {
#define X(V, T) { V, T, tune_run_scalar_##V##_##T },
	BC_CUDA_SCALAR_VARIANTS(X)
#undef X
};
#define CUDA_SCALAR_CANDIDATE_COUNT \
	((int)(sizeof(cuda_scalar_candidates) / sizeof(cuda_scalar_candidates[0])))

#define X(W) \
static void tune_run_wmma_##W(const bc_act *source, bc_act *destination, \
		const bc_act *weights, int dilation) { \
	gpu_wmma(W, source, destination, weights, NULL, NULL, NULL, CUDA_ROWS, dilation); \
}
BC_CUDA_WMMA_VARIANTS(X)
#undef X

static const cuda_tune_candidate cuda_wmma_candidates[] = {
#define X(W) { W, 0, tune_run_wmma_##W },
	BC_CUDA_WMMA_VARIANTS(X)
#undef X
};
#define CUDA_WMMA_CANDIDATE_COUNT \
	((int)(sizeof(cuda_wmma_candidates) / sizeof(cuda_wmma_candidates[0])))

static void cuda_launch_wmma(int warps, const bc_act *source,
		bc_act *destination, const bc_act *weights, double *sums,
		const unsigned int *row_list, const float *row_weights,
		int row_count, int dilation) {
	gpu_wmma(warps, source, destination, weights, sums, row_list, row_weights,
		row_count, dilation);
}

static const cuda_conv_variant cuda_variants[] = {
#define X(V, T) { V, T, launch_conv_##V##_##T },
	BC_CUDA_SCALAR_VARIANTS(X)
#undef X
};
#define CUDA_VARIANT_COUNT \
	((int)(sizeof(cuda_variants) / sizeof(cuda_variants[0])))

static int cuda_default_variant(void) {
	for (int i = 0; i < CUDA_VARIANT_COUNT; i++)
		if (cuda_variants[i].vpt == CUDA_CONV_VPT &&
				cuda_variants[i].threads == CUDA_CONV_THREADS)
			return i;
	return 0;
}

typedef struct { int scalar, warps; } cuda_layer_choice;
static int cuda_parse_schedule(const char *text, int layers, int wmma_available,
		cuda_layer_choice *out) {
	if (!text || !*text || layers != BC_MODEL_NHIDDEN) return 1;
	char buffer[2048];
	if (strlen(text) >= sizeof(buffer)) return 1;
	strcpy(buffer, text);
	int count = 0;
	char *token = buffer;
	while (token) {
		char *next = strchr(token, ',');
		if (next) *next++ = 0;
		if (count >= layers) return 1;
		int v = 0, t = 0, w = 0, used = 0;
		cuda_layer_choice choice = {cuda_default_variant(), 0};
		if (sscanf(token, "s%dx%d%n", &v, &t, &used) == 2 && used > 0 && !token[used]) {
			choice.scalar = -1;
			for (int i = 0; i < CUDA_VARIANT_COUNT; i++)
				if (cuda_variants[i].vpt == v && cuda_variants[i].threads == t) choice.scalar = i;
			if (choice.scalar < 0) return 1;
		} else {
			used = 0;
			if (sscanf(token, "w%d%n", &w, &used) != 1 || used <= 0 || token[used] || !wmma_available) return 1;
			int found = 0;
			for (int i = 0; i < CUDA_WMMA_CANDIDATE_COUNT; i++)
				if (cuda_wmma_candidates[i].first == w) found = 1;
			size_t w_shared = (CUDA_CS == 24) ? 0 : (size_t)27 * CUDA_CS * CUDA_CS * sizeof(bc_act);
			if (!found || w_shared + (size_t)w * CUDA_WMMA_SCRATCH > 48 * 1024) return 1;
			choice.warps = w;
		}
		out[count++] = choice;
		token = next;
	}
	if (count == 1) {
		for (int i = 1; i < layers; i++) out[i] = out[0];
		out[0].warps = 0;
	} else if (count != layers || out[0].warps) return 1;
	return 0;
}

static int checksum_of(const bc_act *buffer, size_t count,
		unsigned long long *out) {
	unsigned long long *device_total = NULL;
	if (gpu_alloc((void **)&device_total, sizeof(*device_total)) != GPU_SUCCESS)
		return 1;
	int failed = gpu_zero(device_total, 0, sizeof(*device_total)) != GPU_SUCCESS;
	if (!failed) {
		gpu_checksum(buffer, device_total, count);
		failed = gpu_sync() != GPU_SUCCESS;
	}
	if (!failed)
		failed = gpu_copy(out, device_total, sizeof(*out),
			GPU_DTOH) != GPU_SUCCESS;
	gpu_free(device_total);
	return failed;
}

/* Bumped for ABI 3 (pre-padded 24-ch WMMA + fused moments). */
#define CUDA_TUNE_STAMP 4

#define BC_STR2(...) #__VA_ARGS__
#define BC_STR(...) BC_STR2(__VA_ARGS__)
#ifndef CUDA_TUNE_CODEGEN
#ifdef __CUDA_ARCH_LIST__
#define CUDA_TUNE_CODEGEN BC_STR(__CUDA_ARCH_LIST__)
#else
#define CUDA_TUNE_CODEGEN "default"
#endif
#endif

#ifndef CUDA_TUNE_REPEATS
#define CUDA_TUNE_REPEATS 3
#endif

static void cuda_device_key(char *device, size_t device_capacity,
		int *major, int *minor, int *driver) {
	gpu_properties properties;
	int ordinal = 0;
	*major = *minor = *driver = 0;
	snprintf(device, device_capacity, "unknown");
	if (gpu_device(&ordinal) != GPU_SUCCESS) return;
	if (gpu_properties_get(&properties, ordinal) == GPU_SUCCESS) {
		snprintf(device, device_capacity, "%s", properties.name);
		*major = properties.major;
		*minor = properties.minor;
	}
	gpu_driver_version(driver);
	for (char *p = device; *p; p++)
		if (*p == '|' || *p == '\n') *p = ' ';
}

static void cuda_cache_key(char *key, size_t capacity, int use_wmma) {
	char device[256];
	int major, minor, driver;
	cuda_device_key(device, sizeof(device), &major, &minor, &driver);
	snprintf(key, capacity, "cuda|%s|%s|%d.%d|%d|%s|%s|%d",
		use_wmma ? "wmma" : "scalar", device, major, minor, driver,
		BC_MODEL_NAME, CUDA_TUNE_CODEGEN, CUDA_TUNE_STAMP);
}

static int cuda_cache_load(const char *key, const cuda_tune_candidate *table,
		int count) {
	int values[2];
	if (bc_accel_tune_load(key, values, 2)) return -1;
	for (int i = 0; i < count; i++)
		if (table[i].first == values[0] && table[i].second == values[1])
			return i;
	return -1;
}

static int cuda_cache_store(const char *key, const cuda_tune_candidate *chosen) {
	int values[2] = { chosen->first, chosen->second };
	return bc_accel_tune_store(key, values, 2);
}

static int cuda_tune(bc_act *scratch_a, bc_act *scratch_b,
		const bc_act *weights, const cuda_tune_candidate *table, int count,
		int use_wmma, int *chosen) {
	unsigned long long reference = 0;
	int have_reference = 0, best = -1;
	double best_ms = 0.0;
	float slowest = 0.0f, best_slowest = 0.0f;
	size_t elements = CUDA_NVOX * (size_t)CUDA_CS;
	const int dilation = 4;
	const char *first_label = use_wmma ? "warps" : "vpt";

	gpu_fill(scratch_a, elements);
	gpu_error_t fill_error = gpu_sync();
	if (fill_error != GPU_SUCCESS) {
		(void)gpu_last_error();
		cuda_error(fill_error, "preparing tuning input");
		return 1;
	}

	fprintf(stderr, "brainchop-core: tuning the %s CUDA convolution for this device "
		"(%d variants, once per device -- caching the result)\n",
		use_wmma ? "tensor-core" : "scalar", count);

	{
		int reference_index = 0;
		for (int i = 0; i < count; i++)
			if (use_wmma ? (table[i].first == CUDA_WMMA_WARPS)
					: (table[i].first == CUDA_CONV_VPT &&
						table[i].second == CUDA_CONV_THREADS))
				reference_index = i;
		table[reference_index].run(scratch_a, scratch_b, weights, dilation);
		if (gpu_sync() == GPU_SUCCESS &&
				!checksum_of(scratch_b, elements, &reference))
			have_reference = 1;
		else
			(void)gpu_last_error();
	}

	for (int i = 0; i < count; i++) {
		float milliseconds = 0.0f;
		unsigned long long digest = 0;

		table[i].run(scratch_a, scratch_b, weights, dilation);
		if (gpu_sync() != GPU_SUCCESS) {
			(void)gpu_last_error();
			if (use_wmma)
				fprintf(stderr, "brainchop-core:   warps=%-3d unavailable\n",
					table[i].first);
			else
				fprintf(stderr, "brainchop-core:   vpt=%-2d threads=%-4d "
					"unavailable\n", table[i].first, table[i].second);
			continue;
		}
		if (checksum_of(scratch_b, elements, &digest)) {
			(void)gpu_last_error();
			continue;
		}
		if (!have_reference) {
			reference = digest;
			have_reference = 1;
		} else if (digest != reference) {
			fprintf(stderr, "brainchop-core:   %s=%-3d rejected (checksum "
				"mismatch)\n", first_label, table[i].first);
			continue;
		}

		for (int repeat = 0; repeat < CUDA_TUNE_REPEATS; repeat++) {
			float sample = 0.0f;
			gpu_event start, stop;
			int bad = gpu_event_create(&start) != GPU_SUCCESS;
			if (!bad) {
				bad = gpu_event_create(&stop) != GPU_SUCCESS;
				if (!bad) {
					bad = gpu_event_record(start) != GPU_SUCCESS;
					if (!bad) table[i].run(scratch_a, scratch_b, weights, dilation);
					bad |= gpu_event_record(stop) != GPU_SUCCESS;
					bad |= gpu_event_sync(stop) != GPU_SUCCESS;
					if (!bad) bad = gpu_event_elapsed(&sample, start, stop) != GPU_SUCCESS;
					bad |= gpu_event_destroy(stop) != GPU_SUCCESS;
				}
				bad |= gpu_event_destroy(start) != GPU_SUCCESS;
			}
			bad |= gpu_last_error() != GPU_SUCCESS;
			if (bad) { milliseconds = 0.0f; break; }
			if (milliseconds <= 0.0f || sample < milliseconds) milliseconds = sample;
			if (sample > slowest) slowest = sample;
		}

		if (milliseconds <= 0.0f) {
			slowest = 0.0f;
			continue;
		}

		if (use_wmma)
			fprintf(stderr, "brainchop-core:   warps=%-3d %7.2f ms\n",
				table[i].first, (double)milliseconds);
		else
			fprintf(stderr, "brainchop-core:   vpt=%-2d threads=%-4d %7.2f ms\n",
				table[i].first, table[i].second, (double)milliseconds);
		if (best < 0 || milliseconds < best_ms) {
			best = i;
			best_ms = milliseconds;
			best_slowest = slowest;
		}
		slowest = 0.0f;
	}
	if (best < 0) return 1;
	if (use_wmma)
		fprintf(stderr, "brainchop-core: chose warps=%d (%.2f ms)\n",
			table[best].first, best_ms);
	else
		fprintf(stderr, "brainchop-core: chose vpt=%d threads=%d (%.2f ms)\n",
			table[best].first, table[best].second, best_ms);
	if (best_slowest > 1.5f * best_ms)
		fprintf(stderr, "brainchop-core: timings varied %.2f-%.2f ms for the same "
			"shape, so this device was busy; re-run with --tune when idle\n",
			best_ms, (double)best_slowest);
	*chosen = best;
	return 0;
}

/* The padded fp16 weight layout, shared with backend_opencl.c. */
#define bc_pad_store bc_act
#define BC_PAD_STORE(x) gpu_half(x)
#define BC_PAD_CS CUDA_CS
#include "accel_pack_padded.h"

#if CUDA_CS == 24
/* Pre-pad hidden-layer 24x24 weights into [27][2][2][16][16] (ldm=16) tiles
   so 24-ch WMMA loads directly from L1/L2 without per-warp tile_b copies. */
static bc_act *cuda_pack_wmma24_weights(const bc_accel_model_spec *spec,
		const bc_padded_weights *packed, size_t *out_count) {
	size_t hidden = (size_t)(spec->layers - 1);
	size_t count = hidden * 27 * 1024;
	bc_act *buf = (bc_act *)calloc(count, sizeof(bc_act));
	if (!buf) return NULL;
	for (int layer = 1; layer < spec->layers; layer++) {
		const bc_act *src = packed->data + packed->layer_offset[layer];
		bc_act *dst = buf + (size_t)(layer - 1) * 27 * 1024;
		for (int tap = 0; tap < 27; tap++) {
			const bc_act *st = src + (size_t)tap * 24 * 24;
			bc_act *dt = dst + (size_t)tap * 1024;
			for (int k = 0; k < 2; k++)
			for (int n = 0; n < 2; n++) {
				bc_act *tile = dt + (size_t)(k * 2 + n) * 256;
				for (int r = 0; r < 16; r++) {
					int ic = k * 16 + r;
					if (ic >= 24) break;
					for (int c = 0; c < 16; c++) {
						int oc = n * 16 + c;
						if (oc >= 24) break;
						tile[r * 16 + c] = st[ic * 24 + oc];
					}
				}
			}
		}
	}
	*out_count = count;
	return buf;
}
#endif

typedef struct {
	const unsigned char *src;
	float *dst;
	size_t n;
	int chunks;
} cuda_u8_to_f32_ctx;

static void cuda_u8_to_f32_chunk(void *context, int lo, int hi) {
	const cuda_u8_to_f32_ctx *u = (const cuda_u8_to_f32_ctx *)context;
	size_t b0 = (u->n * (size_t)lo) / (size_t)u->chunks;
	size_t b1 = (u->n * (size_t)(hi + 1)) / (size_t)u->chunks;
	for (size_t i = b0; i < b1; i++) u->dst[i] = (float)u->src[i];
}

static int cuda_finish(int failed) {
	if (gpu_end())
		fprintf(stderr, "brainchop-core: warning: CUDA cleanup did not complete\n");
	return failed ? 1 : 0;
}

static int cuda_segment(const unsigned char *conformed,
		float *result, const float *weights, const bc_inference_options *options) {
	const bc_accel_model_spec *spec = bc_accel_spec();
	if (!conformed || !result || !weights || !spec) return 1;
	if (options && (options->roi_enabled || options->class_writer)) return 2;
	if (options && options->tissue_planes && (spec->classes != 18 || BC_MODEL_OUTPUT_IS_MASK)) return 2;
	if (gpu_begin(options ? options->adapter : -1))
		return 1;
	memset(g_stage_ms, 0, sizeof(g_stage_ms));
	g_timing_error = 0;
	float *host_input = bc_accel_normalize_input(conformed);
	float *host_weights = bc_accel_copy_weights(weights);
	bc_act *device_a = NULL, *device_b = NULL, *device_weights = NULL;
	bc_act *device_wmma_weights = NULL;
	float *device_affine = NULL;
	bc_act *half_input = NULL;
	unsigned char *host_labels_u8 = NULL;
	bc_padded_weights packed;
	memset(&packed, 0, sizeof(packed));
	double *device_sums = NULL;
	float *device_mul = NULL, *device_add = NULL;
	unsigned int *device_tc_list = NULL;
	float *device_tc_weight = NULL;
	bc_tc_plan tc;
	memset(&tc, 0, sizeof(tc));
	int use_cone = BC_TURBO_CONE && !(options && options->no_turbo);
	int failed = !host_input || !host_weights;
	if (!failed && use_cone) {
		/* Exact pure-background skip (allow_cone=0) preserves strict CUDA voxel
		   budgets (64/1024) and PVE RMSE <= 2/255 with zero moment drift. */
		if (bc_tc_make_ex(conformed, spec->dilation, &tc, 0)) failed = 1;
	}
	if (!failed) failed = bc_pack_padded_weights(spec, host_weights, &packed);
	size_t activation_bytes = CUDA_NVOX * (size_t)CUDA_CS * sizeof(bc_act);
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_a, activation_bytes),
			"allocating first activation");
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_b, activation_bytes),
			"allocating second activation");
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_weights,
			packed.count * sizeof(bc_act)), "allocating weights");
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_sums,
			(size_t)CUDA_MOMENT_STRIPES * CUDA_CS * 2 * sizeof(double)),
			"allocating normalization workspace");
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_mul,
			(size_t)CUDA_CS * sizeof(float)), "allocating channel mul") ||
			cuda_error(gpu_alloc((void **)&device_add,
			(size_t)CUDA_CS * sizeof(float)), "allocating channel add");
#if CUDA_CS == 24
	if (!failed) {
		size_t wmma_count = 0;
		bc_act *host_wmma = cuda_pack_wmma24_weights(spec, &packed, &wmma_count);
		if (!host_wmma) failed = 1;
		else {
			failed = cuda_error(gpu_alloc((void **)&device_wmma_weights,
				wmma_count * sizeof(bc_act)), "allocating 24ch WMMA weights") ||
				cuda_error(gpu_copy(device_wmma_weights, host_wmma,
				wmma_count * sizeof(bc_act), GPU_HTOD), "copying 24ch WMMA weights");
			free(host_wmma);
		}
	}
#endif
#if BC_MODEL_HAS_AFFINE
	if (!failed) {
		/* Pre-pack all layers' affine (scale, bias) into one contiguous device
		   buffer, eliminating 26 synchronous HtoD copies inside the layer loop. */
		float host_affine[BC_MODEL_NHIDDEN * 2 * BC_MODEL_CHAN];
		size_t raw_off = 0;
		for (int layer = 0; layer < spec->layers; layer++) {
			raw_off += (size_t)27 * (layer == 0 ? 1 : spec->channels) * spec->channels;
			float *ls = host_affine + (size_t)layer * 2 * BC_MODEL_CHAN;
			float *lb = ls + BC_MODEL_CHAN;
			for (int c = 0; c < BC_MODEL_CHAN; c++) {
				ls[c] = host_weights[raw_off + (size_t)c * BC_MODEL_CHAN + c];
				lb[c] = host_weights[raw_off + (size_t)BC_MODEL_CHAN * BC_MODEL_CHAN + c];
			}
			raw_off += (size_t)BC_MODEL_CHAN * BC_MODEL_CHAN + BC_MODEL_CHAN;
		}
		failed = cuda_error(gpu_alloc((void **)&device_affine, sizeof(host_affine)),
			"allocating affine parameters") ||
			cuda_error(gpu_copy(device_affine, host_affine, sizeof(host_affine), GPU_HTOD),
			"copying affine parameters");
	}
#endif
	if (!failed && use_cone && tc.n > 0) {
		failed = cuda_error(gpu_alloc((void **)&device_tc_list,
			tc.n * sizeof(unsigned int)), "allocating cone row list") ||
			cuda_error(gpu_alloc((void **)&device_tc_weight,
			tc.n * sizeof(float)), "allocating cone row weights") ||
			cuda_error(gpu_copy(device_tc_list, tc.list,
			tc.n * sizeof(unsigned int), GPU_HTOD), "copying cone row list") ||
			cuda_error(gpu_copy(device_tc_weight, tc.weight,
			tc.n * sizeof(float), GPU_HTOD), "copying cone row weights");
	}
	if (!failed) {
		half_input = (bc_act *)bc_accel_narrow_input(host_input);
		if (!half_input) failed = 1;
	}
	if (!failed) {
		failed = cuda_error(gpu_copy(device_weights, packed.data,
			packed.count * sizeof(bc_act), GPU_HTOD), "copying weights");
	}

	int use_wmma = 0;
	if (!failed) {
		int ordinal = 0;
		gpu_properties properties;
		const char *request = getenv("BRAINCHOPC_CUDA_WMMA");
		if (gpu_device(&ordinal) == GPU_SUCCESS &&
				gpu_properties_get(&properties, ordinal) == GPU_SUCCESS)
			use_wmma = properties.major >= 7;
		else
			(void)gpu_last_error();
		if (use_wmma && !cuda_wmma_compiled()) {
			fprintf(stderr, "brainchop-core: this build has no tensor-core kernel "
				"for %s (compute capability %d.%d); using the scalar "
				"convolution.\n",
				properties.name, properties.major, properties.minor);
			use_wmma = 0;
		}
		if (request && !strcmp(request, "0")) use_wmma = 0;
	}

	const char *schedule_text = getenv("BRAINCHOPC_CUDA_LAYERS");
	cuda_layer_choice schedule[BC_MODEL_NHIDDEN];
	if (!failed && schedule_text && cuda_parse_schedule(schedule_text, spec->layers, use_wmma, schedule)) {
		fprintf(stderr, "brainchop-core: invalid or unsupported BRAINCHOPC_CUDA_LAYERS schedule\n");
		failed = 1;
	}
	const cuda_tune_candidate *table = use_wmma
		? cuda_wmma_candidates : cuda_scalar_candidates;
	int table_count = use_wmma
		? CUDA_WMMA_CANDIDATE_COUNT : CUDA_SCALAR_CANDIDATE_COUNT;
	int chosen_index = -1;
	for (int i = 0; i < table_count; i++)
		if (use_wmma ? (table[i].first == CUDA_WMMA_WARPS)
				: (table[i].first == CUDA_CONV_VPT &&
					table[i].second == CUDA_CONV_THREADS))
			chosen_index = i;
	if (chosen_index < 0) chosen_index = 0;

	if (!failed && !schedule_text) {
		const char *request = getenv("BRAINCHOPC_TUNE");
		int forced = (options && options->force_tuning) ||
			(request && !strcmp(request, "1"));
		int disabled = request && !strcmp(request, "0") && !forced;
		char key[768];
		cuda_cache_key(key, sizeof(key), use_wmma);
		int cached = disabled ? -1
			: cuda_cache_load(key, table, table_count);
		if (cached >= 0 && !forced) {
			chosen_index = cached;
		} else if (!disabled) {
			int picked = chosen_index;
			const bc_act *tune_weights = (use_wmma && CUDA_CS == 24)
				? device_wmma_weights : (device_weights + packed.layer_offset[1]);
			if (!cuda_tune(device_a, device_b, tune_weights, table,
					table_count, use_wmma, &picked)) {
				chosen_index = picked;
				if (cuda_cache_store(key, &table[picked]))
					fprintf(stderr, "brainchop-core: could not save the tuning "
						"result; it will be measured again next run\n");
			} else {
				fprintf(stderr, "brainchop-core: tuning failed; using the "
					"compiled-in convolution shape\n");
			}
		}
	}
	int chosen_warps = use_wmma ? table[chosen_index].first : 0;
	BC_CUDA_ASSERT(CUDA_SCALAR_CANDIDATE_COUNT == CUDA_VARIANT_COUNT,
		"the scalar candidate and launcher tables come from one list");
	int scalar_variant = use_wmma ? cuda_default_variant() : chosen_index;

	if (!failed)
		failed = cuda_error(gpu_copy(device_b, half_input,
			CUDA_NVOX * sizeof(bc_act), GPU_HTOD), "copying input");

	if (schedule_text)
		snprintf(g_variant_label, sizeof(g_variant_label), "offline layer schedule");
	else if (use_wmma)
		snprintf(g_variant_label, sizeof(g_variant_label), "wmma warps=%d",
			chosen_warps);
	else
		snprintf(g_variant_label, sizeof(g_variant_label), "vpt=%d threads=%d",
			table[chosen_index].first, table[chosen_index].second);

	const size_t sums_bytes = (size_t)CUDA_MOMENT_STRIPES * CUDA_CS * 2 * sizeof(double);
	for (int layer = 0; layer < spec->layers && !failed; layer++) {
		int input_stride = layer == 0 ? 1 : CUDA_CS;
		bc_act *source = layer ? ((layer & 1) ? device_a : device_b) : device_b;
		bc_act *destination = (layer & 1) ? device_b : device_a;
		size_t shared_bytes = (size_t)27 * (layer == 0 ? 1 : CUDA_CS) * CUDA_CS *
			sizeof(bc_act);
		int layer_warps = schedule_text ? schedule[layer].warps : (layer > 0 && use_wmma ? chosen_warps : 0);
		int layer_scalar = schedule_text ? schedule[layer].scalar : scalar_variant;
		const unsigned int *row_list = NULL;
		const float *row_weights = NULL;
		int row_count = CUDA_ROWS;
		unsigned int pure_off = 0, npure = 0, rep = 0;
		if (use_cone) {
			const bc_tc_layer *tl = &tc.layer[layer];
			if (tl->count < (uint32_t)CUDA_ROWS || tl->npure > 0) {
				row_list = device_tc_list + tl->list_off;
				row_weights = device_tc_weight + tl->list_off;
				row_count = (int)tl->count;
				pure_off = tl->pure_off;
				npure = tl->npure;
				rep = tl->rep;
			}
		}
		if (cuda_error(gpu_zero(device_sums, 0, sums_bytes), "clearing moment stripes")) {
			failed = 1;
			break;
		}
		double before_ms = g_stage_ms[0];
		cuda_stage stage;
		stage_begin(&stage);
		if (layer_warps) {
			const bc_act *w_ptr = (CUDA_CS == 24)
				? (device_wmma_weights + (size_t)(layer - 1) * 27 * 1024)
				: (device_weights + packed.layer_offset[layer]);
			cuda_launch_wmma(layer_warps, source, destination, w_ptr,
				device_sums, row_list, row_weights, row_count,
				spec->dilation[layer]);
		} else {
			cuda_variants[layer_scalar].launch(source, destination,
				device_weights + packed.layer_offset[layer],
				device_sums, row_list, row_weights, row_count,
				layer == 0 ? 1 : CUDA_CS, spec->channels,
				input_stride, spec->dilation[layer], shared_bytes);
		}
		stage_end(&stage, 0);
		if (cuda_profiling() && !g_timing_error)
			fprintf(stderr, "brainchop-core: CUDA layer=%d dilation=%d rows=%d conv_ms=%.6f\n",
				layer, spec->dilation[layer], row_count, g_stage_ms[0] - before_ms);
		failed = cuda_error(gpu_last_error(), "convolution kernel") || g_timing_error;
		if (failed) break;

		const float *layer_scale = NULL, *layer_bias = NULL;
#if BC_MODEL_HAS_AFFINE
		layer_scale = device_affine + (size_t)layer * 2 * BC_MODEL_CHAN;
		layer_bias = layer_scale + BC_MODEL_CHAN;
#endif
		stage_begin(&stage);
		gpu_moments(device_sums, device_mul, device_add, layer_scale, layer_bias, CUDA_CS);
		stage_end(&stage, 1);

		/* Last layer fuses GroupNorm+GELU directly into classify_u8 or tissue_priors. */
		if (layer + 1 < spec->layers) {
			stage_begin(&stage);
			gpu_norm(destination, device_mul, device_add, row_list, row_count);
			if (npure > 0)
				gpu_fill_rows(destination, device_tc_list + pure_off, (int)npure, (int)rep);
			stage_end(&stage, 2);
		}
		failed = cuda_error(gpu_last_error(), "normalization kernel") || g_timing_error;
	}

	if (!failed) {
		bc_act *last = (spec->layers & 1) ? device_a : device_b;
		bc_act *label_buffer = (spec->layers & 1) ? device_b : device_a;
		const bc_act *classifier_weights = device_weights + packed.classifier_offset;
		const bc_act *classifier_bias = spec->classifier_bias
			? classifier_weights + (size_t)spec->channels * spec->classes : NULL;
		const unsigned int *cls_list = NULL;
		int cls_rows = CUDA_ROWS;
		unsigned int last_pure_off = 0, last_npure = 0, last_rep = 0;
		if (use_cone) {
			const bc_tc_layer *tl = &tc.layer[spec->layers - 1];
			if (tc.ncls < (uint32_t)CUDA_ROWS || tl->npure > 0) {
				cls_list = device_tc_list + tc.cls_off;
				cls_rows = (int)tc.ncls;
				last_pure_off = tl->pure_off;
				last_npure = tl->npure;
				last_rep = tl->rep;
			}
		}
		int priors = options && options->tissue_planes;
		BC_CUDA_ASSERT(CUDA_CS * sizeof(bc_act) >= BC_PRIOR_PLANES * sizeof(float),
			"inactive activation must hold all tissue planes");
		if (priors) {
			if (cls_list)
				failed = cuda_error(gpu_zero(label_buffer, 0,
					CUDA_NVOX * BC_PRIOR_PLANES * sizeof(float)), "clearing tissue planes");
			if (!failed) {
				gpu_priors(last, (float *)label_buffer, device_mul, device_add,
					classifier_weights, classifier_bias, cls_list, cls_rows,
					spec->channels, spec->classes);
				if (last_npure > 0)
					gpu_fill_priors((float *)label_buffer, device_tc_list + last_pure_off,
						(int)last_npure, (int)last_rep);
				if (options->tissue_priors_prepared) {
					/* Reuse dead `last` (805 MB) as 3-plane blur scratch (201 MB). */
					float *plane_a = (float *)label_buffer;
					float *plane_b = (float *)last;
					gpu_pve_blur(plane_a, plane_b, 0);
					gpu_pve_blur(plane_b, plane_a, 1);
					gpu_pve_blur(plane_a, plane_b, 2);
					gpu_pve_support(plane_b, plane_a);
				}
				failed = cuda_error(gpu_last_error(), "tissue prior kernel");
			}
			if (!failed) {
				failed = cuda_error(gpu_copy(options->tissue_planes, label_buffer,
					CUDA_NVOX * sizeof(float) * BC_PRIOR_PLANES, GPU_DTOH),
					"copying tissue priors");
				if (!failed && options->tissue_priors_prepared)
					*options->tissue_priors_prepared = 1;
			}
		} else {
			if (cls_list)
				failed = cuda_error(gpu_zero(label_buffer, 0, CUDA_NVOX), "clearing u8 labels");
			if (!failed) {
				gpu_classify_u8(last, (unsigned char *)label_buffer, device_mul, device_add,
					classifier_weights, classifier_bias, cls_list, cls_rows,
					spec->channels, spec->classes);
				if (last_npure > 0)
					gpu_fill_labels_u8((unsigned char *)label_buffer,
						device_tc_list + last_pure_off, (int)last_npure, (int)last_rep);
				failed = cuda_error(gpu_last_error(), "classifier kernel");
			}
			if (!failed) {
				host_labels_u8 = (unsigned char *)malloc(CUDA_NVOX);
				if (!host_labels_u8) failed = 1;
				else failed = cuda_error(gpu_copy(host_labels_u8, label_buffer,
					CUDA_NVOX, GPU_DTOH), "copying u8 labels");
			}
			if (!failed) {
				cuda_u8_to_f32_ctx uctx = { host_labels_u8, result, CUDA_NVOX, 32 };
#ifdef BC_THREADS
				bc_parallel_for(0, uctx.chunks - 1, 0, cuda_u8_to_f32_chunk, &uctx);
#else
				cuda_u8_to_f32_chunk(&uctx, 0, uctx.chunks - 1);
#endif
			}
		}
	}
	stage_report();
	free(host_labels_u8);
	gpu_free(device_tc_weight);
	gpu_free(device_tc_list);
	gpu_free(device_affine);
	gpu_free(device_wmma_weights);
	gpu_free(device_add);
	gpu_free(device_mul);
	gpu_free(device_sums);
	gpu_free(device_weights);
	gpu_free(device_b);
	gpu_free(device_a);
	bc_tc_free(&tc);
	free(half_input);
	bc_free_padded_weights(&packed);
	free(host_weights);
	free(host_input);
	return cuda_finish(failed);
}

const bc_backend bc_cuda_backend = {
	"cuda",
	cuda_available,
	cuda_segment,
	cuda_adapter_count,
	cuda_adapter_name
};
