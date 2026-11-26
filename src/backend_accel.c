/*
 * fdopen() is POSIX, not ISO C, and this file is built with -std=c11, which
 * hides it. The tuning cache needs it to turn an O_EXCL|O_NOFOLLOW descriptor
 * into a FILE* -- opening by name with fopen would reintroduce exactly the
 * symlink race that flag combination exists to close. Must precede every
 * include. MSVC exposes _fdopen unconditionally, so this is POSIX-only.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include "backend_accel.h"
#include "model_specs.h"
#include "bc_threads.h"
#include "f16.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
/* MSVC has no <unistd.h>: open/close/getpid live in <io.h> and <process.h>,
   mkdir in <direct.h>, all under underscore-prefixed names. <windows.h> is here
   for MoveFileEx alone; see accel_cache_replace for why rename() is not usable. */
#include <windows.h>
#include <direct.h>
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

#define ACCEL_DIM BRAINCHOP_MODEL_DIM
#define ACCEL_NVOX ((size_t)ACCEL_DIM * ACCEL_DIM * ACCEL_DIM)

/*
 * The two elementwise maps below hand this range to bc_parallel_for, whose
 * bounds are int. At DIM 256 that is 16777216, comfortably inside INT_MAX --
 * asserted rather than assumed, because the failure would be a truncated bound
 * silently narrowing the range rather than anything that looks like an error.
 */
_Static_assert(ACCEL_NVOX <= (size_t)INT_MAX,
	"the volume no longer fits bc_parallel_for's int bounds");

/* One model per executable, so one spec, built from the generated descriptor. */
static const bc_accel_model_spec model_spec = {
	BC_MODEL_CHAN, BC_MODEL_NHIDDEN, BC_MODEL_NCLASS,
	/* MindGrab has a classifier bias without hidden-layer affine parameters;
	   GPU backends must use this independent flag for the final logits. */
	BC_MODEL_CLASSIFIER_BIAS,
	BC_MODEL_WEIGHT_FLOATS, brainchop_model_dilation
};

const bc_accel_model_spec *bc_accel_spec(void) {
	return &model_spec;
}

#if BC_MODEL_QUANTILE_MODE == 0
static double accel_quantile(const int64_t *cumulative, size_t count, double quantile, int upper) {
	(void)upper;
	double rank = quantile * (double)(count - 1);
	double base = floor(rank);
	double fraction = rank - base;
	size_t index = (size_t)base;
	int low = 0;
	while (low < 255 && cumulative[low] <= (int64_t)index) low++;
	int high = low;
	while (high < 255 && cumulative[high] <= (int64_t)(index + 1)) high++;
	if (fraction < 0.5)
		return (double)low + (double)(high - low) * fraction;
	return (double)high - (double)(high - low) * (1.0 - fraction);
}
#else
/* Plain rank selection. `upper` says which end this is, rather than inferring
   it from the value: the upper rank uses ceil()-1 and the lower a plain
   truncation, and guessing from `q > 0.5` would silently pick the wrong form
   for a model whose quantile pair sits entirely above or below the midpoint. */
static double accel_quantile(const int64_t *cumulative, size_t count, double quantile, int upper) {
	double raw = upper ? ceil((double)count * quantile) : (double)count * quantile;
	size_t rank = upper ? (raw > 0.0 ? (size_t)raw - 1 : 0) : (size_t)raw;
	if (rank >= count) rank = count - 1;
	int bin = 0;
	while (bin < 255 && cumulative[bin] <= (int64_t)rank) bin++;
	return (double)bin;
}
#endif

/*
 * The GPU backends' input normalization, parameterized exactly as the CPU
 * engine's is. Both read the same descriptor, so the two cannot drift apart
 * into normalizing differently -- which, being a scale factor on the input,
 * would show up as a subtly wrong segmentation rather than an error.
 */
/*
 * The two elementwise maps, hoisted so the pool can hand out index ranges. Both
 * bodies are unchanged from when they carried a `#pragma omp parallel for`.
 *
 * Named `_range` rather than the `_rows` used in meshnet_cpu.c and conform.c
 * because these walk a flat voxel index, not z rows, and `_rows` would be a
 * lie -- so the context is `_args` rather than `_slab` for the same reason.
 * `accel_` prefixed like every other static in this file.
 *
 * The index is plain `int`, not the `int64_t` these loops carried before the
 * pool: the bounds now arrive as int and the _Static_assert above is what makes
 * that safe. Keeping the wider type would have implied a range the API cannot
 * express.
 *
 * The normalization body is deliberately identical to mn_normalize_rows in
 * meshnet_cpu.c, including the BC_MODEL_NORM_CLAMP guard -- both read the same
 * descriptor and must agree, since a divergence is a scale factor on the input
 * and shows up as a subtly wrong segmentation rather than an error.
 */
typedef struct {
	const unsigned char *source;
	float *normalized;
	float low;
	float denominator;
} accel_normalize_args;

static void accel_normalize_range(void *context, int low_index, int high_index) {
	const accel_normalize_args *args = (const accel_normalize_args *)context;
	const unsigned char *source = args->source;
	float *normalized = args->normalized;
	float low = args->low, denominator = args->denominator;
	for (int i = low_index; i <= high_index; i++) {
		float value = ((float)source[i] - low) / denominator;
#if BC_MODEL_NORM_CLAMP
		if (!(value > 0.0f)) value = 0.0f;
		if (value > 1.0f) value = 1.0f;
#endif
		normalized[i] = value;
	}
}

typedef struct {
	const float *normalized;
	uint16_t *narrow;
} accel_narrow_args;

static void accel_narrow_range(void *context, int low_index, int high_index) {
	const accel_narrow_args *args = (const accel_narrow_args *)context;
	const float *normalized = args->normalized;
	uint16_t *narrow = args->narrow;
	for (int i = low_index; i <= high_index; i++)
		narrow[i] = bc_f32_to_f16(normalized[i]);
}

float *bc_accel_normalize_input(const unsigned char *source) {
	if (!source) return NULL;
	int64_t histogram[256] = {0}, cumulative[256], run = 0;
	for (size_t i = 0; i < ACCEL_NVOX; i++) histogram[source[i]]++;
	for (int i = 0; i < 256; i++) {
		run += histogram[i];
		cumulative[i] = run;
	}
	float low = (float)accel_quantile(cumulative, ACCEL_NVOX, BC_MODEL_NORM_Q_LOW, 0);
	float high = (float)accel_quantile(cumulative, ACCEL_NVOX, BC_MODEL_NORM_Q_HIGH, 1);
	if (high <= low && histogram[0] < (int64_t)ACCEL_NVOX) {
		int64_t fg_cum[256];
		fg_cum[0] = 0;
		for (int i = 1; i < 256; i++) fg_cum[i] = cumulative[i] - histogram[0];
		size_t fg_n = ACCEL_NVOX - (size_t)histogram[0];
		low = (float)accel_quantile(fg_cum, fg_n, BC_MODEL_NORM_Q_LOW, 0);
		high = (float)accel_quantile(fg_cum, fg_n, BC_MODEL_NORM_Q_HIGH, 1);
		if (high <= low && high > 0.0f) low = 0.0f;
	}
	float denominator = high - low + BC_MODEL_NORM_DENOM_EPS;
	if (!(denominator > 0.0f)) denominator = 1.0f;
	float *normalized = (float *)malloc(ACCEL_NVOX * sizeof(*normalized));
	if (!normalized) return NULL;
	/*
	 * Parallel, and bit-exact for being so: this is a pure elementwise map with
	 * no reduction, so every element's value is independent of how the range is
	 * split. The histogram above is NOT parallelised because it is a reduction
	 * into 256 bins, which needs per-thread private bins merged at the end
	 * rather than a bare `parallel for`. It was once quoted as "~1% of this
	 * function" -- do NOT trust that now: it was measured against the SERIAL map
	 * below, and parallelising the map made the histogram a much larger share of
	 * what remains. It is the obvious next target here, and a private-bin merge
	 * is trivially bit-exact for integer counts.
	 *
	 * Worth doing because the host side of a GPU run stopped being noise: a
	 * tuned CUDA run is ~1.08 s of which only ~0.35 s is on the GPU, and none
	 * of the host stages had a single omp pragma while meshnet_cpu.c is
	 * thoroughly parallel. the project's "conform is ~0.1%" was measured against
	 * a 140 s CPU run.
	 */
	accel_normalize_args args = { .source = source, .normalized = normalized,
		.low = low, .denominator = denominator };
	bc_parallel_for(0, (int)ACCEL_NVOX - 1, 0, accel_normalize_range, &args);
	return normalized;
}

float *bc_accel_copy_weights(const float *weights) {
	if (!weights) return NULL;
	float *packed = (float *)malloc(BC_MODEL_WEIGHT_FLOATS * sizeof(*packed));
	if (!packed) return NULL;
#if BC_MODEL_LOADER_TRANSPOSE
	/* Already in NIfTI tap order by the time it reaches here. */
	memcpy(packed, weights, BC_MODEL_WEIGHT_FLOATS * sizeof(*packed));
#else
	size_t offset = 0;
	for (int layer = 0; layer < BC_MODEL_NHIDDEN; layer++) {
		int input_channels = layer == 0 ? 1 : BC_MODEL_CHAN;
		for (int dz = -1; dz <= 1; dz++)
			for (int dy = -1; dy <= 1; dy++)
				for (int dx = -1; dx <= 1; dx++) {
					int source_tap = (dx + 1) * 9 + (dy + 1) * 3 + dz + 1;
					int destination_tap = (dz + 1) * 9 + (dy + 1) * 3 + dx + 1;
					size_t count = (size_t)input_channels * BC_MODEL_CHAN;
					memcpy(packed + offset + (size_t)destination_tap * count,
						weights + offset + (size_t)source_tap * count,
						count * sizeof(*packed));
				}
		offset += (size_t)27 * input_channels * BC_MODEL_CHAN;
#if BC_MODEL_HAS_AFFINE
		size_t affine_count =
			(size_t)BC_MODEL_CHAN * BC_MODEL_CHAN + BC_MODEL_CHAN;
		memcpy(packed + offset, weights + offset, affine_count * sizeof(*packed));
		offset += affine_count;
#endif
	}
	memcpy(packed + offset, weights + offset,
		((size_t)BC_MODEL_WEIGHT_FLOATS - offset) * sizeof(*packed));
#endif
	return packed;
}

uint16_t *bc_accel_narrow_input(const float *normalized) {
	if (!normalized) return NULL;
	uint16_t *narrow = (uint16_t *)malloc(ACCEL_NVOX * sizeof(*narrow));
	if (!narrow) return NULL;
	accel_narrow_args args = { .normalized = normalized, .narrow = narrow };
	bc_parallel_for(0, (int)ACCEL_NVOX - 1, 0, accel_narrow_range, &args);
	return narrow;
}

/* ---------------------------------------------------------- tuning cache
 *
 * See backend_accel.h for the contract. Moved here from backend_cuda.cu when
 * OpenCL became the second backend that wanted it.
 */
#define BC_TUNE_MAX_VALUES 8
#ifdef _WIN32
#define BC_FDOPEN _fdopen
#else
#define BC_FDOPEN fdopen
#endif

static int accel_cache_directory(char *buffer, size_t capacity, char *parent,
		size_t parent_capacity) {
	const char *base = getenv("XDG_CACHE_HOME");
	if (base && *base) {
		if (snprintf(parent, parent_capacity, "%s", base) >= (int)parent_capacity)
			return 1;
		return snprintf(buffer, capacity, "%s/brainchop", base) >= (int)capacity;
	}
	const char *home = getenv("HOME");
	if (!home || !*home) return 1;
	if (snprintf(parent, parent_capacity, "%s/.cache", home) >= (int)parent_capacity)
		return 1;
	return snprintf(buffer, capacity, "%s/.cache/brainchop", home) >= (int)capacity;
}

static int accel_cache_path(char *path, size_t capacity, char *directory,
		size_t directory_capacity, char *parent, size_t parent_capacity) {
	if (accel_cache_directory(directory, directory_capacity, parent,
			parent_capacity))
		return 1;
	return snprintf(path, capacity, "%s/tune.txt", directory) >= (int)capacity;
}

#ifdef _WIN32
/* rename() fails on Windows when the destination exists; MoveFileEx with
   MOVEFILE_REPLACE_EXISTING is the atomic replace. */
static int accel_cache_replace(const char *from, const char *to) {
	return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING) ? 0 : 1;
}
static int accel_mkdir(const char *path) { return _mkdir(path); }
static int accel_getpid(void) { return _getpid(); }
#else
static int accel_cache_replace(const char *from, const char *to) {
	return rename(from, to);
}
static int accel_mkdir(const char *path) { return mkdir(path, 0755); }
static int accel_getpid(void) { return (int)getpid(); }
#endif

int bc_accel_tune_load(const char *key, int *values, int count) {
	char directory[512], parent[512], path[600], line[1024];
	if (!key || !values || count <= 0 || count > BC_TUNE_MAX_VALUES) return 1;
	if (accel_cache_path(path, sizeof(path), directory, sizeof(directory),
			parent, sizeof(parent)))
		return 1;
	FILE *file = fopen(path, "r");
	if (!file) return 1;
	int found = 1;
	size_t key_length = strlen(key);
	while (fgets(line, sizeof(line), file)) {
		if (strncmp(line, key, key_length) || line[key_length] != '|') continue;
		const char *cursor = line + key_length;
		int parsed = 0;
		for (; parsed < count && *cursor == '|'; parsed++) {
			char *stop = NULL;
			long value = strtol(cursor + 1, &stop, 10);
			if (stop == cursor + 1) break;
			values[parsed] = (int)value;
			cursor = stop;
		}
		if (parsed == count) found = 0;
		break;
	}
	fclose(file);
	return found;
}

int bc_accel_tune_store(const char *key, const int *values, int count) {
	char directory[512], parent[512], path[600], temporary[640], line[1024];
	if (!key || !values || count <= 0 || count > BC_TUNE_MAX_VALUES) return 1;
	if (accel_cache_path(path, sizeof(path), directory, sizeof(directory),
			parent, sizeof(parent)))
		return 1;
	if (snprintf(temporary, sizeof(temporary), "%s.%d", path, accel_getpid()) >=
			(int)sizeof(temporary))
		return 1;
	accel_mkdir(parent);
	accel_mkdir(directory);

	/* Everything up to and including the separator before the LAST field. Only
	   that field is collected: a key of a|b|c|stamp has prefix "a|b|c|", so a
	   change to `stamp` replaces its predecessor, while a change to any earlier
	   field (driver version, codegen) makes a different prefix and leaves the
	   old line behind. See Known gaps. */
	const char *last = strrchr(key, '|');
	size_t prefix_length = last ? (size_t)(last - key) + 1 : strlen(key);

	/*
	 * O_EXCL|O_NOFOLLOW rather than fopen("w"): the temporary name ends in our
	 * pid and is therefore predictable, and fopen would happily follow a
	 * symlink planted there and truncate whatever it pointed at. Under the
	 * default $HOME/.cache that is only same-user mischief, but XDG_CACHE_HOME
	 * pointing somewhere shared is a routine container and HPC setting.
	 */
	int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_NOFOLLOW
	flags |= O_NOFOLLOW;
#endif
	int descriptor = open(temporary, flags, 0644);
	/*
	 * The READ needs O_NOFOLLOW too, not just the write. Every line that does
	 * not match this key is copied verbatim into the new tune.txt, which is a
	 * fresh 0644 regular file -- so a symlink planted at tune.txt would have
	 * its target's contents republished as world-readable. Same threat model
	 * the write guards against: XDG_CACHE_HOME pointing somewhere shared.
	 */
	int existing_fd = open(path, O_RDONLY
#ifdef O_NOFOLLOW
		| O_NOFOLLOW
#endif
	);
	FILE *existing = existing_fd >= 0 ? BC_FDOPEN(existing_fd, "r") : NULL;
	if (existing_fd >= 0 && !existing) close(existing_fd);
	FILE *file = descriptor >= 0 ? BC_FDOPEN(descriptor, "w") : NULL;
	if (!file) {
		if (descriptor >= 0) {
			close(descriptor);
			remove(temporary);
		}
		if (existing) fclose(existing);
		return 1;
	}
	fputs(key, file);
	for (int i = 0; i < count; i++) fprintf(file, "|%d", values[i]);
	fputc('\n', file);
	if (existing) {
		while (fgets(line, sizeof(line), existing))
			if (strncmp(line, key, prefix_length))
				fputs(line, file);
		fclose(existing);
	}
	int failed = ferror(file) != 0;
	if (fclose(file) != 0) failed = 1;
	/* Written via a temporary and replaced atomically, so two executables
	   tuning concurrently -- which is what `make benchmark` does -- cannot
	   interleave into a half-written line. */
	if (failed || accel_cache_replace(temporary, path) != 0) {
		remove(temporary);
		return 1;
	}
	return 0;
}
