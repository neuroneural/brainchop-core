#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
/* ROCm/HIP module loader for checked-in, architecture-exact HSA code objects.
 * It uses dlopen only: normal users need neither HIP headers nor a compiler. */
#include "backend.h"
#include "backend_accel.h"
#include "cat_lite.h"
#include "f16.h"
#include "model_specs.h"
#include "pve_constants.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int hipError_t;
typedef int32_t HIPdevice;
typedef struct ihipCtx_t *HIPcontext;
typedef struct ihipModule_t *HIPmodule;
typedef struct ihipModuleSymbol_t *HIPfunction;
typedef struct ihipStream_t *HIPstream;
typedef void *HIPdeviceptr;
typedef int HIPjit_option;

enum {
	BC_HIP_SUCCESS = 0,
	BC_HIP_JIT_ERROR_LOG_BUFFER = 5,
	BC_HIP_JIT_ERROR_LOG_SIZE = 6,
	BC_HIP_ARG_BUFFER = 1,
	BC_HIP_ARG_SCALAR = 2,
	BC_HIP_F32 = 0,
	BC_HIP_F16 = 1,
	BC_HIP_DISPATCH_ALWAYS = 0,
	BC_HIP_DISPATCH_LABELS = 1,
	BC_HIP_DISPATCH_PVE = 2
};

#include "hip_device_properties.h"

typedef struct { uint64_t offset, size; const char *entry; } bc_hip_kernel;
typedef struct {
	uint8_t kind, buffer; uint64_t offset, size, scalar;
} bc_hip_argument;
typedef struct {
	uint16_t kernel, first_arg, arg_count;
	uint32_t grid[3], block[3], dynamic_lds_bytes;
	uint8_t mode;
} bc_hip_dispatch;
typedef struct { uint64_t offset, size; uint8_t dtype; } bc_hip_weight;
typedef struct {
	const char *model, *architecture, *descriptor_sha256;
	uint16_t dimension, channels, hidden_layers, classes;
	uint8_t has_affine, classifier_bias;
	uint64_t arena_bytes, input_bytes, output_bytes;
	const bc_hip_kernel *kernels; size_t kernel_count;
	const bc_hip_argument *args; size_t arg_count;
	const bc_hip_dispatch *dispatches; size_t dispatch_count;
	const bc_hip_weight *weights; size_t weight_count;
	uint8_t supports_pve;
	uint64_t pve_output_bytes;
} bc_hip_profile;

#ifndef BC_HIP_CATALOG_HEADER
#error "BC_HIP_CATALOG_HEADER must name a verified artifact catalog"
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include BC_HIP_CATALOG_HEADER
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

extern const unsigned char bc_hip_kernels_begin[], bc_hip_kernels_end[];

typedef struct {
	void *library;
	int loaded;
	hipError_t (*init)(unsigned int);
	hipError_t (*device_count)(int *);
	hipError_t (*device_get)(HIPdevice *, int);
	hipError_t (*properties)(bc_hip_device_prop *, int);
	hipError_t (*ctx_retain)(HIPcontext *, HIPdevice);
	hipError_t (*ctx_release)(HIPdevice);
	hipError_t (*ctx_set)(HIPcontext);
	hipError_t (*synchronize)(HIPstream);
	hipError_t (*malloc_fn)(HIPdeviceptr *, size_t);
	hipError_t (*mem_get_info)(size_t *, size_t *);
	hipError_t (*free_fn)(HIPdeviceptr);
	hipError_t (*copy_htod)(HIPdeviceptr, const void *, size_t);
	hipError_t (*copy_dtoh)(void *, HIPdeviceptr, size_t);
	hipError_t (*module_load)(HIPmodule *, const void *, unsigned int,
		HIPjit_option *, void **);
	hipError_t (*module_unload)(HIPmodule);
	hipError_t (*function_get)(HIPfunction *, HIPmodule, const char *);
	hipError_t (*launch)(HIPfunction, unsigned int, unsigned int, unsigned int,
		unsigned int, unsigned int, unsigned int, unsigned int, HIPstream,
		void **, void **);
	const char *(*error_name)(hipError_t);
	const char *(*error_string)(hipError_t);
} bc_hip_driver;

static bc_hip_driver d;

static int symbol(void *destination, size_t size, const char *name) {
	void *address = dlsym(d.library, name);
	if (!address || size != sizeof(address)) return 1;
	memcpy(destination, &address, size);
	return 0;
}

#define LOAD(member, name) do { \
	if (symbol(&d.member, sizeof(d.member), name)) goto fail; \
} while (0)

static int driver_load(void) {
	if (d.loaded) return !d.library;
	d.loaded = 1;
	d.library = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
	if (!d.library) {
		char path[512];
		const char *root = getenv("ROCM_PATH");
		snprintf(path, sizeof(path), "%s/lib/libamdhip64.so",
			root && *root ? root : "/opt/rocm");
		d.library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	}
	if (!d.library) return 1;
	LOAD(init, "hipInit");
	LOAD(device_count, "hipGetDeviceCount");
	LOAD(device_get, "hipDeviceGet");
	if (symbol(&d.properties, sizeof(d.properties),
			"hipGetDevicePropertiesR0600") &&
		symbol(&d.properties, sizeof(d.properties), "hipGetDeviceProperties"))
		goto fail;
	LOAD(ctx_retain, "hipDevicePrimaryCtxRetain");
	LOAD(ctx_release, "hipDevicePrimaryCtxRelease");
	LOAD(ctx_set, "hipCtxSetCurrent");
	LOAD(synchronize, "hipStreamSynchronize");
	LOAD(malloc_fn, "hipMalloc");
	/* Optional diagnostics must not change driver availability. */
	(void)symbol(&d.mem_get_info, sizeof(d.mem_get_info), "hipMemGetInfo");
	LOAD(free_fn, "hipFree");
	LOAD(copy_htod, "hipMemcpyHtoD");
	LOAD(copy_dtoh, "hipMemcpyDtoH");
	LOAD(module_load, "hipModuleLoadDataEx");
	LOAD(module_unload, "hipModuleUnload");
	LOAD(function_get, "hipModuleGetFunction");
	LOAD(launch, "hipModuleLaunchKernel");
	LOAD(error_name, "hipGetErrorName");
	LOAD(error_string, "hipGetErrorString");
	if (d.init(0) != BC_HIP_SUCCESS) goto fail;
	return 0;
fail:
	dlclose(d.library);
	d.library = NULL;
	return 1;
}

static void base_arch(char *destination, size_t capacity, const char *source) {
	size_t length = 0;
	while (source[length] && source[length] != ':' && length + 1 < capacity)
		length++;
	memcpy(destination, source, length);
	destination[length] = '\0';
}

static int inspect(int index, HIPdevice *device, bc_hip_device_prop *properties,
		char *architecture, size_t capacity) {
	if (d.device_get(device, index) != BC_HIP_SUCCESS) return 1;
	memset(properties, 0, sizeof(*properties));
	if (d.properties(properties, index) != BC_HIP_SUCCESS) return 1;
	base_arch(architecture, capacity, properties->gcn_arch_name);
	return !architecture[0];
}

static int hip_adapter_count(void) {
	int count = 0;
	if (driver_load() || d.device_count(&count) != BC_HIP_SUCCESS || count < 0)
		return 0;
	return count;
}

static int hip_adapter_name(int index, char *description, size_t capacity) {
	HIPdevice device;
	bc_hip_device_prop properties;
	char architecture[256];
	if (index < 0 || index >= hip_adapter_count() ||
		inspect(index, &device, &properties, architecture, sizeof(architecture)))
		return 1;
	if (description && capacity)
		snprintf(description, capacity, "%s (%.2f GB, %s%s)", properties.name,
			(double)properties.total_global_mem / (1024.0 * 1024.0 * 1024.0),
			architecture, strcmp(architecture, bc_hip_artifact_profile.architecture) ?
			", no embedded artifact" : "");
	return 0;
}

static int compatible_device(int requested, HIPdevice *device,
		bc_hip_device_prop *properties) {
	int count = hip_adapter_count();
	int first = requested >= 0 ? requested : 0;
	int last = requested >= 0 ? requested + 1 : count;
	for (int index = first; index < last; index++) {
		char architecture[256];
		if (!inspect(index, device, properties, architecture, sizeof(architecture)) &&
			!strcmp(architecture, bc_hip_artifact_profile.architecture))
			return index;
	}
	return -1;
}

static int hip_available(void) {
	HIPdevice device;
	bc_hip_device_prop properties;
	return compatible_device(-1, &device, &properties) >= 0;
}

static int hip_error(hipError_t error, const char *operation) {
	if (error == BC_HIP_SUCCESS) return 0;
	fprintf(stderr, "brainchop-core: HIP %s failed: %s: %s (%d)\n", operation,
		d.error_name ? d.error_name(error) : "error",
		d.error_string ? d.error_string(error) : "unknown", error);
	return 1;
}

static int allocate_buffers(HIPdeviceptr buffers[3], const uint64_t sizes[3]) {
    static const char *names[3] = {"arena", "input", "output"};
    uint64_t planned = 0, allocated = 0;
    for (int i = 0; i < 3; i++) {
        if (sizes[i] > SIZE_MAX || sizes[i] > UINT64_MAX - planned) return 1;
        planned += sizes[i];
    }
    for (int i = 0; i < 3; i++) {
        hipError_t status = d.malloc_fn(&buffers[i], (size_t)sizes[i]);
        if (status != BC_HIP_SUCCESS) {
            char operation[64];
            snprintf(operation, sizeof(operation), "allocating %s", names[i]);
            (void)hip_error(status, operation);
            fprintf(stderr, "brainchop-core: HIP memory: %s request=%llu bytes (%.3f GiB); "
                "planned persistent=%llu bytes (%.3f GiB); already allocated=%llu bytes; ",
                names[i], (unsigned long long)sizes[i], (double)sizes[i] / 1073741824.0,
                (unsigned long long)planned, (double)planned / 1073741824.0,
                (unsigned long long)allocated);
            /* UMA driver free-memory counters need not reflect host pressure. */
            FILE *meminfo = fopen("/proc/meminfo", "r");
            if (meminfo) {
                char line[256]; unsigned long long host_kib = 0;
                while (fgets(line, sizeof(line), meminfo))
                    if (sscanf(line, "MemAvailable: %llu kB", &host_kib) == 1) {
                        if (host_kib <= UINT64_MAX / 1024)
                            fprintf(stderr, "host MemAvailable snapshot=%llu bytes; ", host_kib * 1024);
                        break;
                    }
                fclose(meminfo);
            }
            size_t available = 0, total = 0;
            if (d.mem_get_info && d.mem_get_info(&available, &total) == BC_HIP_SUCCESS)
                fprintf(stderr, "driver free snapshot=%llu of %llu bytes (%.3f of %.3f GiB); "
                    "shared-device snapshot, not a reservation\n",
                    (unsigned long long)available, (unsigned long long)total,
                    (double)available / 1073741824.0, (double)total / 1073741824.0);
            else fprintf(stderr, "driver free snapshot unavailable\n");
            return 1;
        }
        allocated += sizes[i];
    }
    return 0;
}

static int profile_valid(void) {
	const bc_hip_profile *profile = &bc_hip_artifact_profile;
	const uint64_t voxels = (uint64_t)BRAINCHOP_MODEL_DIM *
		BRAINCHOP_MODEL_DIM * BRAINCHOP_MODEL_DIM;
	size_t bundle_size = (size_t)(bc_hip_kernels_end - bc_hip_kernels_begin);
	if (strcmp(profile->model, BC_MODEL_NAME) ||
		profile->dimension != BRAINCHOP_MODEL_DIM ||
		profile->channels != BC_MODEL_CHAN ||
		profile->hidden_layers != BC_MODEL_NHIDDEN ||
		profile->classes != BC_MODEL_NCLASS ||
		profile->has_affine != BC_MODEL_HAS_AFFINE ||
		profile->classifier_bias != BC_MODEL_CLASSIFIER_BIAS ||
		profile->supports_pve != BC_HAS_PVE ||
		profile->input_bytes != voxels * sizeof(uint16_t) ||
		profile->output_bytes != (BC_HAS_PVE ?
			voxels * BC_PRIOR_PLANES * sizeof(float) : voxels) ||
		profile->pve_output_bytes != (BC_HAS_PVE ?
			voxels * BC_PRIOR_PLANES * sizeof(float) : 0) ||
		profile->arena_bytes > SIZE_MAX)
		return 0;
	for (size_t index = 0; index < profile->kernel_count; index++)
		if (!profile->kernels[index].size || !profile->kernels[index].entry ||
			profile->kernels[index].offset > bundle_size ||
			profile->kernels[index].size >
				bundle_size - profile->kernels[index].offset)
			return 0;
	for (size_t index = 0; index < profile->dispatch_count; index++) {
		const bc_hip_dispatch *dispatch = &profile->dispatches[index];
		if (dispatch->kernel >= profile->kernel_count ||
			dispatch->mode > BC_HIP_DISPATCH_PVE ||
			dispatch->first_arg > profile->arg_count ||
			dispatch->arg_count > profile->arg_count - dispatch->first_arg)
			return 0;
		for (int axis = 0; axis < 3; axis++)
			if (!dispatch->grid[axis] || !dispatch->block[axis]) return 0;
	}
	return 1;
}

static int upload(void *arena, uint64_t arena_bytes, const bc_hip_weight *place,
		const float *values, size_t count, void *temporary) {
	if (place->offset > arena_bytes || place->size > arena_bytes - place->offset)
		return 1;
	if (place->dtype == BC_HIP_F16) {
		if (place->size != count * sizeof(uint16_t)) return 1;
		uint16_t *halves = (uint16_t *)temporary;
		for (size_t index = 0; index < count; index++)
			halves[index] = bc_f32_to_f16(values[index]);
	} else if (place->dtype == BC_HIP_F32) {
		if (place->size != count * sizeof(float)) return 1;
		memcpy(temporary, values, count * sizeof(float));
	} else return 1;
	return hip_error(d.copy_htod((char *)arena + place->offset,
		temporary, (size_t)place->size), "uploading weights");
}

static int upload_weights(void *arena, const float *weights) {
	const bc_hip_profile *profile = &bc_hip_artifact_profile;
	size_t expected = (size_t)BC_MODEL_NHIDDEN *
		(BC_MODEL_HAS_AFFINE ? 3u : 1u) + 1u +
		(BC_MODEL_CLASSIFIER_BIAS ? 1u : 0u);
	if (profile->weight_count != expected) return 1;
	size_t maximum = (size_t)27 * BC_MODEL_CHAN * BC_MODEL_CHAN;
	float *temporary = (float *)malloc(maximum * sizeof(*temporary));
	if (!temporary) return 1;
	size_t source_offset = 0, placement = 0;
	int failed = 0;
	for (int layer = 0; layer < BC_MODEL_NHIDDEN && !failed; layer++) {
		int input_channels = layer ? BC_MODEL_CHAN : 1;
		size_t count = (size_t)27 * input_channels * BC_MODEL_CHAN;
		size_t destination = 0;
		for (int output = 0; output < BC_MODEL_CHAN; output++)
			for (int input = 0; input < input_channels; input++)
				for (int depth = 0; depth < 3; depth++)
					for (int height = 0; height < 3; height++)
						for (int width = 0; width < 3; width++) {
							int source_depth = BC_MODEL_LOADER_TRANSPOSE ? depth : width;
							int source_width = BC_MODEL_LOADER_TRANSPOSE ? width : depth;
							size_t source = source_offset +
								((((size_t)source_depth * 3 + height) * 3 + source_width) *
								input_channels + input) * BC_MODEL_CHAN + output;
							temporary[destination++] = weights[source];
						}
		failed = upload(arena, profile->arena_bytes,
			&profile->weights[placement++], temporary, count, temporary);
		source_offset += count;
#if BC_MODEL_HAS_AFFINE
		if (!failed) {
			float diagonal[BC_MODEL_CHAN];
			for (int channel = 0; channel < BC_MODEL_CHAN; channel++)
				diagonal[channel] = weights[source_offset +
					(size_t)channel * BC_MODEL_CHAN + channel];
			failed = upload(arena, profile->arena_bytes,
				&profile->weights[placement++], diagonal, BC_MODEL_CHAN, temporary);
		}
		source_offset += (size_t)BC_MODEL_CHAN * BC_MODEL_CHAN;
		if (!failed)
			failed = upload(arena, profile->arena_bytes,
				&profile->weights[placement++], weights + source_offset,
				BC_MODEL_CHAN, temporary);
		source_offset += BC_MODEL_CHAN;
#endif
	}
	if (!failed) {
		size_t count = (size_t)BC_MODEL_CHAN * BC_MODEL_NCLASS;
		for (int output = 0; output < BC_MODEL_NCLASS; output++)
			for (int input = 0; input < BC_MODEL_CHAN; input++)
				temporary[(size_t)output * BC_MODEL_CHAN + input] =
					weights[source_offset + (size_t)input * BC_MODEL_NCLASS + output];
		failed = upload(arena, profile->arena_bytes,
			&profile->weights[placement++], temporary, count, temporary);
#if BC_MODEL_CLASSIFIER_BIAS
		source_offset += count;
#endif
	}
#if BC_MODEL_CLASSIFIER_BIAS
	if (!failed)
		failed = upload(arena, profile->arena_bytes,
			&profile->weights[placement++], weights + source_offset,
			BC_MODEL_NCLASS, temporary);
#endif
	free(temporary);
	return failed || placement != expected;
}

static int hip_segment(const unsigned char *conformed, float *result,
		const float *weights, const bc_inference_options *options) {
	const bc_hip_profile *profile = &bc_hip_artifact_profile;
	if (!conformed || !result || !weights || !profile_valid()) return 1;
	const int pve = options && options->tissue_planes;
	if (options && (options->roi_enabled || options->class_writer)) return 2;
	if (pve && !profile->supports_pve) return 2;
#if !defined(BC_HIP_PVE_REVISION) || BC_HIP_PVE_REVISION < BC_PVE_PRIOR_REVISION
	if (pve) return 2;
#endif
	HIPdevice device;
	bc_hip_device_prop properties;
	int requested = options ? options->adapter : -1;
	int ordinal = compatible_device(requested, &device, &properties);
	if (ordinal < 0) {
		fprintf(stderr, "brainchop-core: no selected HIP adapter matches %s\n",
			profile->architecture);
		return 1;
	}
	fprintf(stderr, "brainchop-core: HIP, %s %s, %zu offline kernels\n",
		properties.name, profile->architecture, profile->kernel_count);

	HIPcontext context = NULL;
	HIPdeviceptr buffers[3] = {NULL, NULL, NULL};
	HIPmodule *modules = NULL;
	HIPfunction *functions = NULL;
	float *normalized = NULL;
	uint16_t *input = NULL;
	uint8_t *labels = NULL;
	int failed = hip_error(d.ctx_retain(&context, device), "retaining context");
	if (!failed) failed = hip_error(d.ctx_set(context), "activating context");
	const uint64_t voxels = (uint64_t)BRAINCHOP_MODEL_DIM *
		BRAINCHOP_MODEL_DIM * BRAINCHOP_MODEL_DIM;
	uint64_t sizes[3] = {profile->arena_bytes, profile->input_bytes,
		pve ? profile->pve_output_bytes : voxels};
	if (!failed) failed = allocate_buffers(buffers, sizes);
	if (!failed && getenv("BC_HIP_MEMORY"))
		fprintf(stderr, "brainchop-core: HIP allocations: arena=%llu input=%llu output=%llu "
			"high_water=%llu bytes (3 live hipMalloc allocations; driver/private excluded) "
			"model=%s mode=%s\n",
			(unsigned long long)sizes[0], (unsigned long long)sizes[1],
			(unsigned long long)sizes[2],
			(unsigned long long)(sizes[0] + sizes[1] + sizes[2]),
			profile->model, pve ? "pve" : "labels");
	if (!failed) normalized = bc_accel_normalize_input(conformed);
	if (!failed && normalized) input = bc_accel_narrow_input(normalized);
	free(normalized);
	if (!failed && !input) failed = 1;
	if (!failed) failed = hip_error(d.copy_htod(buffers[1], input,
		(size_t)profile->input_bytes), "uploading input");
	free(input);
	if (!failed) failed = upload_weights(buffers[0], weights);

	if (!failed) {
		modules = (HIPmodule *)calloc(profile->kernel_count, sizeof(*modules));
		functions = (HIPfunction *)calloc(profile->kernel_count, sizeof(*functions));
		failed = !modules || !functions;
	}
	for (size_t index = 0; index < profile->kernel_count && !failed; index++) {
		const bc_hip_kernel *kernel = &profile->kernels[index];
		char log[8192] = {0};
		HIPjit_option options_array[2] = {
			BC_HIP_JIT_ERROR_LOG_SIZE, BC_HIP_JIT_ERROR_LOG_BUFFER};
		void *option_values[2] = {(void *)(uintptr_t)sizeof(log), log};
		hipError_t status = d.module_load(&modules[index],
			bc_hip_kernels_begin + kernel->offset, 2, options_array, option_values);
		if (status != BC_HIP_SUCCESS) {
			(void)hip_error(status, "loading code object");
			if (log[0]) fprintf(stderr, "brainchop-core: HIP load log: %s\n", log);
			failed = 1;
		} else failed = hip_error(d.function_get(&functions[index], modules[index],
			kernel->entry), "resolving kernel");
	}
	const int time_gpu = getenv("BC_TIME_GPU") != NULL;
	struct timespec start = {0, 0}, end = {0, 0};
	size_t launched = 0;
	if (time_gpu && !failed) {
		(void)d.synchronize(NULL);
		clock_gettime(CLOCK_MONOTONIC, &start);
	}
	for (size_t index = 0; index < profile->dispatch_count && !failed; index++) {
		const bc_hip_dispatch *dispatch = &profile->dispatches[index];
		if ((dispatch->mode == BC_HIP_DISPATCH_LABELS && pve) ||
			(dispatch->mode == BC_HIP_DISPATCH_PVE && !pve)) continue;
		uint64_t *values = (uint64_t *)calloc(dispatch->arg_count, sizeof(*values));
		void **parameters = (void **)calloc(dispatch->arg_count, sizeof(*parameters));
		if (!values || !parameters) failed = 1;
		for (size_t argument = 0; argument < dispatch->arg_count && !failed;
				argument++) {
			const bc_hip_argument *item =
				&profile->args[dispatch->first_arg + argument];
			if (item->kind == BC_HIP_ARG_BUFFER && item->buffer < 3 &&
				item->offset <= sizes[item->buffer] &&
				item->size <= sizes[item->buffer] - item->offset) {
				HIPdeviceptr pointer = (char *)buffers[item->buffer] + item->offset;
				memcpy(&values[argument], &pointer, sizeof(pointer));
			} else if (item->kind == BC_HIP_ARG_SCALAR) values[argument] = item->scalar;
			else failed = 1;
			parameters[argument] = &values[argument];
		}
		if (!failed)
			failed = hip_error(d.launch(functions[dispatch->kernel],
				dispatch->grid[0], dispatch->grid[1], dispatch->grid[2],
				dispatch->block[0], dispatch->block[1], dispatch->block[2],
				dispatch->dynamic_lds_bytes, NULL, parameters, NULL),
				"launching kernel");
		if (!failed) launched++;
		free(parameters); free(values);
	}
	if (!failed) failed = hip_error(d.synchronize(NULL), "synchronizing kernels");
	if (time_gpu && !failed) {
		clock_gettime(CLOCK_MONOTONIC, &end);
		fprintf(stderr, "brainchop-core: HIP %zu dispatches in %.3f s\n",
			launched, (double)(end.tv_sec - start.tv_sec) +
			1e-9 * (double)(end.tv_nsec - start.tv_nsec));
	}
	if (!failed && pve)
		failed = hip_error(d.copy_dtoh(options->tissue_planes, buffers[2],
			(size_t)profile->pve_output_bytes), "downloading tissue priors");
	if (!failed && !pve) labels = (uint8_t *)malloc((size_t)voxels);
	if (!failed && !pve && !labels) failed = 1;
	if (!failed && !pve) failed = hip_error(d.copy_dtoh(labels, buffers[2],
		(size_t)voxels), "downloading labels");
	if (!failed && !pve)
		for (size_t index = 0; index < (size_t)voxels; index++)
			result[index] = (float)labels[index];
	free(labels);
	if (modules)
		for (size_t index = profile->kernel_count; index-- > 0;)
			if (modules[index]) (void)d.module_unload(modules[index]);
	free(functions); free(modules);
	for (int index = 3; index-- > 0;)
		if (buffers[index]) (void)d.free_fn(buffers[index]);
	if (context) {
		(void)d.ctx_set(NULL);
		(void)d.ctx_release(device);
	}
	return failed ? 1 : 0;
}

const bc_backend bc_hip_backend = {
	"hip", hip_available, hip_segment, hip_adapter_count, hip_adapter_name
};
