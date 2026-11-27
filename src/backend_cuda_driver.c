/* Driver-only host adapter for the checked-in native modules. No CUDA SDK/cudart.
 * The inference loop and tuner are shared with backend_cuda.cu. As with the
 * legacy backend, calls to this backend must be serialized. */
#include "backend.h"
#include "backend_accel.h"
#include "cuda_driver_min.h"
#include "cuda_modules/catalog.h"
#include "cuda_variants.h"
#include "f16.h"
#include "model_specs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE gpu_library;
#define GPU_OPEN() LoadLibraryA("nvcuda.dll")
#define GPU_SYMBOL GetProcAddress
#define GPU_CLOSE FreeLibrary
typedef FARPROC gpu_symbol;
#else
#include <dlfcn.h>
typedef void *gpu_library;
#define GPU_OPEN() dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL)
#define GPU_SYMBOL dlsym
#define GPU_CLOSE dlclose
typedef void *gpu_symbol;
#endif

/* The generated geometry is paired with the embedded module. */
typedef uint16_t bc_act;
#define gpu_half bc_f32_to_f16
#define BC_CUDA_ASSERT _Static_assert
#define CUDA_DIM BC_CUDA_MODULE_DIM
#define CUDA_MAX_CHANNELS BC_CUDA_MODULE_MAX_CHANNELS
#define CUDA_CS BC_CUDA_MODULE_CS
#define CUDA_THREADS BC_CUDA_MODULE_THREADS
#define CUDA_REDUCE_THREADS BC_CUDA_MODULE_REDUCE_THREADS
#define CUDA_REDUCE_BLOCKS BC_CUDA_MODULE_REDUCE_BLOCKS
#define CUDA_CONV_VPT BC_CUDA_MODULE_CONV_VPT
#define CUDA_CONV_THREADS BC_CUDA_MODULE_CONV_THREADS
#define CUDA_WMMA_TILE BC_CUDA_MODULE_WMMA_TILE
#define CUDA_WMMA_WARPS BC_CUDA_MODULE_WMMA_WARPS
#define CUDA_WMMA_SCRATCH BC_CUDA_MODULE_WMMA_SCRATCH
#define CUDA_NVOX ((size_t)CUDA_DIM * CUDA_DIM * CUDA_DIM)
#define CUDA_ROWS ((int)(CUDA_DIM * CUDA_DIM))
#define CUDA_MOMENT_STRIPES 256
_Static_assert(CUDA_DIM == BRAINCHOP_MODEL_DIM, "module/model dimensions differ");
_Static_assert(BC_MODEL_CHAN <= CUDA_CS, "model exceeds module stride");
_Static_assert(!BC_MODEL_HAS_AFFINE || BC_MODEL_CHAN == CUDA_CS, "affine stride mismatch");
_Static_assert(sizeof(void *) == 8 && sizeof(size_t) == 8, "64-bit module ABI required");
_Static_assert(BC_CUDA_MODULE_ABI == 3, "review host launch contract for new module ABI");

typedef CUresult gpu_error_t;
typedef CUevent gpu_event;
enum { GPU_SUCCESS = 0, GPU_HTOD = 1, GPU_DTOH = 2 };
typedef struct { char name[256]; size_t totalGlobalMem; int major, minor; } gpu_properties;
static bc_cuda_driver d;
static gpu_library library;
static CUcontext context, previous;
static CUmodule module;
static int selected;
static CUresult launch_error;
static char module_key[160];
#define CUDA_TUNE_CODEGEN module_key
extern const unsigned char bc_cuda_module_begin[], bc_cuda_module_end[];

#define X(V, T, symbol) static CUfunction scalar_##V##_##T;
BC_CUDA_MODULE_SCALAR(X)
#undef X
#define X(W, symbol) static CUfunction wmma_##W;
BC_CUDA_MODULE_WMMA(X)
#undef X
#define X(id, symbol) static CUfunction id;
BC_CUDA_MODULE_AUX(X)
#undef X

static int cuda_error(CUresult code, const char *operation) {
    if (!code) return 0;
    const char *message = NULL;
    if (d.error_string) d.error_string(code, &message);
    fprintf(stderr, "brainchop-core: CUDA Driver API %s failed: %d (%s)\n",
        operation, code, message ? message : "unknown error");
    return 1;
}

static int driver_load(void) {
    if (library) return 0;
    gpu_library handle = GPU_OPEN();
    if (!handle) return 1;
    bc_cuda_driver loaded;
#define X(name, symbol, args) do { \
    gpu_symbol address = GPU_SYMBOL(handle, #symbol); \
    _Static_assert(sizeof(address) == sizeof(loaded.name), "driver function pointer ABI"); \
    if (!address) { GPU_CLOSE(handle); return 1; } \
    memcpy(&loaded.name, &address, sizeof(address)); \
} while (0);
    BC_CUDA_DRIVER_FUNCTIONS(X)
#undef X
    int version = 0;
    if (loaded.init(0) || loaded.driver_version(&version) || version < BC_CUDA_MODULE_DRIVER_API) {
        GPU_CLOSE(handle);
        return 1;
    }
    d = loaded;
    library = handle;
    return 0;
}

static int cuda_adapter_count(void) {
    int count = 0;
    if (driver_load() || d.device_count(&count)) return 0;
    return count;
}
static int cuda_available(void) { return cuda_adapter_count() > 0; }
static CUresult gpu_properties_get(gpu_properties *props, int ordinal) {
    CUdevice device;
    CUresult error = d.device_get(&device, ordinal);
    if (!error) error = d.device_name(props->name, sizeof(props->name), device);
    if (!error) error = d.device_memory(&props->totalGlobalMem, device);
    if (!error) error = d.device_attribute(&props->major, 75, device);
    if (!error) error = d.device_attribute(&props->minor, 76, device);
    return error;
}
static int cuda_adapter_name(int index, char *description, size_t capacity) {
    gpu_properties props;
    if (index < 0 || index >= cuda_adapter_count() || gpu_properties_get(&props, index)) return 1;
    if (description && capacity)
        snprintf(description, capacity, "%s (%.2f GB, sm_%d%d)", props.name,
            (double)props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), props.major, props.minor);
    return 0;
}
static CUresult gpu_device(int *ordinal) { *ordinal = selected; return 0; }
static CUresult gpu_driver_version(int *version) { return d.driver_version(version); }

static int gpu_end(void) {
    int failed = 0;
    if (module) failed |= cuda_error(d.module_unload(module), "unloading module");
    if (context) {
        failed |= cuda_error(d.ctx_destroy(context), "destroying context");
        failed |= cuda_error(d.ctx_set(previous), "restoring prior context");
    }
    module = NULL;
    context = previous = NULL;
    launch_error = 0;
    return failed;
}
static int env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && !strcmp(value, "1");
}
static int gpu_begin(int ordinal) {
    if (driver_load()) {
        fprintf(stderr, "brainchop-core: CUDA modules require an NVIDIA driver with API >= %d\n",
            BC_CUDA_MODULE_DRIVER_API);
        return 1;
    }
    selected = ordinal < 0 ? 0 : ordinal;
    CUdevice device;
    gpu_properties props;
    if (cuda_error(d.device_get(&device, selected), "selecting device") ||
        cuda_error(gpu_properties_get(&props, selected), "reading device properties")) return 1;
    if (props.major < 6) {
        fprintf(stderr, "brainchop-core: CUDA module requires sm60 or newer; found %s sm%d%d\n",
            props.name, props.major, props.minor);
        return 1;
    }
    if (cuda_error(d.ctx_current(&previous), "saving prior context") ||
        cuda_error(d.ctx_create(&context, 0, device), "creating context")) return 1;
    if (cuda_error(d.module_load(&module, bc_cuda_module_begin), "loading embedded module")) goto fail;
#define X(V, T, symbol) if (cuda_error(d.function_get(&scalar_##V##_##T, module, symbol), symbol)) goto fail;
    BC_CUDA_MODULE_SCALAR(X)
#undef X
#define X(W, symbol) if (cuda_error(d.function_get(&wmma_##W, module, symbol), symbol)) goto fail;
    BC_CUDA_MODULE_WMMA(X)
#undef X
#define X(id, symbol) if (cuda_error(d.function_get(&id, module, symbol), symbol)) goto fail;
    BC_CUDA_MODULE_AUX(X)
#undef X
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t bytes = (uintptr_t)bc_cuda_module_end - (uintptr_t)bc_cuda_module_begin;
    for (size_t i = 0; i < bytes; i++) hash = (hash ^ bc_cuda_module_begin[i]) * UINT64_C(1099511628211);
    snprintf(module_key, sizeof(module_key), "module-abi%d-%016llx-jit%d%d%d%d",
        BC_CUDA_MODULE_ABI, (unsigned long long)hash,
        env_enabled("CUDA_FORCE_PTX_JIT"), env_enabled("CUDA_DISABLE_PTX_JIT"),
        env_enabled("CUDA_FORCE_JIT"), env_enabled("CUDA_DISABLE_JIT"));
    fprintf(stderr, "brainchop-core: CUDA Driver API, %s sm%d%d, module ABI %d\n",
        props.name, props.major, props.minor, BC_CUDA_MODULE_ABI);
    return 0;
fail:
    gpu_end();
    return 1;
}

static CUresult gpu_alloc(void **pointer, size_t bytes) {
    CUdeviceptr address = 0;
    CUresult error = d.alloc(&address, bytes);
    void *value = (void *)(uintptr_t)address;
    memcpy(pointer, &value, sizeof(value));
    return error;
}
static CUresult gpu_free(void *pointer) { return pointer ? d.free((CUdeviceptr)(uintptr_t)pointer) : 0; }
static CUresult gpu_zero(void *pointer, int value, size_t bytes) {
    return d.memset_d8((CUdeviceptr)(uintptr_t)pointer, (unsigned char)value, bytes);
}
static CUresult gpu_copy(void *dest, const void *src, size_t bytes, int kind) {
    return kind == GPU_HTOD ? d.copy_htod((CUdeviceptr)(uintptr_t)dest, src, bytes)
        : d.copy_dtoh(dest, (CUdeviceptr)(uintptr_t)src, bytes);
}
static CUresult remember(CUresult error) {
    if (error && !launch_error) launch_error = error;
    return error;
}
static CUresult gpu_last_error(void) { CUresult error = launch_error; launch_error = 0; return error; }
static CUresult gpu_sync(void) { remember(d.sync()); return launch_error; }
static CUresult gpu_event_create(gpu_event *event) { return remember(d.event_create(event, 0)); }
static CUresult gpu_event_destroy(gpu_event event) { return remember(d.event_destroy(event)); }
static CUresult gpu_event_record(gpu_event event) { return remember(d.event_record(event, NULL)); }
static CUresult gpu_event_sync(gpu_event event) { return remember(d.event_sync(event)); }
static CUresult gpu_event_elapsed(float *ms, gpu_event start, gpu_event stop) {
    return remember(d.event_elapsed(ms, start, stop));
}
static void launch(CUfunction function, unsigned blocks, unsigned threads, size_t shared, void **args) {
    if (!launch_error)
        remember(d.launch(function, blocks, 1, 1, threads, 1, 1, (unsigned)shared, NULL, args, NULL));
}
static void gpu_conv(int vpt, int threads, const bc_act *source, bc_act *dest,
        const bc_act *weights, double *sums, const unsigned int *row_list,
        const float *row_weights, int row_count, int ci, int co, int stride,
        int dilation, size_t shared) {
    CUfunction function = NULL;
#define X(V, T, symbol) if (vpt == V && threads == T) function = scalar_##V##_##T;
    BC_CUDA_MODULE_SCALAR(X)
#undef X
    if (!function) { remember(1); return; }
    size_t mom_shared = sums ? (size_t)(threads / 32) * 2 * CUDA_CS * sizeof(double) : 0;
    size_t sm = shared > mom_shared ? shared : mom_shared;
    size_t total_vox = row_list ? (size_t)row_count * CUDA_DIM : CUDA_NVOX;
    void *args[] = {&source, &dest, &weights, &sums, &row_list, &row_weights,
        &row_count, &ci, &co, &stride, &dilation};
    unsigned blocks = (unsigned)((total_vox + (size_t)threads * vpt - 1) / ((size_t)threads * vpt));
    launch(function, blocks, (unsigned)threads, sm, args);
}
static void gpu_wmma(int warps, const bc_act *source, bc_act *dest, const bc_act *weights,
        double *sums, const unsigned int *row_list, const float *row_weights,
        int row_count, int dilation) {
    CUfunction function = NULL;
#define X(W, symbol) if (warps == W) function = wmma_##W;
    BC_CUDA_MODULE_WMMA(X)
#undef X
    if (!function) { remember(1); return; }
    size_t w_shared = (CUDA_CS == 24) ? 0 : (size_t)27 * CUDA_CS * CUDA_CS * sizeof(bc_act);
    size_t shared = w_shared + (size_t)warps * CUDA_WMMA_SCRATCH;
    size_t tiles = row_list ? (size_t)row_count * (CUDA_DIM / CUDA_WMMA_TILE)
        : (CUDA_NVOX / CUDA_WMMA_TILE);
    void *args[] = {&source, &dest, &weights, &sums, &row_list, &row_weights,
        &row_count, &dilation};
    launch(function, (unsigned)((tiles + (size_t)warps - 1) / (size_t)warps),
        (unsigned)(warps * 32), shared, args);
}
static void gpu_probe(int *flag) { void *args[] = {&flag}; launch(wmma_probe_kernel, 1, 1, 0, args); }
static void gpu_fill(bc_act *buffer, size_t count) {
    void *args[] = {&buffer, &count}; launch(fill_pattern_kernel, 256, 256, 0, args);
}
static void gpu_checksum(const bc_act *buffer, unsigned long long *sum, size_t count) {
    void *args[] = {&buffer, &sum, &count}; launch(checksum_kernel, 256, 256, 0, args);
}
static void gpu_moments(const double *sums, float *mul, float *add,
        const float *scale, const float *bias, int channels) {
    void *args[] = {&sums, &mul, &add, &scale, &bias, &channels};
    launch(moments_finish_kernel, 1, CUDA_MAX_CHANNELS, 0, args);
}
static void gpu_norm(bc_act *buffer, const float *mul, const float *add,
        const unsigned int *row_list, int row_count) {
    size_t pairs = (size_t)row_count * CUDA_DIM * (CUDA_CS / 2);
    unsigned blocks = (unsigned)((pairs + CUDA_THREADS - 1) / CUDA_THREADS);
    void *args[] = {&buffer, &mul, &add, &row_list, &row_count};
    launch(norm_gelu_kernel, blocks, CUDA_THREADS, 0, args);
}
static void gpu_fill_rows(bc_act *buffer, const unsigned int *pure_list, int npure, int rep_row) {
    size_t u4s = (size_t)npure * ((CUDA_DIM * CUDA_CS * sizeof(bc_act)) / 16);
    unsigned blocks = (unsigned)((u4s + CUDA_THREADS - 1) / CUDA_THREADS);
    void *args[] = {&buffer, &pure_list, &npure, &rep_row};
    launch(fill_rows_kernel, blocks, CUDA_THREADS, 0, args);
}
static void gpu_classify_u8(const bc_act *input, unsigned char *labels,
        const float *mul, const float *add, const bc_act *weights, const bc_act *bias,
        const unsigned int *row_list, int row_count, int channels, int classes) {
    size_t vox = (size_t)row_count * CUDA_DIM;
    unsigned blocks = (unsigned)((vox + CUDA_THREADS - 1) / CUDA_THREADS);
    void *args[] = {&input, &labels, &mul, &add, &weights, &bias, &row_list, &row_count,
        &channels, &classes};
    launch(classify_u8_kernel, blocks, CUDA_THREADS, 0, args);
}
static void gpu_fill_labels_u8(unsigned char *labels, const unsigned int *pure_list,
        int npure, int rep_row) {
    size_t u4s = (size_t)npure * (CUDA_DIM / 16);
    unsigned blocks = (unsigned)((u4s + CUDA_THREADS - 1) / CUDA_THREADS);
    void *args[] = {&labels, &pure_list, &npure, &rep_row};
    launch(fill_labels_u8_kernel, blocks, CUDA_THREADS, 0, args);
}
static void gpu_priors(const bc_act *input, float *planes, const float *mul, const float *add,
        const bc_act *weights, const bc_act *bias, const unsigned int *row_list,
        int row_count, int channels, int classes) {
    size_t vox = (size_t)row_count * CUDA_DIM;
    unsigned blocks = (unsigned)((vox + CUDA_THREADS - 1) / CUDA_THREADS);
    void *args[] = {&input, &planes, &mul, &add, &weights, &bias, &row_list, &row_count,
        &channels, &classes};
    launch(tissue_priors_kernel, blocks, CUDA_THREADS, 0, args);
}
static void gpu_fill_priors(float *planes, const unsigned int *pure_list, int npure, int rep_row) {
    size_t vox = (size_t)npure * CUDA_DIM;
    unsigned blocks = (unsigned)((vox + CUDA_THREADS - 1) / CUDA_THREADS);
    void *args[] = {&planes, &pure_list, &npure, &rep_row};
    launch(fill_priors_kernel, blocks, CUDA_THREADS, 0, args);
}
static void gpu_pve_blur(const float *src, float *dst, int axis) {
    unsigned blocks = (unsigned)((3 * CUDA_NVOX + CUDA_THREADS - 1) / CUDA_THREADS);
    void *args[] = {&src, &dst, &axis};
    launch(pve_blur_kernel, blocks, CUDA_THREADS, 0, args);
}
static void gpu_pve_support(const float *blurred, float *planes) {
    unsigned blocks = (unsigned)((3 * CUDA_NVOX + CUDA_THREADS - 1) / CUDA_THREADS);
    void *args[] = {&blurred, &planes};
    launch(pve_support_kernel, blocks, CUDA_THREADS, 0, args);
}

#include "backend_cuda_host.h"
