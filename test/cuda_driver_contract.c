/* Driver failure/dispatch tests, no NVIDIA driver or GPU required. Include the
 * adapter so errors can be injected at its private Driver API boundary. */
#define _POSIX_C_SOURCE 200809L
#include "../src/backend_cuda_driver.c"
#include <assert.h>

static int lookups, missing, unloaded, destroyed, restores, event_failure, cleanup_failure;
static CUresult launch_failure;
static CUcontext caller = (CUcontext)(uintptr_t)0x1230;
static CUfunction observed;
static unsigned grid, block, shared_bytes;
static void *first_pointer;
static int argument_dilation;
static CUresult ok(void) { return 0; }
static CUresult get_device(CUdevice *device, int ordinal) { *device = ordinal; return ordinal < 2 ? 0 : 101; }
static CUresult get_name(char *name, int capacity, CUdevice device) {
    snprintf(name, capacity, "test GPU %d", device); return 0;
}
static CUresult memory(size_t *bytes, CUdevice device) { (void)device; *bytes = (size_t)8 << 30; return 0; }
static CUresult attribute(int *value, int id, CUdevice device) { (void)device; *value = id == 75 ? 8 : 6; return 0; }
static CUresult current(CUcontext *ctx) { *ctx = caller; return 0; }
static CUresult create(CUcontext *ctx, unsigned flags, CUdevice device) {
    assert(flags == 0 && device == 1); *ctx = (CUcontext)(uintptr_t)0x2000; return 0;
}
static CUresult destroy(CUcontext ctx) { assert(ctx); destroyed++; return 0; }
static CUresult restore(CUcontext ctx) { assert(ctx == caller); restores++; return 0; }
static CUresult load_module(CUmodule *mod, const void *data) {
    assert(data == bc_cuda_module_begin); *mod = (CUmodule)(uintptr_t)0x3000; return 0;
}
static CUresult unload(CUmodule mod) { assert(mod); unloaded++; return cleanup_failure ? 1 : 0; }
static CUresult function(CUfunction *out, CUmodule mod, const char *name) {
    assert(mod && name && *name); lookups++;
    if (lookups == missing) return 500;
    *out = (CUfunction)(uintptr_t)(0x4000 + lookups); return 0;
}
static CUresult allocation(CUdeviceptr *ptr, size_t bytes) { assert(bytes == 8); *ptr = 0x5000; return 0; }
static CUresult release(CUdeviceptr ptr) { assert(ptr); return 0; }
static CUresult zero(CUdeviceptr ptr, unsigned char value, size_t bytes) {
    assert(ptr && value == 0 && bytes == 8); return 0;
}
static CUresult readback(void *dest, CUdeviceptr src, size_t bytes) {
    assert(src && bytes == 8); *(unsigned long long *)dest = 42; return 0;
}
static CUresult event_new(CUevent *event, unsigned flags) {
    assert(flags == 0); *event = (CUevent)(uintptr_t)0x6000; return event_failure ? 2 : 0;
}
static CUresult event_drop(CUevent event) { assert(event); return 0; }
static CUresult event_mark(CUevent event, CUstream stream) { assert(event && !stream); return 0; }
static CUresult event_wait(CUevent event) { assert(event); return 0; }
static CUresult elapsed(float *ms, CUevent start, CUevent stop) { assert(start && stop); *ms = 1; return 0; }
static CUresult kernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz, unsigned sm, CUstream stream, void **args, void **extra) {
    assert(f && gx && bx && gy == 1 && gz == 1 && by == 1 && bz == 1 && !stream && !extra);
    observed = f; grid = gx; block = bx; shared_bytes = sm;
    memcpy(&first_pointer, args[0], sizeof(first_pointer));
    if (f == scalar_3_512) {
        assert(*(int *)args[6] == CUDA_ROWS && *(int *)args[7] == 1 && *(int *)args[8] == BC_MODEL_CHAN && *(int *)args[9] == 1);
        argument_dilation = *(int *)args[10];
    } else if (f == wmma_24) argument_dilation = *(int *)args[7];
    return launch_failure;
}
static void bad_candidate(const bc_act *a, bc_act *b, const bc_act *w, int dilation) {
    (void)a; (void)b; (void)w; (void)dilation;
    remember(701); /* cuCtxSynchronize will still succeed: stale output hazard. */
}
static void good_candidate(const bc_act *a, bc_act *b, const bc_act *w, int dilation) {
    (void)a; (void)b; (void)w; (void)dilation;
}
int main(void) {
    library = (gpu_library)(uintptr_t)1; /* driver_load already succeeded */
    d.device_get = get_device; d.device_name = get_name; d.device_memory = memory;
    d.device_attribute = attribute; d.ctx_current = current; d.ctx_create = create;
    d.ctx_destroy = destroy; d.ctx_set = restore; d.module_load = load_module;
    d.module_unload = unload; d.function_get = function; d.sync = ok;
    d.alloc = allocation; d.free = release; d.memset_d8 = zero; d.copy_dtoh = readback;
    d.event_create = event_new; d.event_destroy = event_drop; d.event_record = event_mark;
    d.event_sync = event_wait; d.event_elapsed = elapsed; d.launch = kernel;


    cuda_layer_choice plan[BC_MODEL_NHIDDEN];
    assert(!cuda_parse_schedule("s3x512", BC_MODEL_NHIDDEN, 0, plan));
    assert(!plan[0].warps && plan[0].scalar == cuda_default_variant());
    assert(!cuda_parse_schedule("w8", BC_MODEL_NHIDDEN, 1, plan));
    assert(!plan[0].warps && plan[1].warps == 8);
    assert(cuda_parse_schedule("w8", BC_MODEL_NHIDDEN, 0, plan));
    assert(cuda_parse_schedule("w0", BC_MODEL_NHIDDEN, 1, plan));
    assert(cuda_parse_schedule("s0x128", BC_MODEL_NHIDDEN, 1, plan));
    assert(cuda_parse_schedule("s3x512garbage", BC_MODEL_NHIDDEN, 1, plan));
    assert(cuda_parse_schedule("s3x512,", BC_MODEL_NHIDDEN, 1, plan));
    assert(cuda_parse_schedule("s3x512,w8", BC_MODEL_NHIDDEN, 1, plan));
    /* At CS==24 weights are no longer staged in shared memory, so w32 fits 32 KiB <= 48 KiB. */
    if (CUDA_CS == 24) assert(!cuda_parse_schedule("w32", BC_MODEL_NHIDDEN, 1, plan));
    missing = 5;
    assert(gpu_begin(1) == 1);
    assert(lookups == 5 && unloaded == 1 && destroyed == 1 && restores == 1 && !module && !context);
    missing = lookups = 0;
    assert(gpu_begin(1) == 0 && lookups == 36 && selected == 1);
    assert(strstr(module_key, "module-abi3-") == module_key);
    bc_act *a = (bc_act *)(uintptr_t)0x7000, *b = (bc_act *)(uintptr_t)0x8000;
    gpu_conv(3, 512, a, b, a, NULL, NULL, NULL, CUDA_ROWS, 1, BC_MODEL_CHAN, 1, 4, 27 * CUDA_CS * 2);
    assert(observed == scalar_3_512 && block == 512 && grid == (CUDA_NVOX + 1535) / 1536);
    assert(shared_bytes == 27 * CUDA_CS * 2 && first_pointer == a && argument_dilation == 4);
    gpu_wmma(24, a, b, a, NULL, NULL, NULL, CUDA_ROWS, 31);
    assert(observed == wmma_24 && block == 768 && grid == ((CUDA_NVOX + 15) / 16 + 23) / 24);
    size_t w_shared = (CUDA_CS == 24) ? 0 : (size_t)27 * CUDA_CS * CUDA_CS * 2;
    assert(shared_bytes == w_shared + 24 * CUDA_WMMA_SCRATCH && argument_dilation == 31);
    gpu_priors(a, (float *)b, NULL, NULL, a, NULL, NULL, 17, BC_MODEL_CHAN, BC_MODEL_NCLASS);
    assert(observed == tissue_priors_kernel && grid == (17u * CUDA_DIM + CUDA_THREADS - 1) / CUDA_THREADS && block == CUDA_THREADS);
    gpu_wmma(0, a, b, a, NULL, NULL, NULL, CUDA_ROWS, 1);
    assert(gpu_sync() != 0 && gpu_last_error() != 0);
    gpu_conv(0, 0, a, b, a, NULL, NULL, NULL, CUDA_ROWS, 1, BC_MODEL_CHAN, 1, 1, 0);
    assert(gpu_sync() != 0 && gpu_last_error() != 0);

    launch_failure = 701;
    gpu_fill(a, 123);
    launch_failure = 0;
    gpu_fill(a, 123); /* subsequent success must not erase the first failure */
    assert(gpu_sync() == 701 && gpu_last_error() == 701 && gpu_last_error() == 0);
    cuda_tune_candidate candidates[] = {{3, 512, bad_candidate}, {1, 128, good_candidate}};
    int chosen = -1;
    launch_failure = 701;
    assert(cuda_tune(a, b, a, candidates, 2, 0, &chosen) == 1);
    launch_failure = 0;
    assert(gpu_last_error() == 0); /* failed initial fill cannot poison fallback */
    observed = NULL;
    gpu_fill(a, 123);
    assert(observed == fill_pattern_kernel && gpu_sync() == 0);
    assert(cuda_tune(a, b, a, candidates, 2, 0, &chosen) == 0 && chosen == 1);
    event_failure = 1;
    assert(cuda_tune(a, b, a, candidates + 1, 1, 0, &chosen) == 1);
    assert(gpu_last_error() == 0); /* candidate failure cannot poison inference */
#ifdef _WIN32
    _putenv_s("BRAINCHOPC_CUDA_PROFILE", "1");
#else
    setenv("BRAINCHOPC_CUDA_PROFILE", "1", 1);
#endif
    cuda_stage stage;
    stage_begin(&stage);
    assert(!stage.active && g_timing_error); /* no use of an uncreated event */
    stage_end(&stage, 0);
    event_failure = 0;
    cleanup_failure = 1;
    /* The unload stub returns 1 and has no cuGetErrorString, so the two stderr
     * lines that follow ("unloading module failed: 1 (unknown error)", "cleanup
     * did not complete") are this test's expected output, not a driver fault. */
    fputs("cuda_driver_contract: injecting a cuModuleUnload failure; the next two warnings are expected\n", stderr);
    assert(cuda_finish(0) == 0 && unloaded == 2 && destroyed == 2 && restores == 2);
    assert(!module && !context); /* all releases attempted despite unload error */
    assert(cuda_finish(1) == 1); /* never hide an earlier inference failure */
    puts("PASS: module resolution/cleanup, adapter binding, launch ABI, rejected launches and failed timing");
    return 0;
}
