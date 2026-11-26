/* Build with the ordinary C compiler and an embedded CUDA fatbin.
 * Resolves the complete catalog, then checks nonzero identity convolutions and
 * the tuning support kernels through the Driver API. No CUDA SDK or cudart. */
#include "cuda_driver_min.h"
#include "model_specs.h"
#include "pve_constants.h"
#include "f16.h"
#include <math.h>
#include "cuda_modules/catalog.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#define LIB_HANDLE HMODULE
#define LIB_OPEN() LoadLibraryA("nvcuda.dll")
#define LIB_SYMBOL FARPROC
#define GET_SYMBOL GetProcAddress
#define LIB_CLOSE FreeLibrary
#else
#include <dlfcn.h>
#define LIB_HANDLE void *
#define LIB_OPEN() dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL)
#define LIB_SYMBOL void *
#define GET_SYMBOL dlsym
#define LIB_CLOSE dlclose
#endif

extern const unsigned char bc_cuda_module_begin[], bc_cuda_module_end[];
_Static_assert(sizeof(size_t) == 8 && sizeof(CUdeviceptr) == 8, "64-bit module ABI required");
static bc_cuda_driver d;

static int failed(CUresult code, const char *operation) {
    if (!code) return 0;
    const char *message = NULL;
    d.error_string(code, &message);
    fprintf(stderr, "%s: CUDA %d (%s)\n", operation, code, message ? message : "unknown error");
    return 1;
}
#define CHECK(call) do { if (failed((call), #call)) goto cleanup; } while (0)

static const char *const symbols[] = {
#define X(v, t, name) name,
    BC_CUDA_MODULE_SCALAR(X)
#undef X
#define X(w, name) name,
    BC_CUDA_MODULE_WMMA(X)
#undef X
#define X(id, name) name,
    BC_CUDA_MODULE_AUX(X)
#undef X
};

int main(int argc, char **argv) {
    size_t bytes = (uintptr_t)bc_cuda_module_end - (uintptr_t)bc_cuda_module_begin;
    if (bytes < 16 || memcmp(bc_cuda_module_begin, "\x50\xed\x55\xba", 4)) {
        fprintf(stderr, "invalid embedded CUDA fatbin\n");
        return 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--embedded-only")) {
        printf("embedded CUDA module: %zu bytes, ABI %d\n", bytes, BC_CUDA_MODULE_ABI);
        return 0;
    }
    int ordinal = 0;
    if (argc > 2) return 2;
    if (argc == 2) {
        char *end = NULL;
        errno = 0;
        long parsed = strtol(argv[1], &end, 10);
        if (errno || end == argv[1] || *end || parsed < 0 || parsed > INT_MAX) return 2;
        ordinal = (int)parsed;
    }
    LIB_HANDLE library = LIB_OPEN();
    if (!library) {
        fprintf(stderr, "NVIDIA driver library unavailable\n");
        return 1;
    }
#define X(name, symbol, args) do { \
        LIB_SYMBOL address = GET_SYMBOL(library, #symbol); \
        _Static_assert(sizeof(address) == sizeof(d.name), "driver pointer ABI"); \
        if (!address) { fprintf(stderr, "missing driver symbol: %s\n", #symbol); LIB_CLOSE(library); return 1; } \
        memcpy(&d.name, &address, sizeof(address)); \
    } while (0);
    BC_CUDA_DRIVER_FUNCTIONS(X)
#undef X
    int result = 1, count = 0, version = 0, major = 0, minor = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUdevice device;
    CUdeviceptr input = 0, output = 0, weights = 0, scratch = 0;
    CUfunction function;
    CHECK(d.init(0));
    CHECK(d.driver_version(&version));
    if (version < BC_CUDA_MODULE_DRIVER_API) {
        fprintf(stderr, "module requires driver API >= %d, found %d\n", BC_CUDA_MODULE_DRIVER_API, version);
        goto cleanup;
    }
    CHECK(d.device_count(&count));
    if (ordinal >= count) { fprintf(stderr, "CUDA adapter %d unavailable (%d devices)\n", ordinal, count); goto cleanup; }
    CHECK(d.device_get(&device, ordinal));
    CHECK(d.device_attribute(&major, 75, device)); /* COMPUTE_CAPABILITY_MAJOR */
    CHECK(d.device_attribute(&minor, 76, device)); /* COMPUTE_CAPABILITY_MINOR */
    if (major < 6) { fprintf(stderr, "module requires sm60 or newer\n"); goto cleanup; }
    char name[256];
    CHECK(d.device_name(name, sizeof(name), device));
    CHECK(d.ctx_create(&context, 0, device));
    CHECK(d.module_load(&module, bc_cuda_module_begin));
    for (size_t i = 0; i < sizeof(symbols) / sizeof(*symbols); i++) {
        if (failed(d.function_get(&function, module, symbols[i]), symbols[i])) goto cleanup;
    }
    printf("%s sm%d%d: resolved all %zu kernels\n", name, major, minor, sizeof(symbols) / sizeof(*symbols));

    /* The model has a fixed 256^3 shape. One block writes just the prefix, but
     * neighbouring convolution reads use the full-volume strides. */
    enum { CS = BC_CUDA_MODULE_CS };
    size_t elements = (size_t)256 * 256 * 256 * CS;
    uint16_t expected[128 * CS], actual[128 * CS], taps[27 * CS * CS] = {0};
    for (int c = 0; c < CS; c++) taps[13 * CS * CS + c * CS + c] = 0x3c00; /* fp16 one */
    CHECK(d.alloc(&input, elements * sizeof(uint16_t)));
    CHECK(d.alloc(&output, sizeof(actual)));
    CHECK(d.alloc(&weights, sizeof(taps)));
    CHECK(d.alloc(&scratch, sizeof(unsigned long long)));
    CHECK(d.copy_htod(weights, taps, sizeof(taps)));
    CHECK(d.function_get(&function, module, "fill_pattern_kernel"));
    void *fill_args[] = {&input, &elements};
    CHECK(d.launch(function, 256, 1, 1, 128, 1, 1, 0, NULL, fill_args, NULL));
    CHECK(d.sync());
    CHECK(d.copy_dtoh(expected, input, sizeof(expected)));
    int nonzero = 0;
    for (size_t i = 0; i < sizeof(expected) / sizeof(*expected); i++) nonzero |= expected[i];
    if (!nonzero) { fprintf(stderr, "empty tuning pattern\n"); goto cleanup; }

    CHECK(d.memset_d8(output, 0xff, sizeof(actual)));
    const char *scalar_symbol = NULL;
#define X(v, t, symbol) if (v == 1 && t == 128) scalar_symbol = symbol;
    BC_CUDA_MODULE_SCALAR(X)
#undef X
    if (!scalar_symbol) { fprintf(stderr, "smoke test needs the 1/128 scalar variant\n"); goto cleanup; }
    CHECK(d.function_get(&function, module, scalar_symbol));
    int channels = CS, stride = CS, dilation = 1;
    void *scalar_args[] = {&input, &output, &weights, &channels, &channels, &stride, &dilation};
    CHECK(d.launch(function, 1, 1, 1, 128, 1, 1, sizeof(taps), NULL, scalar_args, NULL));
    CHECK(d.sync());
    CHECK(d.copy_dtoh(actual, output, sizeof(actual)));
    if (memcmp(actual, expected, sizeof(actual))) { fprintf(stderr, "scalar identity convolution failed\n"); goto cleanup; }

    size_t prefix = 128 * CS;
    unsigned long long expected_sum = 0, actual_sum = 0;
    for (size_t i = 0; i < prefix; i++) expected_sum += expected[i];
    CHECK(d.memset_d8(scratch, 0, sizeof(actual_sum)));
    CHECK(d.function_get(&function, module, "checksum_kernel"));
    void *checksum_args[] = {&output, &scratch, &prefix};
    CHECK(d.launch(function, 1, 1, 1, 128, 1, 1, 0, NULL, checksum_args, NULL));
    CHECK(d.sync());
    CHECK(d.copy_dtoh(&actual_sum, scratch, sizeof(actual_sum)));
    if (actual_sum != expected_sum) { fprintf(stderr, "checksum kernel failed\n"); goto cleanup; }

    int wmma = -1;
    CHECK(d.function_get(&function, module, "wmma_probe_kernel"));
    void *probe_args[] = {&scratch};
    CHECK(d.launch(function, 1, 1, 1, 1, 1, 1, 0, NULL, probe_args, NULL));
    CHECK(d.sync());
    CHECK(d.copy_dtoh(&wmma, scratch, sizeof(wmma)));
    if (wmma != (major >= 7)) { fprintf(stderr, "WMMA compiled-code probe mismatch\n"); goto cleanup; }
    if (wmma) {
        const char *wmma_symbol = NULL;
#define X(w, symbol) if (w == 2) wmma_symbol = symbol;
        BC_CUDA_MODULE_WMMA(X)
#undef X
        if (!wmma_symbol) { fprintf(stderr, "smoke test needs the two-warp variant\n"); goto cleanup; }
        CHECK(d.memset_d8(output, 0xff, sizeof(actual)));
        CHECK(d.function_get(&function, module, wmma_symbol));
        void *wmma_args[] = {&input, &output, &weights, &dilation};
        CHECK(d.launch(function, 1, 1, 1, 64, 1, 1, sizeof(taps) + 2 * BC_CUDA_MODULE_WMMA_SCRATCH, NULL, wmma_args, NULL));
        CHECK(d.sync());
        CHECK(d.copy_dtoh(actual, output, 32 * CS * sizeof(uint16_t)));
        if (memcmp(actual, expected, 32 * CS * sizeof(uint16_t))) { fprintf(stderr, "WMMA identity convolution failed\n"); goto cleanup; }
    }
    /* Shifted, channel-permuting taps exercise the 24-channel K/N tails,
     * odd 48-byte-row alignment, clipping and the largest dilation. */
    {
        uint16_t prefix_values[192 * CS];
        CHECK(d.copy_dtoh(prefix_values, input, sizeof(prefix_values)));
        for (int side = -1; side <= 1; side += 2)
        for (int distance = 1; distance <= 31; distance += 30) {
            memset(taps, 0, sizeof(taps));
            for (int c = 0; c < CS; c++)
                taps[(13 + side) * CS * CS + ((c + 17) % CS) * CS + c] = 0x3c00;
            CHECK(d.copy_htod(weights, taps, sizeof(taps)));
            dilation = distance;
            for (int v = 0; v < 128; v++)
                for (int c = 0; c < CS; c++) {
                    int x = v + side * distance;
                    expected[v * CS + c] = x < 0 ? 0 : prefix_values[x * CS + (c + 17) % CS];
                }
            CHECK(d.function_get(&function, module, scalar_symbol));
            CHECK(d.launch(function, 1, 1, 1, 128, 1, 1, sizeof(taps), NULL, scalar_args, NULL));
            CHECK(d.sync());
            CHECK(d.copy_dtoh(actual, output, sizeof(actual)));
            if (memcmp(actual, expected, sizeof(actual))) {
                fprintf(stderr, "scalar shifted channel-tail convolution failed\n"); goto cleanup;
            }
            if (wmma) {
                const char *symbol = NULL;
#define X(w, name) if (w == 2) symbol = name;
                BC_CUDA_MODULE_WMMA(X)
#undef X
                CHECK(d.function_get(&function, module, symbol));
                void *args[] = {&input, &output, &weights, &dilation};
                CHECK(d.launch(function, 1, 1, 1, 64, 1, 1,
                    sizeof(taps) + 2 * BC_CUDA_MODULE_WMMA_SCRATCH, NULL, args, NULL));
                CHECK(d.sync());
                CHECK(d.copy_dtoh(actual, output, 32 * CS * sizeof(uint16_t)));
                if (memcmp(actual, expected, 32 * CS * sizeof(uint16_t))) {
                    fprintf(stderr, "WMMA shifted channel-tail convolution failed\n"); goto cleanup;
                }
            }
        }
#if BC_MODEL_NCLASS == 18 && !BC_MODEL_OUTPUT_IS_MASK
        /* An independent double-precision oracle checks grouping, both
         * temperatures, bias and the last logical channel of the PVE head. */
        uint16_t classifier[BC_MODEL_CHAN * 18 + 18] = {0};
        for (int c = 0; c < 18; c++) classifier[BC_MODEL_CHAN * 18 + c] = bc_f32_to_f16(c * 0.125f);
        classifier[(BC_MODEL_CHAN - 1) * 18 + 17] = 0x3c00;
        CHECK(d.copy_htod(weights, classifier, sizeof(classifier)));
        CHECK(d.free(output)); output = 0;
        size_t voxels = (size_t)256 * 256 * 256;
        CHECK(d.alloc(&output, BC_PRIOR_PLANES * voxels * sizeof(float)));
        CUdeviceptr bias = weights + BC_MODEL_CHAN * 18 * sizeof(uint16_t);
        int logical = BC_MODEL_CHAN, classes = 18;
        void *args[] = {&input, &output, &weights, &bias, &logical, &classes};
        CHECK(d.function_get(&function, module, "tissue_priors_kernel"));
        CHECK(d.launch(function, 1, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
        CHECK(d.sync());
        float priors[4][128];
        for (int t = 0; t < 4; t++)
            CHECK(d.copy_dtoh(priors[t], output + t * voxels * sizeof(float), sizeof(priors[t])));
        const unsigned masks[] = {BC_PRIOR_GM_MASK, BC_PRIOR_WM_MASK, BC_PRIOR_CSF_MASK};
        for (int v = 0; v < 128; v++) {
            double denom = 0, support = 0, num[3] = {0};
            for (int c = 0; c < 18; c++) {
                double logit = c * 0.125 + (c == 17 ? bc_f16_to_f32(prefix_values[v * CS + BC_MODEL_CHAN - 1]) : 0);
                double probability = exp(logit / 2);
                denom += probability; support += exp(logit);
                for (int t = 0; t < 3; t++) if (masks[t] & (1u << c)) num[t] += probability;
                if (BC_PRIOR_MIXED_GWM_MASK & (1u << c)) {
                    double half = 0.5 * probability;
                    num[0] += half;
                    num[1] += half;
                }
            }
            for (int t = 0; t < 4; t++) {
                double want = t < 3 ? num[t] / denom : 1 - 1 / support;
                if (!isfinite(priors[t][v]) || fabs(priors[t][v] - want) > 2e-6) {
                    fprintf(stderr, "PVE head mismatch at tissue %d voxel %d\n", t, v); goto cleanup;
                }
            }
        }
        puts("PASS: tissue-prior head, bias, support and final logical channel");
#endif
    }

    printf("PASS: scalar, %s, fill and checksum; %zu-byte embedded module\n", wmma ? "WMMA" : "sm60 WMMA exclusion", bytes);
    result = 0;
cleanup:
    if (scratch) d.free(scratch);
    if (weights) d.free(weights);
    if (output) d.free(output);
    if (input) d.free(input);
    if (module) d.module_unload(module);
    if (context) d.ctx_destroy(context);
    LIB_CLOSE(library);
    return result;
}
