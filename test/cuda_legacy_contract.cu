/* Compile through nvcc; invalid dispatch must fail before touching any buffer.
 * No GPU is needed: the software dispatch error takes precedence over probing
 * an absent CUDA device. Link the model's backend_accel/model_specs/bc_threads. */
#include "../src/backend_cuda.cu"
#include <assert.h>
int main(void) {
    const int bad_warps[] = {0, 3, -1};
    for (unsigned i = 0; i < sizeof(bad_warps) / sizeof(*bad_warps); i++) {
        gpu_wmma(bad_warps[i], NULL, NULL, NULL, NULL, NULL, NULL, CUDA_ROWS, 1);
        assert(gpu_sync() == cudaErrorInvalidConfiguration);
        assert(gpu_last_error() == cudaErrorInvalidConfiguration);
    }
    gpu_conv(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, CUDA_ROWS, 1, 16, 1, 1, 0);
    assert(gpu_sync() == cudaErrorInvalidConfiguration);
    assert(gpu_last_error() == cudaErrorInvalidConfiguration);
    puts("PASS: invalid legacy scalar/WMMA dispatch fails explicitly");
    return 0;
}
