#include "backend.h"
#include "backend_accel.h"
#include "cuda_variants.h"
#if defined(BC_MODEL_IS_MINDMAP_OMNI) && BC_MODEL_IS_MINDMAP_OMNI && !defined(BC_MODEL_IS_MINDMAP)
#define BC_MODEL_IS_MINDMAP 1
#include "cuda_kernels.cuh"
#undef BC_MODEL_IS_MINDMAP
#else
#include "cuda_kernels.cuh"
#endif
#include "model_specs.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cuda_error(cudaError_t error, const char *operation) {
	if (error == cudaSuccess) return 0;
	fprintf(stderr, "brainchop-core: CUDA %s failed: %s\n",
		operation, cudaGetErrorString(error));
	if (error == cudaErrorNoKernelImageForDevice) {
		int ordinal = 0;
		cudaDeviceProp properties;
		if (cudaGetDevice(&ordinal) == cudaSuccess &&
				cudaGetDeviceProperties(&properties, ordinal) == cudaSuccess)
			fprintf(stderr, "brainchop-core: this build has no kernel for %s "
				"(compute capability %d.%d). Rebuild with "
				"CUDA_ARCH=sm_%d%d, or use -backend cpu.\n",
				properties.name, properties.major, properties.minor,
				properties.major, properties.minor);
	}
	return 1;
}

static int cuda_available(void) {
	int count = 0;
	cudaError_t error = cudaGetDeviceCount(&count);
	if (error != cudaSuccess) {
		(void)cudaGetLastError();
		return 0;
	}
	return count > 0;
}

static int cuda_adapter_count(void) {
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess) {
		(void)cudaGetLastError();
		return 0;
	}
	return count;
}

static int cuda_adapter_name(int index, char *description, size_t capacity) {
	int count = cuda_adapter_count();
	cudaDeviceProp properties;
	if (index < 0 || index >= count) return 1;
	if (cudaGetDeviceProperties(&properties, index) != cudaSuccess) {
		(void)cudaGetLastError();
		return 1;
	}
	if (description && capacity)
		snprintf(description, capacity, "%s (%.2f GB, sm_%d%d)",
			properties.name,
			(double)properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
			properties.major, properties.minor);
	return 0;
}

/* Host primitives used by the common inference policy. */
#define gpu_error_t cudaError_t
#define GPU_SUCCESS cudaSuccess
#define gpu_event cudaEvent_t
#define gpu_alloc cudaMalloc
#define gpu_free cudaFree
#define gpu_zero cudaMemset
#define gpu_copy cudaMemcpy
#define GPU_DTOH cudaMemcpyDeviceToHost
#define GPU_HTOD cudaMemcpyHostToDevice
static cudaError_t dispatch_error = cudaSuccess;
static cudaError_t gpu_sync(void) {
    cudaError_t error = cudaDeviceSynchronize();
    return dispatch_error != cudaSuccess ? dispatch_error : error;
}
static cudaError_t gpu_last_error(void) {
    cudaError_t error = cudaGetLastError();
    if (dispatch_error != cudaSuccess) error = dispatch_error;
    dispatch_error = cudaSuccess;
    return error;
}
#define gpu_event_create cudaEventCreate
#define gpu_event_destroy cudaEventDestroy
#define gpu_event_record cudaEventRecord
#define gpu_event_sync cudaEventSynchronize
#define gpu_event_elapsed cudaEventElapsedTime
#define gpu_device cudaGetDevice
#define gpu_properties_get cudaGetDeviceProperties
#define gpu_properties cudaDeviceProp
#define gpu_driver_version cudaDriverGetVersion
#define BC_CUDA_ASSERT static_assert
#define gpu_half __float2half
static int gpu_begin(int ordinal) {
    return ordinal >= 0 ? cuda_error(cudaSetDevice(ordinal), "selecting device") : 0;
}
static int gpu_end(void) { return 0; }

static void gpu_conv(int vpt, int threads, const bc_act *source, bc_act *destination,
        const bc_act *weights, double *sums, const unsigned int *row_list,
        const float *row_weights, int row_count, int input_channels,
        int output_channels, int input_stride, int dilation, size_t shared) {
    size_t total_vox = row_list ? (size_t)row_count * CUDA_DIM : CUDA_NVOX;
#define X(V, T) if (vpt == V && threads == T) { \
    size_t mom_shared = sums ? (size_t)(T / 32) * 2 * CUDA_CS * sizeof(double) : 0; \
    size_t sm = shared > mom_shared ? shared : mom_shared; \
    unsigned blocks = (unsigned)((total_vox + (size_t)T * V - 1) / ((size_t)T * V)); \
    conv3d_kernel<V, T><<<blocks, T, sm>>>(source, destination, weights, sums, \
        row_list, row_weights, row_count, input_channels, output_channels, input_stride, dilation); \
    return; }
    BC_CUDA_SCALAR_VARIANTS(X)
#undef X
    dispatch_error = cudaErrorInvalidConfiguration;
}

static void gpu_wmma(int warps, const bc_act *source, bc_act *destination,
        const bc_act *weights, double *sums, const unsigned int *row_list,
        const float *row_weights, int row_count, int dilation) {
    if (warps <= 0) { dispatch_error = cudaErrorInvalidConfiguration; return; }
    size_t w_shared = (CUDA_CS == 24) ? 0 : (size_t)27 * CUDA_CS * CUDA_CS * sizeof(bc_act);
    size_t shared = w_shared + (size_t)warps * CUDA_WMMA_SCRATCH;
    size_t tiles = row_list ? (size_t)row_count * (CUDA_DIM / CUDA_WMMA_TILE)
        : (CUDA_NVOX / CUDA_WMMA_TILE);
    unsigned blocks = (unsigned)((tiles + (size_t)warps - 1) / (size_t)warps);
#define X(W) if (warps == W) { \
    conv3d_wmma_kernel<W><<<blocks, W * 32, shared>>>(source, destination, weights, \
        sums, row_list, row_weights, row_count, dilation); return; }
    BC_CUDA_WMMA_VARIANTS(X)
#undef X
    dispatch_error = cudaErrorInvalidConfiguration;
}

static void gpu_probe(int *flag) { wmma_probe_kernel<<<1, 1>>>(flag); }
static void gpu_fill(bc_act *buffer, size_t count) { fill_pattern_kernel<<<256, 256>>>(buffer, count); }
static void gpu_checksum(const bc_act *buffer, unsigned long long *sum, size_t count) {
    checksum_kernel<<<256, 256>>>(buffer, sum, count);
}
static void gpu_moments(const double *sums, float *mul, float *add,
        const float *scale, const float *bias, int channels) {
    moments_finish_kernel<<<1, CUDA_MAX_CHANNELS>>>(sums, mul, add, scale, bias, channels);
}
static void gpu_norm(bc_act *buffer, const float *mul, const float *add,
        const unsigned int *row_list, int row_count) {
    size_t pairs = (size_t)row_count * CUDA_DIM * (CUDA_CS / 2);
    unsigned blocks = (unsigned)((pairs + CUDA_THREADS - 1) / CUDA_THREADS);
    norm_gelu_kernel<<<blocks, CUDA_THREADS>>>(buffer, mul, add, row_list, row_count);
}
static void gpu_fill_rows(bc_act *buffer, const unsigned int *pure_list, int npure, int rep_row) {
    size_t u4s = (size_t)npure * ((CUDA_DIM * CUDA_CS * sizeof(bc_act)) / sizeof(uint4));
    unsigned blocks = (unsigned)((u4s + CUDA_THREADS - 1) / CUDA_THREADS);
    fill_rows_kernel<<<blocks, CUDA_THREADS>>>(buffer, pure_list, npure, rep_row);
}
static void gpu_classify_u8(const bc_act *input, unsigned char *labels,
        const float *mul, const float *add, const bc_act *weights, const bc_act *bias,
        const unsigned int *row_list, int row_count, int channels, int classes) {
    size_t vox = (size_t)row_count * CUDA_DIM;
    unsigned blocks = (unsigned)((vox + CUDA_THREADS - 1) / CUDA_THREADS);
    classify_u8_kernel<<<blocks, CUDA_THREADS>>>(input, labels, mul, add, weights, bias,
        row_list, row_count, channels, classes);
}
static void gpu_fill_labels_u8(unsigned char *labels, const unsigned int *pure_list,
        int npure, int rep_row) {
    size_t u4s = (size_t)npure * (CUDA_DIM / sizeof(uint4));
    unsigned blocks = (unsigned)((u4s + CUDA_THREADS - 1) / CUDA_THREADS);
    fill_labels_u8_kernel<<<blocks, CUDA_THREADS>>>(labels, pure_list, npure, rep_row);
}
static void gpu_priors(const bc_act *input, float *planes, const float *mul, const float *add,
        const bc_act *weights, const bc_act *bias, const unsigned int *row_list,
        int row_count, int channels, int classes) {
    size_t vox = (size_t)row_count * CUDA_DIM;
    unsigned blocks = (unsigned)((vox + CUDA_THREADS - 1) / CUDA_THREADS);
    tissue_priors_kernel<<<blocks, CUDA_THREADS>>>(input, planes, mul, add, weights, bias,
        row_list, row_count, channels, classes);
}
static void gpu_fill_priors(float *planes, const unsigned int *pure_list, int npure, int rep_row) {
    size_t vox = (size_t)npure * CUDA_DIM;
    unsigned blocks = (unsigned)((vox + CUDA_THREADS - 1) / CUDA_THREADS);
    fill_priors_kernel<<<blocks, CUDA_THREADS>>>(planes, pure_list, npure, rep_row);
}
static void gpu_pve_blur(const float *src, float *dst, int axis) {
    unsigned blocks = (unsigned)((3 * CUDA_NVOX + CUDA_THREADS - 1) / CUDA_THREADS);
    pve_blur_kernel<<<blocks, CUDA_THREADS>>>(src, dst, axis);
}
static void gpu_pve_support(const float *blurred, float *planes) {
    unsigned blocks = (unsigned)((3 * CUDA_NVOX + CUDA_THREADS - 1) / CUDA_THREADS);
    pve_support_kernel<<<blocks, CUDA_THREADS>>>(blurred, planes);
}

#include "backend_cuda_host.h"
