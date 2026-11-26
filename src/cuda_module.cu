/* Maintainer-only, device-only translation unit. nvcc --fatbin discards host
 * code and resolves device math at compilation time; no runtime compiler or
 * CUDA runtime library is needed to load the result through the Driver API. */
#define BC_CUDA_MODULE_EXPORT 1
#include "cuda_kernels.cuh"
#include "cuda_variants.h"

#define X(V, T) template __global__ void conv3d_kernel<V, T>( \
    const bc_act *, bc_act *, const bc_act *, double *, const unsigned int *, \
    const float *, int, int, int, int, int);
BC_CUDA_SCALAR_VARIANTS(X)
#undef X

#define X(W) template __global__ void conv3d_wmma_kernel<W>( \
    const bc_act *, bc_act *, const bc_act *, double *, const unsigned int *, \
    const float *, int, int);
BC_CUDA_WMMA_VARIANTS(X)
#undef X

/* Reject any compile-time override that disagrees with the recorded layout. */
static_assert(CUDA_DIM == BC_CUDA_GEOMETRY_DIM, "module geometry mismatch: DIM");
static_assert(CUDA_MAX_CHANNELS == BC_CUDA_GEOMETRY_MAX_CHANNELS, "module geometry mismatch: MAX_CHANNELS");
static_assert(CUDA_CS == BC_CUDA_GEOMETRY_CS, "module geometry mismatch: CS");
static_assert(CUDA_THREADS == BC_CUDA_GEOMETRY_THREADS, "module geometry mismatch: THREADS");
static_assert(CUDA_REDUCE_THREADS == BC_CUDA_GEOMETRY_REDUCE_THREADS, "module geometry mismatch: REDUCE_THREADS");
static_assert(CUDA_REDUCE_BLOCKS == BC_CUDA_GEOMETRY_REDUCE_BLOCKS, "module geometry mismatch: REDUCE_BLOCKS");
static_assert(CUDA_CONV_VPT == BC_CUDA_GEOMETRY_CONV_VPT, "module geometry mismatch: CONV_VPT");
static_assert(CUDA_CONV_THREADS == BC_CUDA_GEOMETRY_CONV_THREADS, "module geometry mismatch: CONV_THREADS");
static_assert(CUDA_WMMA_TILE == BC_CUDA_GEOMETRY_WMMA_TILE, "module geometry mismatch: WMMA_TILE");
static_assert(CUDA_WMMA_WARPS == BC_CUDA_GEOMETRY_WMMA_WARPS, "module geometry mismatch: WMMA_WARPS");
static_assert(CUDA_WMMA_SCRATCH == BC_CUDA_GEOMETRY_WMMA_SCRATCH, "module geometry mismatch: WMMA_SCRATCH");
