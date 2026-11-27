#ifndef BRAINCHOP_CUDA_KERNELS_CUH
#define BRAINCHOP_CUDA_KERNELS_CUH

/*
 * Device code shared by the CUDA runtime backend and offline module generation.
 * Keep host allocation, launchers, profiling and tuning policy in
 * backend_cuda_host.h.
 *
 * Key performance properties (ABI 3):
 *  - Pre-padded 24-ch WMMA weights ([27][2][2][16][16], ldm=16, 54 KiB in L1/L2)
 *    eliminate per-warp shared-memory tile_b re-copies and direct-load k=0 from
 *    16-byte-aligned global rows while vectorizing k=1 via uint4.
 *  - GroupNorm moments are fused into the conv epilogue (FP32 warp sum -> FP64
 *    block sum -> 256-stripe FP64 atomicAdd), eliminating reduce_channels_kernel's
 *    full-volume HBM re-read while keeping 53-bit FP64 invariance across WARPS.
 *  - turbo_cone.h row lists and pure-background fills skip empty background rows.
 *  - norm_gelu_kernel walks coalesced half2 pairs using pre-fused (mul, add).
 *  - Last layer fuses norm+gelu directly into classify_u8_kernel (16.8 MB DtoH)
 *    or tissue_priors_kernel + GPU pve_blur_kernel / pve_support_kernel.
 */
#include "model_specs.h"
#include "cuda_geometry.h"
#include "pve_constants.h"

#include <cuda_fp16.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef BC_CUDA_MODULE_EXPORT
#define BC_CUDA_GLOBAL extern "C"
#define BC_CUDA_TEMPLATE
#else
#define BC_CUDA_GLOBAL static
#define BC_CUDA_TEMPLATE static
#endif

#define CUDA_DIM BRAINCHOP_MODEL_DIM
#define CUDA_NVOX ((size_t)CUDA_DIM * CUDA_DIM * CUDA_DIM)
#define CUDA_ROWS ((int)(CUDA_DIM * CUDA_DIM))
#define CUDA_MOMENT_STRIPES 256
#ifndef CUDA_THREADS
#define CUDA_THREADS BC_CUDA_GEOMETRY_THREADS
#endif
#ifndef CUDA_REDUCE_THREADS
#define CUDA_REDUCE_THREADS BC_CUDA_GEOMETRY_REDUCE_THREADS
#endif
#ifndef CUDA_REDUCE_BLOCKS
#define CUDA_REDUCE_BLOCKS BC_CUDA_GEOMETRY_REDUCE_BLOCKS
#endif
#define CUDA_MAX_CHANNELS BC_CUDA_GEOMETRY_MAX_CHANNELS

typedef __half bc_act;
static_assert(sizeof(bc_act) == sizeof(uint16_t),
	"bc_accel_narrow_input returns uint16_t; a wider bc_act would read past it");
#define CUDA_CS CUDA_MAX_CHANNELS
static_assert((CUDA_CS & 1) == 0, "CUDA_CS must be even for coalesced half2 norm");
static_assert(!BC_MODEL_HAS_AFFINE || BC_MODEL_CHAN == CUDA_MAX_CHANNELS,
	"an affine model must fill the activation stride");
static_assert(BC_MODEL_CHAN <= CUDA_MAX_CHANNELS,
	"model channels exceed CUDA_MAX_CHANNELS");

#ifndef CUDA_CONV_VPT
#define CUDA_CONV_VPT BC_CUDA_GEOMETRY_CONV_VPT
#endif
#ifndef CUDA_CONV_THREADS
#define CUDA_CONV_THREADS BC_CUDA_GEOMETRY_CONV_THREADS
#endif

/*
 * Scalar NDHWC 3x3x3 convolution (Layer 0 on all cards, hidden fallback on sm_60).
 * Supports optional turbo_cone row lists and fused block-reduced FP64 moments.
 */
template <int VPT, int THREADS>
__global__ __launch_bounds__(THREADS) BC_CUDA_TEMPLATE void conv3d_kernel(
		const bc_act *__restrict__ source,
		bc_act *__restrict__ destination, const bc_act *__restrict__ weights,
		double *__restrict__ sums, const unsigned int *__restrict__ row_list,
		const float *__restrict__ row_weights, int row_count,
		int input_channels, int output_channels, int input_stride, int dilation) {
	(void)output_channels;
	extern __shared__ __align__(16) char conv_shared[];
	bc_act *shared_weights = (bc_act *)conv_shared;
	int weight_count = 27 * input_channels * CUDA_CS;
	for (int i = (int)threadIdx.x; i < weight_count; i += THREADS)
		shared_weights[i] = weights[i];
	__syncthreads();

	size_t total_vox = row_list ? (size_t)row_count * CUDA_DIM : CUDA_NVOX;
	size_t base = (size_t)blockIdx.x * ((size_t)THREADS * VPT) + threadIdx.x;
	float accumulator[VPT][CUDA_MAX_CHANNELS];
#pragma unroll
	for (int v = 0; v < VPT; v++)
#pragma unroll
		for (int channel = 0; channel < CUDA_MAX_CHANNELS; channel++)
			accumulator[v][channel] = 0.0f;

	size_t out_voxel[VPT];
	int x[VPT], y[VPT], z[VPT], valid[VPT];
	float rw[VPT];
#pragma unroll
	for (int v = 0; v < VPT; v++) {
		size_t idx = base + (size_t)v * THREADS;
		valid[v] = idx < total_vox;
		if (valid[v]) {
			if (row_list) {
				size_t ridx = idx >> 8;
				unsigned int row = row_list[ridx];
				x[v] = (int)(idx & 255);
				y[v] = (int)(row & 255);
				z[v] = (int)(row >> 8);
				out_voxel[v] = (size_t)row * CUDA_DIM + (size_t)x[v];
				rw[v] = row_weights ? row_weights[ridx] : 1.0f;
			} else {
				x[v] = (int)(idx & 255);
				y[v] = (int)((idx >> 8) & 255);
				z[v] = (int)(idx >> 16);
				out_voxel[v] = idx;
				rw[v] = 1.0f;
			}
		} else {
			x[v] = y[v] = z[v] = 0;
			out_voxel[v] = 0;
			rw[v] = 0.0f;
		}
	}

	for (int tz = -1; tz <= 1; tz++) {
		for (int ty = -1; ty <= 1; ty++) {
			for (int tx = -1; tx <= 1; tx++) {
				const bc_act *tap = shared_weights +
					(size_t)((tz + 1) * 9 + (ty + 1) * 3 + tx + 1) *
					input_channels * CUDA_CS;
				const bc_act *value[VPT];
#pragma unroll
				for (int v = 0; v < VPT; v++) {
					int input_x = x[v] + tx * dilation;
					int input_y = y[v] + ty * dilation;
					int input_z = z[v] + tz * dilation;
					int inside = valid[v] & ((unsigned)input_x < CUDA_DIM) &
						((unsigned)input_y < CUDA_DIM) &
						((unsigned)input_z < CUDA_DIM);
					size_t input_voxel = (size_t)input_z * CUDA_DIM * CUDA_DIM +
						(size_t)input_y * CUDA_DIM + (size_t)input_x;
					value[v] = inside ? source + input_voxel * input_stride : NULL;
				}
				for (int input_channel = 0; input_channel < input_channels;
						input_channel++) {
					float channel_weights[CUDA_MAX_CHANNELS];
#pragma unroll
					for (int output_channel = 0; output_channel < CUDA_MAX_CHANNELS;
							output_channel++)
						channel_weights[output_channel] =
							__half2float(tap[(size_t)input_channel *
								CUDA_CS + output_channel]);
#pragma unroll
					for (int v = 0; v < VPT; v++) {
						if (!value[v]) continue;
						float input_value = __half2float(value[v][input_channel]);
#pragma unroll
						for (int output_channel = 0;
								output_channel < CUDA_MAX_CHANNELS; output_channel++)
							accumulator[v][output_channel] = fmaf(input_value,
								channel_weights[output_channel],
								accumulator[v][output_channel]);
					}
				}
			}
		}
	}

	double tsum[CUDA_CS], tsq[CUDA_CS];
#pragma unroll
	for (int c = 0; c < CUDA_CS; c++) { tsum[c] = 0.0; tsq[c] = 0.0; }
#pragma unroll
	for (int v = 0; v < VPT; v++) {
		if (!valid[v]) continue;
		bc_act *output = destination + out_voxel[v] * CUDA_CS;
		double wv = (double)rw[v];
#pragma unroll
		for (int channel = 0; channel < CUDA_CS; channel++) {
			bc_act h = __float2half(accumulator[v][channel]);
			output[channel] = h;
			if (sums && rw[v] != 0.0f) {
				double hv = (double)__half2float(h);
				tsum[channel] += hv * wv;
				tsq[channel] += hv * hv * wv;
			}
		}
	}

	if (sums) {
		__syncthreads();
		enum { NWARPS = THREADS / 32 };
		double *sm_sum = (double *)conv_shared;
		double *sm_sq = sm_sum + (size_t)NWARPS * CUDA_CS;
		int warp = (int)threadIdx.x >> 5;
		int lane = (int)threadIdx.x & 31;
		for (int c = 0; c < CUDA_CS; c++) {
			double s = tsum[c], q = tsq[c];
#pragma unroll
			for (int off = 16; off > 0; off >>= 1) {
				s += __shfl_down_sync(0xffffffffu, s, off);
				q += __shfl_down_sync(0xffffffffu, q, off);
			}
			if (lane == 0) {
				sm_sum[warp * CUDA_CS + c] = s;
				sm_sq[warp * CUDA_CS + c] = q;
			}
		}
		__syncthreads();
		if ((int)threadIdx.x < CUDA_CS) {
			int c = (int)threadIdx.x;
			double bs = 0.0, bq = 0.0;
			for (int w = 0; w < NWARPS; w++) {
				bs += sm_sum[w * CUDA_CS + c];
				bq += sm_sq[w * CUDA_CS + c];
			}
			int slot = (int)(blockIdx.x & (CUDA_MOMENT_STRIPES - 1));
			atomicAdd(sums + (size_t)slot * 2 * CUDA_CS + c, bs);
			atomicAdd(sums + (size_t)slot * 2 * CUDA_CS + CUDA_CS + c, bq);
		}
	}
}

#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
#include <mma.h>
#endif

BC_CUDA_GLOBAL __global__ void wmma_probe_kernel(int *out) {
#if __CUDA_ARCH__ >= 700
	*out = 1;
#else
	*out = 0;
#endif
}

#define CUDA_WMMA_TILE BC_CUDA_GEOMETRY_WMMA_TILE
#ifndef CUDA_WMMA_WARPS
#define CUDA_WMMA_WARPS BC_CUDA_GEOMETRY_WMMA_WARPS
#endif
#define CUDA_WMMA_SCRATCH (CUDA_WMMA_TILE * 16 * sizeof(float))

template <int WARPS>
__global__ __launch_bounds__(WARPS * 32) BC_CUDA_TEMPLATE void conv3d_wmma_kernel(
		const bc_act *__restrict__ source, bc_act *__restrict__ destination,
		const bc_act *__restrict__ weights, double *__restrict__ sums,
		const unsigned int *__restrict__ row_list,
		const float *__restrict__ row_weights, int row_count, int dilation) {
#if __CUDA_ARCH__ >= 700
	using namespace nvcuda;
	extern __shared__ __align__(32) char wmma_shared[];
	int warp = (int)threadIdx.x >> 5;
	int lane = (int)threadIdx.x & 31;

#if CUDA_CS == 24
	/* Pre-padded [27][2][2][16][16] weights (54 KiB) stay in L1/L2 cache with
	   ldm=16, removing 108 shared tile_b copies/tile and quadrupling SM occupancy. */
	char *scratch_base = wmma_shared;
#else
	bc_act *staged = (bc_act *)wmma_shared;
	const int weight_count = 27 * CUDA_CS * CUDA_CS;
	char *scratch_base = wmma_shared + (size_t)weight_count * sizeof(bc_act);
	for (int i = (int)threadIdx.x; i < weight_count; i += WARPS * 32)
		staged[i] = weights[i];
	__syncthreads();
#endif

	char *scratch = scratch_base + (size_t)warp * CUDA_WMMA_SCRATCH;
	bc_act *tile_staging = (bc_act *)scratch;
	float *tile_output = (float *)scratch;

	size_t tile = (size_t)blockIdx.x * WARPS + warp;
	size_t total_tiles = row_list ? (size_t)row_count * (CUDA_DIM / CUDA_WMMA_TILE)
		: (CUDA_NVOX / CUDA_WMMA_TILE);
	int active = tile < total_tiles;
	size_t base = 0;
	int x0 = 0, y0 = 0, z0 = 0;
	float rw = 0.0f;
	if (active) {
		if (row_list) {
			size_t ridx = tile >> 4;
			unsigned int row = row_list[ridx];
			x0 = (int)((tile & 15) * CUDA_WMMA_TILE);
			y0 = (int)(row & (CUDA_DIM - 1));
			z0 = (int)(row >> 8);
			base = (size_t)row * CUDA_DIM + (size_t)x0;
			rw = row_weights ? row_weights[ridx] : 1.0f;
		} else {
			base = tile * CUDA_WMMA_TILE;
			x0 = (int)(base & (CUDA_DIM - 1));
			y0 = (int)((base >> 8) & (CUDA_DIM - 1));
			z0 = (int)(base >> 16);
			rw = 1.0f;
		}
	}

	double wsum[2] = { 0.0, 0.0 }, wsq[2] = { 0.0, 0.0 };
	int oc_lane = lane & 15;
	int v0 = (lane >> 4) * 8;

#if CUDA_CS == 24
	if (active) {
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
		wmma::fill_fragment(acc[0], 0.0f);
		wmma::fill_fragment(acc[1], 0.0f);
		uint4 *stg4 = (uint4 *)tile_staging;
		const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
		for (int tz = -1; tz <= 1; tz++) {
			int iz = z0 + tz * dilation;
			if ((unsigned)iz >= CUDA_DIM) continue;
			for (int ty = -1; ty <= 1; ty++) {
				int iy = y0 + ty * dilation;
				if ((unsigned)iy >= CUDA_DIM) continue;
				size_t row = ((size_t)iz * CUDA_DIM + (size_t)iy) * CUDA_DIM;
				int tap_zy = (tz + 1) * 9 + (ty + 1) * 3;
				for (int tx = -1; tx <= 1; tx++) {
					int ix = x0 + tx * dilation;
					const bc_act *tap_w = weights + (size_t)(tap_zy + tx + 1) * 1024;
					wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
					wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b0, b1;
					if (ix >= 0 && ix + CUDA_WMMA_TILE <= CUDA_DIM) {
						/* Each 24-half voxel is 48 bytes (multiple of 16), so k=0
						   loads directly with ldm=24 without staging. */
						wmma::load_matrix_sync(a0, source + (row + (size_t)ix) * 24, 24);
						wmma::load_matrix_sync(b0, tap_w, 16);
						wmma::load_matrix_sync(b1, tap_w + 256, 16);
						wmma::mma_sync(acc[0], a0, b0, acc[0]);
						wmma::mma_sync(acc[1], a0, b1, acc[1]);
						/* k=1 tail (ic=16..23) is 16-byte aligned: 1 uint4 load/lane. */
						if (lane < 16)
							stg4[lane * 2] = *(const uint4 *)(source + (row + (size_t)(ix + lane)) * 24 + 16);
						else
							stg4[(lane - 16) * 2 + 1] = z4;
						__syncwarp();
						wmma::load_matrix_sync(a1, tile_staging, 16);
						wmma::load_matrix_sync(b0, tap_w + 512, 16);
						wmma::load_matrix_sync(b1, tap_w + 768, 16);
						wmma::mma_sync(acc[0], a1, b0, acc[0]);
						wmma::mma_sync(acc[1], a1, b1, acc[1]);
						__syncwarp();
					} else {
						int r = lane >> 1;
						int half_row = lane & 1;
						int xx = ix + r;
						stg4[lane] = ((unsigned)xx < CUDA_DIM)
							? *(const uint4 *)(source + (row + (size_t)xx) * 24 + half_row * 8)
							: z4;
						__syncwarp();
						wmma::load_matrix_sync(a0, tile_staging, 16);
						wmma::load_matrix_sync(b0, tap_w, 16);
						wmma::load_matrix_sync(b1, tap_w + 256, 16);
						wmma::mma_sync(acc[0], a0, b0, acc[0]);
						wmma::mma_sync(acc[1], a0, b1, acc[1]);
						__syncwarp();
						if (lane < 16) {
							int x1 = ix + lane;
							stg4[lane * 2] = ((unsigned)x1 < CUDA_DIM)
								? *(const uint4 *)(source + (row + (size_t)x1) * 24 + 16)
								: z4;
						} else {
							stg4[(lane - 16) * 2 + 1] = z4;
						}
						__syncwarp();
						wmma::load_matrix_sync(a1, tile_staging, 16);
						wmma::load_matrix_sync(b0, tap_w + 512, 16);
						wmma::load_matrix_sync(b1, tap_w + 768, 16);
						wmma::mma_sync(acc[0], a1, b0, acc[0]);
						wmma::mma_sync(acc[1], a1, b1, acc[1]);
						__syncwarp();
					}
				}
			}
		}
#pragma unroll
		for (int n = 0; n < 2; n++) {
			wmma::store_matrix_sync(tile_output, acc[n], 16, wmma::mem_row_major);
			__syncwarp();
			int oc = n * 16 + oc_lane;
			float ls = 0.0f, lq = 0.0f;
			if (oc < CUDA_CS) {
#pragma unroll
				for (int v = 0; v < 8; v++) {
					bc_act h = __float2half(tile_output[(v0 + v) * 16 + oc_lane]);
					destination[(base + (size_t)(v0 + v)) * CUDA_CS + oc] = h;
					float hv = __half2float(h);
					ls += hv;
					lq = fmaf(hv, hv, lq);
				}
			}
			ls += __shfl_xor_sync(0xffffffffu, ls, 16);
			lq += __shfl_xor_sync(0xffffffffu, lq, 16);
			if (lane < 16 && oc < CUDA_CS) {
				wsum[n] = (double)ls * (double)rw;
				wsq[n] = (double)lq * (double)rw;
			}
			__syncwarp();
		}
	}
#else
	if (active) {
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
		wmma::fill_fragment(acc, 0.0f);
		for (int tz = -1; tz <= 1; tz++) {
			int iz = z0 + tz * dilation;
			if ((unsigned)iz >= CUDA_DIM) continue;
			for (int ty = -1; ty <= 1; ty++) {
				int iy = y0 + ty * dilation;
				if ((unsigned)iy >= CUDA_DIM) continue;
				size_t row = ((size_t)iz * CUDA_DIM + (size_t)iy) * CUDA_DIM;
				int tap_zy = (tz + 1) * 9 + (ty + 1) * 3;
				for (int tx = -1; tx <= 1; tx++) {
					int ix = x0 + tx * dilation;
					const bc_act *tap = staged + (size_t)(tap_zy + tx + 1) * CUDA_CS * CUDA_CS;
					wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
					wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
					if (ix >= 0 && ix + CUDA_WMMA_TILE <= CUDA_DIM) {
						wmma::load_matrix_sync(a, source + (row + (size_t)ix) * CUDA_CS, CUDA_CS);
					} else {
						uint4 *stg4 = (uint4 *)tile_staging;
						int r = lane >> 1;
						int half_row = lane & 1;
						int xx = ix + r;
						stg4[lane] = ((unsigned)xx < CUDA_DIM)
							? *(const uint4 *)(source + (row + (size_t)xx) * 16 + half_row * 8)
							: make_uint4(0u, 0u, 0u, 0u);
						__syncwarp();
						wmma::load_matrix_sync(a, tile_staging, CUDA_CS);
						__syncwarp();
					}
					wmma::load_matrix_sync(b, tap, CUDA_CS);
					wmma::mma_sync(acc, a, b, acc);
				}
			}
		}
		wmma::store_matrix_sync(tile_output, acc, CUDA_CS, wmma::mem_row_major);
		__syncwarp();
		float ls = 0.0f, lq = 0.0f;
#pragma unroll
		for (int v = 0; v < 8; v++) {
			bc_act h = __float2half(tile_output[(v0 + v) * 16 + oc_lane]);
			destination[(base + (size_t)(v0 + v)) * CUDA_CS + oc_lane] = h;
			float hv = __half2float(h);
			ls += hv;
			lq = fmaf(hv, hv, lq);
		}
		ls += __shfl_xor_sync(0xffffffffu, ls, 16);
		lq += __shfl_xor_sync(0xffffffffu, lq, 16);
		if (lane < 16) {
			wsum[0] = (double)ls * (double)rw;
			wsq[0] = (double)lq * (double)rw;
		}
	}
#endif

	if (sums) {
		__syncthreads();
		double *sm_sum = (double *)scratch_base;
		double *sm_sq = sm_sum + (size_t)WARPS * CUDA_CS;
#if CUDA_CS == 24
		if (lane < 16) {
			sm_sum[warp * CUDA_CS + lane] = wsum[0];
			sm_sq[warp * CUDA_CS + lane] = wsq[0];
			if (lane < 8) {
				sm_sum[warp * CUDA_CS + 16 + lane] = wsum[1];
				sm_sq[warp * CUDA_CS + 16 + lane] = wsq[1];
			}
		}
#else
		if (lane < 16) {
			sm_sum[warp * CUDA_CS + lane] = wsum[0];
			sm_sq[warp * CUDA_CS + lane] = wsq[0];
		}
#endif
		__syncthreads();
		if ((int)threadIdx.x < CUDA_CS) {
			int c = (int)threadIdx.x;
			double bs = 0.0, bq = 0.0;
			for (int w = 0; w < WARPS; w++) {
				bs += sm_sum[w * CUDA_CS + c];
				bq += sm_sq[w * CUDA_CS + c];
			}
			int slot = (int)(blockIdx.x & (CUDA_MOMENT_STRIPES - 1));
			atomicAdd(sums + (size_t)slot * 2 * CUDA_CS + c, bs);
			atomicAdd(sums + (size_t)slot * 2 * CUDA_CS + CUDA_CS + c, bq);
		}
	}
#else
	(void)source; (void)destination; (void)weights; (void)sums;
	(void)row_list; (void)row_weights; (void)row_count; (void)dilation;
#endif
}

#if BC_MODEL_GELU_TANH_RATIONAL
#define CUDA_TANH_LIMIT 9.0f
__device__ static inline float cuda_gelu(float x) {
	float t = 0.797884583f * (x + 0.044715f * (x * x * x));
	if (t > CUDA_TANH_LIMIT) t = CUDA_TANH_LIMIT;
	if (t < -CUDA_TANH_LIMIT) t = -CUDA_TANH_LIMIT;
	float u = t * t;
	float p = -8.29118133e-14f;
	p = fmaf(p, u, 5.19263868e-11f);
	p = fmaf(p, u, -2.00294448e-08f);
	p = fmaf(p, u, 1.11017944e-05f);
	p = fmaf(p, u, 0.00309865153f);
	p = fmaf(p, u, 0.130791619f);
	p = fmaf(p, u, 0.99999994f);
	float q = 0.000253859733f;
	q = fmaf(q, u, 0.024473751f);
	q = fmaf(q, u, 0.464124829f);
	q = fmaf(q, u, 1.0f);
	return (0.5f * x) * (1.0f + t * p / q);
}
#elif BC_MODEL_GELU_EXP2_SIGMOID
__device__ static inline float cuda_gelu(float x) {
	float u = x + 0.044715f * x * x * x;
	return x / (1.0f + exp2f(-2.302208198144325f * u));
}
#else
#error "model_meta.json named a GELU flavour with no implementation here"
#endif

/* Fold 256 FP64 moment stripes into fused (mul, add) = (invstd*gamma, beta - mean*mul). */
BC_CUDA_GLOBAL __global__ void moments_finish_kernel(const double *__restrict__ sums,
		float *__restrict__ mul_out, float *__restrict__ add_out,
		const float *__restrict__ scale, const float *__restrict__ bias, int channels) {
	int c = (int)threadIdx.x;
	if (c >= channels) return;
	double sum = 0.0, sq = 0.0;
	for (int s = 0; s < CUDA_MOMENT_STRIPES; s++) {
		sum += sums[(size_t)s * 2 * CUDA_CS + c];
		sq += sums[(size_t)s * 2 * CUDA_CS + CUDA_CS + c];
	}
	double mean = sum / (double)CUDA_NVOX;
	double variance = sq / (double)CUDA_NVOX - mean * mean;
	if (variance < 0.0) variance = 0.0;
	float invstd = (float)rsqrt(variance + 1e-5);
	float gamma = scale ? scale[c] : 1.0f;
	float beta = bias ? bias[c] : 0.0f;
	float mul = invstd * gamma;
	mul_out[c] = mul;
	add_out[c] = beta - (float)mean * mul;
}

/* 100% coalesced half2 GroupNorm+affine+GELU over active rows. */
BC_CUDA_GLOBAL __global__ void norm_gelu_kernel(bc_act *__restrict__ buffer,
		const float *__restrict__ mul, const float *__restrict__ add,
		const unsigned int *__restrict__ row_list, int row_count) {
	enum { PAIRS_PER_VOX = CUDA_CS / 2, PAIRS_PER_ROW = CUDA_DIM * PAIRS_PER_VOX };
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = (size_t)row_count * PAIRS_PER_ROW;
	if (idx >= total) return;
	size_t r = idx / PAIRS_PER_ROW;
	size_t off = idx - r * PAIRS_PER_ROW;
	unsigned int row = row_list ? row_list[r] : (unsigned int)r;
	int c0 = (int)((off % PAIRS_PER_VOX) * 2);
	__half2 *ptr = (__half2 *)(buffer + (size_t)row * CUDA_DIM * CUDA_CS) + off;
	float2 f = __half22float2(*ptr);
	f.x = cuda_gelu(fmaf(f.x, mul[c0], add[c0]));
	f.y = cuda_gelu(fmaf(f.y, mul[c0 + 1], add[c0 + 1]));
	*ptr = __float22half2_rn(f);
}

/* Coalesced uint4 copy from representative pure background row to npure rows. */
BC_CUDA_GLOBAL __global__ void fill_rows_kernel(bc_act *__restrict__ buffer,
		const unsigned int *__restrict__ pure_list, int npure, int rep_row) {
	enum { U4_PER_ROW = (CUDA_DIM * CUDA_CS * sizeof(bc_act)) / sizeof(uint4) };
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = (size_t)npure * U4_PER_ROW;
	if (idx >= total) return;
	size_t r = idx / U4_PER_ROW;
	size_t off = idx - r * U4_PER_ROW;
	unsigned int dst_row = pure_list[r];
	const uint4 *src = (const uint4 *)(buffer + (size_t)rep_row * CUDA_DIM * CUDA_CS);
	uint4 *dst = (uint4 *)(buffer + (size_t)dst_row * CUDA_DIM * CUDA_CS);
	dst[off] = src[off];
}

/* Fused final GroupNorm+GELU + shared-weight 1x1 classifier -> uint8 labels. */
BC_CUDA_GLOBAL __global__ void classify_u8_kernel(const bc_act *__restrict__ source,
		unsigned char *__restrict__ labels, const float *__restrict__ mul,
		const float *__restrict__ add, const bc_act *__restrict__ weights,
		const bc_act *__restrict__ bias, const unsigned int *__restrict__ row_list,
		int row_count, int channels, int classes) {
	__shared__ bc_act sh_w[BC_MODEL_CHAN * BC_MODEL_NCLASS];
	__shared__ bc_act sh_b[BC_MODEL_NCLASS];
	int wc = channels * classes;
	for (int i = (int)threadIdx.x; i < wc; i += (int)blockDim.x) sh_w[i] = weights[i];
	if (bias) {
		for (int i = (int)threadIdx.x; i < classes; i += (int)blockDim.x) sh_b[i] = bias[i];
	}
	__syncthreads();

	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = (size_t)row_count * CUDA_DIM;
	if (idx >= total) return;
	unsigned int row = row_list ? row_list[idx >> 8] : (unsigned int)(idx >> 8);
	size_t voxel = (size_t)row * CUDA_DIM + (idx & 255);
	const bc_act *src = source + voxel * CUDA_CS;
	float val[BC_MODEL_CHAN];
#pragma unroll
	for (int c = 0; c < BC_MODEL_CHAN; c++) {
		float v = __half2float(src[c]);
		if (mul) v = cuda_gelu(fmaf(v, mul[c], add[c]));
		val[c] = v;
	}
	int best_class = 0;
	float best_value = bias ? __half2float(sh_b[0]) : 0.0f;
#pragma unroll
	for (int c = 0; c < BC_MODEL_CHAN; c++)
		best_value = fmaf(val[c], __half2float(sh_w[c * classes]), best_value);
	for (int k = 1; k < classes; k++) {
		float cand = bias ? __half2float(sh_b[k]) : 0.0f;
#pragma unroll
		for (int c = 0; c < BC_MODEL_CHAN; c++)
			cand = fmaf(val[c], __half2float(sh_w[c * classes + k]), cand);
		if (cand > best_value) { best_value = cand; best_class = k; }
	}
	labels[voxel] = (unsigned char)best_class;
}

BC_CUDA_GLOBAL __global__ void fill_labels_u8_kernel(unsigned char *__restrict__ labels,
		const unsigned int *__restrict__ pure_list, int npure, int rep_row) {
	enum { U4_PER_ROW = CUDA_DIM / sizeof(uint4) };
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = (size_t)npure * U4_PER_ROW;
	if (idx >= total) return;
	size_t r = idx / U4_PER_ROW;
	size_t off = idx - r * U4_PER_ROW;
	const uint4 *src = (const uint4 *)(labels + (size_t)rep_row * CUDA_DIM);
	uint4 *dst = (uint4 *)(labels + (size_t)pure_list[r] * CUDA_DIM);
	dst[off] = src[off];
}

/* Fused final GroupNorm+GELU (when mul != NULL) + grouped-softmax tissue priors. */
BC_CUDA_GLOBAL __global__ void tissue_priors_kernel(const bc_act *__restrict__ source,
		float *__restrict__ planes, const float *__restrict__ mul,
		const float *__restrict__ add, const bc_act *__restrict__ weights,
		const bc_act *__restrict__ bias, const unsigned int *__restrict__ row_list,
		int row_count, int channels, int classes) {
#if BC_MODEL_NCLASS == 18 && !BC_MODEL_OUTPUT_IS_MASK
	__shared__ bc_act sh_w[BC_MODEL_CHAN * 18];
	__shared__ bc_act sh_b[18];
	int wc = channels * 18;
	for (int i = (int)threadIdx.x; i < wc; i += (int)blockDim.x) sh_w[i] = weights[i];
	if (bias && (int)threadIdx.x < 18) sh_b[threadIdx.x] = bias[threadIdx.x];
	__syncthreads();

	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = (size_t)row_count * CUDA_DIM;
	if (idx >= total) return;
	unsigned int row = row_list ? row_list[idx >> 8] : (unsigned int)(idx >> 8);
	size_t voxel = (size_t)row * CUDA_DIM + (idx & 255);
	const bc_act *src = source + voxel * CUDA_CS;
	float val[BC_MODEL_CHAN];
#pragma unroll
	for (int c = 0; c < BC_MODEL_CHAN; c++) {
		float v = __half2float(src[c]);
		if (mul) v = cuda_gelu(fmaf(v, mul[c], add[c]));
		val[c] = v;
	}
	float logits[18], maximum = -INFINITY;
	for (int k = 0; k < 18; k++) {
		float sum = bias ? __half2float(sh_b[k]) : 0.0f;
#pragma unroll
		for (int ch = 0; ch < BC_MODEL_CHAN; ch++)
			sum = fmaf(val[ch], __half2float(sh_w[ch * classes + k]), sum);
		logits[k] = sum;
		maximum = fmaxf(maximum, sum);
	}
	const unsigned masks[3] = {BC_PRIOR_GM_MASK, BC_PRIOR_WM_MASK, BC_PRIOR_CSF_MASK};
	float denominator = 0.0f, support_denominator = 0.0f;
	float numerator[3] = {0.0f, 0.0f, 0.0f}, background = 0.0f;
	for (int k = 0; k < 18; k++) {
		float centered = logits[k] - maximum;
		float weight = expf(centered / BC_PRIOR_TEMPERATURE);
		float support = expf(centered / BC_PRIOR_SUPPORT_TEMPERATURE);
		denominator += weight;
		support_denominator += support;
		if (k == 0) background = support;
		for (int t = 0; t < 3; t++)
			if (masks[t] & (1u << k)) numerator[t] += weight;
		if (BC_PRIOR_MIXED_GWM_MASK & (1u << k)) {
			float half = 0.5f * weight;
			numerator[0] += half;
			numerator[1] += half;
		}
	}
	for (int t = 0; t < 3; t++) planes[(size_t)t * CUDA_NVOX + voxel] = numerator[t] / denominator;
	planes[3 * CUDA_NVOX + voxel] = fminf(1.0f, fmaxf(0.0f, 1.0f - background / support_denominator));
#else
	(void)source; (void)planes; (void)mul; (void)add;
	(void)weights; (void)bias; (void)row_list; (void)row_count; (void)channels; (void)classes;
#endif
}

BC_CUDA_GLOBAL __global__ void fill_priors_kernel(float *__restrict__ planes,
		const unsigned int *__restrict__ pure_list, int npure, int rep_row) {
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = (size_t)npure * CUDA_DIM;
	if (idx >= total) return;
	unsigned int dst_row = pure_list[idx >> 8];
	size_t x = idx & 255;
	size_t src_v = (size_t)rep_row * CUDA_DIM + x;
	size_t dst_v = (size_t)dst_row * CUDA_DIM + x;
	for (int p = 0; p < 4; p++)
		planes[(size_t)p * CUDA_NVOX + dst_v] = planes[(size_t)p * CUDA_NVOX + src_v];
}

/* GPU 3-axis separable Gaussian blur + support gate for --pve (matching Metal/WebGPU). */
BC_CUDA_GLOBAL __global__ void pve_blur_kernel(const float *__restrict__ src,
		float *__restrict__ dst, int axis) {
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 3 * CUDA_NVOX) return;
	size_t p = idx / CUDA_NVOX;
	size_t v = idx - p * CUDA_NVOX;
	size_t stride = axis == 0 ? 1u : (axis == 1 ? 256u : 65536u);
	int coord = (int)((v / stride) & 255u);
	/* Matches BC_PVE_BLUR_SIGMA (0.65) in cat_lite.h. */
	const float sigma = 0.65f;
	const int radius = (int)(3.0f * sigma + 0.5f);
	const float inv2s2 = -0.5f / (sigma * sigma);
	const float *plane = src + p * CUDA_NVOX;
	float sum = 0.0f, total = 0.0f;
	for (int d = -radius; d <= radius; d++) {
		float wt = expf((float)(d * d) * inv2s2);
		total += wt;
		int q = coord + d;
		if ((unsigned)q < 256u)
			sum += wt * plane[(size_t)((ptrdiff_t)v + (ptrdiff_t)d * (ptrdiff_t)stride)];
	}
	dst[idx] = sum / total;
}

BC_CUDA_GLOBAL __global__ void pve_support_kernel(const float *__restrict__ blurred,
		float *__restrict__ planes) {
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 3 * CUDA_NVOX) return;
	size_t v = idx % CUDA_NVOX;
	float sup = fminf(1.0f, fmaxf(0.0f, planes[3 * CUDA_NVOX + v]));
	planes[idx] = blurred[idx] * sup;
}

BC_CUDA_GLOBAL __global__ void checksum_kernel(const bc_act *__restrict__ source,
		unsigned long long *__restrict__ total, size_t count) {
	size_t start = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	unsigned long long local = 0;
	for (size_t i = start; i < count; i += stride)
		local += (unsigned short)__half_as_ushort(source[i]);
	atomicAdd(total, local);
}

BC_CUDA_GLOBAL __global__ void fill_pattern_kernel(bc_act *__restrict__ buffer,
		size_t count) {
	size_t start = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	for (size_t i = start; i < count; i += stride) {
		unsigned x = (unsigned)i * 2654435761u;
		buffer[i] = __float2half((float)((int)(x & 1023u) - 512) * 0.001f);
	}
}

#endif
