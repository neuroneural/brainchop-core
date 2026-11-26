#ifndef BRAINCHOP_CUDA_KERNELS_CUH
#define BRAINCHOP_CUDA_KERNELS_CUH

/*
 * Device code shared by the CUDA runtime backend and future offline module
 * generation. Keep host allocation, launchers, profiling and tuning policy in
 * backend_cuda.cu. The scalar and WMMA template variants are instantiated by
 * the launchers expanding cuda_variants.h.
 */
#include "model_specs.h"
#include "cuda_geometry.h"
#include "pve_constants.h"

#include <cuda_fp16.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Only the offline module exports kernels; the runtime backend retains its
 * original internal linkage. Templates keep their nvcc ABI names, recorded
 * and checked in the module catalog; the seven ordinary entries use C names. */
#ifdef BC_CUDA_MODULE_EXPORT
#define BC_CUDA_GLOBAL extern "C"
#define BC_CUDA_TEMPLATE
#else
#define BC_CUDA_GLOBAL static
#define BC_CUDA_TEMPLATE static
#endif

#define CUDA_DIM BRAINCHOP_MODEL_DIM
#define CUDA_NVOX ((size_t)CUDA_DIM * CUDA_DIM * CUDA_DIM)
/* Block size for the elementwise kernels (norm+gelu, classify). Unrelated to
   the convolution's block size, which the tuner owns. */
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

/*
 * Activation storage is fp16 on a 16-wide channel stride, which is two changes
 * with one motivation.
 *
 * These models are DESIGNED for fp16 -- it is what upstream brainchop ships and
 * what Metal and WebGPU already do -- so fp32 activations were paying bandwidth
 * for precision the models do not use. Accumulation stays fp32, and the
 * GroupNorm moments fp64, exactly as Metal keeps its reduction wider: the
 * squares in the variance overflow fp16 long before the activations do.
 *
 * The stride is padded to 16 rather than following the logical channel count,
 * for the same reason MESHNET_CPU_CHANNEL_STRIDE and BC_WG_CS are 16: MindGrab
 * is a 15-channel network and 15 tiles into nothing. The pad lane is inert
 * because GroupNorm here is one group per channel, so a zeroed channel has mean
 * 0 and variance 0, normalises to (0-0)/sqrt(0+eps) = 0, and gelu(0) = 0 -- it
 * stays identically zero for every layer. CLAUDE.md records this being verified
 * bit-identical in tinygrad when MindGrab was re-exported padded for Metal.
 *
 * CUDA_CS is the STRIDE; spec->channels remains the logical COUNT. Those were
 * the same number before this change and are not any more, so read every index
 * carefully: activations stride by CUDA_CS, weights and the classifier index by
 * the logical count.
 */
typedef __half bc_act;
/*
 * bc_accel_narrow_input allocates ACCEL_NVOX * sizeof(uint16_t) and
 * backend_cuda.cu copies CUDA_NVOX * sizeof(bc_act) out of it. The counts agree
 * (both are BRAINCHOP_MODEL_DIM cubed); the SIZES agree only because bc_act is
 * 16 bits, which nothing else here asserts. Before the narrowing moved to
 * backend_accel.c the malloc and the memcpy shared one expression and could not
 * disagree.
 */
static_assert(sizeof(bc_act) == sizeof(uint16_t),
	"bc_accel_narrow_input returns uint16_t; a wider bc_act would read past it");
#define CUDA_CS CUDA_MAX_CHANNELS

/*
 * NDHWC convolution using [dz][dy][dx][in][out] weights.
 *
 * A thread-per-voxel version of this loop is load/store bound rather than FMA
 * bound: it re-reads all 27*Cin*Cout weights from memory for every voxel, which
 * is roughly one load per FMA, and an SM has far fewer load/store units than
 * FMA units. Two changes remove that bottleneck:
 *
 *   1. The layer's weights are staged once per block into shared memory.
 *   2. Each thread owns VPT voxels, and one tap's output-channel weights
 *      are hoisted into registers so they amortise across all of them.
 *
 * The voxels a thread owns are strided by the block size so global loads stay
 * coalesced. Accumulation order over (tap, input channel, output channel) is
 * unchanged, so results are bit-identical to the simple version -- and, more
 * usefully, bit-identical across every (VPT, THREADS) pair, since changing them
 * only changes WHICH thread owns a voxel, never the order in which that voxel's
 * own products are summed. cuda_tune() leans on that: it can demand an exact
 * checksum match from a candidate rather than a tolerance.
 *
 * VPT and THREADS are template parameters so the tuner can pick them per
 * device. __launch_bounds__ is what makes the wide blocks usable at all: it
 * tells ptxas the occupancy it must hit, so it caps registers (spilling if it
 * must) instead of emitting a kernel that cannot launch. Without it,
 * VPT=8/THREADS=512 failed with cudaErrorLaunchOutOfResources.
 *
 * Measured on an RTX 4070 Ti SUPER, alternating A/B: the shipped 8/128 pair ran
 * the MindGrab convolutions in 1531 ms and the tuner's 3/512 in 650 ms (2.36x);
 * model16 is 786 -> 377 ms (2.08x). Both models want the same pair, but they do
 * not gain equally.
 */
/*
 * The convolution shape used until the tuner has an answer -- and the one
 * BRAINCHOPC_TUNE=0 pins, so it should be the best portable guess rather than a
 * conservative one. 3/512 is the measured optimum on an RTX 4070 Ti SUPER for
 * both models; __launch_bounds__ makes it launchable regardless of register
 * file size, so a device that dislikes it pays in spills, not in failure.
 */
#ifndef CUDA_CONV_VPT
#define CUDA_CONV_VPT BC_CUDA_GEOMETRY_CONV_VPT
#endif
#ifndef CUDA_CONV_THREADS
#define CUDA_CONV_THREADS BC_CUDA_GEOMETRY_CONV_THREADS
#endif
template <int VPT, int THREADS>
__global__ __launch_bounds__(THREADS) BC_CUDA_TEMPLATE void conv3d_kernel(
		const bc_act *__restrict__ source,
		bc_act *__restrict__ destination, const bc_act *__restrict__ weights,
		int input_channels, int output_channels, int input_stride, int dilation) {
	(void)output_channels;
	/* Staged in fp16, which also halves the shared-memory footprint from 27 KB
	   to 13.5 KB. That is the resource the launch shape was competing for --
	   the tuner picked wide blocks precisely to amortise it -- so the occupancy
	   ceiling moves as a side effect of the precision change, and the tuned
	   shape has to be re-measured rather than carried over. */
	extern __shared__ bc_act shared_weights[];
	int weight_count = 27 * input_channels * CUDA_CS;
	for (int i = (int)threadIdx.x; i < weight_count; i += THREADS)
		shared_weights[i] = weights[i];
	__syncthreads();

	size_t base = (size_t)blockIdx.x * ((size_t)THREADS * VPT) + threadIdx.x;
	float accumulator[VPT][CUDA_MAX_CHANNELS];
#pragma unroll
	for (int v = 0; v < VPT; v++)
#pragma unroll
		for (int channel = 0; channel < CUDA_MAX_CHANNELS; channel++)
			accumulator[v][channel] = 0.0f;

	size_t voxel[VPT];
	int x[VPT], y[VPT], z[VPT];
#pragma unroll
	for (int v = 0; v < VPT; v++) {
		voxel[v] = base + (size_t)v * THREADS;
		x[v] = (int)(voxel[v] & 255);
		y[v] = (int)((voxel[v] >> 8) & 255);
		z[v] = (int)(voxel[v] >> 16);
	}

	for (int tz = -1; tz <= 1; tz++) {
		for (int ty = -1; ty <= 1; ty++) {
			for (int tx = -1; tx <= 1; tx++) {
				const bc_act *tap = shared_weights +
					(size_t)((tz + 1) * 9 + (ty + 1) * 3 + tx + 1) *
					input_channels * CUDA_CS;
				/* NULL marks a tap that falls outside the volume for this voxel. */
				const bc_act *value[VPT];
#pragma unroll
				for (int v = 0; v < VPT; v++) {
					int input_x = x[v] + tx * dilation;
					int input_y = y[v] + ty * dilation;
					int input_z = z[v] + tz * dilation;
					int inside = ((unsigned)input_x < CUDA_DIM) &
						((unsigned)input_y < CUDA_DIM) &
						((unsigned)input_z < CUDA_DIM) & (voxel[v] < CUDA_NVOX);
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
						/* The weight buffer is pre-padded to CUDA_CS wide with
						   zeros (bc_pack_padded_weights), so the tail lanes
						   contribute nothing and no runtime bound is needed.
						   Indexing this by output_channels instead is silently
						   correct for a 16-channel model and silently WRONG for
						   MindGrab's 15 -- it cost 1.3M voxels once. */
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
#pragma unroll
	for (int v = 0; v < VPT; v++) {
		if (voxel[v] >= CUDA_NVOX) continue;
		bc_act *output = destination + voxel[v] * CUDA_CS;
		/* All CUDA_CS lanes, not just the logical channels: the pad lane's
		   accumulator is zero (its weights were zeroed above), and writing it
		   keeps the store contiguous and the buffer fully defined. */
#pragma unroll
		for (int channel = 0; channel < CUDA_CS; channel++)
			output[channel] = __float2half(accumulator[v][channel]);
	}
}

/*
 * Tensor-core convolution, for the hidden layers on sm_70 and newer.
 *
 * The whole convolution is a matmul hiding behind a stencil: for one tap,
 * out[voxel][oc] += sum_ic in[voxel + tapoffset][ic] * w[tap][ic][oc]. With the
 * channel stride padded to 16 that is exactly an m16n16k16 fragment -- M is 16
 * voxels, N the 16 output channels, K the 16 input channels -- so a warp
 * computes 16 voxels x 16 outputs by issuing 27 mma instructions, one per tap,
 * instead of 27 * 16 * 16 scalar FMAs.
 *
 * Three facts make the tiles line up, and all three are load-bearing:
 *
 *   - 16 divides DIM, so a 16-voxel tile never straddles a row. x is the
 *     fastest axis, so the tile's 16 voxels are 16 consecutive x positions in
 *     one (y, z) row and its 16x16 channel block is contiguous.
 *   - the activation stride is CUDA_CS, which is the fragment's leading
 *     dimension, so an in-bounds tile loads straight out of global memory.
 *   - the weights are already padded to 16x16 per tap, which is the only reason
 *     MindGrab's 15 channels can be expressed at all: a fragment has no way to
 *     say 15.
 *
 * Accumulation is fp32 inside the fragment. The ORDER differs from the scalar
 * kernel -- the hardware reduces K its own way -- so the two do not agree
 * bit-for-bit, which is why this kernel is not part of the tuner's
 * checksum-equality family.
 *
 * Layer 0 is deliberately excluded: it has one input channel and a stride-1
 * source, so K would be 1 and the fragment 15/16 zero. It stays on the scalar
 * kernel, which is also what keeps a working path on sm_60.
 */
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
#include <mma.h>
#endif

/*
 * Does the code that will actually RUN on this device contain tensor-core
 * instructions? That is not the same question as "does the device have tensor
 * cores", and confusing the two produced a silently wrong segmentation.
 *
 * conv3d_wmma_kernel's body is compiled out below sm_70. A build restricted to
 * an older architecture -- `make CUDA_ARCH=sm_60`, which this project
 * advertises as the fast-iteration flag -- still RUNS on a modern card, because
 * compute_60 PTX JITs forward. The kernel that arrives is then the empty
 * fallback. Probing only properties.major would launch it, get cudaSuccess, and
 * never write the destination: measured 2,157,365 of 16,777,216 voxels wrong on
 * an RTX 4070 Ti SUPER, exit status 0.
 *
 * So ask the compiled code, not the device. This kernel is subject to exactly
 * the same #if as the convolution, so whatever the loader selects for this
 * device answers for itself.
 */
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
		const bc_act *__restrict__ weights, int dilation) {
#if __CUDA_ARCH__ >= 700
	using namespace nvcuda;
	extern __shared__ __align__(32) char wmma_shared[];
	bc_act *staged = (bc_act *)wmma_shared;
	const int weight_count = 27 * CUDA_CS * CUDA_CS;
	char *scratch_base = wmma_shared + (size_t)weight_count * sizeof(bc_act);

	int warp = (int)threadIdx.x / 32;
	int lane = (int)threadIdx.x % 32;
	/* One scratch slot per warp, used first to stage a clipped input tile and
	   afterwards to land the accumulator. The two uses do not overlap in time,
	   so they share the space. */
	char *scratch = scratch_base + (size_t)warp * CUDA_WMMA_SCRATCH;
	bc_act *tile_staging = (bc_act *)scratch;
	float *tile_output = (float *)scratch;

	for (int i = (int)threadIdx.x; i < weight_count; i += WARPS * 32)
		staged[i] = weights[i];
	__syncthreads();

	size_t tile = (size_t)blockIdx.x * WARPS + warp;
	size_t base = tile * CUDA_WMMA_TILE;
	if (base >= CUDA_NVOX) return;
	int x0 = (int)(base & (CUDA_DIM - 1));
	int y0 = (int)((base >> 8) & (CUDA_DIM - 1));
	int z0 = (int)(base >> 16);

#if CUDA_CS == 24
	/* Global rows stay compact. Both K and N tails are zero-padded in
	 * aligned per-warp scratch; a 48-byte global row is not WMMA aligned
	 * at every dilated spatial offset. A/B staging shares its 1 KiB with
	 * the final fp32 output tile, whose lifetime starts after the last MMA. */
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
	wmma::fill_fragment(acc[0], 0.0f);
	wmma::fill_fragment(acc[1], 0.0f);
	bc_act *tile_b = tile_staging + 256;
	for (int tz = -1; tz <= 1; tz++)
	for (int ty = -1; ty <= 1; ty++)
	for (int tx = -1; tx <= 1; tx++) {
		int iy = y0 + ty * dilation, iz = z0 + tz * dilation;
		int ix = x0 + tx * dilation;
		if ((unsigned)iy >= CUDA_DIM || (unsigned)iz >= CUDA_DIM) continue;
		const bc_act *tap = staged +
			(size_t)((tz + 1) * 9 + (ty + 1) * 3 + tx + 1) * CUDA_CS * CUDA_CS;
		size_t row = ((size_t)iz * CUDA_DIM + iy) * CUDA_DIM;
#pragma unroll
		for (int k = 0; k < 2; k++) {
			for (int e = lane; e < 256; e += 32) {
				int xx = ix + e / 16, ic = k * 16 + e % 16;
				tile_staging[e] = ((unsigned)xx < CUDA_DIM && ic < CUDA_CS)
					? source[(row + xx) * CUDA_CS + ic] : __float2half(0.0f);
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
			wmma::load_matrix_sync(a, tile_staging, 16);
			__syncwarp();
#pragma unroll
			for (int n = 0; n < 2; n++) {
				for (int e = lane; e < 256; e += 32) {
					int ic = k * 16 + e / 16, oc = n * 16 + e % 16;
					tile_b[e] = (ic < CUDA_CS && oc < CUDA_CS)
						? tap[ic * CUDA_CS + oc] : __float2half(0.0f);
				}
				__syncwarp();
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
				wmma::load_matrix_sync(b, tile_b, 16);
				wmma::mma_sync(acc[n], a, b, acc[n]);
				__syncwarp();
			}
		}
	}
#pragma unroll
	for (int n = 0; n < 2; n++) {
		wmma::store_matrix_sync(tile_output, acc[n], 16, wmma::mem_row_major);
		__syncwarp();
		for (int e = lane; e < 256; e += 32) {
			int oc = n * 16 + e % 16;
			if (oc < CUDA_CS)
				destination[(base + e / 16) * CUDA_CS + oc] = __float2half(tile_output[e]);
		}
		__syncwarp();
	}
#else
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
	wmma::fill_fragment(acc, 0.0f);

	for (int tz = -1; tz <= 1; tz++) {
		for (int ty = -1; ty <= 1; ty++) {
			for (int tx = -1; tx <= 1; tx++) {
				int iy = y0 + ty * dilation;
				int iz = z0 + tz * dilation;
				int ix = x0 + tx * dilation;
				if ((unsigned)iy >= CUDA_DIM || (unsigned)iz >= CUDA_DIM)
					continue;   /* whole tile outside: contributes nothing */
				const bc_act *tap = staged +
					(size_t)((tz + 1) * 9 + (ty + 1) * 3 + tx + 1) *
					CUDA_CS * CUDA_CS;
				size_t row = ((size_t)iz * CUDA_DIM + iy) * CUDA_DIM;
				wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
					wmma::row_major> a;
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
					wmma::row_major> b;
				if (ix >= 0 && ix + CUDA_WMMA_TILE <= CUDA_DIM) {
					/* Uniform across the warp, so this branch never diverges. */
					wmma::load_matrix_sync(a,
						source + (row + ix) * CUDA_CS, CUDA_CS);
				} else {
					for (int e = lane; e < CUDA_WMMA_TILE * CUDA_CS; e += 32) {
						int voxel_in_tile = e / CUDA_CS;
						int channel = e % CUDA_CS;
						int xx = ix + voxel_in_tile;
						tile_staging[e] = ((unsigned)xx < CUDA_DIM)
							? source[(row + xx) * CUDA_CS + channel]
							: __float2half(0.0f);
					}
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
	bc_act *out = destination + base * CUDA_CS;
	for (int e = lane; e < CUDA_WMMA_TILE * CUDA_CS; e += 32)
		out[e] = __float2half(tile_output[e]);
#endif
#else
	/* Never launched below sm_70; the host checks the capability first. The
	   body still has to compile for every architecture in the fatbin. */
	(void)source; (void)destination; (void)weights; (void)dilation;
#endif
}

/*
 * Per-channel sum and sum-of-squares over the activation buffer.
 *
 * The buffer is [voxel][channel] with the channel contiguous, so the obvious
 * shape -- one block row per channel, indexing source[i * channels + channel]
 * -- gives every lane of a warp a 64-byte stride. That is one 32-byte sector
 * fetched per useful 4-byte value, and it repeats for all `channels` rows, so
 * the pass moved roughly 16x the activation buffer.
 *
 * Instead each thread walks the buffer FLAT, so a warp reads 128 contiguous
 * bytes per instruction and the buffer is read exactly once. The trick that
 * makes this legal: the grid stride (gridDim.x * blockDim.x) is a multiple of
 * `channels`, so a thread's flat index keeps the same residue mod `channels`
 * for its whole walk -- one thread accumulates one channel, as before. The
 * block reduction then folds down to `channels` slots rather than to one.
 *
 * This DOES change the summation order, so the fp64 totals may differ in the
 * last bits from the row-per-channel version. That was already true run to run
 * -- the atomicAdd order across blocks is nondeterministic -- and both models
 * still agree with the CPU reference exactly. Measured: 308 ms -> 80 ms.
 *
 * The channel count need NOT be a power of two -- MindGrab's is 15 -- which is
 * why the fold below is an arithmetic walk rather than the usual halving tree.
 * The one requirement, that the grid stride be a multiple of it, is enforced at
 * the launch site.
 */
static_assert(BC_MODEL_CHAN <= CUDA_REDUCE_THREADS,
	"the block fold needs at least one thread per channel");
/*
 * The real ceiling in this file. conv3d_kernel's accumulator is
 * [VPT][CUDA_MAX_CHANNELS] and is indexed by output channel, and
 * moments_finish_kernel launches exactly CUDA_MAX_CHANNELS threads, so a wider
 * model would overrun a register array and silently drop the tail channels'
 * moments. CLAUDE.md promises that adding a model needs no shared C changes;
 * this is one of the places that promise has a bound.
 */
/*
 * norm_gelu_kernel indexes scale/bias by the 16-wide stride, so a model with an
 * affine must fill that stride exactly; otherwise the pad lane would read past
 * the compact arrays. True of model16 (16 channels, affine) and vacuous for
 * MindGrab (15 channels, no affine), which is why no padding is synthesised.
 */
static_assert(!BC_MODEL_HAS_AFFINE || BC_MODEL_CHAN == CUDA_MAX_CHANNELS,
	"an affine model must fill the activation stride; pad its scale/bias "
	"before widening CUDA_CS beyond its channel count");
static_assert(BC_MODEL_CHAN <= CUDA_MAX_CHANNELS,
	"this backend's accumulator and moments kernel are sized by "
	"CUDA_MAX_CHANNELS; raise it for a wider model");
/*
 * There is deliberately NO `static_assert(BC_MODEL_NHIDDEN & 1)` in either
 * accelerator backend, though both carried one for a while. It guarded classify
 * against aliasing its source with its destination at an even layer count -- a
 * real race, since thread v writes labels[v] while reading
 * source[v * channels ...], and both classify_kernel parameters are
 * __restrict__. Neither needs it now: each derives its label buffer from the
 * SAME layer-count parity as `last`, so the two are never the same pointer at
 * ANY parity (see the comment at the classify_kernel launch). The restriction
 * on the model is gone rather than asserted away.
 */

BC_CUDA_GLOBAL __global__ void reduce_channels_kernel(const bc_act *__restrict__ source,
		double *__restrict__ sums, int channels) {
	__shared__ double shared_sum[CUDA_REDUCE_THREADS];
	__shared__ double shared_square[CUDA_REDUCE_THREADS];
	size_t total = CUDA_NVOX * (size_t)channels;
	size_t start = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	double sum = 0.0, square = 0.0;
	for (size_t i = start; i < total; i += stride) {
		double value = __half2float(source[i]);
		sum += value;
		square += value * value;
	}
	shared_sum[threadIdx.x] = sum;
	shared_square[threadIdx.x] = square;
	__syncthreads();
	/*
	 * A halving tree cannot be used here: it pairs lane t with lane t + step,
	 * which hold DIFFERENT channels unless the channel count divides every
	 * step. MindGrab's is 15. So the first `channels` lanes each walk their own
	 * arithmetic sequence instead -- about blockDim/channels adds, once per
	 * block, which is nothing beside the buffer walk above.
	 *
	 * Lane t of this block holds channel (blockIdx.x * blockDim.x + t) %
	 * channels, so the lanes belonging to channel c start at the offset that
	 * makes that residue come out to c.
	 */
	int base = (int)(((size_t)blockIdx.x * blockDim.x) % (size_t)channels);
	if ((int)threadIdx.x < channels) {
		int first = ((int)threadIdx.x - base + channels) % channels;
		double lane_sum = 0.0, lane_square = 0.0;
		for (int i = first; i < (int)blockDim.x; i += channels) {
			lane_sum += shared_sum[i];
			lane_square += shared_square[i];
		}
		atomicAdd(sums + threadIdx.x, lane_sum);
		atomicAdd(sums + channels + threadIdx.x, lane_square);
	}
}

/*
 * GELU, in the flavour this model declares -- the same two approximations
 * meshnet_cpu.c and webgpu_kernels.h select between, keyed off the same
 * generated macros.
 *
 * This used to be a single hardcoded expression, model16's, which meant a
 * MindGrab CUDA build silently ran the OTHER model's approximation. The two
 * agree closely (2.302208198144325 = 2 * 0.797884583 / ln 2), so it was
 * absorbed by the voxel tolerance rather than producing a visibly wrong
 * answer, which is exactly why it survived unnoticed: nothing in CI has ever
 * compiled this file.
 */
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

/*
 * Reduce the fp64 moment sums to per-channel (mean, inverse deviation).
 *
 * This used to live inside norm_gelu_kernel's voxel loop, which recomputed both
 * -- two fp64 divisions and an fp64 rsqrt -- once per voxel per channel, i.e.
 * 268M times per layer for a quantity that has only `channels` distinct values.
 * Consumer NVIDIA parts run fp64 at 1/64 rate, so that dominated the pass.
 * Hoisting it here is bit-identical: the arithmetic and the order of the casts
 * to float are unchanged, only the number of times it runs. Measured on an
 * RTX 4070 Ti SUPER: 457 ms -> 174 ms per MindGrab inference.
 *
 * One block of CUDA_MAX_CHANNELS threads; the work is 16 lanes wide.
 */
BC_CUDA_GLOBAL __global__ void moments_finish_kernel(const double *__restrict__ sums,
		float *__restrict__ mean_out, float *__restrict__ invstd_out,
		int channels) {
	int channel = (int)threadIdx.x;
	if (channel >= channels) return;
	double mean = sums[channel] / (double)CUDA_NVOX;
	double variance = sums[channels + channel] / (double)CUDA_NVOX - mean * mean;
	if (variance < 0.0) variance = 0.0;
	mean_out[channel] = (float)mean;
	invstd_out[channel] = (float)rsqrt(variance + 1e-5);
}

BC_CUDA_GLOBAL __global__ void norm_gelu_kernel(bc_act *buffer,
		const float *__restrict__ mean, const float *__restrict__ invstd,
		const float *scale, const float *bias, int channels) {
	size_t voxel = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (voxel >= CUDA_NVOX) return;
	bc_act *row = buffer + voxel * CUDA_CS;
	/* Runs to CUDA_CS, not the logical count, so the loop bound is a literal
	   and the pad lane keeps its inert zero rather than being left undefined.
	   The affine arrays are indexed by the same bound, which the static_assert
	   below pins to models whose channel count already fills the stride. */
#pragma unroll
	for (int channel = 0; channel < CUDA_CS; channel++) {
		float value = (__half2float(row[channel]) - mean[channel]) *
			invstd[channel];
		if (scale) value = value * scale[channel] + bias[channel];
		row[channel] = __float2half(cuda_gelu(value));
	}
	(void)channels;
}

BC_CUDA_GLOBAL __global__ void classify_kernel(const bc_act *__restrict__ source,
		float *__restrict__ labels, const bc_act *__restrict__ weights,
		const bc_act *__restrict__ bias, int channels, int classes) {
	size_t voxel = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (voxel >= CUDA_NVOX) return;
	/* Strides by CUDA_CS but sums only the LOGICAL channels: the classifier
	   weight matrix is [channels][classes] and has no row for the pad lane, so
	   running this loop to CUDA_CS would read past it. */
	const bc_act *value = source + voxel * CUDA_CS;
	int best_class = 0;
	float best_value = bias ? __half2float(bias[0]) : 0.0f;
	for (int channel = 0; channel < channels; channel++)
		best_value = fmaf(__half2float(value[channel]),
			__half2float(weights[channel * classes]), best_value);
	for (int class_index = 1; class_index < classes; class_index++) {
		float candidate = bias ? __half2float(bias[class_index]) : 0.0f;
		for (int channel = 0; channel < channels; channel++)
			candidate = fmaf(__half2float(value[channel]),
				__half2float(weights[channel * classes + class_index]), candidate);
		if (candidate > best_value) {
			best_value = candidate;
			best_class = class_index;
		}
	}
	labels[voxel] = (float)best_class;
}

/* Four priors, never 18 materialized logit volumes. The destination is the
 * dead activation buffer, disjoint from the final hidden activation. */
BC_CUDA_GLOBAL __global__ void tissue_priors_kernel(const bc_act *__restrict__ source,
		float *__restrict__ planes, const bc_act *__restrict__ weights,
		const bc_act *__restrict__ bias, int channels, int classes) {
#if BC_MODEL_NCLASS == 18 && !BC_MODEL_OUTPUT_IS_MASK
	size_t voxel = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (voxel >= CUDA_NVOX) return;
	const bc_act *value = source + voxel * CUDA_CS;
	float logits[18], maximum = -INFINITY;
	for (int c = 0; c < 18; c++) {
		float sum = bias ? __half2float(bias[c]) : 0.0f;
		for (int ch = 0; ch < channels; ch++)
			sum = fmaf(__half2float(value[ch]), __half2float(weights[ch * classes + c]), sum);
		logits[c] = sum;
		maximum = fmaxf(maximum, sum);
	}
	const unsigned masks[3] = {BC_PRIOR_GM_MASK, BC_PRIOR_WM_MASK, BC_PRIOR_CSF_MASK};
	float denominator = 0.0f, support_denominator = 0.0f;
	float numerator[3] = {0.0f, 0.0f, 0.0f}, background = 0.0f;
	for (int c = 0; c < 18; c++) {
		float centered = logits[c] - maximum;
		float weight = expf(centered / BC_PRIOR_TEMPERATURE);
		float support = expf(centered / BC_PRIOR_SUPPORT_TEMPERATURE);
		denominator += weight;
		support_denominator += support;
		if (c == 0) background = support;
		for (int t = 0; t < 3; t++)
			if (masks[t] & (1u << c)) numerator[t] += weight;
		if (BC_PRIOR_MIXED_GWM_MASK & (1u << c)) {
			float half = 0.5f * weight;
			numerator[0] += half;
			numerator[1] += half;
		}
	}
	for (int t = 0; t < 3; t++) planes[(size_t)t * CUDA_NVOX + voxel] = numerator[t] / denominator;
	planes[3 * CUDA_NVOX + voxel] = fminf(1.0f, fmaxf(0.0f, 1.0f - background / support_denominator));
#else
	(void)source; (void)planes; (void)weights; (void)bias; (void)channels; (void)classes;
#endif
}

/*
 * Fingerprint the convolution output EXACTLY.
 *
 * Summing the floats would not work: a candidate is timed against a reference
 * whose blocks finish in a different order, and fp addition is not associative,
 * so a correct variant would look wrong. Reinterpreting each float as an
 * integer and summing that in 64 bits is associative and commutative, so the
 * checksum is independent of block scheduling while still changing if any
 * single output bit changes.
 */
BC_CUDA_GLOBAL __global__ void checksum_kernel(const bc_act *__restrict__ source,
		unsigned long long *__restrict__ total, size_t count) {
	size_t start = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	unsigned long long local = 0;
	for (size_t i = start; i < count; i += stride)
		local += (unsigned long long)__half_as_ushort(source[i]);
	atomicAdd(total, local);
}

/* A deterministic, non-degenerate pattern to tune against. */
BC_CUDA_GLOBAL __global__ void fill_pattern_kernel(bc_act *buffer, size_t count) {
	size_t start = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	for (size_t i = start; i < count; i += stride) {
		unsigned int h = (unsigned int)i * 2654435761u;
		buffer[i] = __float2half((float)(h >> 8) * (1.0f / 16777216.0f) - 0.5f);
	}
}

#undef BC_CUDA_GLOBAL
#undef BC_CUDA_TEMPLATE

#endif /* BRAINCHOP_CUDA_KERNELS_CUH */
