#ifndef BRAINCHOPC_ACCEL_PACK_WEIGHTS_H
#define BRAINCHOPC_ACCEL_PACK_WEIGHTS_H

/*
 * The GPU weight layout, shared by every backend that packs f16 weights.
 *
 * SHARED, NOT COPIED, and the reason is recorded in development notes: the GELU
 * approximation was copied across backends rather than shared, silently
 * diverged, and a MindGrab build ran model16's function. This is that hazard
 * with more surface -- it encodes the 3x3x3 tap permutation, the
 * BC_MODEL_LOADER_TRANSPOSE convention, the affine diagonal and the classifier
 * order, any one of which produces a plausible-looking wrong volume rather than
 * an error. Worse, the two backends are NEVER compiled together (-DHAVE_WEBGPU
 * and -DHAVE_WEBGL2 are mutually exclusive), so no build links both and no test
 * compares them: a divergence could only be found by noticing that one
 * backend's output had quietly moved.
 *
 * It began as a verbatim 60-line copy in backend_webgl2.c whose only real
 * difference was five lines of row padding, which is what `pad_to` is for.
 *
 * USAGE. There is no bc_store type here on purpose: the element type is the
 * backend's (f16 or, under BC_WG_F32/BC_WGL_F32, f32). Define `bc_store` and
 * `BC_STORE(x)` and `BC_CS`, then include this -- the same "define the alias,
 * then include the generic" idiom model_def.h already uses. Exactly one
 * translation unit per build includes it, so the function is static.
 *
 * This is a pure function: same weights in, same bytes out, no GL or WebGPU
 * state. A mistake here is a compile error or a gross voxel change, not a
 * subtle one -- both make test-wasm-gpu (37/682/88) and make test-wasm-gl
 * (33/710/94) exquisitely sensitive to it.
 */

#include "model_specs.h"
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	uint32_t kernel_off;
	uint32_t affine_off;
	uint32_t bias_off;
} bc_layer_offsets;

/*
 * Pack model.bin into the layout the kernels index:
 * ((tap * in_channels) + ic) * BC_CS + oc, which is meshnet_cpu_pack_kernel's.
 *
 * `pad_to` rounds the allocation up to a multiple of that many elements, so a
 * backend uploading into a 2D texture gets whole rows and can do it in one
 * call. 0 allocates exactly what is used. The tail is zero and never indexed.
 *
 * `count_out` receives the USED length in elements, which is not the allocated
 * length when pad_to is nonzero.
 */
static bc_store *bc_pack_weights(const float *weights, bc_layer_offsets *offsets,
		uint32_t *classifier_off, uint32_t *classifier_bias_off,
		size_t *count_out, size_t pad_to) {
	/* Worst case: every tap slot padded to CS inputs and CS outputs. */
	size_t capacity = 0;
	for (int layer = 0; layer < BC_MODEL_NHIDDEN; layer++) {
		size_t cin = layer == 0 ? 1 : (size_t)BC_CS;
		capacity += 27 * cin * BC_CS;
#if BC_MODEL_HAS_AFFINE
		capacity += 2 * BC_CS;
#endif
	}
	capacity += (size_t)BC_CS * BC_MODEL_NCLASS + BC_MODEL_NCLASS;

	size_t allocate = capacity;
	if (pad_to) allocate = ((capacity + pad_to - 1) / pad_to) * pad_to;
	bc_store *packed = (bc_store *)calloc(allocate, sizeof(*packed));
	if (!packed) return NULL;

	size_t src = 0, dst = 0;
	for (int layer = 0; layer < BC_MODEL_NHIDDEN; layer++) {
		int cin = layer == 0 ? 1 : BC_MODEL_CHAN;
		int cin_slots = layer == 0 ? 1 : BC_CS;
		offsets[layer].kernel_off = (uint32_t)dst;
		for (int dz = -1; dz <= 1; dz++)
		for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++) {
			int dst_tap = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
#if BC_MODEL_LOADER_TRANSPOSE
			int src_tap = dst_tap;   /* the loader already swapped the axes */
#else
			int src_tap = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1);
#endif
			for (int ic = 0; ic < cin; ic++)
				for (int oc = 0; oc < BC_MODEL_CHAN; oc++)
					packed[dst + ((size_t)dst_tap * cin_slots + ic) * BC_CS + oc] =
						BC_STORE(weights[src +
							((size_t)src_tap * cin + ic) * BC_MODEL_CHAN + oc]);
		}
		dst += (size_t)27 * cin_slots * BC_CS;
		src += (size_t)27 * cin * BC_MODEL_CHAN;
#if BC_MODEL_HAS_AFFINE
		/* Only the diagonal of the 1x1 affine is used, as on the CPU. */
		offsets[layer].affine_off = (uint32_t)dst;
		for (int c = 0; c < BC_MODEL_CHAN; c++)
			packed[dst + c] = BC_STORE(weights[src + c * BC_MODEL_CHAN + c]);
		dst += BC_CS;
		src += (size_t)BC_MODEL_CHAN * BC_MODEL_CHAN;
		offsets[layer].bias_off = (uint32_t)dst;
		for (int c = 0; c < BC_MODEL_CHAN; c++)
			packed[dst + c] = BC_STORE(weights[src + c]);
		dst += BC_CS;
		src += BC_MODEL_CHAN;
#else
		offsets[layer].affine_off = 0;
		offsets[layer].bias_off = 0;
#endif
	}

	*classifier_off = (uint32_t)dst;
	for (int c = 0; c < BC_MODEL_CHAN; c++)
		for (int k = 0; k < BC_MODEL_NCLASS; k++)
			packed[dst + (size_t)c * BC_MODEL_NCLASS + k] =
				BC_STORE(weights[src + (size_t)c * BC_MODEL_NCLASS + k]);
	dst += (size_t)BC_MODEL_CHAN * BC_MODEL_NCLASS;
	src += (size_t)BC_MODEL_CHAN * BC_MODEL_NCLASS;
	*classifier_bias_off = (uint32_t)dst;
#if BC_MODEL_CLASSIFIER_BIAS
	for (int k = 0; k < BC_MODEL_NCLASS; k++)
		packed[dst + k] = BC_STORE(weights[src + k]);
#endif
	dst += BC_MODEL_NCLASS;

	*count_out = dst;
	return packed;
}

#endif
