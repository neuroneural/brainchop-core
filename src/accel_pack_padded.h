#ifndef BRAINCHOPC_ACCEL_PACK_PADDED_H
#define BRAINCHOPC_ACCEL_PACK_PADDED_H

/*
 * The padded f16 convolution-weight layout shared by CUDA and OpenCL.
 *
 * SHARED, NOT COPIED, for the reason accel_pack_weights.h states at length and
 * benchmarking records: the GELU was copied across backends, silently diverged,
 * and a MindGrab build ran model16's function. This is a second instance of the
 * same hazard -- it encodes the tap stride, the layer-0 special case, the affine
 * skip and the classifier's exclusion from padding, any of which produces a
 * plausible-looking wrong volume.
 *
 * WHY THIS IS NOT accel_pack_weights.h. The index formula is the same, but that
 * header is the WASM packer and does three things these backends must not do:
 * it applies the BC_MODEL_LOADER_TRANSPOSE tap permutation itself (CUDA and
 * OpenCL get pre-transposed weights from bc_accel_copy_weights, so including it
 * would double-transpose), it packs the affine diagonal and bias into the same
 * buffer (these upload the affine separately, per layer, as fp32), and it
 * carries a `pad_to` for texture-row alignment that neither needs. Folding all
 * four into one is blocked on unifying the kernel transpose, which this design
 * already lists under Known gaps -- that is the real prerequisite, not a header
 * move.
 *
 * WHAT THE PADDING IS FOR. The channel stride is widened to BC_PAD_CS (16)
 * because MindGrab is a 15-channel network and 15 tiles into nothing: an
 * m16n16k16 tensor-core fragment has no way to express it, and the scalar
 * kernels get a literal inner bound out of it. The pad lane is inert because
 * GroupNorm here is one group per channel, so a zeroed channel has mean 0 and
 * variance 0, normalises to (0-0)/sqrt(0+eps) = 0, and gelu(0) = 0 -- unless
 * the model has an affine, where bias would reach it. Each backend asserts
 * BC_MODEL_CHAN == its stride for affine models, which is what keeps that
 * caveat from mattering.
 *
 * Layer 0 keeps ONE input channel: padding it to 16 would be 16x the weights
 * for the one layer that has a single input, and it is also the layer the
 * tensor-core path skips. Its OUTPUT is padded, so the activation it writes
 * matches every later layer.
 *
 * USAGE, following the idiom accel_pack_weights.h establishes: define
 * `bc_pad_store` (the element type), `BC_PAD_STORE(x)` (the narrowing) and
 * `BC_PAD_CS` (the stride), then include this. CUDA uses __half/__float2half;
 * OpenCL uses uint16_t/bc_f32_to_f16. Both round to nearest even. Exactly one
 * translation unit per build includes it, so the function is static.
 *
 * A divergence here would be LOUD, unlike the GELU's: a structural mistake
 * moves values by order 1, far outside the 64/1024 fp16 budgets, so
 * make test-cuda and make test-opencl catch it. But those targets print "not
 * enabled" and pass when the backend is off, and nothing in CI compiles either
 * file -- which is exactly the gap the GELU divergence survived in. Sharing the
 * code is what removes the reliance on that gate.
 */

#include "backend_accel.h"
#include "model_specs.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if !defined(bc_pad_store) || !defined(BC_PAD_STORE) || !defined(BC_PAD_CS)
#error "define bc_pad_store, BC_PAD_STORE(x) and BC_PAD_CS before including"
#endif

typedef struct {
	bc_pad_store *data;
	size_t *layer_offset;      /* padded element offset of each layer's conv */
	size_t classifier_offset;
	size_t count;
} bc_padded_weights;

static void bc_free_padded_weights(bc_padded_weights *packed) {
	free(packed->data);
	free(packed->layer_offset);
	packed->data = NULL;
	packed->layer_offset = NULL;
}

static int bc_pack_padded_weights(const bc_accel_model_spec *spec,
		const float *source, bc_padded_weights *packed) {
	memset(packed, 0, sizeof(*packed));
	packed->layer_offset = (size_t *)calloc((size_t)spec->layers,
		sizeof(*packed->layer_offset));
	if (!packed->layer_offset) return 1;

	size_t padded = 0;
	for (int layer = 0; layer < spec->layers; layer++) {
		packed->layer_offset[layer] = padded;
		padded += (size_t)27 * (layer == 0 ? 1 : BC_PAD_CS) * BC_PAD_CS;
	}
	/* The classifier is NOT padded: it is indexed by the logical channel count
	   and has no row for the pad lane, so widening it would be read as data. */
	packed->classifier_offset = padded;
	padded += (size_t)spec->channels * spec->classes +
		(spec->classifier_bias ? (size_t)spec->classes : 0);
	packed->count = padded;

	/* calloc, not malloc: the pad lanes must be exactly zero. That is what
	   makes them contribute nothing, rather than whatever was on the heap. */
	packed->data = (bc_pad_store *)calloc(padded, sizeof(bc_pad_store));
	if (!packed->data) {
		bc_free_padded_weights(packed);
		return 1;
	}

	size_t raw = 0;
	for (int layer = 0; layer < spec->layers; layer++) {
		int input_channels = layer == 0 ? 1 : spec->channels;
		int padded_inputs = layer == 0 ? 1 : BC_PAD_CS;
		bc_pad_store *destination = packed->data + packed->layer_offset[layer];
		for (int tap = 0; tap < 27; tap++)
			for (int ic = 0; ic < input_channels; ic++)
				for (int oc = 0; oc < spec->channels; oc++)
					destination[((size_t)tap * padded_inputs + ic) * BC_PAD_CS + oc] =
						BC_PAD_STORE(source[raw +
							((size_t)tap * input_channels + ic) *
							spec->channels + oc]);
		raw += (size_t)27 * input_channels * spec->channels;
#if BC_MODEL_HAS_AFFINE
		/* Stepped over, not copied: run_normalize reads the affine from the
		   fp32 host copy and uploads it per layer. */
		raw += (size_t)BC_MODEL_CHAN * BC_MODEL_CHAN + BC_MODEL_CHAN;
#endif
	}
	size_t classifier = (size_t)spec->channels * spec->classes +
		(spec->classifier_bias ? (size_t)spec->classes : 0);
	for (size_t i = 0; i < classifier; i++)
		packed->data[packed->classifier_offset + i] =
			BC_PAD_STORE(source[raw + i]);
	return 0;
}

/*
 * Undefined so they cannot leak into the rest of the including file -- both
 * backends include this mid-file, and a stray BC_PAD_CS several hundred lines
 * later would read as a general constant rather than this header's parameter.
 * The include guard means a second include with DIFFERENT macros is a silent
 * no-op, so one translation unit gets one element type, by construction.
 */
#undef bc_pad_store
#undef BC_PAD_STORE
#undef BC_PAD_CS

#endif
