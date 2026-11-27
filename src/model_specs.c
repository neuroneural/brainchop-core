#include "chan_stride.h"
#include "model_specs.h"

/*
 * The dilation schedule comes from the generated header rather than being
 * transcribed here, so a model whose model.json changes its schedule cannot
 * silently keep running the old one.
 */
const int brainchop_model_dilation[BC_MODEL_NHIDDEN] = BC_MODEL_DILATION;

/*
 * Cross-checks between the offset arithmetic meshnet_cpu_prepare() performs and
 * the offset TABLE the generator derived independently by walking the weights
 * manifest. The walk encodes an assumption about layout; the table encodes what
 * model.json actually says. Asserting they agree turns a silent mislayout --
 * reading weights at the wrong offset, which yields a plausible but wrong
 * segmentation rather than a crash -- into a compile error.
 */
#if BC_MODEL_IS_MINDGRAB
_Static_assert(BC_MINDGRAB_W_CONV3D_0_GN_KERNEL_OFF == 0,
	"first layer must start at offset zero");
_Static_assert(BC_MINDGRAB_W_CONV3D_1_GN_KERNEL_OFF ==
		BC_MINDGRAB_W_CONV3D_0_GN_KERNEL_LEN,
	"MindGrab layer stride disagrees with the model.json manifest");
_Static_assert(BC_MINDGRAB_W_CONV3D_0_GN_KERNEL_LEN == 27 * BC_MODEL_CHAN,
	"first-layer kernel size disagrees with the model.json manifest");
_Static_assert(BC_MINDGRAB_W_CONV3D_1_GN_KERNEL_LEN ==
		27 * BC_MODEL_CHAN * BC_MODEL_CHAN,
	"hidden-layer kernel size disagrees with the model.json manifest");
_Static_assert(BC_MINDGRAB_W_OUTPUT_KERNEL_LEN == BC_MODEL_CHAN * BC_MODEL_NCLASS,
	"classifier kernel size disagrees with the model.json manifest");
_Static_assert(BC_MINDGRAB_W_OUTPUT_BIAS_OFF + BC_MINDGRAB_W_OUTPUT_BIAS_LEN ==
		BC_MODEL_WEIGHT_FLOATS,
	"weights do not end with the classifier bias");
_Static_assert(BC_MODEL_CLASSIFIER_BIAS == 1,
	"MindGrab must apply the trained classifier bias");
_Static_assert(BC_MODEL_HAS_AFFINE == 0, "MindGrab has no post-GroupNorm affine");
#endif

#if BC_MODEL_IS_MODEL16CHAN18CLS
_Static_assert(BC_MODEL16_W_CONV3D_0_GN_KERNEL_OFF == 0,
	"first layer must start at offset zero");
_Static_assert(BC_MODEL16_W_CONV3D_1_GN_KERNEL_OFF ==
		BC_MODEL16_W_CONV3D_0_GN_KERNEL_LEN +
		BC_MODEL16_W_AFFINE_0_KERNEL_LEN + BC_MODEL16_W_AFFINE_0_BIAS_LEN,
	"model16 layer stride disagrees with the model.json manifest");
_Static_assert(BC_MODEL16_W_AFFINE_0_KERNEL_LEN == BC_MODEL_CHAN * BC_MODEL_CHAN,
	"affine is not a full CHAN x CHAN matrix");
_Static_assert(BC_MODEL16_W_CONV3D_FINAL_KERNEL_LEN ==
		BC_MODEL_CHAN * BC_MODEL_NCLASS,
	"classifier kernel size disagrees with the model.json manifest");
_Static_assert(BC_MODEL16_W_CONV3D_FINAL_BIAS_OFF +
		BC_MODEL16_W_CONV3D_FINAL_BIAS_LEN == BC_MODEL_WEIGHT_FLOATS,
	"weights do not end with the classifier bias");
_Static_assert(BC_MODEL_HAS_AFFINE == 1,
	"model16 is expected to carry a post-GroupNorm affine");
#endif

#if BC_MODEL_IS_MINDMAP
_Static_assert(BC_MINDMAP_W_CONV3D_0_GN_KERNEL_OFF == 0,
	"first layer must start at offset zero");
_Static_assert(BC_MINDMAP_W_CONV3D_1_GN_KERNEL_OFF ==
		BC_MINDMAP_W_CONV3D_0_GN_KERNEL_LEN +
		BC_MINDMAP_W_AFFINE_0_KERNEL_LEN + BC_MINDMAP_W_AFFINE_0_BIAS_LEN,
	"mindmap layer stride disagrees with the model.json manifest");
_Static_assert(BC_MINDMAP_W_AFFINE_0_KERNEL_LEN == BC_MODEL_CHAN * BC_MODEL_CHAN,
	"affine is not a full CHAN x CHAN matrix");
_Static_assert(BC_MINDMAP_W_CONV3D_FINAL_KERNEL_LEN ==
		BC_MODEL_CHAN * BC_MODEL_NCLASS,
	"classifier kernel size disagrees with the model.json manifest");
_Static_assert(BC_MINDMAP_W_CONV3D_FINAL_BIAS_OFF +
		BC_MINDMAP_W_CONV3D_FINAL_BIAS_LEN == BC_MODEL_WEIGHT_FLOATS,
	"weights do not end with the classifier bias");
_Static_assert(BC_MODEL_HAS_AFFINE == 1,
	"mindmap is expected to carry a post-GroupNorm affine");
/* 24 channels exactly fill the stride, so this model has no inert pad
   lanes -- and therefore cannot detect a backend that confuses the stride
   with the logical count. mindgrab is the only model that can. */
_Static_assert(BC_CHAN_STRIDE == 24, "mindmap must stride activations by 24");
#endif

#if BC_MODEL_IS_MINDMAP_OMNI
_Static_assert(BC_MINDMAP_OMNI_W_CONV3D_0_GN_KERNEL_OFF == 0,
	"first layer must start at offset zero");
_Static_assert(BC_MINDMAP_OMNI_W_CONV3D_1_GN_KERNEL_OFF ==
		BC_MINDMAP_OMNI_W_CONV3D_0_GN_KERNEL_LEN +
		BC_MINDMAP_OMNI_W_AFFINE_0_KERNEL_LEN + BC_MINDMAP_OMNI_W_AFFINE_0_BIAS_LEN,
	"mindmap_omni layer stride disagrees with the model.json manifest");
_Static_assert(BC_MINDMAP_OMNI_W_AFFINE_0_KERNEL_LEN == BC_MODEL_CHAN * BC_MODEL_CHAN,
	"affine is not a full CHAN x CHAN matrix");
_Static_assert(BC_MINDMAP_OMNI_W_CONV3D_FINAL_KERNEL_LEN ==
		BC_MODEL_CHAN * BC_MODEL_NCLASS,
	"classifier kernel size disagrees with the model.json manifest");
_Static_assert(BC_MINDMAP_OMNI_W_CONV3D_FINAL_BIAS_OFF +
		BC_MINDMAP_OMNI_W_CONV3D_FINAL_BIAS_LEN == BC_MODEL_WEIGHT_FLOATS,
	"weights do not end with the classifier bias");
_Static_assert(BC_MODEL_HAS_AFFINE == 1,
	"mindmap_omni is expected to carry a post-GroupNorm affine");
_Static_assert(BC_CHAN_STRIDE == 24, "mindmap_omni must stride activations by 24");
#endif

#if BC_MODEL_IS_MINDSNAP
_Static_assert(BC_MINDSNAP_W_CONV3D_0_GN_KERNEL_OFF == 0,
	"first layer must start at offset zero");
_Static_assert(BC_MINDSNAP_W_CONV3D_1_GN_KERNEL_OFF ==
		BC_MINDSNAP_W_CONV3D_0_GN_KERNEL_LEN +
		BC_MINDSNAP_W_AFFINE_0_KERNEL_LEN + BC_MINDSNAP_W_AFFINE_0_BIAS_LEN,
	"mindsnap layer stride disagrees with the model.json manifest");
_Static_assert(BC_MINDSNAP_W_AFFINE_0_KERNEL_LEN == BC_MODEL_CHAN * BC_MODEL_CHAN,
	"affine is not a full CHAN x CHAN matrix");
_Static_assert(BC_MINDSNAP_W_CONV3D_FINAL_KERNEL_LEN ==
		BC_MODEL_CHAN * BC_MODEL_NCLASS,
	"classifier kernel size disagrees with the model.json manifest");
_Static_assert(BC_MINDSNAP_W_CONV3D_FINAL_BIAS_OFF +
		BC_MINDSNAP_W_CONV3D_FINAL_BIAS_LEN == BC_MODEL_WEIGHT_FLOATS,
	"weights do not end with the classifier bias");
_Static_assert(BC_MODEL_HAS_AFFINE == 1,
	"mindsnap is expected to carry a post-GroupNorm affine");
_Static_assert(BC_CHAN_STRIDE == 24, "mindsnap must stride activations by 24");
#endif
