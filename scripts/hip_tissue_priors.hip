#include <hip/hip_runtime.h>
#include <math.h>

#include "pve_constants.h"

#ifndef BC_HIP_NVOX
#error "BC_HIP_NVOX must name the model voxel count"
#endif

/* Convert class-major f32 logits to the four CAT-lite input planes without
 * materializing probabilities. The class grouping and temperatures are shared
 * with the CPU, CUDA, WebGPU and WebGL2 implementations. */
extern "C" __global__ void bc_hip_tissue_priors(
		const float *__restrict__ logits, float *__restrict__ planes) {
	size_t voxel = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (voxel >= (size_t)BC_HIP_NVOX) return;
	float values[18], maximum = -INFINITY;
	for (int c = 0; c < 18; c++) {
		float value = logits[(size_t)c * BC_HIP_NVOX + voxel];
		values[c] = value;
		maximum = fmaxf(maximum, value);
	}
	const unsigned masks[3] = {
		BC_PRIOR_GM_MASK, BC_PRIOR_WM_MASK, BC_PRIOR_CSF_MASK
	};
	float denominator = 0.0f, support_denominator = 0.0f;
	float numerator[3] = {0.0f, 0.0f, 0.0f}, background = 0.0f;
	for (int c = 0; c < 18; c++) {
		float centered = values[c] - maximum;
		float weight = expf(centered / BC_PRIOR_TEMPERATURE);
		float support = expf(centered / BC_PRIOR_SUPPORT_TEMPERATURE);
		denominator += weight;
		support_denominator += support;
		if (c == 0) background = support;
		for (int tissue = 0; tissue < 3; tissue++)
			if (masks[tissue] & (1u << c)) numerator[tissue] += weight;
		if (BC_PRIOR_MIXED_GWM_MASK & (1u << c)) {
			float half = 0.5f * weight;
			numerator[0] += half;
			numerator[1] += half;
		}
	}
	for (int tissue = 0; tissue < 3; tissue++)
		planes[(size_t)tissue * BC_HIP_NVOX + voxel] =
			numerator[tissue] / denominator;
	planes[3 * (size_t)BC_HIP_NVOX + voxel] =
		fminf(1.0f, fmaxf(0.0f,
			1.0f - background / support_denominator));
}
