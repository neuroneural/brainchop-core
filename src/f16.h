#ifndef BRAINCHOPC_F16_H
#define BRAINCHOPC_F16_H

#include <stdint.h>

/*
 * IEEE half conversion, shared by every backend that stores activations or
 * weights as f16.
 *
 * SHARED ON PURPOSE. This began as a static function inside backend_webgpu.c
 * and was pulled out when the WebGL2 backend needed the identical conversion.
 * benchmarking records what happens otherwise: the GELU approximation was copied
 * rather than shared and silently diverged across four backends, so a MindGrab
 * build ran model16's function. A rounding rule that differs by backend is the
 * same defect one order of magnitude quieter -- it would not produce an error,
 * only a slightly wrong volume on one backend.
 */

/*
 * float -> IEEE half, by hand rather than via _Float16/__fp16.
 *
 * The conversion has to agree with what the GPU reads, and doing it in bits
 * keeps that independent of whether a given compiler exposes a half type for
 * this target. Round-to-nearest-even, with overflow saturating to infinity and
 * subnormals handled, matching the hardware conversion the shader would do.
 */
static inline uint16_t bc_f32_to_f16(float value) {
	union { float f; uint32_t u; } in = { value };
	uint32_t sign = (in.u >> 16) & 0x8000u;
	int32_t exponent = (int32_t)((in.u >> 23) & 0xffu) - 127 + 15;
	uint32_t mantissa = in.u & 0x7fffffu;

	if (((in.u >> 23) & 0xffu) == 0xffu)          /* inf or NaN */
		return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
	if (exponent >= 0x1f) return (uint16_t)(sign | 0x7c00u);   /* overflow */
	if (exponent <= 0) {
		if (exponent < -10) return (uint16_t)sign;             /* underflow */
		mantissa |= 0x800000u;
		uint32_t shift = (uint32_t)(14 - exponent);
		uint32_t half = mantissa >> shift;
		/*
		 * Round-to-nearest-EVEN, matching the normal path below and the GPU's
		 * own conversion. This used to round half-up, which disagreed with both.
		 *
		 * Written branchless deliberately. The obvious
		 * `if (rem > mid || (rem == mid && (half & 1)))` costs 2x on this loop
		 * -- not for the arithmetic, which is three integer ops on a branch
		 * taken 529 times in 16.7M, but because the short-circuit `||` becomes a
		 * second unpredictable branch. Measured over 16.7M values: half-up
		 * 20.3 ms, branching RTNE 39.5 ms, this 18.9 ms.
		 */
		uint32_t rem = mantissa & ((1u << shift) - 1u);
		uint32_t mid = 1u << (shift - 1);
		half += (uint32_t)((rem > mid) | ((rem == mid) & (half & 1u)));
		return (uint16_t)(sign | half);
	}
	uint16_t half = (uint16_t)(sign | ((uint32_t)exponent << 10) |
		(mantissa >> 13));
	if ((mantissa & 0x1fffu) > 0x1000u ||
	    ((mantissa & 0x1fffu) == 0x1000u && (half & 1u))) half++;
	return half;
}

/*
 * IEEE half -> float, exact in every case because f32 covers f16's whole range.
 *
 * Needed only to ask "what value did the GPU actually see", which is what
 * separates a plumbing bug from f16 rounding when a backend is compared against
 * the fp32 CPU reference. Handles subnormals by normalising them, which the
 * naive exponent-rebias does not.
 */
static inline float bc_f16_to_f32(uint16_t half) {
	uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
	uint32_t exponent = (half >> 10) & 0x1fu;
	uint32_t mantissa = half & 0x3ffu;
	union { uint32_t u; float f; } out;

	if (exponent == 0) {
		if (mantissa == 0) { out.u = sign; return out.f; }
		/* Subnormal: shift the mantissa up until its leading 1 becomes
		   implicit, decrementing the exponent to match. */
		exponent = 1;
		while (!(mantissa & 0x400u)) { mantissa <<= 1; exponent--; }
		mantissa &= 0x3ffu;
		out.u = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
		return out.f;
	}
	if (exponent == 0x1f) {                         /* inf or NaN */
		out.u = sign | 0x7f800000u | (mantissa << 13);
		return out.f;
	}
	out.u = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
	return out.f;
}

#endif
