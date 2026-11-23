#ifndef BRAINCHOPC_WEBGPU_KERNELS_H
#define BRAINCHOPC_WEBGPU_KERNELS_H

#include "chan_stride.h"
#include "model_specs.h"

/*
 * The WGSL compute kernels, written directly rather than generated.
 *
 * These mirror src/meshnet_cpu.c stage for stage -- conv_first, conv_hidden,
 * moments, norm, classify -- because that file is the reference the whole
 * project is gated against, and a reader who knows one should recognise the
 * other. The C is the specification; this is the same specification expressed
 * for a GPU with storage buffers.
 *
 * WHY NOT GENERATED. tinygrad can emit WGSL, and brainchop-test ships
 * tinygrad-exported runners for both models. Those are BEAM-tuned and will be
 * faster than this to begin with. They are not used here because generating
 * them requires tinygrad plus a Dawn shared library on the build machine, which
 * is a large build-time dependency for kernels this simple, and because the
 * result is opaque: 195 KB of machine-written WGSL that cannot be read against
 * the C. The tinygrad runners remain useful as an independent correctness
 * oracle and as the performance target.
 *
 * RELATION TO tfjs. tfjs has TWO Conv3D implementations. Its WebGL one is a
 * fragment shader over packed textures (`Conv3DProgram` in conv_gpu.js), and
 * almost all of that code exists because WebGL has neither compute shaders nor
 * storage buffers -- ivec5 coordinate unpacking, texture layout. None of it is
 * needed here. Its WebGPU one is `conv3d_naive_webgpu.ts` (Apache-2.0, Google;
 * also vendored by onnxruntime-web), and the convolution below deliberately
 * follows its structure: workgroup size 64, bounds-checked padding, and
 * accumulation as vec4 dot products over the input channels. See
 * THIRD_PARTY_NOTICES.md.
 *
 * Neither project in this tree installs `tfjs-backend-webgpu` -- brainchop and
 * BrowserQC both run the WebGL backend -- which is why the WebGPU kernel is
 * easy to miss. It exists.
 *
 * What differs is specialisation. The tfjs kernel is general: arbitrary
 * channels, strides, dilations and batch, so it carries uniform-driven index
 * helpers and a remainder path for input-channel counts that are not a multiple
 * of four. Here the shape is fixed and the channel stride is exactly 16 = 4
 * vec4s, so the remainder path cannot be reached and is omitted rather than
 * written and left dead.
 *
 * PRECISION. Storage is f16 and arithmetic f32, matching the Metal backend:
 * f16 halves the 1 GiB activation working set, while accumulating a 3x3x3x16
 * dot product or a 16.7M-element variance in f16 would lose too much. The CPU
 * reference accumulates GroupNorm moments in f64, which WGSL does not have, so
 * this path is tolerance-checked rather than bit-exact -- as the Metal path
 * already is.
 *
 * LAYOUT. Channel-last with a stride of BC_WG_CS, identical to the CPU engine,
 * so the 16 channels of one voxel are contiguous and a workgroup's reads
 * coalesce. Kernel weights keep meshnet_cpu_pack_kernel's order:
 * ((tap * in_channels) + ic) * CS + oc.
 */

#define BC_WG_STR2(x) #x
#define BC_WG_STR(x) BC_WG_STR2(x)

/* The packed channel stride, matching MESHNET_CPU_CHANNEL_STRIDE. */
#define BC_WG_CS BC_CHAN_STRIDE

/*
 * Activation storage precision. f16 by default, matching the Metal backend and
 * halving a 1 GiB working set. BC_WG_F32=1 switches to f32 and is a DIAGNOSTIC:
 * it separates error inherent to fp16 storage from an actual defect in these
 * kernels, which is otherwise guesswork. It doubles the device memory required.
 */
#ifndef BC_WG_F32
#define BC_WG_F32 0
#endif
#if BC_WG_F32
#define BC_WG_ENABLE_F16 ""
#define BC_WG_STORE "f32"
#else
#define BC_WG_ENABLE_F16 "enable f16;\n"
#define BC_WG_STORE "f16"
#endif

/* One invocation per voxel, 64 per workgroup along x. 256^3 / 64 workgroups
   dispatched as (4, 256, 256): every dimension stays under WebGPU's 65535
   per-dimension limit, which a flat 1-D dispatch of 262144 would not. */
#define BC_WG_LOCAL_X 64

/*
 * Voxels per invocation in the CONVOLUTION only (the other kernels stay at one).
 *
 * This is the register-blocking factor, and it is the single thing that decides
 * whether the convolution is FMA-bound or load-bound. A thread computing one
 * voxel must read all 27x16x16 = 6912 weights to produce its 16 outputs -- one
 * weight load per multiply-add, so the kernel issues as many loads as FMAs and
 * runs at load-issue rate no matter how fast the ALU is. Computing VOX voxels
 * per thread reads those weights ONCE and reuses them across all VOX, dividing
 * weight traffic by VOX while the FMA count per voxel stays the same.
 *
 * The reasoning is MN_DEFINE_CONV's, transplanted: benchmarking records MN_BLOCK=4
 * as what keeps the CPU kernel at 98% of its FMA roofline, for exactly this
 * reason. The GPU's register budget per thread is tighter, so the optimum here
 * is 2, not 4.
 *
 * MEASURED on an M4 Pro at 256^3, GPU time for the whole run, BOTH models --
 * the optimum is 2 for each, so the shared constant is measured rather than
 * inherited:
 *   model16   VOX=1 1.047s  VOX=2 0.906s  VOX=4 1.169s
 *   mindgrab  VOX=1 1.912s  VOX=2 1.635s  VOX=4 2.107s
 * 4 is past the cliff -- 16 vec4 accumulators plus the live weight quad spill,
 * and the spill costs more than the halved weight traffic saves. Do not raise
 * it without re-measuring.
 */
#ifndef BC_WG_VOX
#define BC_WG_VOX 2
#endif

/*
 * Injected into the WGSL as real constants so each build's kernels are
 * specialised exactly as the C is. WGSL has no preprocessor; this is the C one
 * doing the substitution before the string ever reaches the driver.
 */
#define BC_WGSL_PREAMBLE \
	BC_WG_ENABLE_F16 \
	"const DIM : i32 = 256;\n" \
	"const CS : u32 = " BC_WG_STR(BC_WG_CS) "u;\n" \
	"const CHAN : u32 = " BC_WG_STR(BC_MODEL_CHAN) "u;\n" \
	"const NCLASS : u32 = " BC_WG_STR(BC_MODEL_NCLASS) "u;\n" \
	"const HAS_AFFINE : bool = " \
		BC_WG_STR(BC_MODEL_HAS_AFFINE) " != 0;\n" \
	"const CLASSIFIER_BIAS : bool = " \
		BC_WG_STR(BC_MODEL_CLASSIFIER_BIAS) " != 0;\n" \
	"const EPS : f32 = 1e-5;\n"

/*
 * GELU. Two different approximations, exactly as in meshnet_cpu.c, selected by
 * the descriptor. They are not interchangeable: each is what its upstream
 * reference computes.
 */
#if BC_MODEL_GELU_TANH_RATIONAL
#define BC_WGSL_GELU \
	"fn mn_tanh(x_in : f32) -> f32 {\n" \
	"  var x = clamp(x_in, -9.0, 9.0);\n" \
	"  let u = x * x;\n" \
	"  var p = -8.29118133e-14;\n" \
	"  p = p * u + 5.19263868e-11;\n" \
	"  p = p * u - 2.00294448e-08;\n" \
	"  p = p * u + 1.11017944e-05;\n" \
	"  p = p * u + 0.00309865153;\n" \
	"  p = p * u + 0.130791619;\n" \
	"  p = p * u + 0.99999994;\n" \
	"  var q = 0.000253859733;\n" \
	"  q = q * u + 0.024473751;\n" \
	"  q = q * u + 0.464124829;\n" \
	"  q = q * u + 1.0;\n" \
	"  return x * p / q;\n" \
	"}\n" \
	"fn gelu(x : f32) -> f32 {\n" \
	"  let u = 0.797884583 * (x + 0.044715 * (x * x * x));\n" \
	"  return (0.5 * x) * (1.0 + mn_tanh(u));\n" \
	"}\n"
#else
/*
 * exp2 form. This reproduces the CPU's mn_fast_exp2 polynomial rather than
 * calling WGSL's exp2 builtin.
 *
 * The builtin is the obvious choice -- the C only hand-rolls this because
 * libm's exp2f blocked autovectorisation -- but the two are not the same
 * function to the last bit, and the difference compounds over 13 layers times
 * 16.7M voxels times 16 channels. Using the builtin measured 1223 differing
 * voxels against the CPU reference; MindGrab, whose GELU IS bit-identical to
 * the C, sat comfortably inside its budget at the same time. Matching the
 * polynomial exactly removes that as a variable.
 */
#define BC_WGSL_GELU \
	"fn mn_fast_exp2(x_in : f32) -> f32 {\n" \
	"  var x = x_in;\n" \
	"  if (!(x > -126.0)) { x = -126.0; }\n" \
	"  if (x > 126.0) { x = 126.0; }\n" \
	"  let f = floor(x);\n" \
	"  let r = x - f;\n" \
	"  var p = 0.000216128448;\n" \
	"  p = p * r + 0.00124678648;\n" \
	"  p = p * r + 0.0096754498;\n" \
	"  p = p * r + 0.0554852814;\n" \
	"  p = p * r + 0.240229305;\n" \
	"  p = p * r + 0.693147044;\n" \
	"  p = p * r + 1.0;\n" \
	"  let scale = bitcast<f32>(u32((i32(f) + 127) << 23u));\n" \
	"  return p * scale;\n" \
	"}\n" \
	"fn gelu(x : f32) -> f32 {\n" \
	"  let u = x + 0.044715 * x * x * x;\n" \
	"  return x / (1.0 + mn_fast_exp2(-2.302208198144325 * u));\n" \
	"}\n"
#endif

/*
 * Bindings, shared by every kernel so one bind-group layout serves all of them:
 *   0 src         f16 activations, or the normalised input for conv_first
 *   1 dst         f16 activations, or the label output for classify
 *   2 weights     f16 kernel/affine/classifier weights for this layer
 *   3 params      dilation, weight offsets, and the moment scratch stride
 *   4 moments     f32 per-channel mean and inverse stddev, and the partials
 *   6 wts4        binding 2's buffer again, typed as vec4
 *   7 src4        binding 0's buffer again, typed as vec4
 *
 * Binding 6 is the SAME buffer as binding 2, bound a second time with a vector
 * element type. WGSL cannot load four adjacent f16 out of an array<f16> in one
 * instruction, and the convolution's inner loop reads the 16 output-channel
 * weights of one tap as four contiguous quads -- so the scalar type costs 4x
 * the load instructions for identical bytes. Aliasing is legal here because
 * both views are read-only; WebGPU only forbids overlap when one side is
 * writable, which is also why `dst` is not aliased the same way.
 */
#define BC_WGSL_BINDINGS \
	"struct Params {\n" \
	"  dilation : i32,\n" \
	"  weight_off : u32,\n" \
	"  affine_off : u32,\n" \
	"  bias_off : u32,\n" \
	"  in_channels : u32,\n" \
	"  in_stride : u32,\n" \
	"  partial_groups : u32,\n" \
	"};\n" \
	"@group(0) @binding(0) var<storage, read> src : array<" BC_WG_STORE ">;\n" \
	"@group(0) @binding(1) var<storage, read_write> dst : array<" BC_WG_STORE ">;\n" \
	"@group(0) @binding(2) var<storage, read> wts : array<" BC_WG_STORE ">;\n" \
	"@group(0) @binding(3) var<uniform> params : Params;\n" \
	"@group(0) @binding(4) var<storage, read_write> mom : array<f32>;\n" \
	"@group(0) @binding(6) var<storage, read> wts4 : array<vec4<" BC_WG_STORE ">>;\n" \
	"@group(0) @binding(7) var<storage, read> src4 : array<vec4<" BC_WG_STORE ">>;\n"

/*
 * Convolution, 3x3x3, `same` zero padding, dilated.
 *
 * Accumulation is vec4 dot products over the input channels, following tfjs's
 * conv3d_naive_webgpu. Because CS is 16 the four-wide loop divides exactly, so
 * unlike the general kernel there is no 1-3 channel remainder to handle.
 *
 * One invocation computes all CS output channels of BC_WG_VOX consecutive
 * voxels, holding them in registers across the 27 taps. This differs from tfjs,
 * which computes one output channel per invocation: that is the right call for
 * a general kernel, but here it would re-read the same 27x16 input window once
 * per output channel. Holding the accumulators live reads each input value once
 * and reuses it across all 16 outputs -- the same reasoning as MN_DEFINE_CONV,
 * and 16.7M voxels leave no shortage of parallelism either way.
 *
 * TWO THINGS THIS KERNEL GETS RIGHT THAT THE OBVIOUS VERSION DOES NOT, both
 * measured, both worth roughly a factor of two together:
 *
 * 1. It vectorises over the OUTPUT channel, not the input channel. The packed
 *    layout is ((tap * in) + ic) * CS + oc, so `oc` is the contiguous axis and
 *    `ic` is strided by CS. Building the vec4 across `ic` -- which reads
 *    naturally, and is what tfjs does because its layout is transposed -- turns
 *    every quad into four strided scalar loads. Across `oc` it is one vector
 *    load of adjacent memory, via the wts4 alias.
 *
 * 2. The accumulators are named vec4s, not `array<f32, 16>` indexed by the loop
 *    variable. A private array under a dynamic index is not guaranteed to live
 *    in registers, and in practice does not: it lands in scratch memory and
 *    every accumulate becomes a round trip. Naming them forces the issue.
 *
 * in_channels and in_stride come from params rather than specialisation
 * constants so the first layer and the hidden layers share one pipeline. The
 * stride matters: the input volume has ONE channel per voxel, and indexing it
 * with the 16-wide activation stride would demand a 512 MiB buffer to hold
 * 32 MiB of data. The bounds test mirrors the CPU's zero padding.
 */
/*
 * Per-voxel fragments of the kernel. `v` is a decimal-digit STRING, so the
 * accumulators come out as distinct identifiers (a00..a33) rather than array
 * slots -- see point 2 above. The C preprocessor is doing the unrolling WGSL
 * has no way to request.
 *
 * The tap's contribution is masked with select() over a clamped index rather
 * than skipped with an `if`. The voxels of one thread straddle the volume edge
 * at different taps, so a branch here would diverge within the block; a clamped
 * read plus a zero select is uniform and reproduces the CPU's zero padding
 * exactly, because a masked lane contributes 0.0 to the FMA.
 */
/*
 * The accumulator fan-out. One vec4 per four output channels, so the count is
 * CS/4 and is NOT four: a 24-channel model needs six. Each entry carries its
 * index and its base output offset, and the three list macros differ only in
 * how many extra parameters the body needs (none, a voxel, a voxel plus a
 * component). Everything below is written against these rather than against a
 * literal a0..a3, which is what a fifth accumulator used to have no way to
 * reach -- the kernel silently computed the first 16 of 24 channels.
 */
#if BC_WG_CS == 16
#define BC_WG_ACC_N(M)        M("0","0")  M("1","4")  M("2","8")  M("3","12")
#define BC_WG_ACC_NV(M, v)    M("0","0",v) M("1","4",v) M("2","8",v) M("3","12",v)
#define BC_WG_ACC_NVC(M, v, c) \
	M("0",v,c) M("1",v,c) M("2",v,c) M("3",v,c)
#elif BC_WG_CS == 24
#define BC_WG_ACC_N(M)        M("0","0")  M("1","4")  M("2","8")  M("3","12") \
                              M("4","16") M("5","20")
#define BC_WG_ACC_NV(M, v)    M("0","0",v) M("1","4",v) M("2","8",v) M("3","12",v) \
                              M("4","16",v) M("5","20",v)
#define BC_WG_ACC_NVC(M, v, c) \
	M("0",v,c) M("1",v,c) M("2",v,c) M("3",v,c) M("4",v,c) M("5",v,c)
#else
#error "BC_WG_CS must be 16 or 24; add an accumulator list for a new width"
#endif

#define BC_WG_ACCW(n, b) \
	"          let w" n " = vec4<f32>(wts4[wq + " n "u]);\n"
#define BC_WG_ACCDECL(n, b, v) "  var a" v n " = vec4<f32>(0.0);\n"
#define BC_WG_VOXDECL(v) BC_WG_ACC_NV(BC_WG_ACCDECL, v)

/*
 * The source quad for voxel `v` at the current tap and input-channel group.
 *
 * Reading the input channels of one voxel as vec4s rather than scalars cuts the
 * source load count fourfold, which matters once the weight loads have been
 * amortised by BC_WG_VOX: at that point the two are the same order and
 * shrinking only one of them stops helping.
 */
#define BC_WG_VOXQUAD(v) \
	"        { let sx = x0 + " v " + dxd;\n" \
	"          let ok = sx >= 0 && sx < DIM;\n" \
	"          let q = vec4<f32>(src4[(plane + u32(clamp(sx, 0, DIM - 1))) *\n" \
	"                                 st4 + ic4]);\n" \
	"          s" v " = select(vec4<f32>(0.0), q, ok); }\n"

#define BC_WG_VOXDECLQ(v)  "  var s" v " = vec4<f32>(0.0);\n"

/* One input channel of the quad (`c` is .x/.y/.z/.w) into voxel v's outputs. */
#define BC_WG_ACCFMA4(n, v, c) \
	"          a" v n " = fma(w" n ", vec4<f32>(s" v "." c "), a" v n ");\n"
#define BC_WG_VOXFMA4(v, c) BC_WG_ACC_NVC(BC_WG_ACCFMA4, v, c)

/* Layer 0 only: a single input channel, so there is no quad to read. */
#define BC_WG_ACCFMA1(n, b, v) \
	"          a" v n " = fma(w" n ", s, a" v n ");\n"
#define BC_WG_VOXFMA1(v) \
	"        { let sx = x0 + " v " + dxd;\n" \
	"          let sv = f32(src[plane + u32(clamp(sx, 0, DIM - 1))]);\n" \
	"          let s = vec4<f32>(select(0.0, sv, sx >= 0 && sx < DIM));\n" \
	BC_WG_ACC_NV(BC_WG_ACCFMA1, v) \
	"        }\n"

/* Offsets are written as base + 0..3 rather than 24 literals; WGSL folds them. */
#define BC_WG_ACCSTORE(n, b, v) \
	"    dst[o + " b "u + 0u] = " BC_WG_STORE "(a" v n ".x);\n" \
	"    dst[o + " b "u + 1u] = " BC_WG_STORE "(a" v n ".y);\n" \
	"    dst[o + " b "u + 2u] = " BC_WG_STORE "(a" v n ".z);\n" \
	"    dst[o + " b "u + 3u] = " BC_WG_STORE "(a" v n ".w);\n"
#define BC_WG_VOXSTORE(v) \
	"  { let o = (obase + " v "u) * CS;\n" \
	BC_WG_ACC_NV(BC_WG_ACCSTORE, v) \
	"  }\n"

#define BC_WG_PERVOX1(M) M("0")
#define BC_WG_PERVOX2(M) M("0") M("1")
#define BC_WG_PERVOX4(M) M("0") M("1") M("2") M("3")
#define BC_WG_PERVOXC1(M, c) M("0", c)
#define BC_WG_PERVOXC2(M, c) M("0", c) M("1", c)
#define BC_WG_PERVOXC4(M, c) M("0", c) M("1", c) M("2", c) M("3", c)
#if BC_WG_VOX != 1 && BC_WG_VOX != 2 && BC_WG_VOX != 4
#error "BC_WG_VOX must be 1, 2 or 4"
#endif
/* Two levels so VOX expands before pasting; it must be a bare literal. */
#define BC_WG_CAT2(a, b) a##b
#define BC_WG_CAT(a, b) BC_WG_CAT2(a, b)
#define BC_WG_PV(VOX) BC_WG_CAT(BC_WG_PERVOX, VOX)
#define BC_WG_PVC(VOX) BC_WG_CAT(BC_WG_PERVOXC, VOX)

/*
 * One input-channel quad: read it for every voxel, then walk its four channels,
 * loading that channel's 16 output weights as four contiguous vec4s and
 * accumulating into every voxel. The weight quad is loaded ONCE and consumed by
 * all BC_WG_VOX voxels -- that reuse is the entire point of the blocking.
 */
#define BC_WG_QUADBODY(VOX) \
	BC_WG_PV(VOX)(BC_WG_VOXQUAD) \
	"        { let wq = wq0 + ((tap * cin) + ic4 * 4u) * (CS / 4u);\n" \
	BC_WG_ACC_N(BC_WG_ACCW) \
	BC_WG_PVC(VOX)(BC_WG_VOXFMA4, "x") \
	"        }\n" \
	"        { let wq = wq0 + ((tap * cin) + ic4 * 4u + 1u) * (CS / 4u);\n" \
	BC_WG_ACC_N(BC_WG_ACCW) \
	BC_WG_PVC(VOX)(BC_WG_VOXFMA4, "y") \
	"        }\n" \
	"        { let wq = wq0 + ((tap * cin) + ic4 * 4u + 2u) * (CS / 4u);\n" \
	BC_WG_ACC_N(BC_WG_ACCW) \
	BC_WG_PVC(VOX)(BC_WG_VOXFMA4, "z") \
	"        }\n" \
	"        { let wq = wq0 + ((tap * cin) + ic4 * 4u + 3u) * (CS / 4u);\n" \
	BC_WG_ACC_N(BC_WG_ACCW) \
	BC_WG_PVC(VOX)(BC_WG_VOXFMA4, "w") \
	"        }\n"

/*
 * No x bounds test on the output. DIM is 256 and BC_WG_LOCAL_X * BC_WG_VOX
 * divides it exactly, so the dispatch covers the row with no remainder; the
 * static assert in backend_webgpu.c holds anyone who changes either constant to
 * that promise rather than letting a partial block write out of range.
 */
#define BC_WGSL_CONV_SHAPE(LX, VOX) \
	"@compute @workgroup_size(" BC_WG_STR(LX) ", 1, 1)\n" \
	"fn conv(@builtin(global_invocation_id) gid : vec3<u32>) {\n" \
	"  let y = i32(gid.y);\n" \
	"  let z = i32(gid.z);\n" \
	"  if (y >= DIM || z >= DIM) { return; }\n" \
	"  let x0 = i32(gid.x) * " BC_WG_STR(VOX) ";\n" \
	"  let d = params.dilation;\n" \
	"  let cin = params.in_channels;\n" \
	"  let st4 = params.in_stride / 4u;\n" \
	"  let cin4 = cin / 4u;\n" \
	"  let wq0 = params.weight_off / 4u;\n" \
	BC_WG_PV(VOX)(BC_WG_VOXDECL) \
	BC_WG_PV(VOX)(BC_WG_VOXDECLQ) \
	"  var tap : u32 = 0u;\n" \
	"  for (var dz : i32 = -1; dz <= 1; dz = dz + 1) {\n" \
	"  for (var dy : i32 = -1; dy <= 1; dy = dy + 1) {\n" \
	"    let sz = z + dz * d;\n" \
	"    let sy = y + dy * d;\n" \
	"    let okyz = sz >= 0 && sz < DIM && sy >= 0 && sy < DIM;\n" \
	"    let plane = (u32(clamp(sz, 0, DIM - 1)) * u32(DIM) +\n" \
	"                 u32(clamp(sy, 0, DIM - 1))) * u32(DIM);\n" \
	"  for (var dx : i32 = -1; dx <= 1; dx = dx + 1) {\n" \
	"    let dxd = dx * d;\n" \
	"    if (okyz) {\n" \
	"      if (cin == 1u) {\n" \
	"        let wq = wq0 + tap * (CS / 4u);\n" \
	BC_WG_ACC_N(BC_WG_ACCW) \
	BC_WG_PV(VOX)(BC_WG_VOXFMA1) \
	"      } else {\n" \
	"      for (var ic4 : u32 = 0u; ic4 < cin4; ic4 = ic4 + 1u) {\n" \
	BC_WG_QUADBODY(VOX) \
	"      }\n" \
	"      }\n" \
	"    }\n" \
	"    tap = tap + 1u;\n" \
	"  }}}\n" \
	"  let obase = (u32(z) * u32(DIM) + u32(y)) * u32(DIM) + u32(x0);\n" \
	BC_WG_PV(VOX)(BC_WG_VOXSTORE) \
	"}\n"
#define BC_WGSL_CONV BC_WGSL_CONV_SHAPE(BC_WG_LOCAL_X, BC_WG_VOX)

/* AMD conv launch shape (WGSL/WebGPU). Bit-identical to 64xVOX: a
   voxel's sums keep their order. Moments stays at BC_WG_LOCAL_X: wider reorders
   GroupNorm and needs 48 KiB of workgroup memory at CS 24 (default 16 KiB). */
#define BC_WG_WIDE_X 256

/*
 * GroupNorm moments, stage 1: per-workgroup partial sums of x and x*x for every
 * channel, accumulated in f32.
 *
 * Each invocation owns its OWN slot in workgroup memory -- lid * CS + channel --
 * and one invocation per channel then folds the 64 lanes. Having every lane
 * accumulate into a single shared wg_sum[c] is a data race: WGSL has no
 * floating-point atomics, so the reads and writes simply interleave and the
 * variance comes out wrong, which shows up as a plausible-looking but badly
 * over-inclusive segmentation rather than an error.
 *
 * Reduces `dst` -- the buffer the convolution just wrote -- not `src`. Reading
 * `src` here computes the statistics of the previous layer's input, and for
 * layer 0 indexes a one-channel buffer with the 16-wide activation stride,
 * which reads out of bounds as zero and silently produces an all-zero volume.
 *
 * The CPU reduces in a fixed slice order so the result does not depend on the
 * thread count. This cannot reproduce that order, which is the main reason the
 * WebGPU path is tolerance-checked rather than bit-exact -- the same trade the
 * Metal backend already makes.
 */
/* The workgroup arrays are LOCAL_X * CS, not a literal 16: the slot index is
   lid * CS, so a wider model indexes straight past the end of a 16-wide one. */
#define BC_WGSL_MOMENTS \
	"var<workgroup> wg_sum : array<f32, " BC_WG_STR(BC_WG_LOCAL_X) " * " \
		BC_WG_STR(BC_WG_CS) ">;\n" \
	"var<workgroup> wg_sq : array<f32, " BC_WG_STR(BC_WG_LOCAL_X) " * " \
		BC_WG_STR(BC_WG_CS) ">;\n" \
	"@compute @workgroup_size(" BC_WG_STR(BC_WG_LOCAL_X) ", 1, 1)\n" \
	"fn moments(@builtin(global_invocation_id) gid : vec3<u32>,\n" \
	"           @builtin(local_invocation_index) lid : u32,\n" \
	"           @builtin(workgroup_id) wid : vec3<u32>,\n" \
	"           @builtin(num_workgroups) nwg : vec3<u32>) {\n" \
	"  let slot = lid * CS;\n" \
	"  for (var c : u32 = 0u; c < CS; c = c + 1u) {\n" \
	"    wg_sum[slot + c] = 0.0;\n" \
	"    wg_sq[slot + c] = 0.0;\n" \
	"  }\n" \
	"  let x = gid.x; let y = gid.y; let z = gid.z;\n" \
	"  if (x < u32(DIM) && y < u32(DIM) && z < u32(DIM)) {\n" \
	"    let base = (z * u32(DIM) * u32(DIM) + y * u32(DIM) + x) * CS;\n" \
	"    for (var c : u32 = 0u; c < CS; c = c + 1u) {\n" \
	"      let v = f32(dst[base + c]);\n" \
	"      wg_sum[slot + c] = v;\n" \
	"      wg_sq[slot + c] = v * v;\n" \
	"    }\n" \
	"  }\n" \
	"  workgroupBarrier();\n" \
	"  if (lid < CS) {\n" \
	"    var s : f32 = 0.0;\n" \
	"    var q : f32 = 0.0;\n" \
	"    for (var l : u32 = 0u; l < " BC_WG_STR(BC_WG_LOCAL_X) "u; l = l + 1u) {\n" \
	"      s = s + wg_sum[l * CS + lid];\n" \
	"      q = q + wg_sq[l * CS + lid];\n" \
	"    }\n" \
	"    let g = wid.z * nwg.y * nwg.x + wid.y * nwg.x + wid.x;\n" \
	"    mom[(g * CS + lid) * 2u] = s;\n" \
	"    mom[(g * CS + lid) * 2u + 1u] = q;\n" \
	"  }\n" \
	"}\n"

/*
 * Stage 2: fold the partials into a mean and inverse stddev per channel.
 *
 * HIERARCHICAL, not a single serial loop, and that is the point. This folds
 * 262144 partials per channel; summed one after another in f32 the running
 * total reaches order 10^7 while the addends are order 1, past where a 24-bit
 * mantissa can resolve them, and the tail is quietly discarded. The CPU
 * reference sidesteps this by accumulating in f64, which WGSL does not have.
 *
 * Kahan compensation was tried first and changed the result by exactly zero
 * voxels -- the WGSL compiler is free to reassociate `(t - s) - v` and evidently
 * does. Splitting the sum into independent stripes and combining them cannot
 * be optimised away, because the stripes are genuinely separate accumulators:
 * it shortens each dependent chain and so cuts the accumulated error without
 * relying on the compiler preserving an algebraic identity.
 *
 * STRIPES is 256 / CS, NOT a literal 16, because the 256-thread workgroup is
 * tiled as stripe-by-channel and `fs` holds exactly 256. At CS 16 that is the
 * same 16 stripes as before. At CS 24 a literal 16 would read fs[15 * 24 + 23]
 * -- index 383 of a 256-entry array -- and leave five stripes never written.
 */
#define BC_WGSL_MOMENTS_FINISH \
	"const STRIPES : u32 = 256u / CS;\n" \
	"var<workgroup> fs : array<f32, 256>;\n" \
	"var<workgroup> fq : array<f32, 256>;\n" \
	"@compute @workgroup_size(256, 1, 1)\n" \
	"fn moments_finish(@builtin(local_invocation_index) lid : u32) {\n" \
	"  let channel = lid % CS;\n" \
	"  let stripe = lid / CS;\n" \
	"  let groups = params.partial_groups;\n" \
	"  var s : f32 = 0.0;\n" \
	"  var q : f32 = 0.0;\n" \
	"  if (stripe < STRIPES) {\n" \
	"    var g : u32 = stripe;\n" \
	"    loop {\n" \
	"      if (g >= groups) { break; }\n" \
	"      s = s + mom[(g * CS + channel) * 2u];\n" \
	"      q = q + mom[(g * CS + channel) * 2u + 1u];\n" \
	"      g = g + STRIPES;\n" \
	"    }\n" \
	"  }\n" \
	"  fs[lid] = s;\n" \
	"  fq[lid] = q;\n" \
	"  workgroupBarrier();\n" \
	"  if (lid < CS) {\n" \
	"    var ts : f32 = 0.0;\n" \
	"    var tq : f32 = 0.0;\n" \
	"    for (var k : u32 = 0u; k < STRIPES; k = k + 1u) {\n" \
	"      ts = ts + fs[k * CS + lid];\n" \
	"      tq = tq + fq[k * CS + lid];\n" \
	"    }\n" \
	"    let n = f32(DIM) * f32(DIM) * f32(DIM);\n" \
	"    let mean = ts / n;\n" \
	"    let varr = max(tq / n - mean * mean, 0.0);\n" \
	"    let base = groups * CS * 2u;\n" \
	"    mom[base + lid * 2u] = mean;\n" \
	"    mom[base + lid * 2u + 1u] = inverseSqrt(varr + EPS);\n" \
	"  }\n" \
	"}\n"

/*
 * Normalise, apply the optional per-channel affine, activate. In place, exactly
 * as mn_norm is.
 *
 * The channel loop runs to the literal CS, never to CHAN. benchmarking records
 * that bounding it by the logical channel count made Clang emit the CPU version
 * entirely scalar; the pad lanes are inert because the packed kernel zeroes
 * them, so the extra work is free and the loop stays uniform.
 */
#define BC_WGSL_NORM \
	"@compute @workgroup_size(" BC_WG_STR(BC_WG_LOCAL_X) ", 1, 1)\n" \
	"fn norm(@builtin(global_invocation_id) gid : vec3<u32>) {\n" \
	"  let x = gid.x; let y = gid.y; let z = gid.z;\n" \
	"  if (x >= u32(DIM) || y >= u32(DIM) || z >= u32(DIM)) { return; }\n" \
	"  let base = (z * u32(DIM) * u32(DIM) + y * u32(DIM) + x) * CS;\n" \
	"  let mbase = params.partial_groups * CS * 2u;\n" \
	"  for (var c : u32 = 0u; c < CS; c = c + 1u) {\n" \
	"    let mean = mom[mbase + c * 2u];\n" \
	"    let inv = mom[mbase + c * 2u + 1u];\n" \
	"    var v = (f32(dst[base + c]) - mean) * inv;\n" \
	"    if (HAS_AFFINE) {\n" \
	"      v = v * f32(wts[params.affine_off + c]) +\n" \
	"          f32(wts[params.bias_off + c]);\n" \
	"    }\n" \
	"    dst[base + c] = " BC_WG_STORE "(gelu(v));\n" \
	"  }\n" \
	"}\n"

/*
 * Classifier: a 1x1 convolution followed by argmax, writing one label per
 * voxel as u32.
 *
 * A tie leaves the lower class index, which is what the CPU's strict `>` does
 * and is why MindGrab's two-class case does not need a special path here
 * either. Labels are u32 rather than packed bytes because WGSL storage buffers
 * address 4-byte words; the host narrows them on readback.
 */
#define BC_WGSL_CLASSIFY \
	"@group(0) @binding(5) var<storage, read_write> labels : array<u32>;\n" \
	"@compute @workgroup_size(" BC_WG_STR(BC_WG_LOCAL_X) ", 1, 1)\n" \
	"fn classify(@builtin(global_invocation_id) gid : vec3<u32>) {\n" \
	"  let x = gid.x; let y = gid.y; let z = gid.z;\n" \
	"  if (x >= u32(DIM) || y >= u32(DIM) || z >= u32(DIM)) { return; }\n" \
	"  let i = z * u32(DIM) * u32(DIM) + y * u32(DIM) + x;\n" \
	"  let base = i * CS;\n" \
	"  var best : u32 = 0u;\n" \
	"  var best_v : f32 = -3.0e38;\n" \
	"  for (var k : u32 = 0u; k < NCLASS; k = k + 1u) {\n" \
	"    var acc : f32 = 0.0;\n" \
	"    if (CLASSIFIER_BIAS) { acc = f32(wts[params.bias_off + k]); }\n" \
	"    for (var c : u32 = 0u; c < CHAN; c = c + 1u) {\n" \
	"      acc = acc + f32(src[base + c]) *\n" \
	"            f32(wts[params.weight_off + c * NCLASS + k]);\n" \
	"    }\n" \
	"    if (acc > best_v) { best_v = acc; best = k; }\n" \
	"  }\n" \
	"  labels[i] = best;\n" \
	"}\n"

/*
 * The whole module as one source string. WebGPU compiles a shader module once
 * and exposes each entry point as its own pipeline, so there is no benefit to
 * splitting these and a real cost in duplicated declarations.
 */
#define BC_WGSL_SOURCE \
	BC_WGSL_PREAMBLE \
	BC_WGSL_BINDINGS \
	BC_WGSL_GELU \
	BC_WGSL_CONV \
	BC_WGSL_MOMENTS \
	BC_WGSL_MOMENTS_FINISH \
	BC_WGSL_NORM \
	BC_WGSL_CLASSIFY

/* A module of its own: Dawn compiles every entry point of a module, so a wide
   conv inside BC_WGSL_SOURCE would cost compile time where it is never used. */
#define BC_WGSL_WIDE_CONV_SOURCE \
	BC_WGSL_PREAMBLE \
	BC_WGSL_BINDINGS \
	BC_WGSL_CONV_SHAPE(BC_WG_WIDE_X, 1)

#ifdef __cplusplus
extern "C" {
#endif

extern const char *const bc_wgsl_source;
extern const char *const bc_wgsl_wide_conv_source;

#ifdef __cplusplus
}
#endif

#endif
