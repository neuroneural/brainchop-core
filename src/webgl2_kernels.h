#ifndef BRAINCHOPC_WEBGL2_KERNELS_H
#define BRAINCHOPC_WEBGL2_KERNELS_H

#include "chan_stride.h"
#include "model_specs.h"

/*
 * The GLSL ES 3.00 kernels for the WebGL2 fallback.
 *
 * These mirror src/meshnet_cpu.c stage for stage, exactly as
 * webgpu_kernels.h does, because that file is the reference the whole project
 * is gated against. Where a choice here differs from the WGSL, the reason is
 * always that WebGL2 has no compute shaders: everything is a fragment shader
 * rendering into a texture, with no storage buffers, no shared memory and no
 * atomics. Six programs rather than five, because GLSL has no entry points and
 * each is its own vertex+fragment pair; the moments reduction became three
 * passes rather than the WGSL's two.
 *
 * WHY NOT SHARED WITH THE WGSL. The two languages are close enough to tempt a
 * common source and far enough apart that it would be a lowest common
 * denominator of both: GLSL ES 3.00 has no fma(), no f16 arithmetic and no
 * storage buffers, while WGSL has no MRT, no fragment stage and no texelFetch
 * on a render target. The kernels are transliterated, not shared. What IS
 * shared is the packed weight layout, which comes from
 * meshnet_cpu_pack_kernel and is honoured identically by both backends.
 *
 * PRECISION. Storage is f16 (RGBA16F textures) and arithmetic f32 (`highp`),
 * matching the Metal and WebGPU backends. Note the shader does not know which:
 * texelFetch on a sampler3D yields a vec4 whatever the internal format is, so
 * the BC_WGL_F32 diagnostic is a C-side change of internalformat only and needs
 * no second shader. That is one place WebGL2 is simpler than WGSL, which had to
 * switch the declared element type.
 *
 * LAYOUT. Channel-last with a stride of BC_WGL_CS, identical to the CPU engine
 * and the WGSL. The 16 channels of one voxel are one RGBA texel in each of four
 * 3D textures, so a "vec4 over the output channel" is a single hardware fetch
 * rather than the aliased second binding the WGSL needed. Kernel weights keep
 * meshnet_cpu_pack_kernel's order: ((tap * in_channels) + ic) * CS + oc.
 */

#define BC_WGL_STR2(x) #x
#define BC_WGL_STR(x) BC_WGL_STR2(x)

/* The packed channel stride, matching MESHNET_CPU_CHANNEL_STRIDE and BC_WG_CS. */
#define BC_WGL_CS BC_CHAN_STRIDE

/*
 * Width, in RGBA texels, of the 2D texture holding every layer's weights.
 *
 * A power of two so the shader's row/column split is a shift and a mask rather
 * than an integer division, and small enough that the tallest model stays well
 * inside MAX_TEXTURE_SIZE: MindGrab packs 166352 floats = 41588 texels = 163
 * rows at this width. check_capabilities() checks that height against
 * MAX_TEXTURE_SIZE at run time -- it is the only dimension the backend derives
 * from the MODEL rather than from a constant, so it is the one that moves when
 * a third model is added.
 */
#define BC_WGL_WTEX_W 256
#define BC_WGL_WTEX_SHIFT 8
#define BC_WGL_WTEX_MASK 255

/*
 * Width of the RGBA8 label texture, the only buffer that leaves the GPU.
 *
 * Four consecutive x voxels per texel, so 256^3 voxels are 4194304 texels, and
 * 2048x2048 is the square that holds them -- comfortably inside the 8192 that
 * Firefox reports and the 16384 Chromium does. backend_webgl2.c asserts the
 * arithmetic rather than trusting this comment.
 */
#define BC_WGL_LAB_W 2048

/*
 * The vertex shader, shared by every program.
 *
 * A fullscreen triangle synthesised from gl_VertexID alone: no vertex buffer,
 * no attributes, no VAO contents. That is what keeps a full-volume pass at
 * three GL calls per z slice (bindFramebuffer, uniform, drawArrays): 256
 * slices x 3 = 768 for a full-volume pass at one voxel per fragment, half that
 * in the shipping VOX=2 build for the hidden convolutions it blocks, and an
 * empty pass measures 4.6 ms. That floor is the premise the per-slice design
 * rests on -- 256 separate draws are only viable because each is this cheap.
 *
 * Three vertices at (0,0), (2,0), (0,2) in unit space map to (-1,-1), (3,-1),
 * (-1,3) in clip space, which covers the viewport with one primitive and no
 * diagonal seam.
 */
#define BC_GLSL_VERTEX \
	"#version 300 es\n" \
	"void main() {\n" \
	"  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n" \
	"  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n" \
	"}\n"

/*
 * Injected as real constants so each build's kernels are specialised exactly as
 * its C is -- the same trick webgpu_kernels.h uses, except that here the C
 * preprocessor is doing it for a language that has its own preprocessor and
 * simply is not reachable from the host at run time.
 *
 * `precision highp` is mandatory, not defensive: GLSL ES has no default float
 * precision in the fragment stage, so omitting it is a compile error, and a
 * mediump fallback would silently compute the convolution at f16.
 */
#define BC_GLSL_PREAMBLE \
	"#version 300 es\n" \
	"precision highp float;\n" \
	"precision highp int;\n" \
	"precision highp sampler2D;\n" \
	"precision highp sampler3D;\n" \
	"const int DIM = " BC_WGL_STR(BRAINCHOP_MODEL_DIM) ";\n" \
	"const int CS = " BC_WGL_STR(BC_WGL_CS) ";\n" \
	"const int CHAN = " BC_WGL_STR(BC_MODEL_CHAN) ";\n" \
	"const int NCLASS = " BC_WGL_STR(BC_MODEL_NCLASS) ";\n" \
	"const float EPS = 1e-5;\n"

/*
 * Weight fetch. `q` is an index in RGBA texels, i.e. a float index divided by
 * four, which is exact because BC_WGL_CS is a multiple of 4 and every offset
 * the host passes is a multiple of CS.
 */
#define BC_GLSL_WFETCH \
	"uniform highp sampler2D wts;\n" \
	"vec4 wf(int q) {\n" \
	"  return texelFetch(wts, ivec2(q & " BC_WGL_STR(BC_WGL_WTEX_MASK) ",\n" \
	"                               q >> " BC_WGL_STR(BC_WGL_WTEX_SHIFT) "), 0);\n" \
	"}\n"

/*
 * The per-plane fan-out: one RGBA attachment per four channels, so the count is
 * CS/4 and is NOT four. Everything below is written against these lists rather
 * than a literal o0..o3 / a0..a3 / s0..s3, which is what a 24-channel model had
 * no way to reach: the shader compiled and framebuffer setup then failed on a
 * GL_NONE attachment.
 *
 * The strides are LITERALS because they are pasted into GLSL source; a
 * stringified `BC_WGL_CS * BC_WGL_CS / 4` would emit the expression text.
 *   BC_WGL_CS4 = CS/4, one input channel's vec4 stride, and conv_first's
 *                per-tap stride since it has a single input channel.
 *   BC_WGL_TAP = CS * CS/4, the hidden convolution's per-tap stride.
 */
#if BC_WGL_CS == 16
#define BC_WGL_PER_PLANE(M) M("0") M("1") M("2") M("3")
#define BC_WGL_PER_QUAD(M)  M("0","0") M("1","16") M("2","32") M("3","48")
#define BC_WGL_CS4 "4"
#define BC_WGL_TAP "64"
#elif BC_WGL_CS == 24
#define BC_WGL_PER_PLANE(M) M("0") M("1") M("2") M("3") M("4") M("5")
#define BC_WGL_PER_QUAD(M)  M("0","0") M("1","24") M("2","48") M("3","72") \
                            M("4","96") M("5","120")
#define BC_WGL_CS4 "6"
#define BC_WGL_TAP "144"
#else
#error "no GLSL plane list for this channel stride; add one beside chan_stride.h"
#endif

/* VOX=2 writes two z slices at once and so needs 2 * CS/4 attachments. Browsers
   report eight, so it exists only at CS 16; at 24 it would want twelve, and the
   backend's runtime probe already declines it. */
#define BC_WGL_HAS_VOX2 (BC_WGL_CS == 16)

#define BC_WGL_OUT(n)     "layout(location = " n ") out vec4 o" n ";\n"
#define BC_WGL_ACC(n)     "  vec4 a" n " = vec4(0.0);\n"
#define BC_WGL_STORE(n)   "  o" n " = a" n ";\n"
#define BC_WGL_FMA(n)     "    a" n " += wf(w + " n ") * v;\n"
#define BC_WGL_FMA_Q(n)   "      a" n " += wf(w + " n ") * v;\n"
#define BC_WGL_SAMPLER(n) "uniform highp sampler3D s" n ";\n"
#define BC_WGL_FETCH(n)   "    vec4 q" n " = texelFetch(s" n ", sc, 0) * m;\n"
#define BC_WGL_NORMG(n)   BC_WGL_NORM_GROUP("v" n, "s" n, n)
#define BC_WGL_NORMOUT(n) "  o" n " = g4(v" n ");\n"
#define BC_WGL_CLSFETCH(n) "    cv[" n "] = texelFetch(s" n ", p, 0);\n"

/*
 * Layer 0: a 3x3x3 dilated convolution from ONE input channel to CS outputs,
 * `same` zero padding.
 *
 * A separate program from the hidden convolution rather than a runtime branch
 * on the input channel count, which is what the WGSL does (`if (cin == 1u)`).
 * WGSL needs one pipeline because a bind-group layout is fixed at pipeline
 * creation; GLSL has no such constraint, and two programs mean neither carries
 * the other's dead loop. It also lets the input volume stay a one-channel R16F
 * texture -- indexing it at the 16-wide activation stride would demand 512 MiB
 * to hold 32 MiB of data.
 *
 * TAP ORDER is (dz, dy, dx) lexicographic, so `tap` increments exactly as
 * meshnet_cpu_pack_kernel's dst_tap = (dz+1)*9 + (dy+1)*3 + (dx+1). The CPU
 * reaches the same number as (tz*3 + ty)*3 + tx. Getting this wrong produces a
 * plausible-looking volume, not an error, so it is stated rather than implied.
 *
 * PADDING is a clamped fetch multiplied by a mask, not a branch. The CPU skips
 * out-of-range taps, which contributes nothing; multiplying by zero is the same
 * arithmetic and stays uniform across the quad. A branch would diverge at the
 * volume edge for no gain.
 *
 * The four accumulators are NAMED vec4s. webgpu_kernels.h records that an
 * `array<f32,16>` under a dynamic index lands in scratch memory rather than
 * registers; the same is true of GLSL, and the four MRT outputs make the naming
 * natural here anyway.
 */
#define BC_GLSL_CONV_FIRST \
	BC_GLSL_PREAMBLE \
	"uniform highp sampler3D src;\n" \
	BC_GLSL_WFETCH \
	"uniform int uZ;\n" \
	"uniform int uDil;\n" \
	"uniform int uWQ0;\n" \
	BC_WGL_PER_PLANE(BC_WGL_OUT) \
	"void main() {\n" \
	"  ivec3 p = ivec3(int(gl_FragCoord.x), int(gl_FragCoord.y), uZ);\n" \
	BC_WGL_PER_PLANE(BC_WGL_ACC) \
	"  int tap = 0;\n" \
	"  for (int dz = -1; dz <= 1; ++dz) {\n" \
	"  for (int dy = -1; dy <= 1; ++dy) {\n" \
	"  for (int dx = -1; dx <= 1; ++dx) {\n" \
	"    ivec3 s = p + ivec3(dx, dy, dz) * uDil;\n" \
	"    bool ok = all(greaterThanEqual(s, ivec3(0))) &&\n" \
	"              all(lessThan(s, ivec3(DIM)));\n" \
	"    float v = texelFetch(src, clamp(s, ivec3(0), ivec3(DIM - 1)), 0).r *\n" \
	"              (ok ? 1.0 : 0.0);\n" \
	"    int w = uWQ0 + tap * " BC_WGL_CS4 ";\n" \
	BC_WGL_PER_PLANE(BC_WGL_FMA) \
	"    ++tap;\n" \
	"  }}}\n" \
	BC_WGL_PER_PLANE(BC_WGL_STORE) \
	"}\n"

/*
 * Hidden layers: 3x3x3 dilated, CS inputs to CS outputs.
 *
 * The 16 input channels of a voxel arrive as four RGBA fetches and the 16
 * outputs leave through four attachments, so one fragment does what one WGSL
 * invocation does. What it CANNOT do is BC_WG_VOX register blocking: a fragment
 * owns exactly one texel per attachment, so the 27x16x16 weights are re-fetched
 * per voxel -- 1728 texel fetches, one per multiply-add. That is the dominant
 * cost, 91-93% of GPU time here, and it is issue-rate bound rather than a
 * locality problem: throughput is flat across dilation 1-31, measured on this
 * backend and not merely inherited from the CPU path's identical finding. It is
 * why THIS kernel runs about 3.2x slower per layer than the WebGPU one, which
 * keeps its register blocking; the shipping VOX=2 build closes about a third of
 * that, leaving the backend near 2x end to end.
 *
 * The input-channel walk is ascending, matching MN_DEFINE_CONV's
 * `for input_channel ... for channel`, so the accumulation order agrees with
 * the CPU as closely as a different data layout allows.
 */
#define BC_WGL_QUAD(n, off) \
	"  { for (int i = 0; i < 4; ++i) {\n" \
	"      float v = q" n "[i];\n" \
	"      int w = b + " off " + i * " BC_WGL_CS4 ";\n" \
	BC_WGL_PER_PLANE(BC_WGL_FMA_Q) \
	"    } }\n"

#define BC_GLSL_CONV_HIDDEN \
	BC_GLSL_PREAMBLE \
	BC_WGL_PER_PLANE(BC_WGL_SAMPLER) \
	BC_GLSL_WFETCH \
	"uniform int uZ;\n" \
	"uniform int uDil;\n" \
	"uniform int uWQ0;\n" \
	BC_WGL_PER_PLANE(BC_WGL_OUT) \
	"void main() {\n" \
	"  ivec3 p = ivec3(int(gl_FragCoord.x), int(gl_FragCoord.y), uZ);\n" \
	BC_WGL_PER_PLANE(BC_WGL_ACC) \
	"  int tap = 0;\n" \
	"  for (int dz = -1; dz <= 1; ++dz) {\n" \
	"  for (int dy = -1; dy <= 1; ++dy) {\n" \
	"  for (int dx = -1; dx <= 1; ++dx) {\n" \
	"    ivec3 s = p + ivec3(dx, dy, dz) * uDil;\n" \
	"    float m = (all(greaterThanEqual(s, ivec3(0))) &&\n" \
	"               all(lessThan(s, ivec3(DIM)))) ? 1.0 : 0.0;\n" \
	"    ivec3 sc = clamp(s, ivec3(0), ivec3(DIM - 1));\n" \
	BC_WGL_PER_PLANE(BC_WGL_FETCH) \
	"    int b = uWQ0 + tap * " BC_WGL_TAP ";\n" \
	BC_WGL_PER_QUAD(BC_WGL_QUAD) \
	"    ++tap;\n" \
	"  }}}\n" \
	BC_WGL_PER_PLANE(BC_WGL_STORE) \
	"}\n"

#if BC_WGL_HAS_VOX2
/*
 * The same convolution computing TWO z slices per fragment.
 *
 * This is the WebGL2 answer to BC_WG_VOX, and getting it to exist at all took
 * choosing the right axis. Blocking over x -- which is what the WGSL does --
 * would need two voxels' 16 channels in eight distinct attachments, and since
 * the four activation textures are already fully used by one voxel, it would
 * mean splitting the activation into even-x and odd-x halves. Every other
 * kernel would then need to know that, and the convolution's source fetch would
 * have to pick a texture by the parity of a runtime dilation.
 *
 * Blocking over z costs none of that. `glFramebufferTextureLayer` attaches an
 * IMAGE, not a texture, so layer z and layer z+1 of the same texture are two
 * different images and may both be attached to one framebuffer. Attachments 0-3
 * take layer z and 4-7 layer z+1. **The activation layout does not change at
 * all**, so moments, norm and classify are untouched.
 *
 * What it buys: the 1728 weight quads are fetched once and consumed by two
 * voxels, halving the weight traffic that dominates GPU time -- 91-93% at one
 * voxel per fragment, 87-89% once this halving is in.
 *
 * What it costs: eight vec4 accumulators plus eight source quads live at once,
 * against four and four. The WGSL found VOX=4 past the spill cliff for the same
 * reason, so this is measured rather than assumed.
 *
 * It needs eight draw buffers, and the GLES 3.0 floor is four. The backend
 * probes and falls back to BC_GLSL_CONV_HIDDEN, which is why both exist.
 */
#define BC_WGL_QUAD2(qa, qb, off) \
	"  { for (int i = 0; i < 4; ++i) {\n" \
	"      float v = " qa "[i];\n" \
	"      float u = " qb "[i];\n" \
	"      int w = b + " off " + i * 4;\n" \
	"      vec4 W0 = wf(w), W1 = wf(w + 1), W2 = wf(w + 2), W3 = wf(w + 3);\n" \
	"      a0 += W0 * v; a1 += W1 * v; a2 += W2 * v; a3 += W3 * v;\n" \
	"      b0 += W0 * u; b1 += W1 * u; b2 += W2 * u; b3 += W3 * u;\n" \
	"    } }\n"

#define BC_GLSL_CONV_HIDDEN2 \
	BC_GLSL_PREAMBLE \
	"uniform highp sampler3D s0, s1, s2, s3;\n" \
	BC_GLSL_WFETCH \
	"uniform int uZ;\n" \
	"uniform int uDil;\n" \
	"uniform int uWQ0;\n" \
	"layout(location = 0) out vec4 o0;\n" \
	"layout(location = 1) out vec4 o1;\n" \
	"layout(location = 2) out vec4 o2;\n" \
	"layout(location = 3) out vec4 o3;\n" \
	"layout(location = 4) out vec4 n0;\n" \
	"layout(location = 5) out vec4 n1;\n" \
	"layout(location = 6) out vec4 n2;\n" \
	"layout(location = 7) out vec4 n3;\n" \
	"void main() {\n" \
	"  int x = int(gl_FragCoord.x);\n" \
	"  int y = int(gl_FragCoord.y);\n" \
	"  vec4 a0 = vec4(0.0), a1 = vec4(0.0), a2 = vec4(0.0), a3 = vec4(0.0);\n" \
	"  vec4 b0 = vec4(0.0), b1 = vec4(0.0), b2 = vec4(0.0), b3 = vec4(0.0);\n" \
	"  int tap = 0;\n" \
	"  for (int dz = -1; dz <= 1; ++dz) {\n" \
	"  for (int dy = -1; dy <= 1; ++dy) {\n" \
	"  for (int dx = -1; dx <= 1; ++dx) {\n" \
	"    int sx = x + dx * uDil;\n" \
	"    int sy = y + dy * uDil;\n" \
	"    int za = uZ + dz * uDil;\n" \
	"    int zb = za + 1;\n" \
	"    bool okxy = sx >= 0 && sx < DIM && sy >= 0 && sy < DIM;\n" \
	"    float ma = (okxy && za >= 0 && za < DIM) ? 1.0 : 0.0;\n" \
	"    float mb = (okxy && zb >= 0 && zb < DIM) ? 1.0 : 0.0;\n" \
	"    int cx = clamp(sx, 0, DIM - 1);\n" \
	"    int cy = clamp(sy, 0, DIM - 1);\n" \
	"    ivec3 ca = ivec3(cx, cy, clamp(za, 0, DIM - 1));\n" \
	"    ivec3 cb = ivec3(cx, cy, clamp(zb, 0, DIM - 1));\n" \
	"    vec4 q0 = texelFetch(s0, ca, 0) * ma;\n" \
	"    vec4 q1 = texelFetch(s1, ca, 0) * ma;\n" \
	"    vec4 q2 = texelFetch(s2, ca, 0) * ma;\n" \
	"    vec4 q3 = texelFetch(s3, ca, 0) * ma;\n" \
	"    vec4 r0 = texelFetch(s0, cb, 0) * mb;\n" \
	"    vec4 r1 = texelFetch(s1, cb, 0) * mb;\n" \
	"    vec4 r2 = texelFetch(s2, cb, 0) * mb;\n" \
	"    vec4 r3 = texelFetch(s3, cb, 0) * mb;\n" \
	"    int b = uWQ0 + tap * 64;\n" \
	BC_WGL_QUAD2("q0", "r0", "0") \
	BC_WGL_QUAD2("q1", "r1", "16") \
	BC_WGL_QUAD2("q2", "r2", "32") \
	BC_WGL_QUAD2("q3", "r3", "48") \
	"    ++tap;\n" \
	"  }}}\n" \
	"  o0 = a0; o1 = a1; o2 = a2; o3 = a3;\n" \
	"  n0 = b0; n1 = b1; n2 = b2; n3 = b3;\n" \
	"}\n"
#endif /* BC_WGL_HAS_VOX2 */

/*
 * GroupNorm moments, in three passes because WebGL2 has neither shared memory
 * nor atomics.
 *
 * The shape is chosen for short accumulation chains, which is the numerical
 * lesson webgpu_kernels.h records: summing 262144 partials serially in f32
 * loses the tail, and Kahan compensation was optimised away. Here every stage
 * sums exactly DIM = 256 values, so no chain is ever longer than that:
 *
 *   A  one fragment per (x, y) column, summing the 256 voxels along z.
 *      65536 partials per channel group. Reads the volume exactly once.
 *   B  collapse each column of A: 256 partials per group.
 *   F  collapse those and form mean and inverse stddev.
 *
 * Pass A is run once per channel group with the group's texture bound to the
 * same sampler and the viewport moved to that group's rows. That avoids
 * indexing an array of samplers by a runtime value, which GLSL ES 3.00 forbids,
 * without needing four separate programs.
 */
#define BC_GLSL_MOMENTS_A \
	BC_GLSL_PREAMBLE \
	"uniform highp sampler3D src;\n" \
	"uniform int uRowBase;\n" \
	"layout(location = 0) out vec4 oSum;\n" \
	"layout(location = 1) out vec4 oSq;\n" \
	"void main() {\n" \
	"  int x = int(gl_FragCoord.x);\n" \
	"  int y = int(gl_FragCoord.y) - uRowBase;\n" \
	"  vec4 s = vec4(0.0), q = vec4(0.0);\n" \
	"  for (int z = 0; z < DIM; ++z) {\n" \
	"    vec4 v = texelFetch(src, ivec3(x, y, z), 0);\n" \
	"    s += v;\n" \
	"    q += v * v;\n" \
	"  }\n" \
	"  oSum = s; oSq = q;\n" \
	"}\n"

#define BC_GLSL_MOMENTS_B \
	BC_GLSL_PREAMBLE \
	"uniform highp sampler2D pSum, pSq;\n" \
	"layout(location = 0) out vec4 oSum;\n" \
	"layout(location = 1) out vec4 oSq;\n" \
	"void main() {\n" \
	"  int x = int(gl_FragCoord.x);\n" \
	"  int g = int(gl_FragCoord.y);\n" \
	"  vec4 s = vec4(0.0), q = vec4(0.0);\n" \
	"  for (int y = 0; y < DIM; ++y) {\n" \
	"    s += texelFetch(pSum, ivec2(x, g * DIM + y), 0);\n" \
	"    q += texelFetch(pSq,  ivec2(x, g * DIM + y), 0);\n" \
	"  }\n" \
	"  oSum = s; oSq = q;\n" \
	"}\n"

/*
 * The variance is formed as E[x^2] - E[x]^2, the cancellation-prone way, which
 * is what the CPU, CUDA and the WGSL all do -- and which the parity contract flags as the
 * standing suspect for OpenCL's looser tolerance, because OpenCL accumulates in
 * fp32 where CPU and CUDA use fp64. This is fp32 too, and has no f64 available.
 * The mitigation here is structural rather than a different formula: the three
 * short chains above keep the sums accurate enough that the subtraction has
 * something left to cancel against. If the fp16 voxel budgets turn out tight,
 * a shifted mean is the next thing to try, not a wider tolerance.
 */
#define BC_GLSL_MOMENTS_FINISH \
	BC_GLSL_PREAMBLE \
	"uniform highp sampler2D pSum, pSq;\n" \
	"layout(location = 0) out vec4 oMean;\n" \
	"layout(location = 1) out vec4 oInv;\n" \
	"void main() {\n" \
	"  int g = int(gl_FragCoord.y);\n" \
	"  vec4 s = vec4(0.0), q = vec4(0.0);\n" \
	"  for (int x = 0; x < DIM; ++x) {\n" \
	"    s += texelFetch(pSum, ivec2(x, g), 0);\n" \
	"    q += texelFetch(pSq,  ivec2(x, g), 0);\n" \
	"  }\n" \
	"  float n = float(DIM) * float(DIM) * float(DIM);\n" \
	"  vec4 mean = s / n;\n" \
	"  vec4 varr = max(q / n - mean * mean, vec4(0.0));\n" \
	"  oMean = mean;\n" \
	"  oInv = inversesqrt(varr + EPS);\n" \
	"}\n"

/*
 * GELU. Two different approximations, exactly as in meshnet_cpu.c and
 * webgpu_kernels.h, selected by the descriptor. They are NOT interchangeable:
 * each is what its upstream reference computes, and benchmarking records that
 * CUDA and OpenCL once hardcoded one of them and silently ran the wrong
 * function for the other model. The #error is the point.
 */
#if BC_MODEL_GELU_TANH_RATIONAL
#define BC_GLSL_GELU \
	"float mn_tanh(float x_in) {\n" \
	"  float x = clamp(x_in, -9.0, 9.0);\n" \
	"  float u = x * x;\n" \
	"  float p = -8.29118133e-14;\n" \
	"  p = p * u + 5.19263868e-11;\n" \
	"  p = p * u - 2.00294448e-08;\n" \
	"  p = p * u + 1.11017944e-05;\n" \
	"  p = p * u + 0.00309865153;\n" \
	"  p = p * u + 0.130791619;\n" \
	"  p = p * u + 0.99999994;\n" \
	"  float q = 0.000253859733;\n" \
	"  q = q * u + 0.024473751;\n" \
	"  q = q * u + 0.464124829;\n" \
	"  q = q * u + 1.0;\n" \
	"  return x * p / q;\n" \
	"}\n" \
	"float gelu(float x) {\n" \
	"  float u = 0.797884583 * (x + 0.044715 * (x * x * x));\n" \
	"  return (0.5 * x) * (1.0 + mn_tanh(u));\n" \
	"}\n"
#elif BC_MODEL_GELU_EXP2_SIGMOID
/*
 * Reproduces the CPU's mn_fast_exp2 polynomial rather than calling exp2().
 * webgpu_kernels.h measured the builtin at 1223 differing voxels against the
 * CPU; matching the polynomial removed that as a variable, and the same
 * argument applies here.
 */
#define BC_GLSL_GELU \
	"float mn_fast_exp2(float x_in) {\n" \
	"  float x = x_in;\n" \
	"  if (!(x > -126.0)) x = -126.0;\n" \
	"  if (x > 126.0) x = 126.0;\n" \
	"  float f = floor(x);\n" \
	"  float r = x - f;\n" \
	"  float p = 0.000216128448;\n" \
	"  p = p * r + 0.00124678648;\n" \
	"  p = p * r + 0.0096754498;\n" \
	"  p = p * r + 0.0554852814;\n" \
	"  p = p * r + 0.240229305;\n" \
	"  p = p * r + 0.693147044;\n" \
	"  p = p * r + 1.0;\n" \
	"  float scale = uintBitsToFloat(uint((int(f) + 127) << 23));\n" \
	"  return p * scale;\n" \
	"}\n" \
	"float gelu(float x) {\n" \
	"  float u = x + 0.044715 * x * x * x;\n" \
	"  return x / (1.0 + mn_fast_exp2(-2.302208198144325 * u));\n" \
	"}\n"
#else
#error "this model declares a GELU flavour the WebGL2 kernels do not implement"
#endif

/*
 * Normalise, apply the optional per-channel affine, activate.
 *
 * NOT in place, unlike mn_norm and the WGSL: WebGL2 forbids reading and writing
 * one texture in a draw, so this reads the convolution's output and writes the
 * other half of the ping-pong. Write traffic is unchanged -- the same 512 MiB
 * is stored either way -- so the only cost is that the schedule has to name
 * which buffer is current, which backend_webgl2.c does.
 *
 * All CS channels are processed, including the pad lanes beyond CHAN. They are
 * inert: a zeroed channel has mean 0 and variance 0, normalises to 0, and
 * gelu(0) = 0 in both flavours, so it stays zero for every layer. That is the
 * same argument MESHNET_CPU_CHANNEL_STRIDE and BC_WG_CS rely on.
 */
#if BC_MODEL_HAS_AFFINE
#define BC_WGL_AFFINE(v, g) \
	"  " v " = " v " * wf(uAffQ + " g ") + wf(uBiasQ + " g ");\n"
#define BC_WGL_AFFINE_UNIFORMS \
	"uniform int uAffQ;\nuniform int uBiasQ;\n"
#else
#define BC_WGL_AFFINE(v, g) ""
#define BC_WGL_AFFINE_UNIFORMS ""
#endif

#define BC_WGL_NORM_GROUP(v, s, g) \
	"  vec4 " v " = (texelFetch(" s ", p, 0) - texelFetch(pMean, ivec2(0, " g "), 0))\n" \
	"              * texelFetch(pInv, ivec2(0, " g "), 0);\n" \
	BC_WGL_AFFINE(v, g)

#define BC_GLSL_NORM \
	BC_GLSL_PREAMBLE \
	BC_WGL_PER_PLANE(BC_WGL_SAMPLER) \
	"uniform highp sampler2D pMean, pInv;\n" \
	BC_GLSL_WFETCH \
	BC_WGL_AFFINE_UNIFORMS \
	"uniform int uZ;\n" \
	BC_GLSL_GELU \
	BC_WGL_PER_PLANE(BC_WGL_OUT) \
	"vec4 g4(vec4 v) {\n" \
	"  return vec4(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w));\n" \
	"}\n" \
	"void main() {\n" \
	"  ivec3 p = ivec3(int(gl_FragCoord.x), int(gl_FragCoord.y), uZ);\n" \
	BC_WGL_PER_PLANE(BC_WGL_NORMG) \
	BC_WGL_PER_PLANE(BC_WGL_NORMOUT) \
	"}\n"

/*
 * Classifier: a 1x1 convolution followed by argmax, four voxels per fragment.
 *
 * The output is an RGBA8 2D texture rather than a 3D one, because this is the
 * only buffer that leaves the GPU and glReadPixels with GL_RGBA/
 * GL_UNSIGNED_BYTE from a normalized attachment is the one combination every
 * implementation must support. Packing four consecutive x voxels per texel
 * makes the readback exactly one byte per voxel instead of four.
 *
 * A tie leaves the LOWER class index, which is what the CPU's strict `>` does.
 * The channel loop runs to CHAN, not CS: the classifier weights are packed at
 * stride NCLASS with only CHAN channels, so a pad lane would index a weight
 * that does not belong to it.
 */
#define BC_GLSL_CLASSIFY \
	BC_GLSL_PREAMBLE \
	BC_WGL_PER_PLANE(BC_WGL_SAMPLER) \
	BC_GLSL_WFETCH \
	"uniform int uWCls;\n" \
	"uniform int uWBias;\n" \
	"out vec4 outColor;\n" \
	"float ws(int i) { return wf(i >> 2)[i & 3]; }\n" \
	"void main() {\n" \
	"  int t = int(gl_FragCoord.y) * " BC_WGL_STR(BC_WGL_LAB_W) " +\n" \
	"          int(gl_FragCoord.x);\n" \
	"  vec4 lab = vec4(0.0);\n" \
	"  for (int j = 0; j < 4; ++j) {\n" \
	"    int v = t * 4 + j;\n" \
	"    ivec3 p = ivec3(v % DIM, (v / DIM) % DIM, v / (DIM * DIM));\n" \
	"    vec4 cv[" BC_WGL_CS4 "];\n" \
	BC_WGL_PER_PLANE(BC_WGL_CLSFETCH) \
	"    float bestv = -3.0e38;\n" \
	"    int best = 0;\n" \
	"    for (int k = 0; k < NCLASS; ++k) {\n" \
	"      float acc = " BC_WGL_STR(BC_MODEL_CLASSIFIER_BIAS) " != 0\n" \
	"                  ? ws(uWBias + k) : 0.0;\n" \
	"      for (int c = 0; c < CHAN; ++c) {\n" \
	"        float sv = cv[c >> 2][c & 3];\n" \
	"        acc += sv * ws(uWCls + c * NCLASS + k);\n" \
	"      }\n" \
	"      if (acc > bestv) { bestv = acc; best = k; }\n" \
	"    }\n" \
	"    lab[j] = float(best) / 255.0;\n" \
	"  }\n" \
	"  outColor = lab;\n" \
	"}\n"

#ifdef __cplusplus
extern "C" {
#endif

extern const char *const bc_glsl_vertex;
extern const char *const bc_glsl_conv_first;
extern const char *const bc_glsl_conv_hidden;
extern const char *const bc_glsl_conv_hidden2;
extern const char *const bc_glsl_moments_a;
extern const char *const bc_glsl_moments_b;
extern const char *const bc_glsl_moments_finish;
extern const char *const bc_glsl_norm;
extern const char *const bc_glsl_classify;

#ifdef __cplusplus
}
#endif

#endif
