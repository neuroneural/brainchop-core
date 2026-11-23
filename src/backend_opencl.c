/*
 * clock_gettime is POSIX, not ISO C, and this file is built with -std=c11 which
 * hides it. The tuner needs a monotonic clock. Must precede every include.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include "opencl_driver.h"

#include "backend.h"
#include "backend_accel.h"
#include "model_specs.h"
#include "f16.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define OCL_DIM BRAINCHOP_MODEL_DIM
#define OCL_NVOX ((size_t)OCL_DIM * OCL_DIM * OCL_DIM)
#define OCL_MAX_CHANNELS 16
/* Channel STRIDE of the activation buffers, distinct from the model's logical
   channel count. See the kernel-source comment for why it is padded to 16. */
#define OCL_CS OCL_MAX_CHANNELS
/* Defaults for the conv launch shape. Both are now -D build options that the
   tuner searches per device, so these are only the starting point and what
   BRAINCHOPC_TUNE=0 pins. */
#ifndef OCL_LOCAL
#define OCL_LOCAL 128
#endif
#ifndef OCL_VPT
#define OCL_VPT 8
#endif
#define OCL_REDUCE_LOCAL 256
#define OCL_REDUCE_BLOCKS 256

/*
 * The bounds this backend's literals impose on a model, asserted rather than
 * assumed. the architecture requires that adding a model needs no shared C change;
 * these are the places that promise has an edge, and backend_cuda.cu already
 * carries the channel one. OpenCL carried none.
 */
_Static_assert(BC_MODEL_CHAN <= OCL_MAX_CHANNELS,
	"the conv accumulator (float acc[VPT][MAXC]) and the host mean/invstd "
	"arrays are sized by OCL_MAX_CHANNELS; raise it for a wider model");
_Static_assert(!BC_MODEL_HAS_AFFINE || BC_MODEL_CHAN == OCL_CS,
	"norm_gelu indexes scale/bias by the CS-wide stride, so an affine model "
	"must fill that stride; pad its compact arrays before widening OCL_CS");
/*
 * The kernel source hardcodes these as literals (see the NVOX/DIM defines in
 * the program text below) while the host derives them from the descriptor. If
 * BRAINCHOP_MODEL_DIM ever moves, host and device would disagree on the grid
 * with no diagnostic at all.
 */
_Static_assert(OCL_DIM == 256,
	"the kernel source hardcodes DIM 256 and NVOX 16777216; update both");

/*
 * GELU, in the flavour this model declares -- the same two approximations
 * meshnet_cpu.c, webgpu_kernels.h and backend_cuda.cu select between, keyed off
 * the same generated macros.
 *
 * This kernel previously hardcoded model16's sigmoid form, so a MindGrab
 * OpenCL build silently evaluated the OTHER model's approximation. The two are
 * close (2.302208198144325 = 2 * 0.797884583 / ln 2), so the voxel tolerance
 * absorbed it instead of failing, which is how it went unnoticed -- nothing in
 * CI compiles this file either.
 *
 * clamp() rather than the CPU's pair of negated comparisons: the program is
 * built with -cl-fast-relaxed-math, which implies -cl-finite-math-only, so the
 * NaN-safe idiom that file relies on has no meaning here.
 */
#if BC_MODEL_GELU_TANH_RATIONAL
#define OCL_GELU \
"float bc_gelu(float x){\n" \
" float t=0.797884583f*(x+0.044715f*x*x*x);\n" \
" t=clamp(t,-9.0f,9.0f);\n" \
" float u=t*t;\n" \
" float p=-8.29118133e-14f;\n" \
" p=fma(p,u,5.19263868e-11f);\n" \
" p=fma(p,u,-2.00294448e-08f);\n" \
" p=fma(p,u,1.11017944e-05f);\n" \
" p=fma(p,u,0.00309865153f);\n" \
" p=fma(p,u,0.130791619f);\n" \
" p=fma(p,u,0.99999994f);\n" \
" float q=0.000253859733f;\n" \
" q=fma(q,u,0.024473751f);\n" \
" q=fma(q,u,0.464124829f);\n" \
" q=fma(q,u,1.0f);\n" \
" return (0.5f*x)*(1.0f+t*p/q);\n" \
"}\n"
#elif BC_MODEL_GELU_EXP2_SIGMOID
#define OCL_GELU \
"float bc_gelu(float x){\n" \
" float u=fma(0.044715f*x*x,x,x);\n" \
" return x/(1.0f+exp2(-2.302208198144325f*u));\n" \
"}\n"
#else
#error "model_meta.json named a GELU flavour with no implementation here"
#endif

/*
 * Kernel source. DIM, NVOX, CS, VPT, LOCAL and RLOCAL all arrive as -D build
 * options rather than being written here, which is what lets the tuner search
 * VPT and LOCAL at run time: this backend compiles its kernels on the device,
 * so its search space is genuinely open where the CUDA backend had to
 * enumerate template instantiations.
 *
 * Activations are stored as `half` and computed in float. That needs NO
 * extension: `vload_half`/`vstore_half` and pointers to `half` are core OpenCL,
 * where the `half` scalar TYPE and half arithmetic would need the optional
 * cl_khr_fp16 that many integrated devices lack. Accumulation is float and the
 * moments are finished in double on the host, so the narrowing is storage-only
 * -- the same split Metal and the CUDA backend use.
 *
 * The channel stride is padded to CS (16) exactly as backend_cuda.cu pads it,
 * and for the same reason: MindGrab is a 15-channel network, the pad lane is
 * inert because GroupNorm is one group per channel, and a zeroed channel
 * normalises to 0 with gelu(0) = 0. `channels` remains the logical COUNT and CS
 * the STRIDE; activations index by CS, the classifier by the count.
 */
static const char opencl_source[] =
"__kernel void conv3d(__global const half *src,__global half *dst,\n"
" __global const half *weights,ulong woff,int cin,int istride,int dil,\n"
" __local half *sw){\n"
" int lid=(int)get_local_id(0),lsz=(int)get_local_size(0);\n"
/* Weights are pre-padded to CS wide, so the inner bounds are literals and no
   runtime channel count is needed -- the tail lanes are zero and contribute
   nothing. Staged as half rather than float: 13.8 KB of local memory instead of
   27.6 KB, which matters on devices that only guarantee 32 KB. */
" int wcount=27*cin*CS;\n"
/* Round-trips through float rather than copying the half directly: without
   cl_khr_fp16 a `half` lvalue cannot be assigned, only loaded and stored via
   the builtins. The conversion is exact in both directions, so this is a copy. */
" for(int i=lid;i<wcount;i+=lsz)vstore_half(vload_half(woff+i,weights),i,sw);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" size_t base=(size_t)get_group_id(0)*((size_t)lsz*VPT)+lid;\n"
" float acc[VPT][CS];\n"
" size_t vox[VPT];int vx[VPT],vy[VPT],vz[VPT];\n"
" for(int v=0;v<VPT;v++){\n"
"  for(int oc=0;oc<CS;oc++)acc[v][oc]=0.0f;\n"
"  vox[v]=base+(size_t)v*lsz;\n"
"  vx[v]=(int)(vox[v]&255);vy[v]=(int)((vox[v]>>8)&255);vz[v]=(int)(vox[v]>>16);\n"
" }\n"
" for(int tz=-1;tz<=1;tz++){\n"
"  for(int ty=-1;ty<=1;ty++){\n"
"   for(int tx=-1;tx<=1;tx++){\n"
"    size_t tap=(size_t)((tz+1)*9+(ty+1)*3+tx+1)*cin*CS;\n"
"    int ok[VPT];size_t iv[VPT];\n"
"    for(int v=0;v<VPT;v++){\n"
"     int ix=vx[v]+tx*dil,iy=vy[v]+ty*dil,iz=vz[v]+tz*dil;\n"
"     ok[v]=((uint)ix<DIM)&&((uint)iy<DIM)&&((uint)iz<DIM)&&(vox[v]<NVOX);\n"
"     iv[v]=(size_t)iz*DIM*DIM+(size_t)iy*DIM+(size_t)ix;\n"
"    }\n"
"    for(int ic=0;ic<cin;ic++){\n"
"     float w[CS];\n"
"     for(int oc=0;oc<CS;oc++)w[oc]=vload_half(tap+(size_t)ic*CS+oc,sw);\n"
"     for(int v=0;v<VPT;v++){\n"
"      if(!ok[v])continue;\n"
"      float q=vload_half(iv[v]*(size_t)istride+ic,src);\n"
"      for(int oc=0;oc<CS;oc++)acc[v][oc]=fma(q,w[oc],acc[v][oc]);\n"
"     }\n"
"    }\n"
"   }\n"
"  }\n"
" }\n"
" for(int v=0;v<VPT;v++){\n"
"  if(vox[v]>=NVOX)continue;\n"
"  for(int oc=0;oc<CS;oc++)vstore_half(acc[v][oc],vox[v]*CS+oc,dst);\n"
" }\n"
"}\n"
/*
 * Per-channel sum and sum-of-squares, walking the buffer FLAT.
 *
 * The previous shape indexed src[i * channels + channel] with one block row per
 * channel, giving every lane a 64-byte stride and reading the whole activation
 * once PER CHANNEL. Walking it flat reads it exactly once, coalesced. The trick
 * that makes it legal is that the grid stride is a multiple of CS, so a
 * work-item keeps the same channel for its whole walk. Worth ~4x on the
 * equivalent CUDA kernel.
 *
 * The block fold is an arithmetic walk rather than a halving tree because the
 * tree would pair lanes holding different channels. With RLOCAL a multiple of
 * CS the base residue is always 0, but it is computed rather than assumed so
 * the kernel stays correct if either constant is tuned.
 */
"__kernel void reduce_channels(__global const half *src,__global float2 *partial){\n"
" int block=(int)get_group_id(0),lid=(int)get_local_id(0);\n"
" __local float sums[RLOCAL];__local float squares[RLOCAL];\n"
" size_t total=(size_t)NVOX*CS;\n"
" size_t start=(size_t)block*RLOCAL+lid;\n"
" size_t stride=(size_t)get_num_groups(0)*RLOCAL;\n"
" float s=0.0f,q=0.0f;\n"
" for(size_t i=start;i<total;i+=stride){float v=vload_half(i,src);s+=v;q=fma(v,v,q);}\n"
" sums[lid]=s;squares[lid]=q;barrier(CLK_LOCAL_MEM_FENCE);\n"
" int rbase=(int)(((size_t)block*RLOCAL)%CS);\n"
" if(lid<CS){\n"
"  int first=(lid-rbase+CS)%CS;\n"
"  float ts=0.0f,tq=0.0f;\n"
"  for(int i=first;i<RLOCAL;i+=CS){ts+=sums[i];tq+=squares[i];}\n"
"  partial[(size_t)block*CS+lid]=(float2)(ts,tq);\n"
" }\n"
"}\n"
OCL_GELU
"__kernel void norm_gelu(__global half *buf,__global const float *mean,\n"
" __global const float *invstd,__global const float *scale,\n"
" __global const float *bias,int affine){size_t voxel=get_global_id(0);\n"
" if(voxel>=NVOX)return;\n"
/* Runs to CS, not the logical count: the pad lane keeps its inert zero rather
   than being left undefined. The affine arrays are indexed by the same bound,
   which the static assert on the host pins to models that fill the stride. */
" for(int c=0;c<CS;c++){float v=(vload_half(voxel*CS+c,buf)-mean[c])*invstd[c];\n"
"  if(affine)v=fma(v,scale[c],bias[c]);\n"
"  vstore_half(bc_gelu(v),voxel*CS+c,buf);}\n"
"}\n"
"__kernel void classify(__global const half *src,__global float *labels,\n"
" __global const half *weights,ulong woff,int channels,int classes,int has_bias){\n"
" size_t voxel=get_global_id(0);if(voxel>=NVOX)return;\n"
" ulong boff=woff+(ulong)channels*classes;\n"
/* -MAXFLOAT, not -INFINITY: the program is built with -cl-fast-relaxed-math,
   which implies -cl-finite-math-only and lets the implementation assume no
   infinities occur. MAXFLOAT is spec-safe and equivalent for real logits. */
" int best=0;float bestv=-MAXFLOAT;\n"
/* Strides by CS but sums only the LOGICAL channels: the classifier weight
   matrix has no row for the pad lane. */
" for(int cls=0;cls<classes;cls++){\n"
"  float v=has_bias?vload_half(boff+cls,weights):0.0f;\n"
"  for(int c=0;c<channels;c++)\n"
"   v=fma(vload_half(voxel*CS+c,src),\n"
"     vload_half(woff+(size_t)c*classes+cls,weights),v);\n"
"  if(v>bestv){bestv=v;best=cls;}}\n"
" labels[voxel]=(float)best;\n"
"}\n"
/* Tuning support. The checksum reads the SAME buffer as ushort rather than
   half: it wants the raw bit pattern, and integer addition is associative and
   commutative even when it wraps, so the fingerprint is independent of how the
   work-groups are scheduled. */
"__kernel void fill_pattern(__global half *buf,ulong count){\n"
" size_t start=get_global_id(0),stride=get_global_size(0);\n"
" for(size_t i=start;i<count;i+=stride){uint h=(uint)i*2654435761u;\n"
"  vstore_half((float)(h>>8)*(1.0f/16777216.0f)-0.5f,i,buf);}\n"
"}\n"
"__kernel void checksum(__global const ushort *src,__global uint *total,\n"
" ulong count){size_t start=get_global_id(0),stride=get_global_size(0);\n"
" uint acc=0;\n"
" for(size_t i=start;i<count;i+=stride)acc+=(uint)src[i];\n"
" atomic_add(total,acc);\n"
"}\n";

typedef struct {
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel convolution;
	cl_kernel reduction;
	cl_kernel normalization;
	cl_kernel classifier;
	cl_kernel fill;
	cl_kernel checksum;
	/* The launch shape this program was BUILT for. The kernels bake VPT and
	   LOCAL in as -D options, so a retune means rebuilding the program, and the
	   host NDRange must be derived from these rather than from the macros. */
	int vpt;
	int local;
} opencl_state;

static int contains_case_insensitive(const char *text, const char *needle) {
	if (!needle || !*needle) return 1;
	size_t length = strlen(needle);
	for (; *text; text++) {
		size_t i = 0;
		while (i < length && text[i] &&
			tolower((unsigned char)text[i]) == tolower((unsigned char)needle[i]))
			i++;
		if (i == length) return 1;
	}
	return 0;
}

/*
 * Every GPU device on every platform, flattened into ONE list in a fixed order:
 * platforms in the order the ICD loader reports them, and devices in the order
 * each platform reports them.
 *
 * That flattening is what makes --adapter meaningful. The index a user reads out
 * of `--adapter -1` has to select the same device on the next run, so listing
 * and selection must walk the hardware identically -- which is why both go
 * through this one function rather than each doing its own nested loop.
 *
 * Fills up to `capacity` entries and returns the TOTAL number present, which may
 * exceed capacity; callers size their array from a first call with capacity 0.
 */
typedef struct {
	cl_device_id id;
	char name[512];
	cl_ulong memory;
	cl_bool unified_memory;
} opencl_adapter;

static int enumerate_devices(opencl_adapter *out, int capacity) {
	if (!bc_opencl_load()) return 0;
	cl_uint platform_count = 0;
	if (clGetPlatformIDs(0, NULL, &platform_count) != CL_SUCCESS ||
			platform_count == 0)
		return 0;
	cl_platform_id *platforms = (cl_platform_id *)calloc(
		platform_count, sizeof(*platforms));
	if (!platforms) return 0;
	if (clGetPlatformIDs(platform_count, platforms, NULL) != CL_SUCCESS) {
		free(platforms);
		return 0;
	}
	int total = 0;
	for (cl_uint platform_index = 0; platform_index < platform_count;
			platform_index++) {
		cl_uint device_count = 0;
		if (clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU,
				0, NULL, &device_count) != CL_SUCCESS || device_count == 0)
			continue;
		cl_device_id *devices = (cl_device_id *)calloc(device_count, sizeof(*devices));
		if (!devices) continue;
		if (clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU,
				device_count, devices, NULL) != CL_SUCCESS) {
			free(devices);
			continue;
		}
		char platform_name[256] = "";
		clGetPlatformInfo(platforms[platform_index], CL_PLATFORM_NAME,
			sizeof(platform_name), platform_name, NULL);
		for (cl_uint device_index = 0; device_index < device_count; device_index++) {
			if (total < capacity && out) {
				opencl_adapter *entry = &out[total];
				char device_name[256] = "";
				entry->id = devices[device_index];
				entry->memory = 0;
				entry->unified_memory = CL_FALSE;
				clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
					sizeof(device_name), device_name, NULL);
				clGetDeviceInfo(devices[device_index], CL_DEVICE_GLOBAL_MEM_SIZE,
					sizeof(entry->memory), &entry->memory, NULL);
				clGetDeviceInfo(devices[device_index], CL_DEVICE_HOST_UNIFIED_MEMORY,
					sizeof(entry->unified_memory), &entry->unified_memory, NULL);
				snprintf(entry->name, sizeof(entry->name), "%s / %s",
					platform_name, device_name);
			}
			total++;
		}
		free(devices);
	}
	free(platforms);
	return total;
}

/* Capacity for the on-stack enumeration below. Machines with more GPUs than
   this exist, but not ones where the extras are worth a heap allocation on a
   path that runs once per invocation. */
#define OPENCL_MAX_ADAPTERS 32

static int opencl_adapter_count(void) {
	/*
	 * Clamped, so that the count, the listing and the selector all agree on the
	 * same set. enumerate_devices reports the TRUE total, which may exceed the
	 * on-stack capacity; returning it unclamped would let bc_backend_adapter_valid
	 * accept an index that select_device then cannot resolve, failing at run time
	 * with the index having been checked. Only reachable with more than
	 * OPENCL_MAX_ADAPTERS GPUs present, but "validated then failed" is the wrong
	 * shape of bug to leave lying around.
	 */
	int count = enumerate_devices(NULL, 0);
	return count > OPENCL_MAX_ADAPTERS ? OPENCL_MAX_ADAPTERS : count;
}

static int opencl_adapter_name(int index, char *description, size_t capacity) {
	opencl_adapter adapters[OPENCL_MAX_ADAPTERS];
	int count = enumerate_devices(adapters, OPENCL_MAX_ADAPTERS);
	if (index < 0 || index >= count || index >= OPENCL_MAX_ADAPTERS) return 1;
	if (description && capacity)
		snprintf(description, capacity, "%s (%.2f GB%s)", adapters[index].name,
			(double)adapters[index].memory / (1024.0 * 1024.0 * 1024.0),
			adapters[index].unified_memory ? ", integrated" : "");
	return 0;
}

/*
 * Pick a device. Three ways in, in descending order of explicitness:
 *
 *   requested >= 0                  --adapter, a position in the list above
 *   BRAINCHOPC_OPENCL_DEVICE        case-insensitive substring of the name
 *   neither                         the most global memory
 *
 * The last is the historical rule and it is WRONG on a machine with both a
 * discrete and an integrated GPU: an iGPU reports a slice of system RAM, so a
 * 2-CU Raphael can report more than a 16 GB discrete card and win. Discrete is
 * now preferred over integrated first, and memory only breaks ties within a
 * class -- measured on the Windows box, where the old rule chose a device about
 * 117x slower than the one beside it.
 */
static cl_device_id select_device(int requested, char *description, size_t capacity) {
	opencl_adapter adapters[OPENCL_MAX_ADAPTERS];
	int count = enumerate_devices(adapters, OPENCL_MAX_ADAPTERS);
	if (count > OPENCL_MAX_ADAPTERS) count = OPENCL_MAX_ADAPTERS;
	if (count <= 0) return NULL;

	int best = -1;
	if (requested >= 0) {
		if (requested >= count) return NULL;
		best = requested;
	} else {
		const char *wanted = getenv("BRAINCHOPC_OPENCL_DEVICE");
		for (int i = 0; i < count; i++) {
			if (wanted && *wanted &&
					!contains_case_insensitive(adapters[i].name, wanted))
				continue;
			if (best < 0) { best = i; continue; }
			int best_discrete = !adapters[best].unified_memory;
			int this_discrete = !adapters[i].unified_memory;
			if (this_discrete != best_discrete) {
				if (this_discrete) best = i;
			} else if (adapters[i].memory > adapters[best].memory) {
				best = i;
			}
		}
	}
	if (best < 0) return NULL;
	if (description && capacity)
		snprintf(description, capacity, "%s", adapters[best].name);
	return adapters[best].id;
}

static int opencl_available(void) {
	return select_device(-1, NULL, 0) != NULL;
}

/* The padded fp16 weight layout, shared with backend_cuda.cu. */
#define bc_pad_store uint16_t
#define BC_PAD_STORE(x) bc_f32_to_f16(x)
#define BC_PAD_CS OCL_CS
#include "accel_pack_padded.h"

static int opencl_error(cl_int error, const char *operation) {
	if (error == CL_SUCCESS) return 0;
	fprintf(stderr, "brainchopc: OpenCL %s failed (%d)\n", operation, (int)error);
	return 1;
}

static void release_program(opencl_state *state);

static void release_state(opencl_state *state) {
	release_program(state);
	if (state->queue) clReleaseCommandQueue(state->queue);
	if (state->context) clReleaseContext(state->context);
	memset(state, 0, sizeof(*state));
}

/*
 * (Re)build the program for a given launch shape and create its kernels.
 *
 * VPT and LOCAL are -D build options, not kernel arguments, so changing the
 * shape means recompiling -- which is the price of the open search space this
 * backend has and the CUDA one does not. Any previous program and kernels are
 * released first, so the tuner can call this once per candidate.
 */
static void release_program(opencl_state *state) {
	if (state->checksum) clReleaseKernel(state->checksum);
	if (state->fill) clReleaseKernel(state->fill);
	if (state->classifier) clReleaseKernel(state->classifier);
	if (state->normalization) clReleaseKernel(state->normalization);
	if (state->reduction) clReleaseKernel(state->reduction);
	if (state->convolution) clReleaseKernel(state->convolution);
	if (state->program) clReleaseProgram(state->program);
	state->checksum = state->fill = NULL;
	state->classifier = state->normalization = NULL;
	state->reduction = state->convolution = NULL;
	state->program = NULL;
}

static int build_program(opencl_state *state, int vpt, int local) {
	cl_int error = CL_SUCCESS;
	release_program(state);
	state->vpt = vpt;
	state->local = local;
	const char *source = opencl_source;
	size_t source_length = strlen(source);
	state->program = clCreateProgramWithSource(state->context, 1,
		&source, &source_length, &error);
	if (opencl_error(error, "program creation")) return 1;
	char options[256];
	snprintf(options, sizeof(options),
		"-cl-fast-relaxed-math -DDIM=%d -DNVOX=%luul -DCS=%d -DVPT=%d "
		"-DLOCAL=%d -DRLOCAL=%d",
		OCL_DIM, (unsigned long)OCL_NVOX, OCL_CS, vpt, local,
		OCL_REDUCE_LOCAL);
	error = clBuildProgram(state->program, 1, &state->device, options, NULL, NULL);
	if (error != CL_SUCCESS) {
		size_t log_length = 0;
		clGetProgramBuildInfo(state->program, state->device,
			CL_PROGRAM_BUILD_LOG, 0, NULL, &log_length);
		char *log = (char *)malloc(log_length + 1);
		if (log) {
			clGetProgramBuildInfo(state->program, state->device,
				CL_PROGRAM_BUILD_LOG, log_length, log, NULL);
			log[log_length] = 0;
			fprintf(stderr, "brainchopc: OpenCL build failed:\n%s\n", log);
			free(log);
		}
		return 1;
	}
#define CREATE_KERNEL(FIELD, NAME) \
	state->FIELD = clCreateKernel(state->program, NAME, &error); \
	if (opencl_error(error, "kernel creation")) return 1
	CREATE_KERNEL(convolution, "conv3d");
	CREATE_KERNEL(reduction, "reduce_channels");
	CREATE_KERNEL(normalization, "norm_gelu");
	CREATE_KERNEL(classifier, "classify");
	CREATE_KERNEL(fill, "fill_pattern");
	CREATE_KERNEL(checksum, "checksum");
#undef CREATE_KERNEL
	return 0;
}

static int initialize_state(opencl_state *state, int adapter, int vpt, int local) {
	char device_name[512];
	cl_int error = CL_SUCCESS;
	memset(state, 0, sizeof(*state));
	state->vpt = vpt;
	state->local = local;
	state->device = select_device(adapter, device_name, sizeof(device_name));
	if (!state->device) return 1;
	fprintf(stderr, "brainchopc: OpenCL device: %s\n", device_name);
	state->context = clCreateContext(NULL, 1, &state->device, NULL, NULL, &error);
	if (opencl_error(error, "context creation")) goto fail;
	state->queue = clCreateCommandQueue(state->context, state->device, 0, &error);
	if (opencl_error(error, "queue creation")) goto fail;
	/*
	 * No program yet. The launch shape is not known until the tune cache has
	 * been read, and building the compiled-in shape here only to rebuild the
	 * cached one moments later cost ~175 ms per run on NVIDIA/Windows -- a
	 * seventh of the segment. opencl_segment builds exactly once per shape.
	 */
	return 0;
fail:
	release_state(state);
	return 1;
}

static int set_argument(cl_kernel kernel, cl_uint index,
		size_t size, const void *value) {
	return opencl_error(clSetKernelArg(kernel, index, size, value),
		"setting kernel argument");
}

/*
 * Normalisation runs over the STRIDE, not the logical channel count.
 *
 * Every buffer here is OCL_CS wide, so a reduction told there are 15 channels
 * would walk 15/16 of the activation and assign each element to the wrong
 * channel. The pad lane's moments come out exactly (0, 0), which normalises and
 * activates back to zero.
 *
 * The reduction is now one-dimensional: the flat walk covers every channel in a
 * single pass, where the old shape needed a block row per channel.
 */
static int run_normalize(opencl_state *state, cl_mem buffer,
		cl_mem partial_buffer, cl_mem mean_buffer, cl_mem invstd_buffer,
		cl_mem scale_buffer, cl_mem bias_buffer,
		const float *scale, const float *bias) {
	size_t global = (size_t)OCL_REDUCE_BLOCKS * OCL_REDUCE_LOCAL;
	size_t local = OCL_REDUCE_LOCAL;
	if (set_argument(state->reduction, 0, sizeof(buffer), &buffer) ||
		set_argument(state->reduction, 1, sizeof(partial_buffer), &partial_buffer) ||
		opencl_error(clEnqueueNDRangeKernel(state->queue, state->reduction,
			1, NULL, &global, &local, 0, NULL, NULL), "normalization reduction"))
		return 1;
	size_t partial_count = (size_t)OCL_CS * OCL_REDUCE_BLOCKS * 2;
	float *partial = (float *)malloc(partial_count * sizeof(*partial));
	if (!partial) return 1;
	cl_int error = clEnqueueReadBuffer(state->queue, partial_buffer, CL_TRUE, 0,
		partial_count * sizeof(*partial), partial, 0, NULL, NULL);
	if (opencl_error(error, "reading normalization reduction")) {
		free(partial);
		return 1;
	}
	/*
	 * The partials are laid out [block][channel], not [channel][block], because
	 * the flat walk assigns a channel per lane within a block. Summed in double
	 * on the host, which is where this backend has always finished the moments.
	 */
	float mean[OCL_CS], invstd[OCL_CS];
	for (int channel = 0; channel < OCL_CS; channel++) {
		double sum = 0.0, square = 0.0;
		for (int block = 0; block < OCL_REDUCE_BLOCKS; block++) {
			size_t index = ((size_t)block * OCL_CS + channel) * 2;
			sum += partial[index];
			square += partial[index + 1];
		}
		double channel_mean = sum / (double)OCL_NVOX;
		double variance = square / (double)OCL_NVOX - channel_mean * channel_mean;
		if (variance < 0.0) variance = 0.0;
		mean[channel] = (float)channel_mean;
		invstd[channel] = (float)(1.0 / sqrt(variance + 1e-5));
	}
	free(partial);
	int affine = scale != NULL;
	/*
	 * Blocking writes. The sources are locals of this function and the caller
	 * reuses the frame on its next layer, so a non-blocking transfer would be
	 * reading freed stack by the time the DMA ran -- the spec requires the host
	 * memory to stay valid until the command completes. Two 64-byte copies per
	 * layer, 13 or 25 times per inference, so the serialization is unmeasurable.
	 *
	 * The affine buffers are only read when the kernel's `affine` flag is set,
	 * so MindGrab skips them entirely instead of uploading a dummy identity.
	 * Their compact arrays are BC_MODEL_CHAN wide, which the static assert at
	 * the top pins equal to OCL_CS for any model that has an affine.
	 */
#define WRITE_PARAMETER(BUFFER, DATA, COUNT, LABEL) \
	if (opencl_error(clEnqueueWriteBuffer(state->queue, BUFFER, CL_TRUE, 0, \
		(size_t)(COUNT) * sizeof(float), DATA, 0, NULL, NULL), LABEL)) return 1
	WRITE_PARAMETER(mean_buffer, mean, OCL_CS, "writing normalization mean");
	WRITE_PARAMETER(invstd_buffer, invstd, OCL_CS, "writing normalization scale");
	if (affine) {
		WRITE_PARAMETER(scale_buffer, scale, BC_MODEL_CHAN, "writing affine scale");
		WRITE_PARAMETER(bias_buffer, bias, BC_MODEL_CHAN, "writing affine bias");
	}
#undef WRITE_PARAMETER
	size_t norm_global = (OCL_NVOX + state->local - 1) / state->local * state->local;
	size_t norm_local = (size_t)state->local;
	if (set_argument(state->normalization, 0, sizeof(buffer), &buffer) ||
		set_argument(state->normalization, 1, sizeof(mean_buffer), &mean_buffer) ||
		set_argument(state->normalization, 2, sizeof(invstd_buffer), &invstd_buffer) ||
		set_argument(state->normalization, 3, sizeof(scale_buffer), &scale_buffer) ||
		set_argument(state->normalization, 4, sizeof(bias_buffer), &bias_buffer) ||
		set_argument(state->normalization, 5, sizeof(affine), &affine) ||
		opencl_error(clEnqueueNDRangeKernel(state->queue, state->normalization,
			1, NULL, &norm_global, &norm_local, 0, NULL, NULL),
			"normalization kernel"))
		return 1;
	return 0;
}

/* ------------------------------------------------------------------ autotune
 *
 * The same idea as backend_cuda.cu's tuner, with one structural advantage: this
 * backend compiles its kernels on the device, so VPT and the local size are -D
 * options and the search space is genuinely open rather than a fixed table of
 * template instantiations. The price is that each candidate needs a program
 * rebuild, which is why the grid is kept small.
 *
 * Every candidate must reproduce the reference checksum exactly. Changing the
 * shape changes only WHICH work-item owns a voxel, never the order in which
 * that voxel's own products are summed, so equality is the right bar rather
 * than a tolerance -- the same property the CUDA tuner relies on.
 */
#define OCL_TUNE_STAMP 1
#ifndef OCL_TUNE_REPEATS
#define OCL_TUNE_REPEATS 3
#endif

static const int opencl_tune_vpt[] = { 1, 2, 3, 4, 6, 8 };
static const int opencl_tune_local[] = { 64, 128, 256 };

static void opencl_cache_key(opencl_state *state, char *key, size_t capacity) {
	char name[256] = "unknown", vendor[128] = "", version[128] = "";
	clGetDeviceInfo(state->device, CL_DEVICE_NAME, sizeof(name), name, NULL);
	clGetDeviceInfo(state->device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
	clGetDeviceInfo(state->device, CL_DRIVER_VERSION, sizeof(version),
		version, NULL);
	/* '|' is the field separator, so it must not survive inside a field. */
	for (char *p = name; *p; p++) if (*p == '|' || *p == '\n') *p = ' ';
	for (char *p = vendor; *p; p++) if (*p == '|' || *p == '\n') *p = ' ';
	for (char *p = version; *p; p++) if (*p == '|' || *p == '\n') *p = ' ';
	snprintf(key, capacity, "opencl|%s|%s|%s|%s|%d",
		vendor, name, version, BC_MODEL_NAME, OCL_TUNE_STAMP);
}

/* Wall-clock around a finished queue. Profiling events would be tidier, but the
   queue is not created with CL_QUEUE_PROFILING_ENABLE and turning it on for
   every run to serve the tuner would be the wrong trade. */
static double opencl_now(void) {
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}

static int opencl_run_conv(opencl_state *state, cl_mem source,
		cl_mem destination, cl_mem weights, int dilation) {
	cl_ulong weight_offset = 0;
	int padded_inputs = OCL_CS, input_stride = OCL_CS;
	size_t conv_items = (OCL_NVOX + state->vpt - 1) / state->vpt;
	size_t conv_global = (conv_items + state->local - 1) / state->local *
		state->local;
	size_t local = (size_t)state->local;
	if (set_argument(state->convolution, 0, sizeof(source), &source) ||
		set_argument(state->convolution, 1, sizeof(destination), &destination) ||
		set_argument(state->convolution, 2, sizeof(weights), &weights) ||
		set_argument(state->convolution, 3, sizeof(weight_offset), &weight_offset) ||
		set_argument(state->convolution, 4, sizeof(padded_inputs), &padded_inputs) ||
		set_argument(state->convolution, 5, sizeof(input_stride), &input_stride) ||
		set_argument(state->convolution, 6, sizeof(dilation), &dilation) ||
		set_argument(state->convolution, 7,
			(size_t)27 * OCL_CS * OCL_CS * sizeof(uint16_t), NULL))
		return 1;
	return clEnqueueNDRangeKernel(state->queue, state->convolution, 1, NULL,
		&conv_global, &local, 0, NULL, NULL) != CL_SUCCESS;
}

static int opencl_checksum(opencl_state *state, cl_mem buffer, cl_uint *out) {
	cl_int error = CL_SUCCESS;
	cl_uint zero = 0;
	cl_ulong count = (cl_ulong)OCL_NVOX * OCL_CS;
	cl_mem total = clCreateBuffer(state->context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(zero), &zero, &error);
	if (error != CL_SUCCESS) return 1;
	size_t global = 65536, local = 256;
	int failed = set_argument(state->checksum, 0, sizeof(buffer), &buffer) ||
		set_argument(state->checksum, 1, sizeof(total), &total) ||
		set_argument(state->checksum, 2, sizeof(count), &count) ||
		clEnqueueNDRangeKernel(state->queue, state->checksum, 1, NULL,
			&global, &local, 0, NULL, NULL) != CL_SUCCESS ||
		clEnqueueReadBuffer(state->queue, total, CL_TRUE, 0, sizeof(*out),
			out, 0, NULL, NULL) != CL_SUCCESS;
	clReleaseMemObject(total);
	return failed;
}

static int opencl_tune(opencl_state *state, cl_mem scratch_a, cl_mem scratch_b,
		cl_mem weights, int *best_vpt, int *best_local) {
	cl_uint reference = 0;
	int have_reference = 0;
	double best_ms = 0.0;
	int chosen_vpt = state->vpt, chosen_local = state->local, found = 0;
	const int dilation = 4;
	const int vpt_count = (int)(sizeof(opencl_tune_vpt) / sizeof(*opencl_tune_vpt));
	const int local_count =
		(int)(sizeof(opencl_tune_local) / sizeof(*opencl_tune_local));

	fprintf(stderr, "brainchopc: tuning OpenCL convolution for this device "
		"(%d variants, once per device -- caching the result)\n",
		vpt_count * local_count);

	for (int vi = 0; vi < vpt_count; vi++) {
		for (int li = 0; li < local_count; li++) {
			int vpt = opencl_tune_vpt[vi], local = opencl_tune_local[li];
			cl_uint digest = 0;
			double best_sample = 0.0;
			if (build_program(state, vpt, local)) {
				fprintf(stderr, "brainchopc:   vpt=%-2d local=%-4d unavailable\n",
					vpt, local);
				continue;
			}
			/* Fill through the freshly built program, so the pattern is written
			   by the same code that will read it. */
			{
				cl_ulong count = (cl_ulong)OCL_NVOX * OCL_CS;
				size_t global = 65536, lsz = 256;
				if (set_argument(state->fill, 0, sizeof(scratch_a), &scratch_a) ||
					set_argument(state->fill, 1, sizeof(count), &count) ||
					clEnqueueNDRangeKernel(state->queue, state->fill, 1, NULL,
						&global, &lsz, 0, NULL, NULL) != CL_SUCCESS)
					continue;
			}
			if (opencl_run_conv(state, scratch_a, scratch_b, weights, dilation) ||
					clFinish(state->queue) != CL_SUCCESS) {
				fprintf(stderr, "brainchopc:   vpt=%-2d local=%-4d unavailable\n",
					vpt, local);
				continue;
			}
			if (opencl_checksum(state, scratch_b, &digest)) continue;
			if (!have_reference) {
				reference = digest;
				have_reference = 1;
			} else if (digest != reference) {
				fprintf(stderr, "brainchopc:   vpt=%-2d local=%-4d rejected "
					"(checksum mismatch)\n", vpt, local);
				continue;
			}
			/*
			 * Best of OCL_TUNE_REPEATS. The winner is cached and then trusted
			 * indefinitely, so a single sample taken while the device was busy
			 * would be baked in permanently -- an audit reproduced exactly that
			 * on the CUDA side, costing 49%.
			 */
			for (int repeat = 0; repeat < OCL_TUNE_REPEATS; repeat++) {
				double start = opencl_now();
				if (opencl_run_conv(state, scratch_a, scratch_b, weights,
						dilation) || clFinish(state->queue) != CL_SUCCESS) {
					best_sample = 0.0;
					break;
				}
				double elapsed = opencl_now() - start;
				if (best_sample <= 0.0 || elapsed < best_sample)
					best_sample = elapsed;
			}
			if (best_sample <= 0.0) continue;
			fprintf(stderr, "brainchopc:   vpt=%-2d local=%-4d %7.2f ms\n",
				vpt, local, best_sample);
			if (!found || best_sample < best_ms) {
				found = 1;
				best_ms = best_sample;
				chosen_vpt = vpt;
				chosen_local = local;
			}
		}
	}
	if (!found) return 1;
	fprintf(stderr, "brainchopc: chose vpt=%d local=%d (%.2f ms)\n",
		chosen_vpt, chosen_local, best_ms);
	*best_vpt = chosen_vpt;
	*best_local = chosen_local;
	return 0;
}

static int opencl_segment(const unsigned char *conformed,
		float *result, const float *weights, const bc_inference_options *options) {
	const bc_accel_model_spec *spec = bc_accel_spec();
	if (!conformed || !result || !weights || !spec) return 1;
	if (options && (options->roi_enabled || options->class_writer || options->tissue_planes)) return 2;
	opencl_state state;
	if (initialize_state(&state, options ? options->adapter : -1,
			OCL_VPT, OCL_LOCAL)) return 1;
	/* OCL_CS, not the logical count: the buffer is padded to the stride. Even
	   so this halves what fp32 used -- MindGrab went 15x4 to 16x2 bytes per
	   voxel, model16 16x4 to 16x2. */
	size_t activation_bytes = OCL_NVOX * (size_t)OCL_CS * sizeof(uint16_t);
	cl_ulong max_allocation = 0, global_memory = 0;
	clGetDeviceInfo(state.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
		sizeof(max_allocation), &max_allocation, NULL);
	clGetDeviceInfo(state.device, CL_DEVICE_GLOBAL_MEM_SIZE,
		sizeof(global_memory), &global_memory, NULL);
	size_t required = 2 * activation_bytes +
		spec->weight_count * sizeof(uint16_t);
	if (max_allocation < activation_bytes || global_memory < required) {
		fprintf(stderr, "brainchopc: OpenCL device memory is insufficient "
			"(needs about %.1f GB, maximum allocation %.1f GB)\n",
			(double)required / 1e9, (double)max_allocation / 1e9);
		release_state(&state);
		return 1;
	}
	float *host_input = bc_accel_normalize_input( conformed);
	float *host_weights = bc_accel_copy_weights( weights);
	bc_padded_weights packed;
	memset(&packed, 0, sizeof(packed));
	uint16_t *half_input = NULL;
	cl_int error = CL_SUCCESS;
	cl_mem first = NULL, second = NULL, weight_buffer = NULL;
	cl_mem partial_buffer = NULL, mean_buffer = NULL, invstd_buffer = NULL;
	cl_mem scale_buffer = NULL, bias_buffer = NULL;
	int failed = !host_input || !host_weights;
	if (!failed) failed = bc_pack_padded_weights(spec, host_weights, &packed);
	if (!failed) {
		/* Shared with backend_cuda.cu and parallel; see bc_accel_narrow_input. */
		half_input = bc_accel_narrow_input(host_input);
		if (!half_input) failed = 1;
	}
#define CREATE_BUFFER(VARIABLE, FLAGS, SIZE, POINTER, LABEL) do { \
	if (!failed) { \
		VARIABLE = clCreateBuffer(state.context, FLAGS, SIZE, POINTER, &error); \
		if (opencl_error(error, LABEL)) failed = 1; \
	} \
} while (0)
	CREATE_BUFFER(first, CL_MEM_READ_WRITE, activation_bytes, NULL,
		"allocating first activation");
	CREATE_BUFFER(second, CL_MEM_READ_WRITE, activation_bytes, NULL,
		"allocating second activation");
	CREATE_BUFFER(weight_buffer, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		packed.count * sizeof(uint16_t), packed.data, "allocating weights");
	/* Sized by OCL_CS, not the logical count. The reduction now folds to the
	   stride, and norm_gelu indexes mean/invstd/scale/bias by it -- sizing
	   these by 15 while the kernel reads 16 is exactly the class of bug that
	   cost a million voxels when the CUDA backend made this move. */
	CREATE_BUFFER(partial_buffer, CL_MEM_READ_WRITE,
		(size_t)OCL_CS * OCL_REDUCE_BLOCKS * 2 * sizeof(float), NULL,
		"allocating reduction workspace");
	CREATE_BUFFER(mean_buffer, CL_MEM_READ_ONLY,
		(size_t)OCL_CS * sizeof(float), NULL, "allocating mean");
	CREATE_BUFFER(invstd_buffer, CL_MEM_READ_ONLY,
		(size_t)OCL_CS * sizeof(float), NULL, "allocating inverse deviation");
	CREATE_BUFFER(scale_buffer, CL_MEM_READ_ONLY,
		(size_t)OCL_CS * sizeof(float), NULL, "allocating affine scale");
	CREATE_BUFFER(bias_buffer, CL_MEM_READ_ONLY,
		(size_t)OCL_CS * sizeof(float), NULL, "allocating affine bias");
#undef CREATE_BUFFER
	/*
	 * Pick the launch shape, tuning first if this device has no cached answer.
	 * Runs BEFORE the input upload because it scribbles on both activation
	 * buffers -- and before the layer loop because a retune rebuilds the
	 * program, which invalidates every kernel handle the loop would use.
	 *
	 * --tune or BRAINCHOPC_TUNE=1 forces a retune; BRAINCHOPC_TUNE=0 pins the
	 * compiled-in shape and skips both the cache and the search.
	 */
	if (!failed) {
		const char *request = getenv("BRAINCHOPC_TUNE");
		int forced = (options && options->force_tuning) ||
			(request && !strcmp(request, "1"));
		int disabled = request && !strcmp(request, "0") && !forced;
		char key[768];
		int values[2] = { OCL_VPT, OCL_LOCAL };
		opencl_cache_key(&state, key, sizeof(key));
		int miss = disabled ? 1 : bc_accel_tune_load(key, values, 2);
		/*
		 * Validate against the search grid before trusting the file. The cache
		 * is plain text a user can edit and bc_accel_tune_load does no range
		 * checking, so a corrupt local size of 0 reaches the NDRange division
		 * and raises SIGFPE -- which contradicts the cache's own contract that
		 * a malformed entry is not an error. CUDA gets this for free by
		 * matching its variant table; OpenCL has to say so.
		 */
		if (!miss) {
			int ok = 0;
			for (int vi = 0; vi < (int)(sizeof(opencl_tune_vpt) /
					sizeof(*opencl_tune_vpt)); vi++)
				for (int li = 0; li < (int)(sizeof(opencl_tune_local) /
						sizeof(*opencl_tune_local)); li++)
					if (values[0] == opencl_tune_vpt[vi] &&
							values[1] == opencl_tune_local[li])
						ok = 1;
			if (!ok) miss = 1;
		}
		/* Choose the shape first, then build it exactly once. The tuner builds
		   its own candidates, so its last one is not necessarily the winner. */
		int vpt = OCL_VPT, local = OCL_LOCAL;
		if (!miss && !forced) {
			vpt = values[0];
			local = values[1];
		} else if (!disabled) {
			if (!opencl_tune(&state, first, second, weight_buffer, &vpt, &local)) {
				int store[2] = { vpt, local };
				if (bc_accel_tune_store(key, store, 2))
					fprintf(stderr, "brainchopc: could not save the tuning "
						"result; it will be measured again next run\n");
			} else {
				fprintf(stderr, "brainchopc: tuning failed; using the "
					"compiled-in convolution shape\n");
				vpt = OCL_VPT;
				local = OCL_LOCAL;
			}
		}
		if (build_program(&state, vpt, local)) failed = 1;
	}

	/* Layer 0 reads this with a stride of ONE, not OCL_CS -- the input is a
	   single channel and only becomes CS-wide once the first convolution has
	   written it, which is why conv3d takes istride separately. */
	if (!failed)
		failed = opencl_error(clEnqueueWriteBuffer(state.queue, second, CL_TRUE, 0,
			OCL_NVOX * sizeof(uint16_t), half_input, 0, NULL, NULL),
			"copying input");

	/* A second walk, over the ORIGINAL fp32 weights. The convolutions index the
	   padded fp16 copy through packed.layer_offset; the affine block is still
	   read from host_weights, so its position is tracked in the source layout. */
	size_t raw_offset = 0;
	size_t global = (OCL_NVOX + state.local - 1) / state.local * state.local;
	size_t conv_items = (OCL_NVOX + state.vpt - 1) / state.vpt;
	size_t conv_global = (conv_items + state.local - 1) / state.local * state.local;
	size_t local = (size_t)state.local;
	for (int layer = 0; layer < spec->layers && !failed; layer++) {
		int input_channels = layer == 0 ? 1 : spec->channels;
		int padded_inputs = layer == 0 ? 1 : OCL_CS;
		int input_stride = layer == 0 ? 1 : OCL_CS;
		cl_mem source = layer ? ((layer & 1) ? first : second) : second;
		cl_mem destination = (layer & 1) ? second : first;
		cl_ulong weight_offset = (cl_ulong)packed.layer_offset[layer];
		int dilation = spec->dilation[layer];
		if (set_argument(state.convolution, 0, sizeof(source), &source) ||
			set_argument(state.convolution, 1, sizeof(destination), &destination) ||
			set_argument(state.convolution, 2, sizeof(weight_buffer), &weight_buffer) ||
			set_argument(state.convolution, 3, sizeof(weight_offset), &weight_offset) ||
			set_argument(state.convolution, 4, sizeof(padded_inputs), &padded_inputs) ||
			set_argument(state.convolution, 5, sizeof(input_stride), &input_stride) ||
			set_argument(state.convolution, 6, sizeof(dilation), &dilation) ||
			set_argument(state.convolution, 7,
				(size_t)27 * padded_inputs * OCL_CS * sizeof(uint16_t), NULL) ||
			opencl_error(clEnqueueNDRangeKernel(state.queue, state.convolution,
				1, NULL, &conv_global, &local, 0, NULL, NULL), "convolution kernel")) {
			failed = 1;
			break;
		}
		raw_offset += (size_t)27 * input_channels * spec->channels;
		const float *layer_scale = NULL, *layer_bias = NULL;
#if BC_MODEL_HAS_AFFINE
		float compact_scale[BC_MODEL_CHAN], compact_bias[BC_MODEL_CHAN];
		for (int channel = 0; channel < BC_MODEL_CHAN; channel++) {
			compact_scale[channel] =
				host_weights[raw_offset + channel * BC_MODEL_CHAN + channel];
			compact_bias[channel] = host_weights[raw_offset +
				BC_MODEL_CHAN * BC_MODEL_CHAN + channel];
		}
		layer_scale = compact_scale;
		layer_bias = compact_bias;
		raw_offset += BC_MODEL_CHAN * BC_MODEL_CHAN + BC_MODEL_CHAN;
#endif
		failed = run_normalize(&state, destination, partial_buffer,
			mean_buffer, invstd_buffer, scale_buffer, bias_buffer,
			layer_scale, layer_bias);
	}
	/* Only read when the model has an affine; MindGrab has none, so the walk
	   still has to happen but its result is unused there. */
	(void)raw_offset;
	if (!failed) {
		cl_mem last = (spec->layers & 1) ? first : second;
		/*
		 * Labels go to the buffer that is NOT the classifier's input. This used
		 * to pass `second` unconditionally and rely on a _Static_assert that the
		 * layer count is odd; deriving it from the same parity as `last` makes
		 * the two provably different at ANY parity, so the restriction on the
		 * model is gone rather than merely asserted. Aliasing them would be a
		 * real race: labels[v] lands at a smaller byte offset than v's own
		 * activation row, so a work-item writing its label can clobber source
		 * data another has not read yet.
		 */
		cl_mem label_buffer = (spec->layers & 1) ? second : first;
		cl_ulong weight_offset = (cl_ulong)packed.classifier_offset;
		int has_bias = spec->classifier_bias;
		if (set_argument(state.classifier, 0, sizeof(last), &last) ||
			set_argument(state.classifier, 1, sizeof(label_buffer), &label_buffer) ||
			set_argument(state.classifier, 2, sizeof(weight_buffer), &weight_buffer) ||
			set_argument(state.classifier, 3, sizeof(weight_offset), &weight_offset) ||
			set_argument(state.classifier, 4, sizeof(spec->channels), &spec->channels) ||
			set_argument(state.classifier, 5, sizeof(spec->classes), &spec->classes) ||
			set_argument(state.classifier, 6, sizeof(has_bias), &has_bias) ||
			opencl_error(clEnqueueNDRangeKernel(state.queue, state.classifier,
				1, NULL, &global, &local, 0, NULL, NULL), "classifier kernel") ||
			opencl_error(clEnqueueReadBuffer(state.queue, label_buffer, CL_TRUE, 0,
				OCL_NVOX * sizeof(float), result, 0, NULL, NULL), "reading labels"))
			failed = 1;
	}
	if (bias_buffer) clReleaseMemObject(bias_buffer);
	if (scale_buffer) clReleaseMemObject(scale_buffer);
	if (invstd_buffer) clReleaseMemObject(invstd_buffer);
	if (mean_buffer) clReleaseMemObject(mean_buffer);
	if (partial_buffer) clReleaseMemObject(partial_buffer);
	if (weight_buffer) clReleaseMemObject(weight_buffer);
	if (second) clReleaseMemObject(second);
	if (first) clReleaseMemObject(first);
	free(half_input);
	bc_free_padded_weights(&packed);
	free(host_weights);
	free(host_input);
	release_state(&state);
	return failed ? 1 : 0;
}

const bc_backend bc_opencl_backend = {
	"opencl",
	opencl_available,
	opencl_segment,
	opencl_adapter_count,
	opencl_adapter_name
};
