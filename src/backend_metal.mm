#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend.h"
#include "cat_lite.h"
#include "model_specs.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BC_DIM BRAINCHOP_MODEL_DIM
#define BC_NVOX ((size_t)BC_DIM*BC_DIM*BC_DIM)

typedef struct {
	uint8_t buffer;
	uint32_t offset;
} bc_metal_arg;

typedef struct {
	uint8_t kernel;
	uint16_t first_arg;
	uint8_t arg_count;
	uint32_t groups[3];
	uint32_t threads[3];
} bc_metal_dispatch;

typedef struct {
	uint32_t offset;
	uint32_t size;
	uint8_t half;
} bc_metal_weight;

/*
 * This model's generated Metal programs -- 124 KB and 175 KB of source and
 * dispatch tables respectively, so compiling only the one that is needed is
 * where most of a per-model executable's size saving comes from.
 *
 * The header name and the symbol prefix inside it are supplied by the Makefile
 * (BC_METAL_HEADER, BC_METAL_SYM, BC_METAL_SYM_UC) because they are properties
 * of a generated artifact, not something this file should enumerate. Pasting
 * them here rather than branching on the model is what keeps adding a model to
 * a Makefile change.
 */
#include BC_METAL_HEADER

#define BC_M_CAT2(a, b) a##b
#define BC_M_CAT(a, b) BC_M_CAT2(a, b)
/* bc_metal_<model>_source, and friends. */
#define BC_M(suffix) BC_M_CAT(BC_M_CAT(bc_metal_, BC_METAL_SYM), suffix)
/* BC_METAL_<MODEL>_ARENA_BYTES, and friends. */
#define BC_M_UC(suffix) BC_M_CAT(BC_M_CAT(BC_METAL_, BC_METAL_SYM_UC), suffix)

static int metal_available(void) {
	@autoreleasepool {
		return MTLCreateSystemDefaultDevice()!=nil;
	}
}

static double quantile_u8(const int64_t *cum, size_t n, double q) {
	double rank=q*(double)(n-1), base=floor(rank), t=rank-base;
	size_t i=(size_t)base;
	int a=0,b=0;
	while(a<255&&cum[a]<=(int64_t)i)a++;
	b=a;
	while(b<255&&cum[b]<=(int64_t)(i+1))b++;
	return (double)a+(b-a)*t;
}

/*
 * The quantile POSITIONS, the denominator guard and the clamp come from the
 * generated descriptor, as they do in the CPU engine and in
 * bc_accel_normalize_input.
 *
 * The quantile FORM does not. quantile_u8 above is always the interpolated
 * form, so a model declaring quantile_mode "rank" -- model16 does -- gets
 * interpolated quantiles here and plain rank selection on the CPU. Metal's
 * linear form also lacks the CPU's `t < 0.5` stable-lerp branch, so even a
 * "linear" model differs slightly. Both predate the per-model split (see
 * `git show origin/main:src/backend_metal.mm`) and are absorbed by the fp16
 * tolerance, which is why they have never shown up as a failure.
 *
 * Do not "fix" this in isolation: it changes Metal output, and the CUDA and
 * OpenCL paths need checking for the same divergence at the same time. See the
 * known-gaps entry in the build notes.
 */
static __fp16 *normalize_input(const unsigned char *src) {
	int64_t hist[256]={0},cum[256],run=0;
	for(size_t i=0;i<BC_NVOX;i++)hist[src[i]]++;
	for(int i=0;i<256;i++){run+=hist[i];cum[i]=run;}
	float lo=(float)quantile_u8(cum,BC_NVOX,BC_MODEL_NORM_Q_LOW);
	float hi=(float)quantile_u8(cum,BC_NVOX,BC_MODEL_NORM_Q_HIGH);
	float den=hi-lo+BC_MODEL_NORM_DENOM_EPS;
	if(!(den>0))den=1;
	__fp16 *out=(__fp16 *)malloc(BC_NVOX*sizeof(*out));
	if(!out)return NULL;
	for(size_t i=0;i<BC_NVOX;i++) {
		float v=((float)src[i]-lo)/den;
#if BC_MODEL_NORM_CLAMP
		if(!(v>0))v=0;
		if(v>1)v=1;
#endif
		out[i]=(__fp16)v;
	}
	return out;
}

/*
 * The manifest must carry one entry per hidden layer plus the classifier.
 * Without this, a short manifest makes the classifier lookup an out-of-bounds
 * read of a static const array, and its garbage .offset is used directly as an
 * arena write pointer.
 */
/* The affine packer advances `placement` THREE times per layer (kernel, affine,
   bias) and then reads the classifier and its bias, so it indexes much further
   than the non-affine one: model16 needs 41 entries, not 14. Asserting only
   NHIDDEN+1 left the packer that reads furthest unguarded. */
_Static_assert(sizeof(BC_M(_weights))/sizeof(BC_M(_weights)[0]) >= (size_t)
	(BC_MODEL_HAS_AFFINE ? 3*BC_MODEL_NHIDDEN+2 : BC_MODEL_NHIDDEN+1+BC_MODEL_CLASSIFIER_BIAS),
	"Metal weight manifest is shorter than the model");

#if !BC_MODEL_HAS_AFFINE
/*
 * The generated kernels may be built for MORE channels than the model has.
 *
 * MindGrab is a 15-channel network exported at 16, because 15 cannot tile into
 * the 8x8x8 operands of simdgroup_multiply_accumulate and BEAM therefore could
 * not use the matrix units at all -- measured, every hidden convolution had
 * zero WMMA against model16's 33-97, which was the whole of MindGrab's 2.9x
 * deficit. The padded channel is inert: the GroupNorm has one group per
 * channel, so a zeroed channel normalises to 0, MindGrab has no affine to shift
 * it, and gelu(0) = 0. Verified bit-identical against the 15-channel model in
 * tinygrad before the export was adopted.
 *
 * The width comes from the generated header rather than BC_MODEL_CHAN, so the
 * two sides cannot silently disagree -- writing 15 channels into a buffer the
 * kernels index as 16 misaligns every later layer and produced 1,745,854 wrong
 * voxels when this packer was left unchanged.
 */
static void pack_model_weights(void *arena, const float *weights) {
	const int packed_chan=BC_M_UC(_CHAN);
	size_t source_offset=0;
	for(int layer=0;layer<BC_MODEL_NHIDDEN;layer++) {
		int cin=layer?BC_MODEL_CHAN:1;
		int packed_cin=layer?packed_chan:1;
		__fp16 *dst=(__fp16 *)((unsigned char *)arena+
			BC_M(_weights)[layer].offset);
		for(int output=0;output<packed_chan;output++)
			for(int input=0;input<packed_cin;input++)
				for(int depth=0;depth<3;depth++)
					for(int height=0;height<3;height++)
						for(int width=0;width<3;width++) {
							if(output>=BC_MODEL_CHAN||input>=cin) { *dst++=(__fp16)0; continue; }
							size_t src=source_offset+
								((((size_t)depth*3+height)*3+width)*cin+input)*
								BC_MODEL_CHAN+output;
							*dst++=(__fp16)weights[src];
						}
		source_offset+=(size_t)27*cin*BC_MODEL_CHAN;
	}
	/*
	 * The classifier is NOT padded, even though the convolutions are.
	 *
	 * SequentialConvArgmax aliases the final conv's weight, and the export
	 * halves the model AFTER constructing it -- so the model's convolutions
	 * become padded fp16 tensors while the classifier keeps its original fp32
	 * buffer at the logical width. Assuming the padded width here wrote 32
	 * floats into a 30-float buffer and produced 35415 wrong voxels.
	 *
	 * So take the width from the manifest rather than from either constant.
	 * `.size` was previously emitted and never read, which is exactly why the
	 * mismatch was silent; reading it makes the two sides impossible to
	 * disagree about.
	 */
	/* Honour .half too, not just .size: dividing a byte count by sizeof(float)
	   when the manifest says the tensor is fp16 yields half the true width and
	   writes floats into a half array -- the same silent-wrong-weights class as
	   the padding bug, and the reason .half is read here rather than assumed. */
	const bc_metal_weight cls=BC_M(_weights)[BC_MODEL_NHIDDEN];
	const size_t cls_elem=cls.half?sizeof(__fp16):sizeof(float);
	const int classifier_chan=(int)(cls.size/cls_elem/BC_MODEL_NCLASS);
	unsigned char *cls_dst=(unsigned char *)arena+cls.offset;
	for(int output=0;output<BC_MODEL_NCLASS;output++)
		for(int input=0;input<classifier_chan;input++) {
			float v=input<BC_MODEL_CHAN?
				weights[source_offset+input*BC_MODEL_NCLASS+output]:0.0f;
			size_t at=(size_t)output*classifier_chan+input;
			if(cls.half) ((__fp16 *)cls_dst)[at]=(__fp16)v;
			else ((float *)cls_dst)[at]=v;
		}
#if BC_MODEL_CLASSIFIER_BIAS
	const bc_metal_weight bias=BC_M(_weights)[BC_MODEL_NHIDDEN+1];
	float *bias_dst=(float *)((unsigned char *)arena+bias.offset);
	for(int output=0;output<BC_MODEL_NCLASS;output++)
		bias_dst[output]=weights[source_offset+BC_MODEL_CHAN*BC_MODEL_NCLASS+output];
#endif
}

#else
static void pack_model_weights(void *arena, const float *weights) {
	size_t source_offset=0;
	int placement=0;
	for(int layer=0;layer<BC_MODEL_NHIDDEN;layer++) {
		int cin=layer?BC_MODEL_CHAN:1;
		__fp16 *dst=(__fp16 *)((unsigned char *)arena+
			BC_M(_weights)[placement++].offset);
		for(int output=0;output<BC_MODEL_CHAN;output++)
			for(int input=0;input<cin;input++)
				for(int depth=0;depth<3;depth++)
					for(int height=0;height<3;height++)
						for(int width=0;width<3;width++) {
							/* TFJS stores x,y,z; tinygrad consumes z,y,x. */
							size_t src=source_offset+
								((((size_t)width*3+height)*3+depth)*cin+input)*
								BC_MODEL_CHAN+output;
							*dst++=(__fp16)weights[src];
						}
		source_offset+=(size_t)27*cin*BC_MODEL_CHAN;

		const float *affine=weights+source_offset;
		__fp16 *scale=(__fp16 *)((unsigned char *)arena+
			BC_M(_weights)[placement++].offset);
		for(int channel=0;channel<BC_MODEL_CHAN;channel++)
			scale[channel]=(__fp16)affine[channel*BC_MODEL_CHAN+channel];
		source_offset+=(size_t)BC_MODEL_CHAN*BC_MODEL_CHAN;

		__fp16 *bias=(__fp16 *)((unsigned char *)arena+
			BC_M(_weights)[placement++].offset);
		for(int channel=0;channel<BC_MODEL_CHAN;channel++)
			bias[channel]=(__fp16)weights[source_offset+channel];
		source_offset+=BC_MODEL_CHAN;
	}
	__fp16 *classifier=(__fp16 *)((unsigned char *)arena+
		BC_M(_weights)[placement++].offset);
	for(int output=0;output<BC_MODEL_NCLASS;output++)
		for(int input=0;input<BC_MODEL_CHAN;input++)
			classifier[output*BC_MODEL_CHAN+input]=
				(__fp16)weights[source_offset+input*BC_MODEL_NCLASS+output];
	source_offset+=(size_t)BC_MODEL_CHAN*BC_MODEL_NCLASS;
	__fp16 *bias=(__fp16 *)((unsigned char *)arena+
		BC_M(_weights)[placement].offset);
	for(int output=0;output<BC_MODEL_NCLASS;output++)
		bias[output]=(__fp16)weights[source_offset+output];
}
#endif

typedef void (*bc_metal_weight_packer)(void *, const float *);

#include "metal_turbo.h"

#if BC_HAS_PVE
/*
 * --pve head: the webgpu_pve.h kernels in MSL. The network runs up to the
 * classifier dispatch; priors then recomputes f32 logits from the post-GELU
 * fp16 activation (channel-planar, the exporter proves which buffer), and the
 * blur and support gate reproduce bc_cat_lite_priors. All of it lives in the
 * arena below the activation, which is dead once the network has finished.
 */
#define BC_MSL_STR2(x) #x
#define BC_MSL_STR(x) BC_MSL_STR2(x)
static const char bc_metal_pve_source[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"constant uint NV=16777216u;\n"
"constant uint CHAN=" BC_MSL_STR(BC_MODEL_CHAN) ";\n"
"constant uint NCLASS=" BC_MSL_STR(BC_MODEL_NCLASS) ";\n"
"constant uint STRIDE=" BC_MSL_STR(BC_M_UC(_CLASSIFIER_INPUT_CHAN_STRIDE)) ";\n"
"constant bool BIAS=" BC_MSL_STR(BC_MODEL_CLASSIFIER_BIAS) "!=0;\n"
"constant float TEMPERATURE=" BC_MSL_STR(BC_PRIOR_TEMPERATURE) ";\n"
"constant float SUPPORT_TEMPERATURE=" BC_MSL_STR(BC_PRIOR_SUPPORT_TEMPERATURE) ";\n"
"constant float SIGMA=" BC_MSL_STR(BC_PVE_BLUR_SIGMA) ";\n"
"constant int RADIUS=int(3.0f*SIGMA+0.5f);\n"
"constant uint MASK[3]={" BC_MSL_STR(BC_PRIOR_GM_MASK) "," BC_MSL_STR(BC_PRIOR_WM_MASK) ","
	BC_MSL_STR(BC_PRIOR_CSF_MASK) "};\n"
/* Classifier as packed for the generated kernels: [class][channel]. Summation
   order matches mn_pve_rows: bias, then channels ascending. */
"kernel void bc_pve_priors(device const half *act [[buffer(0)]],\n"
"  device const half *w [[buffer(1)]], device const half *b [[buffer(2)]],\n"
"  device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
" float a[CHAN]; for(uint c=0;c<CHAN;c++)a[c]=float(act[c*STRIDE+i]);\n"
" float logit[NCLASS]; float maximum=-INFINITY;\n"
" for(uint k=0;k<NCLASS;k++){\n"
"  float sum=BIAS?float(b[k]):0.0f;\n"
"  for(uint c=0;c<CHAN;c++)sum+=a[c]*float(w[k*CHAN+c]);\n"
"  logit[k]=sum; maximum=max(maximum,sum);\n"
" }\n"
" float total=0,support_total=0,background=0; float tissue[3]={0,0,0};\n"
" for(uint k=0;k<NCLASS;k++){\n"
"  float centered=logit[k]-maximum;\n"
"  float v=exp(centered/TEMPERATURE), s=exp(centered/SUPPORT_TEMPERATURE);\n"
"  total+=v; support_total+=s; if(k==0)background=s;\n"
"  for(uint t=0;t<3;t++)if(MASK[t]&(1u<<k))tissue[t]+=v;\n"
" }\n"
" for(uint t=0;t<3;t++)out[t*NV+i]=tissue[t]/total;\n"
" out[3*NV+i]=clamp(1.0f-background/support_total,0.0f,1.0f);\n"
"}\n"
/* cl_blur: out-of-volume taps still count toward the normaliser, as in the
   CPU and WGSL versions. */
"kernel void bc_pve_blur(device const float *src [[buffer(0)]],\n"
"  device float *dst [[buffer(1)]], constant uint &axis [[buffer(2)]],\n"
"  uint2 g [[thread_position_in_grid]]) {\n"
" uint i=g.x, t=g.y; uint stride=axis==0?1u:(axis==1?256u:65536u);\n"
" int coordinate=int((i/stride)%256u); float sum=0,total=0;\n"
" for(int d=-RADIUS;d<=RADIUS;d++){\n"
"  float w=exp(-0.5f*float(d*d)/(SIGMA*SIGMA)); total+=w;\n"
"  int q=coordinate+d; if(q>=0&&q<256)sum+=w*src[t*NV+uint(int(i)+d*int(stride))];\n"
" }\n"
" dst[t*NV+i]=sum/total;\n"
"}\n"
"kernel void bc_pve_support(device const float *blurred [[buffer(0)]],\n"
"  device float *planes [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
" planes[g.y*NV+g.x]=blurred[g.y*NV+g.x]*clamp(planes[3*NV+g.x],0.0f,1.0f);\n"
"}\n";

_Static_assert(BC_MODEL_HAS_AFFINE, "the PVE head reads the affine packer's classifier");
/* Planes A (4) and blur scratch B (3) must sit below the activation. */
_Static_assert(7ULL*BC_NVOX*sizeof(float) <= BC_M_UC(_CLASSIFIER_INPUT_OFFSET),
	"PVE planes would overwrite the classifier input");

/* The manifest facts the head assumes: no packed weight inside the planes,
   and an fp16 [NCLASS][CHAN] classifier with an fp16 bias. The static arrays
   are not constant expressions, so this runs once per call, before any work. */
static int pve_layout_invalid(void) {
	const bc_metal_weight *w=BC_M(_weights), *classifier=&w[3*BC_MODEL_NHIDDEN];
	int bad=!classifier[0].half||!classifier[1].half||
		classifier[0].size!=BC_MODEL_NCLASS*BC_MODEL_CHAN*sizeof(__fp16)||
		classifier[1].size!=BC_MODEL_NCLASS*sizeof(__fp16);
	for(size_t k=0;k<sizeof(BC_M(_weights))/sizeof(w[0]);k++)
		bad|=w[k].offset<7ULL*BC_NVOX*sizeof(float);
	if(bad)fprintf(stderr,"brainchopc: Metal export does not match the PVE head's layout\n");
	return bad;
}

/* Returns 0 on success; `prepare` adds the blur and support gate. */
static int encode_pve_head(id<MTLDevice> metal, id<MTLComputeCommandEncoder> encoder,
		id<MTLBuffer> arena, int prepare) {
	NSError *error=nil;
	MTLCompileOptions *options=[MTLCompileOptions new];
	options.fastMathEnabled=NO;
	id<MTLLibrary> library=[metal newLibraryWithSource:
		[NSString stringWithUTF8String:bc_metal_pve_source] options:options error:&error];
	if(!library) {
		fprintf(stderr,"brainchopc: Metal PVE compilation failed: %s\n",
			error.localizedDescription.UTF8String);
		return 1;
	}
	id<MTLComputePipelineState> pipelines[3];
	const char *names[3]={"bc_pve_priors","bc_pve_blur","bc_pve_support"};
	for(int k=0;k<3;k++) {
		id<MTLFunction> function=[library newFunctionWithName:[NSString stringWithUTF8String:names[k]]];
		pipelines[k]=function?[metal newComputePipelineStateWithFunction:function error:&error]:nil;
		if(!pipelines[k]) {
			fprintf(stderr,"brainchopc: Metal pipeline %s failed: %s\n",names[k],
				error.localizedDescription.UTF8String);
			return 1;
		}
	}
	const NSUInteger plane=BC_NVOX*sizeof(float), a=0, b=4*plane;
	const bc_metal_weight *classifier=&BC_M(_weights)[3*BC_MODEL_NHIDDEN];
	const MTLSize group=MTLSizeMake(256,1,1);
	[encoder setComputePipelineState:pipelines[0]];
	[encoder setBuffer:arena offset:BC_M_UC(_CLASSIFIER_INPUT_OFFSET) atIndex:0];
	[encoder setBuffer:arena offset:classifier[0].offset atIndex:1];
	[encoder setBuffer:arena offset:classifier[1].offset atIndex:2];
	[encoder setBuffer:arena offset:a atIndex:3];
	[encoder dispatchThreadgroups:MTLSizeMake(BC_NVOX/256,1,1) threadsPerThreadgroup:group];
	if(!prepare)return 0;
	/* A -> B -> A -> B along x, y, z; support then writes B * A[3] into A. */
	[encoder setComputePipelineState:pipelines[1]];
	for(uint32_t axis=0;axis<3;axis++) {
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		[encoder setBuffer:arena offset:axis==1?b:a atIndex:0];
		[encoder setBuffer:arena offset:axis==1?a:b atIndex:1];
		[encoder setBytes:&axis length:sizeof(axis) atIndex:2];
		[encoder dispatchThreadgroups:MTLSizeMake(BC_NVOX/256,3,1) threadsPerThreadgroup:group];
	}
	[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
	[encoder setComputePipelineState:pipelines[2]];
	[encoder setBuffer:arena offset:b atIndex:0];
	[encoder setBuffer:arena offset:a atIndex:1];
	[encoder dispatchThreadgroups:MTLSizeMake(BC_NVOX/256,3,1) threadsPerThreadgroup:group];
	return 0;
}
#endif

static int run_generated(id<MTLDevice> metal,
		const unsigned char *conformed, float *result, const float *weights,
		const char *source_text, const char *const *kernel_names,
		size_t kernel_count, const bc_metal_arg *args,
		const bc_metal_dispatch *dispatches, size_t dispatch_count,
		size_t arena_bytes, size_t input_bytes, size_t output_bytes,
		bc_metal_weight_packer pack_weights, float *tissue_planes,
		int *priors_prepared) {
	(void)priors_prepared; /* only read by --pve builds */
	NSTimeInterval start=[NSDate timeIntervalSinceReferenceDate];
	__fp16 *input=normalize_input(conformed);
	if(!input)return 1;

	NSError *error=nil;
	MTLCompileOptions *options=[MTLCompileOptions new];
	options.fastMathEnabled=NO;
	id<MTLLibrary> library=[metal
		newLibraryWithSource:[NSString stringWithUTF8String:source_text]
		options:options error:&error];
	if(!library) {
		fprintf(stderr,"brainchopc: Metal shader compilation failed: %s\n",
			error.localizedDescription.UTF8String);
		free(input);
		return 1;
	}

	NSMutableArray<id<MTLComputePipelineState>> *pipelines=
		[NSMutableArray arrayWithCapacity:kernel_count];
	for(size_t i=0;i<kernel_count;i++) {
		id<MTLFunction> function=[library newFunctionWithName:
			[NSString stringWithUTF8String:kernel_names[i]]];
		id<MTLComputePipelineState> pipeline=function?
			[metal newComputePipelineStateWithFunction:function error:&error]:nil;
		if(!pipeline) {
			fprintf(stderr,"brainchopc: Metal pipeline %s failed: %s\n",
				kernel_names[i],error.localizedDescription.UTF8String);
			free(input);
			return 1;
		}
		[pipelines addObject:pipeline];
	}
	NSTimeInterval compiled=[NSDate timeIntervalSinceReferenceDate];

	id<MTLBuffer> buffers[3]={
		[metal newBufferWithLength:(NSUInteger)arena_bytes
			options:MTLResourceStorageModeShared],
		[metal newBufferWithBytes:input length:(NSUInteger)input_bytes
			options:MTLResourceStorageModeShared],
		/* --pve stops before the classifier and never writes labels. */
		tissue_planes?nil:[metal newBufferWithLength:(NSUInteger)output_bytes
			options:MTLResourceStorageModeShared]
	};
	free(input);
	if(!buffers[0]||!buffers[1]||(!buffers[2]&&!tissue_planes)) {
		fprintf(stderr,"brainchopc: unable to allocate Metal working memory\n");
		return 1;
	}
	pack_weights(buffers[0].contents,weights);

	id<MTLCommandQueue> queue=[metal newCommandQueue];
	id<MTLCommandBuffer> command=[queue commandBuffer];
	id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
	for(size_t i=0;i<dispatch_count;i++) {
		const bc_metal_dispatch *dispatch=&dispatches[i];
		[encoder setComputePipelineState:pipelines[dispatch->kernel]];
		for(unsigned int a=0;a<dispatch->arg_count;a++) {
			const bc_metal_arg *arg=&args[dispatch->first_arg+a];
			[encoder setBuffer:buffers[arg->buffer] offset:arg->offset atIndex:a];
		}
		[encoder dispatchThreadgroups:MTLSizeMake(dispatch->groups[0],
			dispatch->groups[1],dispatch->groups[2])
			threadsPerThreadgroup:MTLSizeMake(dispatch->threads[0],
				dispatch->threads[1],dispatch->threads[2])];
		if(i+1<dispatch_count)
			[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
	}
#if BC_HAS_PVE
	if(tissue_planes) {
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		if(encode_pve_head(metal,encoder,buffers[0],priors_prepared!=NULL)) {
			[encoder endEncoding];
			return 1;
		}
	}
#endif
	[encoder endEncoding];
	[command commit];
	[command waitUntilCompleted];
	if(command.status==MTLCommandBufferStatusError) {
		fprintf(stderr,"brainchopc: Metal execution failed: %s\n",
			command.error.localizedDescription.UTF8String);
		return 1;
	}
	NSTimeInterval finished=[NSDate timeIntervalSinceReferenceDate];
#if BC_HAS_PVE
	if(tissue_planes) {
		/* Planes A sit at arena offset 0 (encode_pve_head). */
		memcpy(tissue_planes,buffers[0].contents,BC_NVOX*BC_PVE_PLANES*sizeof(float));
		if(priors_prepared)*priors_prepared=1;
	} else
#endif
	memcpy(result,buffers[2].contents,output_bytes);
	if(getenv("BRAINCHOPC_METAL_PROFILE")) {
		NSTimeInterval end=[NSDate timeIntervalSinceReferenceDate];
		fprintf(stderr,"brainchopc: Metal compile %.3fs, run %.3fs, read %.3fs\n",
			compiled-start,finished-compiled,end-finished);
	}
	return 0;
}

static int metal_segment(const unsigned char *conformed, float *result,
		const float *weights, const bc_inference_options *options) {
	if(!conformed||!result||!weights)return 1;
	if(options&&(options->roi_enabled||options->class_writer))return 2;
	float *planes=options?options->tissue_planes:NULL;
	size_t dispatch_count=sizeof(BC_M(_dispatches))/sizeof(BC_M(_dispatches)[0]);
#if BC_HAS_PVE
	if(planes) {
		if(pve_layout_invalid())return 1;
		dispatch_count=BC_M_UC(_CLASSIFIER_DISPATCH);
	}
#else
	if(planes)return 2;
#endif
	@autoreleasepool {
		@try {
			id<MTLDevice> metal=MTLCreateSystemDefaultDevice();
			if(!metal)return 1;
			/* The 8-bit path has no tissue-prior head: --pve stays fp16. */
			if(!planes&&!(options&&options->no_turbo))
				return run_turbo(metal,conformed,result,weights);
			return run_generated(metal,conformed,result,weights,
				BC_M(_source),BC_M(_kernel_names),
				sizeof(BC_M(_kernel_names))/sizeof(BC_M(_kernel_names)[0]),
				BC_M(_args),BC_M(_dispatches),dispatch_count,
				BC_M_UC(_ARENA_BYTES),BC_M_UC(_INPUT_BYTES),
				BC_M_UC(_OUTPUT_BYTES),pack_model_weights,planes,
				planes?options->tissue_priors_prepared:NULL);
		} @catch(NSException *exception) {
			fprintf(stderr,"brainchopc: Metal backend error: %s\n",
				exception.reason.UTF8String);
			return 1;
		}
	}
}

const bc_backend bc_metal_backend={
	"metal",
	metal_available,
	metal_segment,
	/* Not wired up. MTLCopyAllDevices could enumerate on an Intel Mac with a
	   discrete GPU, but Apple Silicon has exactly one device, which is the only
	   configuration this backend is built for today. */
	NULL,
	NULL
};
