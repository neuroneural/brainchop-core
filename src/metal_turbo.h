/*
 * Metal turbo: hand-written 8-bit-activation MeshNet after brainchop-test's V4
 * engine (webgpu_runners/meshnet_fast_v4_runner.js). Included only by
 * backend_metal.mm; the fp16 tinygrad export stays behind --no-turbo and --pve.
 *
 * STORAGE is 8-bit; hidden convs use fp16 matrix operands with fp32
 * accumulation (conv0 accumulates in fp16), statistics are fp32:
 *  - conv output y: u8 per channel, scaled by the min/span of its 64-voxel block
 *    (one simdgroup), stored with the block's sum/sumsq so moments are fused;
 *  - post-GELU activation: u8 per channel, square-law code
 *    a = qs * u|u|, u = (q - ZP) / 16, qs from the channel's GELU range and
 *    folded into the next layer's weights (prep). Linear codes (V4's) cost
 *    0.2-0.7 points of foreground agreement and failed the native gates.
 * The last GroupNorm/GELU is fused into the classifier, unquantized.
 *
 * Every kernel walks the row lists of turbo_cone.h (V4's cone skip: pure
 * background rows copied from one representative, and a backward cone with
 * sampled exterior moments for the last layers).
 *
 * Hidden convolutions use the same SIMD-group matrix layout as the tinygrad
 * export. Results narrow to fp16 before the unchanged block quantization.
 * test-metal gates Dice / agreement (test/nii_dice.py); measurements and
 * rejected alternatives are in docs/audit-history.md.
 */
#ifndef BRAINCHOPC_METAL_TURBO_H
#define BRAINCHOPC_METAL_TURBO_H

#include "chan_stride.h"
#include "pve_constants.h"

#define BC_TS2(x) #x
#define BC_TS(x) BC_TS2(x)

#if BC_MODEL_GELU_TANH_RATIONAL
#define BC_MSL_GELU \
"static inline float gelu(float x){\n" \
" float t=clamp(0.797884583f*(x+0.044715f*(x*x*x)),-9.0f,9.0f), u=t*t;\n" \
" float p=-8.29118133e-14f; p=p*u+5.19263868e-11f; p=p*u-2.00294448e-08f;\n" \
" p=p*u+1.11017944e-05f; p=p*u+0.00309865153f; p=p*u+0.130791619f; p=p*u+0.99999994f;\n" \
" float q=0.000253859733f; q=q*u+0.024473751f; q=q*u+0.464124829f; q=q*u+1.0f;\n" \
" return (0.5f*x)*(1.0f+t*p/q);\n" \
"}\n"
#else
#define BC_MSL_GELU \
"static inline float gelu(float x){\n" \
" float e=clamp(-2.302208198144325f*(x+0.044715f*x*x*x),-126.0f,126.0f);\n" \
" float f=floor(e), r=e-f, p=0.000216128448f; p=p*r+0.00124678648f;\n" \
" p=p*r+0.0096754498f; p=p*r+0.0554852814f; p=p*r+0.240229305f; p=p*r+0.693147044f; p=p*r+1.0f;\n" \
" return x/(1.0f+p*as_type<float>(uint((int(f)+127)<<23)));\n" \
"}\n"
#endif

/* Voxels per lane; a simdgroup's 32 x VOX voxels are one x-block (the y8
   scale and moment granule, the matrix tile) and a threadgroup is one row.
   Other tilings lost (docs/audit-history.md). */
#define BC_MT_VOX 2
#define BC_MT_BLK (32*BC_MT_VOX)

/* First convolution: `TAP` reads voxel `v`'s input. */
#define BC_MSL_CONV(NAME, SRC, W, TAP) \
"kernel void " NAME "(" SRC ", device uchar4 *dst [[buffer(1)]],\n" \
"  device float4 *mom [[buffer(2)]], " W ", constant Layer &L [[buffer(4)]],\n" \
"  device const uint *list [[buffer(6)]], uint g [[threadgroup_position_in_grid]],\n" \
"  uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],\n" \
"  uint sg [[simdgroup_index_in_threadgroup]]) {\n" \
" uint row=simd_broadcast_first(list[L.list_off+g]); /* uniform: scalar tap branches */\n" \
" int y=int(row%256u), z=int(row/256u), x0=int(t*VOX), d=L.dil;\n" \
" half4 a[VOX][CS4]; for(uint v=0;v<VOX;v++) for(uint o=0;o<CS4;o++) a[v][o]=0.0h;\n" \
" uint tap=0;\n" \
" for(int dz=-1;dz<=1;dz++) for(int dy=-1;dy<=1;dy++){\n" \
"  int sz=z+dz*d, sy=y+dy*d; bool ok=sz>=0&&sz<256&&sy>=0&&sy<256;\n" \
"  uint plane=(uint(clamp(sz,0,255))*256u+uint(clamp(sy,0,255)))*256u;\n" \
"  for(int dx=-1;dx<=1;dx++,tap++){\n" \
"   if(!ok) continue;\n" \
TAP \
"  }\n" \
" }\n" \
" epilogue(a,row*256u+uint(x0),row*(256u/BLK)+sg,lane,dst,mom);\n" \
"}\n"

static const char bc_metal_turbo_source[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"constant constexpr uint NV=16777216u, CS=" BC_TS(BC_CHAN_STRIDE) ", CS4=CS/4,\n"
"  CHAN=" BC_TS(BC_MODEL_CHAN) ", NCLASS=" BC_TS(BC_MODEL_NCLASS) ",\n"
"  VOX=" BC_TS(BC_MT_VOX) ", BLK=32u*VOX, NBLK=NV/BLK;\n"
"constant constexpr bool AFFINE=" BC_TS(BC_MODEL_HAS_AFFINE) "!=0,\n"
"  CBIAS=" BC_TS(BC_MODEL_CLASSIFIER_BIAS) "!=0;\n"
/* GELU's minimum is -0.17; the 32 codes below zero reach it at the qs floor. */
"constant constexpr float ZP=32.0f, EPS=1e-5f;\n"
BC_MSL_GELU
/* Rows come from the turbo_cone.h schedule: list[list_off..+count) computed
   (moment weights in rw), list[pure_off..+npure) copied from row rep,
   list[cls_off..+ncls) labelled. */
"struct Layer { int dil; uint w_off, aff_off, bias_off, next_w_off,\n"
"  list_off, count, pure_off, npure, rep, cls_off; };\n"
"constant constexpr uint ROW=256u*CS4;\n"
/* Per channel: block moments for GroupNorm, then the block's u8 code. mom is [c][block]. */
"static inline void epilogue(thread half4 (&a)[VOX][CS4], uint v0, uint blk, uint lane,\n"
"  device uchar4 *dst, device float4 *mom) {\n"
" for(uint o=0;o<CS4;o++){\n"
"  float4 s=0.0f, q=0.0f, mn=3.0e38f, mx=-3.0e38f;\n"
"  for(uint v=0;v<VOX;v++){ float4 x=float4(a[v][o]); s+=x; q+=x*x; mn=min(mn,x); mx=max(mx,x); }\n"
"  s=simd_sum(s); q=simd_sum(q); mn=simd_min(mn);\n"
"  float4 sp=simd_max(mx)-mn, inv=select(float4(0.0f),255.0f/sp,sp>1e-30f);\n"
"  for(uint v=0;v<VOX;v++) dst[(v0+v)*CS4+o]=uchar4(rint((float4(a[v][o])-mn)*inv));\n"
"  if(lane==0) for(uint j=0;j<4;j++) mom[(o*4+j)*NBLK+blk]=float4(s[j],q[j],mn[j],sp[j]);\n"
" }\n"
"}\n"
/* Pre-norm value of channel c at voxel v from its block's u8 code. */
"static inline float pre(uint v, uint c, float code, device const float4 *mom) {\n"
" float4 m=mom[c*NBLK+v/BLK];\n"
" return code*(1.0f/255.0f)*m.w+m.z;\n"
"}\n"
BC_MSL_CONV("conv0", "device const half *in [[buffer(0)]]",
	"device const float *w [[buffer(3)]]",
"   device const float4 *wt=(device const float4 *)(w+L.w_off)+tap*CS4;\n"
"   for(uint v=0;v<VOX;v++){ int sx=x0+int(v)+dx*d; half sv=(sx>=0&&sx<256)?in[plane+uint(sx)]:0.0h;\n"
"    for(uint o=0;o<CS4;o++) a[v][o]=fma(half4(wt[o]),sv,a[v][o]); }\n")
/* Each lane owns adjacent columns of an 8x8 matrix (the export's layout).
   Fully unroll matrix-array indices: the compiler's default was 8x slower. */
"kernel void conv(device const uchar2 *src [[buffer(0)]], device uchar4 *dst [[buffer(1)]],\n"
" device float4 *mom [[buffer(2)]], device const half2 *w [[buffer(3)]],\n"
" constant Layer &L [[buffer(4)]], device const uint *list [[buffer(6)]],\n"
" uint g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],\n"
" uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {\n"
" uint row=simd_broadcast_first(list[L.list_off+g]);\n"
" int y=int(row%256u), z=int(row/256u), d=L.dil;\n"
" uint mr=(lane>>1&3u)|(lane>>2&4u), mc=(lane&1u)*2u+(lane>>3&1u)*4u;\n"
" simdgroup_float8x8 acc[BLK/8][CS/8];\n"
" _Pragma(\"clang loop unroll(full)\") for(uint v=0;v<BLK/8;v++) _Pragma(\"clang loop unroll(full)\") for(uint o=0;o<CS/8;o++) acc[v][o]=make_filled_simdgroup_matrix<float,8,8>(0.0f);\n"
" uint tap=0;\n"
" for(int dz=-1;dz<=1;dz++) for(int dy=-1;dy<=1;dy++){\n"
"  int sz=z+dz*d, sy=y+dy*d; bool ok=sz>=0&&sz<256&&sy>=0&&sy<256;\n"
"  uint plane=(uint(clamp(sz,0,255))*256u+uint(clamp(sy,0,255)))*256u;\n"
"  for(int dx=-1;dx<=1;dx++,tap++){\n"
"   if(!ok) continue;\n"
"   _Pragma(\"clang loop unroll(full)\") for(uint i=0;i<CS;i+=8){\n"
"    simdgroup_half8x8 wt[CS/8];\n"
"    _Pragma(\"clang loop unroll(full)\") for(uint o=0;o<CS/8;o++){\n"
"     half2 h=w[((tap*CS+i+mr)*CS+o*8+mc)/2];\n"
"     wt[o].thread_elements()[0]=h.x; wt[o].thread_elements()[1]=h.y;\n"
"    }\n"
"    _Pragma(\"clang loop unroll(full)\") for(uint v=0;v<BLK/8;v++){\n"
"     int sx=int(sg*BLK+v*8u+mr)+dx*d;\n"
"     half2 u=fma(half2(src[((plane+uint(clamp(sx,0,255)))*CS+i+mc)/2]),0.0625h,-half(ZP/16.0f))*half(uint(sx)<256u);\n"
"     half2 h=u*abs(u);\n"
"     simdgroup_half8x8 a;\n"
"     a.thread_elements()[0]=h.x; a.thread_elements()[1]=h.y;\n"
"     _Pragma(\"clang loop unroll(full)\") for(uint o=0;o<CS/8;o++) simdgroup_multiply_accumulate(acc[v][o],a,wt[o],acc[v][o]);\n"
"    }\n"
"   }\n"
"  }\n"
" }\n"
" threadgroup half out[256*CS];\n"
" _Pragma(\"clang loop unroll(full)\") for(uint v=0;v<BLK/8;v++) _Pragma(\"clang loop unroll(full)\") for(uint o=0;o<CS/8;o++){\n"
"  uint idx=(sg*BLK+v*8u+mr)*CS+o*8u+mc;\n"
"  out[idx]=half(acc[v][o].thread_elements()[0]); out[idx+1]=half(acc[v][o].thread_elements()[1]);\n"
" }\n"
/* threadgroup_barrier (not simdgroup_barrier): on Apple7 (M1),
   simdgroup_barrier(mem_threadgroup) did not reliably order the half stores
   before the half4 loads in epilogue, causing run-to-run non-determinism
   (MindMap Dice fluctuating 0.913-0.992). threadgroup_barrier is bit-exact. */
" threadgroup_barrier(mem_flags::mem_threadgroup);\n"
" half4 a[VOX][CS4];\n"
" for(uint v=0;v<VOX;v++) for(uint o=0;o<CS4;o++) a[v][o]=((threadgroup half4 *)out)[(t*VOX+v)*CS4+o];\n"
" epilogue(a,row*256u+t*VOX,row*(256u/BLK)+sg,lane,dst,mom);\n"
"}\n"
/* One threadgroup per channel: fold the block moments, derive the
   GroupNorm scale/shift and the u8 activation scale qs.
   stats[c] = (scale, shift, 1/qs, qs). */
"kernel void finish(device const float4 *mom [[buffer(0)]], device float4 *stats [[buffer(1)]],\n"
"  device const float *w [[buffer(2)]], constant Layer &L [[buffer(3)]],\n"
"  device const uint *list [[buffer(6)]], device const float *rw [[buffer(7)]],\n"
"  uint c [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],\n"
"  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {\n"
" threadgroup float4 sh[32];\n"
" float s=0.0f, q=0.0f, mn=3.0e38f, mx=-3.0e38f;\n"
" for(uint i=t;i<L.count;i+=1024u){ uint b0=c*NBLK+list[L.list_off+i]*(256u/BLK); float wt=rw[L.list_off+i];\n"
"  for(uint b=0;b<256u/BLK;b++){ float4 m=mom[b0+b]; s+=wt*m.x; q+=wt*m.y; mn=min(mn,m.z); mx=max(mx,m.z+m.w); } }\n"
" s=simd_sum(s); q=simd_sum(q); mn=simd_min(mn); mx=simd_max(mx);\n"
" if(lane==0) sh[sg]=float4(s,q,mn,mx);\n"
" threadgroup_barrier(mem_flags::mem_threadgroup);\n"
" if(t!=0) return;\n"
" float4 r=sh[0];\n"
" for(uint k=1;k<32;k++){ r.x+=sh[k].x; r.y+=sh[k].y; r.z=min(r.z,sh[k].z); r.w=max(r.w,sh[k].w); }\n"

" float mean=r.x/float(NV), var=max(r.y/float(NV)-mean*mean,0.0f), inv=rsqrt(var+EPS);\n"
" float gamma=1.0f, beta=0.0f;\n"
" if(AFFINE){ gamma=w[L.aff_off+c]; beta=w[L.bias_off+c]; }\n"
" float scale=inv*gamma, shift=beta-mean*scale;\n"
" float vmax=max(0.25f,max(gelu(r.z*scale+shift),gelu(r.w*scale+shift)));\n"
" float qs=c<CHAN?max(vmax/(223.0f*223.0f),0.1705f/(ZP*ZP)):1.0f;\n"
" stats[c]=float4(scale,shift,1.0f/qs,qs);\n"
"}\n"
/* Next layer's weights times its input channel's qs (x256 for the /16 decode). */
"kernel void prep(device const float *w [[buffer(0)]], device const float4 *stats [[buffer(1)]],\n"
"  device half *weff [[buffer(2)]], constant Layer &L [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
" if(i<27u*CS*CS) weff[i]=half(w[L.next_w_off+i]*stats[(i/CS)%CS].w*256.0f);\n"
"}\n"
"kernel void norm(device uchar4 *act [[buffer(0)]], device const float4 *mom [[buffer(1)]],\n"
"  device const float4 *stats [[buffer(2)]], constant Layer &L [[buffer(3)]],\n"
"  device const uint *list [[buffer(6)]], uint t [[thread_position_in_grid]]) {\n"
" uint i=list[L.list_off+t/ROW]*ROW+t%ROW, v=i/CS4, c0=(i%CS4)*4u; float4 q=float4(act[i]), r;\n"
" for(uint j=0;j<4;j++){ float4 st=stats[c0+j];\n"
"  float a=gelu(pre(v,c0+j,q[j],mom)*st.x+st.y)*st.z;\n"
"  r[j]=clamp(rint(sign(a)*sqrt(abs(a))+ZP),0.0f,255.0f); }\n"
" act[i]=uchar4(r);\n"
"}\n"
"kernel void fill(device uchar4 *act [[buffer(0)]], constant Layer &L [[buffer(3)]],\n"
"  device const uint *list [[buffer(6)]], uint t [[thread_position_in_grid]]) {\n"
" act[list[L.pure_off+t/ROW]*ROW+t%ROW]=act[L.rep*ROW+t%ROW];\n"
"}\n"
"kernel void fill_labels(device uchar *labels [[buffer(5)]], constant Layer &L [[buffer(3)]],\n"
"  device const uint *list [[buffer(6)]], uint t [[thread_position_in_grid]]) {\n"
" labels[list[L.pure_off+t/256u]*256u+t%256u]=labels[L.rep*256u+t%256u];\n"
"}\n"
/* Last GroupNorm + GELU, unquantized, then 1x1 conv and argmax (first max). */
"kernel void classify(device const uchar4 *y8 [[buffer(0)]], device const float4 *mom [[buffer(1)]],\n"
"  device const float4 *stats [[buffer(2)]], constant Layer &L [[buffer(3)]],\n"
"  device const float *w [[buffer(4)]], device uchar *labels [[buffer(5)]],\n"
"  device const uint *list [[buffer(6)]], uint t [[thread_position_in_grid]]) {\n"
" uint v=list[L.cls_off+t/256u]*256u+t%256u;\n"
" float a[CS];\n"
" for(uint o=0;o<CS4;o++){ float4 q=float4(y8[v*CS4+o]);\n"
"  for(uint j=0;j<4;j++){ uint c=o*4+j; float4 st=stats[c]; a[c]=gelu(pre(v,c,q[j],mom)*st.x+st.y); } }\n"
" uint best=0; float bv=-3.0e38f;\n"
" for(uint k=0;k<NCLASS;k++){\n"
"  float acc=CBIAS?w[L.aff_off+k]:0.0f;\n"
"  for(uint c=0;c<CHAN;c++) acc+=a[c]*w[L.next_w_off+c*NCLASS+k];\n"
"  if(acc>bv){bv=acc;best=k;}\n"
" }\n"
" labels[v]=uchar(best);\n"
"}\n"
#if BC_HAS_PVE
/* Grouped-softmax tissue priors directly from Layer-12 turbo y8 output,
   followed by separable 3-axis Gaussian blur and support gate in compact
   turbo buffers (~1.2 GB vs ~4.0 GB FP16 engine). */
"constant constexpr float PVE_TEMP=" BC_TS(BC_PRIOR_TEMPERATURE) ";\n"
"constant constexpr float PVE_SUP_TEMP=" BC_TS(BC_PRIOR_SUPPORT_TEMPERATURE) ";\n"
"constant constexpr float PVE_SIGMA=" BC_TS(BC_PVE_BLUR_SIGMA) ";\n"
"constant constexpr int PVE_RADIUS=int(3.0f*PVE_SIGMA+0.5f);\n"
"constant constexpr uint PVE_MASK[3]={" BC_TS(BC_PRIOR_GM_MASK) "," BC_TS(BC_PRIOR_WM_MASK) ","
	BC_TS(BC_PRIOR_CSF_MASK) "};\n"
"constant constexpr uint PVE_MIXED=" BC_TS(BC_PRIOR_MIXED_GWM_MASK) ";\n"
"kernel void priors_turbo(device const uchar4 *y8 [[buffer(0)]], device const float4 *mom [[buffer(1)]],\n"
"  device const float4 *stats [[buffer(2)]], constant Layer &L [[buffer(3)]],\n"
"  device const float *w [[buffer(4)]], device float *planes [[buffer(5)]],\n"
"  device const uint *list [[buffer(6)]], uint t [[thread_position_in_grid]]) {\n"
" uint v=list[L.cls_off+t/256u]*256u+t%256u;\n"
" float a[CS];\n"
" for(uint o=0;o<CS4;o++){ float4 q=float4(y8[v*CS4+o]);\n"
"  for(uint j=0;j<4;j++){ uint c=o*4+j; float4 st=stats[c]; a[c]=gelu(pre(v,c,q[j],mom)*st.x+st.y); } }\n"
" float logit[NCLASS]; float mx=-3.0e38f;\n"
" for(uint k=0;k<NCLASS;k++){\n"
"  float acc=CBIAS?w[L.aff_off+k]:0.0f;\n"
"  for(uint c=0;c<CHAN;c++) acc+=a[c]*w[L.next_w_off+c*NCLASS+k];\n"
"  logit[k]=acc; mx=max(mx,acc);\n"
" }\n"
" float total=0.0f, sup_total=0.0f, bg=0.0f; float tissue[3]={0.0f,0.0f,0.0f};\n"
" for(uint k=0;k<NCLASS;k++){\n"
"  float d=logit[k]-mx;\n"
"  float pv=exp(d/PVE_TEMP), sv=exp(d/PVE_SUP_TEMP);\n"
"  total+=pv; sup_total+=sv; if(k==0) bg=sv;\n"
"  for(uint p=0;p<3;p++) if(PVE_MASK[p]&(1u<<k)) tissue[p]+=pv;\n"
"  if(PVE_MIXED&(1u<<k)){ float hv=0.5f*pv; tissue[0]+=hv; tissue[1]+=hv; }\n"
" }\n"
" for(uint p=0;p<3;p++) planes[p*NV+v]=tissue[p]/total;\n"
" planes[3u*NV+v]=clamp(1.0f-bg/sup_total,0.0f,1.0f);\n"
"}\n"
"kernel void fill_priors(device float *planes [[buffer(5)]], constant Layer &L [[buffer(3)]],\n"
"  device const uint *list [[buffer(6)]], uint t [[thread_position_in_grid]]) {\n"
" uint dst=list[L.pure_off+t/256u]*256u+t%256u, src=L.rep*256u+t%256u;\n"
" for(uint p=0;p<4;p++) planes[p*NV+dst]=planes[p*NV+src];\n"
"}\n"
"kernel void pve_blur_turbo(device const float *src [[buffer(0)]],\n"
"  device float *dst [[buffer(1)]], constant uint &axis [[buffer(2)]],\n"
"  uint2 g [[thread_position_in_grid]]) {\n"
" uint i=g.x, p=g.y; uint stride=axis==0?1u:(axis==1?256u:65536u);\n"
" int coord=int((i/stride)%256u); float sum=0.0f, total=0.0f;\n"
" for(int d=-PVE_RADIUS;d<=PVE_RADIUS;d++){\n"
"  float wt=exp(-0.5f*float(d*d)/(PVE_SIGMA*PVE_SIGMA)); total+=wt;\n"
"  int q=coord+d; if(q>=0&&q<256) sum+=wt*src[p*NV+uint(int(i)+d*int(stride))];\n"
" }\n"
" dst[p*NV+i]=sum/total;\n"
"}\n"
"kernel void pve_support_turbo(device const float *blurred [[buffer(0)]],\n"
"  device float *planes [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
" planes[g.y*NV+g.x]=blurred[g.y*NV+g.x]*clamp(planes[3u*NV+g.x],0.0f,1.0f);\n"
"}\n"
#endif
;

typedef float bc_store;
#define BC_STORE(x) (x)
#define BC_CS BC_CHAN_STRIDE
#include "accel_pack_weights.h"

typedef struct {
	int32_t dil;
	uint32_t w_off, aff_off, bias_off, next_w_off, list_off, count, pure_off, npure, rep, cls_off;
} bc_turbo_layer;

#include "turbo_cone.h"

static int run_turbo(id<MTLDevice> metal, const unsigned char *conformed, float *result,
		const float *weights, float *tissue_planes, int *priors_prepared) {
	enum { CS=BC_CHAN_STRIDE, NL=BC_MODEL_NHIDDEN };
	static const int dilation[]=BC_MODEL_DILATION;
	(void)priors_prepared;
	NSTimeInterval start=[NSDate timeIntervalSinceReferenceDate];
	bc_layer_offsets off[NL];
	uint32_t cls_off,cls_bias_off;
	size_t wcount;
	float *packed=bc_pack_weights(weights,off,&cls_off,&cls_bias_off,&wcount,0);
	__fp16 *input=normalize_input(conformed);
	bc_tc_plan plan;
	if(!packed||!input||bc_tc_make(conformed,dilation,&plan)) { free(packed); free(input); return 1; }

	NSError *error=nil;
	/* Fast math (the default) is 3-5% faster; the kernels therefore use finite
	   min/max sentinels, never INFINITY. */
	MTLCompileOptions *options=[MTLCompileOptions new];
	id<MTLLibrary> library=[metal newLibraryWithSource:
		[NSString stringWithUTF8String:bc_metal_turbo_source] options:options error:&error];
	enum {
		K_CONV0, K_CONV, K_FINISH, K_PREP, K_NORM, K_CLASSIFY, K_FILL, K_FILL_LABELS,
#if BC_HAS_PVE
		K_PRIORS, K_FILL_PRIORS, K_PVE_BLUR, K_PVE_SUPPORT,
#endif
		K_N
	};
	const char *names[K_N]={
		"conv0","conv","finish","prep","norm","classify","fill","fill_labels"
#if BC_HAS_PVE
		,"priors_turbo","fill_priors","pve_blur_turbo","pve_support_turbo"
#endif
	};
	id<MTLComputePipelineState> pipe[K_N];
	for(int k=0;k<K_N&&library;k++) {
		id<MTLFunction> f=[library newFunctionWithName:[NSString stringWithUTF8String:names[k]]];
		pipe[k]=f?[metal newComputePipelineStateWithFunction:f error:&error]:nil;
		/* The epilogue, finish and matrix lane layout assume 32-wide simdgroups. */
		if(!pipe[k]||pipe[k].maxTotalThreadsPerThreadgroup<(k==K_FINISH?1024u:256u)||
				pipe[k].threadExecutionWidth!=32) library=nil;
	}
	if(!library) {
		fprintf(stderr,"brainchop-core: Metal turbo pipeline failed: %s\n",
			error?error.localizedDescription.UTF8String:"missing kernel or unsupported threadgroup/SIMD width");
		free(packed); free(input); bc_tc_free(&plan);
		return 1;
	}
	NSTimeInterval compiled=[NSDate timeIntervalSinceReferenceDate];

	const MTLResourceOptions shared=MTLResourceStorageModeShared, priv=MTLResourceStorageModePrivate;
	id<MTLBuffer> in=[metal newBufferWithBytes:input length:BC_NVOX*sizeof(__fp16) options:shared];
	id<MTLBuffer> wbuf=[metal newBufferWithBytes:packed length:wcount*sizeof(float) options:shared];
	id<MTLBuffer> lbuf=[metal newBufferWithBytes:plan.list length:plan.n*sizeof(uint32_t) options:shared];
	id<MTLBuffer> rwbuf=[metal newBufferWithBytes:plan.weight length:plan.n*sizeof(float) options:shared];
	free(input); free(packed);
	id<MTLBuffer> act[2]={[metal newBufferWithLength:BC_NVOX*CS options:priv],
		[metal newBufferWithLength:BC_NVOX*CS options:priv]};
	id<MTLBuffer> mom=[metal newBufferWithLength:BC_NVOX/BC_MT_BLK*CS*16 options:priv];
	id<MTLBuffer> stats=[metal newBufferWithLength:CS*16 options:priv];
	id<MTLBuffer> weff=[metal newBufferWithLength:27*CS*CS*2 options:priv];
	NSUInteger out_bytes=tissue_planes?(NSUInteger)BC_NVOX*BC_PRIOR_PLANES*sizeof(float):(NSUInteger)BC_NVOX;
	id<MTLBuffer> labels=[metal newBufferWithLength:out_bytes options:shared];
	if(!in||!wbuf||!lbuf||!rwbuf||!act[0]||!act[1]||!mom||!stats||!weff||!labels) {
		fprintf(stderr,"brainchop-core: unable to allocate Metal turbo memory\n");
		bc_tc_free(&plan);
		return 1;
	}

	id<MTLCommandQueue> queue=[metal newCommandQueue];
	id<MTLCommandBuffer> command=[queue commandBuffer];
	/* Rows the cone leaves unlabelled are background (0 labels or 0 priors). */
	id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
	[blit fillBuffer:labels range:NSMakeRange(0,out_bytes) value:0];
	[blit endEncoding];
	id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
	/* BRAINCHOPC_METAL_PROFILE=2: one command buffer per kernel, GPU time per kind. */
	const char *prof=getenv("BRAINCHOPC_METAL_PROFILE");
	const int split=prof&&!strcmp(prof,"2");
	double gpu[K_N]={0};
	int last=-1, failed=0;
	auto stage=[&](int k) {
		if(split&&last>=0) {
			[e endEncoding]; [command commit]; [command waitUntilCompleted];
			failed|=command.status==MTLCommandBufferStatusError;
			gpu[last]+=command.GPUEndTime-command.GPUStartTime;
			command=[queue commandBuffer]; e=[command computeCommandEncoder];
		}
		last=k;
		[e setComputePipelineState:pipe[k]];
		[e setBuffer:lbuf offset:0 atIndex:6];
		[e setBuffer:rwbuf offset:0 atIndex:7];
	};
	const MTLSize row=MTLSizeMake(256/BC_MT_VOX,1,1), g256=MTLSizeMake(256,1,1);
	const NSUInteger rowq=CS/4; /* 256-thread threadgroups per row of uchar4 */
	for(int l=0;l<NL;l++) {
		const bc_tc_layer *t=&plan.layer[l];
		bc_turbo_layer layer={dilation[l],off[l].kernel_off,off[l].affine_off,off[l].bias_off,
			l+1<NL?off[l+1].kernel_off:0,t->list_off,t->count,t->pure_off,t->npure,t->rep,plan.cls_off},
			*p=&layer;
		id<MTLBuffer> dst=act[l&1];
		stage(l?K_CONV:K_CONV0);
		[e setBuffer:l?act[(l+1)&1]:in offset:0 atIndex:0];
		[e setBuffer:dst offset:0 atIndex:1];
		[e setBuffer:mom offset:0 atIndex:2];
		[e setBuffer:l?weff:wbuf offset:0 atIndex:3];
		[e setBytes:p length:sizeof *p atIndex:4];
		[e dispatchThreadgroups:MTLSizeMake(t->count,1,1) threadsPerThreadgroup:row];
		[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
		stage(K_FINISH);
		[e setBuffer:mom offset:0 atIndex:0];
		[e setBuffer:stats offset:0 atIndex:1];
		[e setBuffer:wbuf offset:0 atIndex:2];
		[e setBytes:p length:sizeof *p atIndex:3];
		[e dispatchThreadgroups:MTLSizeMake(CS,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
		[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
		if(l+1<NL) {
			stage(K_PREP);
			[e setBuffer:wbuf offset:0 atIndex:0];
			[e setBuffer:stats offset:0 atIndex:1];
			[e setBuffer:weff offset:0 atIndex:2];
			[e setBytes:p length:sizeof *p atIndex:3];
			[e dispatchThreadgroups:MTLSizeMake((27*CS*CS+255)/256,1,1) threadsPerThreadgroup:g256];
			stage(K_NORM);
			[e setBuffer:dst offset:0 atIndex:0];
			[e setBuffer:mom offset:0 atIndex:1];
			[e setBuffer:stats offset:0 atIndex:2];
			[e setBytes:p length:sizeof *p atIndex:3];
			[e dispatchThreadgroups:MTLSizeMake(t->count*rowq,1,1) threadsPerThreadgroup:g256];
			if(t->npure) {
				[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
				stage(K_FILL);
				[e setBuffer:dst offset:0 atIndex:0];
				[e setBytes:p length:sizeof *p atIndex:3];
				[e dispatchThreadgroups:MTLSizeMake(t->npure*rowq,1,1) threadsPerThreadgroup:g256];
			}
		} else {
			/* Classifier offsets ride in the spare fields. */
			bc_turbo_layer c=*p;
			c.next_w_off=cls_off;
			c.aff_off=cls_bias_off;
#if BC_HAS_PVE
			if(tissue_planes) {
				/* Dead activation buffer holds 3-plane blur scratch (201 MB <= 403 MB). */
				_Static_assert((size_t)BC_NVOX * CS >= 3ULL * BC_NVOX * sizeof(float),
					"inactive turbo activation must hold 3 float blur planes");
				id<MTLBuffer> blur_scratch=act[(l+1)&1];
				stage(K_PRIORS);
				[e setBuffer:dst offset:0 atIndex:0];
				[e setBuffer:mom offset:0 atIndex:1];
				[e setBuffer:stats offset:0 atIndex:2];
				[e setBytes:&c length:sizeof c atIndex:3];
				[e setBuffer:wbuf offset:0 atIndex:4];
				[e setBuffer:labels offset:0 atIndex:5];
				[e dispatchThreadgroups:MTLSizeMake(plan.ncls,1,1) threadsPerThreadgroup:g256];
				if(t->npure) {
					[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
					stage(K_FILL_PRIORS);
					[e setBytes:&c length:sizeof c atIndex:3];
					[e setBuffer:labels offset:0 atIndex:5];
					[e dispatchThreadgroups:MTLSizeMake(t->npure,1,1) threadsPerThreadgroup:g256];
				}
				if(priors_prepared) {
					for(uint32_t axis=0;axis<3;axis++) {
						[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
						stage(K_PVE_BLUR);
						[e setBuffer:axis==1?blur_scratch:labels offset:0 atIndex:0];
						[e setBuffer:axis==1?labels:blur_scratch offset:0 atIndex:1];
						[e setBytes:&axis length:sizeof(axis) atIndex:2];
						[e dispatchThreadgroups:MTLSizeMake(BC_NVOX/256,3,1) threadsPerThreadgroup:g256];
					}
					[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
					stage(K_PVE_SUPPORT);
					[e setBuffer:blur_scratch offset:0 atIndex:0];
					[e setBuffer:labels offset:0 atIndex:1];
					[e dispatchThreadgroups:MTLSizeMake(BC_NVOX/256,3,1) threadsPerThreadgroup:g256];
				}
			} else
#endif
			{
				stage(K_CLASSIFY);
				[e setBuffer:dst offset:0 atIndex:0];
				[e setBuffer:mom offset:0 atIndex:1];
				[e setBuffer:stats offset:0 atIndex:2];
				[e setBytes:&c length:sizeof c atIndex:3];
				[e setBuffer:wbuf offset:0 atIndex:4];
				[e setBuffer:labels offset:0 atIndex:5];
				[e dispatchThreadgroups:MTLSizeMake(plan.ncls,1,1) threadsPerThreadgroup:g256];
				if(t->npure) {
					[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
					stage(K_FILL_LABELS);
					[e setBytes:&c length:sizeof c atIndex:3];
					[e setBuffer:labels offset:0 atIndex:5];
					[e dispatchThreadgroups:MTLSizeMake(t->npure,1,1) threadsPerThreadgroup:g256];
				}
			}
		}
		[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
	}
	[e endEncoding];
	[command commit];
	[command waitUntilCompleted];
	if(split) {
		gpu[last]+=command.GPUEndTime-command.GPUStartTime;
		for(int k=0;k<K_N;k++) fprintf(stderr,"brainchop-core: turbo %-14s %7.1f ms\n",names[k],gpu[k]*1e3);
	}
	if(failed||command.status==MTLCommandBufferStatusError) {
		fprintf(stderr,"brainchop-core: Metal turbo execution failed: %s\n",
			command.error?command.error.localizedDescription.UTF8String:"an earlier command buffer failed");
		bc_tc_free(&plan);
		return 1;
	}
	NSTimeInterval finished=[NSDate timeIntervalSinceReferenceDate];
	size_t work=0;
	for(int l=0;l<NL;l++) work+=plan.layer[l].count;
	bc_tc_free(&plan);
#if BC_HAS_PVE
	if(tissue_planes) {
		memcpy(tissue_planes,labels.contents,(size_t)BC_NVOX*BC_PRIOR_PLANES*sizeof(float));
		if(priors_prepared)*priors_prepared=1;
	} else
#endif
	{
		const uint8_t *lab=(const uint8_t *)labels.contents;
		for(size_t i=0;i<BC_NVOX;i++) result[i]=lab[i];
	}
	if(prof)
		fprintf(stderr,"brainchop-core: Metal turbo plan+compile %.3fs, run %.3fs, read %.3fs, "
			"%.0f%% of rows computed\n",compiled-start,finished-compiled,
			[NSDate timeIntervalSinceReferenceDate]-finished,100.0*work/(NL*65536.0));
	return 0;
}

#endif
