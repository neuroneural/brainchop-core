/* GPU tissue preparation; activations remain compact. The two dead network
   buffers alternate as four f32 planes. No full-volume logits are allocated. */
#include "cat_lite.h"
#if BC_HAS_PVE
static const char bc_pve_head[] = BC_WG_ENABLE_F16
"const CHAN:u32=" BC_WG_STR(BC_MODEL_CHAN) "u;\n"
"const CS:u32=" BC_WG_STR(BC_WG_CS) "u;\n"
"const BIAS:bool=" BC_WG_STR(BC_MODEL_CLASSIFIER_BIAS) "!=0;\n"
"const TEMPERATURE:f32=" BC_WG_STR(BC_PRIOR_TEMPERATURE) ";\n"
"const SUPPORT_TEMPERATURE:f32=" BC_WG_STR(BC_PRIOR_SUPPORT_TEMPERATURE) ";\n"
"const GM_MASK:u32=" BC_WG_STR(BC_PRIOR_GM_MASK) ";\n"
"const WM_MASK:u32=" BC_WG_STR(BC_PRIOR_WM_MASK) ";\n"
"const MIXED_GWM_MASK:u32=" BC_WG_STR(BC_PRIOR_MIXED_GWM_MASK) ";\n"
"const CSF_MASK:u32=" BC_WG_STR(BC_PRIOR_CSF_MASK) ";\n"
"struct P { weight_off:u32, bias_off:u32, axis:u32, pad:u32 };\n"
"@group(0) @binding(3) var<uniform> p:P;\n"
"const NV:u32=16777216u;\n"
"@group(0) @binding(0) var<storage,read> input:array<" BC_WG_STORE ">;\n"
"@group(0) @binding(1) var<storage,read> weights:array<" BC_WG_STORE ">;\n"
"@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n"
"@compute @workgroup_size(64)\n"
"fn priors(@builtin(global_invocation_id) gid:vec3<u32>) {\n"
" let i=gid.x+gid.y*16384u;\n"
" if(i>=NV){return;}\n"
" var logits:array<f32,18>;\n"
" var maximum:f32=-3.0e38;\n"
" for(var k=0u;k<18u;k++) {\n"
"  var acc=0.0;\n"
"  if(BIAS){acc=f32(weights[p.bias_off+k]);}\n"
"  for(var c=0u;c<CHAN;c++){acc+=f32(input[i*CS+c])*f32(weights[p.weight_off+c*18u+k]);}\n"
"  logits[k]=acc;maximum=max(maximum,acc);\n"
" }\n"
" var total=0.0;var support_total=0.0;var background=0.0;var tissue=vec3<f32>(0.0);\n"
" for(var k=0u;k<18u;k++) {\n"
"  let v=exp((logits[k]-maximum)/TEMPERATURE);\n"
"  let s=exp((logits[k]-maximum)/SUPPORT_TEMPERATURE);\n"
"  total+=v;support_total+=s;\n"
"  if(k==0u){background=s;}\n"
"  if((GM_MASK&(1u<<k))!=0u){tissue.x+=v;}\n"
"  if((WM_MASK&(1u<<k))!=0u){tissue.y+=v;}\n"
"  if((MIXED_GWM_MASK&(1u<<k))!=0u){let hv=0.5*v;tissue.x+=hv;tissue.y+=hv;}\n"
"  if((CSF_MASK&(1u<<k))!=0u){tissue.z+=v;}\n"
" }\n"
" output[i]=tissue.x/total;output[NV+i]=tissue.y/total;output[2u*NV+i]=tissue.z/total;\n"
" output[3u*NV+i]=clamp(1.0-background/support_total,0.0,1.0);\n"
"}\n";
static const char bc_pve_tail[] =
"struct P { weight_off:u32, bias_off:u32, axis:u32, pad:u32 };\n"
"@group(0) @binding(3) var<uniform> p:P;\n"
"const NV:u32=16777216u;\n"
"@group(0) @binding(0) var<storage,read> input:array<f32>;\n"
"@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n"
"@compute @workgroup_size(64)\n"
"fn blur(@builtin(global_invocation_id) gid:vec3<u32>) {\n"
" let i=gid.x+gid.y*16384u;let tissue=gid.z;\n"
" if(i>=NV || tissue>=3u){return;}\n"
" let stride=select(select(65536u,256u,p.axis==1u),1u,p.axis==0u);\n"
" let coordinate=(i/stride)%256u;\n"
" var sum=0.0;var total=0.0;\n"
" for(var delta=-2;delta<=2;delta++) {\n"
"  let w=exp(-0.5*f32(delta*delta)/(0.65*0.65));total+=w;\n"
"  let q=i32(coordinate)+delta;\n"
"  if(q>=0 && q<256){sum+=w*input[tissue*NV+u32(i32(i)+delta*i32(stride))];}\n"
" }\n"
" output[tissue*NV+i]=sum/total;\n"
" if(tissue==0u){output[3u*NV+i]=input[3u*NV+i];}\n"
"}\n"
"@compute @workgroup_size(64)\n"
"fn support(@builtin(global_invocation_id) gid:vec3<u32>) {\n"
" let i=gid.x+gid.y*16384u;let tissue=gid.z;\n"
" if(i>=NV || tissue>=3u){return;}\n"
" let s=clamp(input[3u*NV+i],0.0,1.0);\n"
" output[tissue*NV+i]=input[tissue*NV+i]*s;\n"
" if(tissue==0u){output[3u*NV+i]=s;}\n"
"}\n";

static int bc_webgpu_prepare_priors(WGPUDevice device, WGPUQueue queue,
        WGPUBuffer activation, WGPUBuffer scratch, WGPUBuffer weights,
        uint32_t weight_off, uint32_t bias_off, int prepare) {
    WGPUShaderModule modules[2]={NULL,NULL};
    WGPUComputePipeline pipelines[3]={NULL,NULL,NULL};
    WGPUBindGroup groups[3]={NULL,NULL,NULL};
    WGPUBindGroupLayout layout=NULL; WGPUPipelineLayout pl=NULL;
    WGPUBuffer uniform=NULL;
    int failed=1;
    const char *sources[2]={bc_pve_head,bc_pve_tail};
    WGPUBindGroupLayoutEntry entries[4]={0};
    for (int i=0;i<4;i++) {
        entries[i].binding=i; entries[i].visibility=WGPUShaderStage_Compute;
        entries[i].buffer.type=i==3?WGPUBufferBindingType_Uniform:
            (i==2?WGPUBufferBindingType_Storage:WGPUBufferBindingType_ReadOnlyStorage);
    }
    WGPUBindGroupLayoutDescriptor ld={0};ld.entryCount=4;ld.entries=entries;
    layout=wgpuDeviceCreateBindGroupLayout(device,&ld);
    WGPUPipelineLayoutDescriptor pd={0};pd.bindGroupLayoutCount=1;pd.bindGroupLayouts=&layout;
    pl=wgpuDeviceCreatePipelineLayout(device,&pd);
    for (int i=0;i<2;i++) {
        WGPUShaderSourceWGSL source={0};source.chain.sType=WGPUSType_ShaderSourceWGSL;source.code=view(sources[i]);
        WGPUShaderModuleDescriptor md={0};md.nextInChain=&source.chain;
        modules[i]=wgpuDeviceCreateShaderModule(device,&md);
    }
    if (!layout||!pl||!modules[0]||!modules[1]) goto done;
    pipelines[0]=make_pipeline(device,pl,modules[0],"priors");
    pipelines[1]=make_pipeline(device,pl,modules[1],"blur");
    pipelines[2]=make_pipeline(device,pl,modules[1],"support");
    uniform=make_buffer(device,16,WGPUBufferUsage_Uniform|WGPUBufferUsage_CopyDst);
    if (!pipelines[0]||!pipelines[1]||!pipelines[2]||!uniform) goto done;
    for (int g=0;g<3;g++) {
        WGPUBindGroupEntry be[4]={0};
        WGPUBuffer bufs[4]={g==1?scratch:activation,weights,g==1?activation:scratch,uniform};
        for(int i=0;i<4;i++){be[i].binding=i;be[i].buffer=bufs[i];be[i].size=wgpuBufferGetSize(bufs[i]);}
        /* Binding only the live plane range avoids a giant float buffer view. */
        if(g!=0)be[0].size=(uint64_t)BC_NVOX*4*sizeof(float);
        be[2].size=(uint64_t)BC_NVOX*4*sizeof(float);
        WGPUBindGroupDescriptor bd={0};bd.layout=layout;bd.entryCount=4;bd.entries=be;
        groups[g]=wgpuDeviceCreateBindGroup(device,&bd);
        if(!groups[g])goto done;
    }
    for(int stage=0;stage<(prepare?5:1);stage++) {
        uint32_t params[4]={weight_off,bias_off,stage?stage-1:0,0};
        wgpuQueueWriteBuffer(queue,uniform,0,params,sizeof(params));
        WGPUCommandEncoder encoder=wgpuDeviceCreateCommandEncoder(device,NULL);
        WGPUComputePassEncoder pass=wgpuCommandEncoderBeginComputePass(encoder,NULL);
        wgpuComputePassEncoderSetPipeline(pass,pipelines[stage==0?0:(stage==4?2:1)]);
        wgpuComputePassEncoderSetBindGroup(pass,0,groups[stage==0?0:(stage%2?1:2)],0,NULL);
        wgpuComputePassEncoderDispatchWorkgroups(pass,256,1024,stage==0?1:3);
        wgpuComputePassEncoderEnd(pass);
        WGPUCommandBuffer command=wgpuCommandEncoderFinish(encoder,NULL);
        wgpuQueueSubmit(queue,1,&command);
        wgpuCommandBufferRelease(command);wgpuComputePassEncoderRelease(pass);wgpuCommandEncoderRelease(encoder);
    }
    failed=0;
done:
    for(int i=0;i<3;i++){wgpuBindGroupRelease(groups[i]);wgpuComputePipelineRelease(pipelines[i]);}
    /* Submitted commands retain their resources; release, do not destroy the
       uniform before completion. The caller waits on the final readback. */
    wgpuBufferRelease(uniform);
    wgpuPipelineLayoutRelease(pl);wgpuBindGroupLayoutRelease(layout);
    for(int i=0;i<2;i++)wgpuShaderModuleRelease(modules[i]);
    return failed;
}
#endif
