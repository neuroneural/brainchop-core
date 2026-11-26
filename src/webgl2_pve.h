/* Grouped tissue head and prior preparation using RGBA32F volume textures.
   Network scratch is released first so these replace, rather than accumulate
   on top of, its dead activation set. */
#include "cat_lite.h"
#if BC_HAS_PVE
static const char bc_gl_pve_head[] = BC_GLSL_PREAMBLE
BC_WGL_PER_PLANE(BC_WGL_SAMPLER) BC_GLSL_WFETCH
"uniform int uZ; uniform int uWCls; uniform int uWBias;\n"
"out vec4 outColor;\n"
"float ws(int i){return wf(i>>2)[i&3];}\n"
"void main(){\n"
" ivec3 p=ivec3(ivec2(gl_FragCoord.xy),uZ);\n"
" vec4 cv[" BC_WGL_CS4 "];\n"
BC_WGL_PER_PLANE(BC_WGL_CLSFETCH)
" float logits[18];float maximum=-3.0e38;\n"
" for(int k=0;k<18;k++){\n"
"  float acc=" BC_WGL_STR(BC_MODEL_CLASSIFIER_BIAS) "!=0?ws(uWBias+k):0.0;\n"
"  for(int c=0;c<CHAN;c++)acc+=cv[c>>2][c&3]*ws(uWCls+c*18+k);\n"
"  logits[k]=acc;maximum=max(maximum,acc);\n"
" }\n"
" float total=0.0,st=0.0,bg=0.0;vec3 tissue=vec3(0.0);\n"
" for(int k=0;k<18;k++){\n"
"  float v=exp((logits[k]-maximum)/" BC_WGL_STR(BC_PRIOR_TEMPERATURE) ");\n"
"  float s=exp((logits[k]-maximum)/" BC_WGL_STR(BC_PRIOR_SUPPORT_TEMPERATURE) ");\n"
"  total+=v;st+=s;if(k==0)bg=s;\n"
"  if((" BC_WGL_STR(BC_PRIOR_GM_MASK) " & (1u<<uint(k)))!=0u)tissue.x+=v;\n"
"  if((" BC_WGL_STR(BC_PRIOR_WM_MASK) " & (1u<<uint(k)))!=0u)tissue.y+=v;\n"
"  if((" BC_WGL_STR(BC_PRIOR_MIXED_GWM_MASK) " & (1u<<uint(k)))!=0u){float hv=0.5*v;tissue.x+=hv;tissue.y+=hv;}\n"
"  if((" BC_WGL_STR(BC_PRIOR_CSF_MASK) " & (1u<<uint(k)))!=0u)tissue.z+=v;\n"
" }\n"
" outColor=vec4(tissue/total,clamp(1.0-bg/st,0.0,1.0));\n"
"}\n";
static const char bc_gl_pve_tail[] = BC_GLSL_PREAMBLE
"uniform highp sampler3D source;uniform int uZ;uniform int axis;\n"
"out vec4 outColor;\n"
"void main(){\n"
" ivec3 p=ivec3(ivec2(gl_FragCoord.xy),uZ);vec4 center=texelFetch(source,p,0);\n"
" if(axis==3){outColor=vec4(center.xyz*clamp(center.w,0.0,1.0),clamp(center.w,0.0,1.0));return;}\n"
" vec3 sum=vec3(0.0);float total=0.0;\n"
" for(int d=-2;d<=2;d++){\n"
"  float w=exp(-0.5*float(d*d)/(0.65*0.65));total+=w;ivec3 q=p;q[axis]+=d;\n"
"  if(q[axis]>=0&&q[axis]<256)sum+=w*texelFetch(source,q,0).xyz;\n"
" }\n"
" outColor=vec4(sum/total,center.w);\n"
"}\n";

static int bc_webgl2_prepare_priors(const GLuint activation[BC_PLANES], GLuint weights,
        uint32_t weight_off, uint32_t bias_off, float *planes, int prepare) {
    GLuint head=0,tail=0,texture[2]={0},fbo=0;
    float *slice=malloc(256u*256u*4u*sizeof(float));
    int failed=1;
    if(!slice)goto done;
    head=link_program(bc_glsl_vertex,bc_gl_pve_head,"tissue priors");
    if(prepare)tail=link_program(bc_glsl_vertex,bc_gl_pve_tail,"tissue blur/support");
    if(!head||(prepare&&!tail))goto done;
    glActiveTexture(GL_TEXTURE0);
    texture[0]=make_tex3d(256,256,256,GL_RGBA32F);
    if(prepare)texture[1]=make_tex3d(256,256,256,GL_RGBA32F);
    glGenFramebuffers(1,&fbo);
    glBindFramebuffer(GL_FRAMEBUFFER,fbo);
    const GLenum one=GL_COLOR_ATTACHMENT0;glDrawBuffers(1,&one);
    if(gl_failed("tissue allocation"))goto done;
    glViewport(0,0,256,256);
    for(int stage=0;stage<(prepare?5:1);stage++) {
        GLuint program=stage?tail:head;
        glUseProgram(program);
        if(!stage) {
            for(int i=0;i<BC_PLANES;i++) {
                char name[4]={'s',(char)('0'+i),0,0};
                glActiveTexture(GL_TEXTURE0+i);glBindTexture(GL_TEXTURE_3D,activation[i]);
                bind_unit(program,name,i);
            }
            glActiveTexture(GL_TEXTURE0+BC_UNIT_WTS);glBindTexture(GL_TEXTURE_2D,weights);
            bind_unit(program,"wts",BC_UNIT_WTS);
            glUniform1i(glGetUniformLocation(program,"uWCls"),(GLint)weight_off);
            glUniform1i(glGetUniformLocation(program,"uWBias"),(GLint)bias_off);
        }else {
            glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_3D,texture[(stage-1)%2]);
            bind_unit(program,"source",0);
            glUniform1i(glGetUniformLocation(program,"axis"),stage-1);
        }
        GLint uz=glGetUniformLocation(program,"uZ");
        for(int z=0;z<256;z++) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,texture[stage%2],0,z);
            if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)goto done;
            glUniform1i(uz,z);glDrawArrays(GL_TRIANGLES,0,3);
        }
        if(gl_failed("tissue draw"))goto done;
    }
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    for(int z=0;z<256;z++) {
        glFramebufferTextureLayer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,texture[0],0,z);
        glReadPixels(0,0,256,256,GL_RGBA,GL_FLOAT,slice);
        if(gl_failed("tissue readback"))goto done;
        for(size_t i=0;i<65536;i++)for(size_t t=0;t<4;t++)
            planes[t*BC_NVOX+(size_t)z*65536+i]=slice[4*i+t];
    }
    failed=0;
done:
    if(fbo)glDeleteFramebuffers(1,&fbo);
    glDeleteTextures(2,texture);
    if(head)glDeleteProgram(head);
    if(tail)glDeleteProgram(tail);
    free(slice);return failed;
}
#endif
