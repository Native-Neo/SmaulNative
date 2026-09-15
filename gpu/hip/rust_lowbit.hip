#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ float fp2(uint8_t q){q&=3; return q==0?-1.0f:q==2?1.0f:0.0f;}
__device__ float fp4(uint8_t q){q&=15; switch(q){case 0:return -2.0f;case 1:return -1.0f;case 2:return -0.5f;case 3:return -0.25f;case 5:return 0.25f;case 6:return 0.5f;case 7:return 1.0f;case 8:return 2.0f;default:return 0.0f;}}
__global__ void packed_linear_kernel(const float*x,const uint8_t*w,const float*s,float*y,int batch,int out_f,int in_f,int bits){int o=blockIdx.x*blockDim.x+threadIdx.x;int n=blockIdx.y;if(o>=out_f||n>=batch)return;int rb=(in_f+(8/bits)-1)/(8/bits);const uint8_t*row=w+(size_t)o*rb;float sum=0.0f;for(int k=0;k<in_f;k++){uint8_t q=bits==2?((row[k>>2]>>(6-((k&3)<<1)))&3):((row[k>>1]>>(4-((k&1)<<2)))&15);sum+=x[(size_t)n*in_f+k]*s[o]*(bits==2?fp2(q):fp4(q));}y[(size_t)n*out_f+o]=sum;}
extern "C" int smaul_hip_packed_linear(const float*x,const uint8_t*w,const float*s,float*y,int batch,int out_f,int in_f,int bits){
    if(bits!=2&&bits!=4)return 1;
    // Same issue as the CUDA backend: x/w/s/y are HOST pointers here. HIP kernels
    // need DEVICE pointers, so allocate device buffers and copy explicitly instead
    // of launching directly on host memory (undefined behavior otherwise).
    size_t rb=(size_t)((in_f+(8/bits)-1)/(8/bits));
    size_t x_bytes=(size_t)batch*(size_t)in_f*sizeof(float);
    size_t w_bytes=rb*(size_t)out_f;
    size_t s_bytes=(size_t)out_f*sizeof(float);
    size_t y_bytes=(size_t)batch*(size_t)out_f*sizeof(float);
    float *dx=nullptr,*ds=nullptr,*dy=nullptr; uint8_t *dw=nullptr;
    hipError_t err=hipSuccess;
    if((err=hipMalloc((void**)&dx,x_bytes))!=hipSuccess) goto fail;
    if((err=hipMalloc((void**)&dw,w_bytes))!=hipSuccess) goto fail;
    if((err=hipMalloc((void**)&ds,s_bytes))!=hipSuccess) goto fail;
    if((err=hipMalloc((void**)&dy,y_bytes))!=hipSuccess) goto fail;
    if((err=hipMemcpy(dx,x,x_bytes,hipMemcpyHostToDevice))!=hipSuccess) goto fail;
    if((err=hipMemcpy(dw,w,w_bytes,hipMemcpyHostToDevice))!=hipSuccess) goto fail;
    if((err=hipMemcpy(ds,s,s_bytes,hipMemcpyHostToDevice))!=hipSuccess) goto fail;
    {
        dim3 block(256,1,1);dim3 grid((out_f+255)/256,batch,1);
        hipLaunchKernelGGL(packed_linear_kernel,grid,block,0,0,dx,dw,ds,dy,batch,out_f,in_f,bits);
        err=hipGetLastError();
        if(err==hipSuccess) err=hipDeviceSynchronize();
    }
    if(err!=hipSuccess) goto fail;
    if((err=hipMemcpy(y,dy,y_bytes,hipMemcpyDeviceToHost))!=hipSuccess) goto fail;
    hipFree(dx); hipFree(dw); hipFree(ds); hipFree(dy);
    return 0;
fail:
    if(dx) hipFree(dx); if(dw) hipFree(dw); if(ds) hipFree(ds); if(dy) hipFree(dy);
    return err!=hipSuccess ? 1 : 0;
}
