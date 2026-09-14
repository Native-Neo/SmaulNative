#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ float fp2(uint8_t q){q&=3; return q==0?-1.0f:q==2?1.0f:0.0f;}
__device__ float fp4(uint8_t q){q&=15; switch(q){case 0:return -2.0f;case 1:return -1.0f;case 2:return -0.5f;case 3:return -0.25f;case 5:return 0.25f;case 6:return 0.5f;case 7:return 1.0f;case 8:return 2.0f;default:return 0.0f;}}
__global__ void packed_linear_kernel(const float*x,const uint8_t*w,const float*s,float*y,int batch,int out_f,int in_f,int bits){int o=blockIdx.x*blockDim.x+threadIdx.x;int n=blockIdx.y;if(o>=out_f||n>=batch)return;int rb=(in_f+(8/bits)-1)/(8/bits);const uint8_t*row=w+(size_t)o*rb;float sum=0.0f;for(int k=0;k<in_f;k++){uint8_t q=bits==2?((row[k>>2]>>(6-((k&3)<<1)))&3):((row[k>>1]>>(4-((k&1)<<2)))&15);sum+=x[(size_t)n*in_f+k]*s[o]*(bits==2?fp2(q):fp4(q));}y[(size_t)n*out_f+o]=sum;}
extern "C" int smaul_hip_packed_linear(const float*x,const uint8_t*w,const float*s,float*y,int batch,int out_f,int in_f,int bits){if(bits!=2&&bits!=4)return 1;dim3 block(256,1,1);dim3 grid((out_f+255)/256,batch,1);hipLaunchKernelGGL(packed_linear_kernel,grid,block,0,0,x,w,s,y,batch,out_f,in_f,bits);return hipGetLastError()!=hipSuccess;}
