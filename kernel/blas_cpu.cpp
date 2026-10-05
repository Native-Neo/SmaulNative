#include <ATen/Parallel.h>
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <torch/extension.h>

// SmaulNative's small Ivy Bridge BLAS subset.
// AVX1 only: no AVX2 and no FMA. This targets the FP32 GEMM shapes used by
// dense projections/backward passes, rather than attempting to reproduce MKL's
// enormous API surface.

namespace {

inline float hsum8(__m256 v) {
    float x[8];
    _mm256_storeu_ps(x, v);
    return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
}

inline void dot8(const float* a, const float* b, int64_t k, float* out) {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= k; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i)));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8)));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16)));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24)));
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    float s = hsum8(acc);
    for (; i < k; ++i) s += a[i] * b[i];
    *out = s;
}

void gemm(const float* A, const float* B, float* C,
          int64_t M, int64_t K, int64_t N,
          int64_t r0, int64_t r1, int64_t n0, int64_t n1) {
    for (int64_t i = r0; i < r1; ++i) {
        const float* ar = A + i * K;
        float* cr = C + i * N;
        for (int64_t j = n0; j < n1; j += 8) {
            if (j + 8 <= n1) {
                __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
                __m256 a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
                int64_t k=0;
                for (; k+4<=K; k+=4) {
                    a0=_mm256_add_ps(a0,_mm256_mul_ps(_mm256_set1_ps(ar[k]),   _mm256_loadu_ps(B+k*N+j)));
                    a1=_mm256_add_ps(a1,_mm256_mul_ps(_mm256_set1_ps(ar[k+1]), _mm256_loadu_ps(B+(k+1)*N+j)));
                    a2=_mm256_add_ps(a2,_mm256_mul_ps(_mm256_set1_ps(ar[k+2]), _mm256_loadu_ps(B+(k+2)*N+j)));
                    a3=_mm256_add_ps(a3,_mm256_mul_ps(_mm256_set1_ps(ar[k+3]), _mm256_loadu_ps(B+(k+3)*N+j)));
                }
                __m256 acc=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
                for (; k<K; ++k)
                    acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_set1_ps(ar[k]),_mm256_loadu_ps(B+k*N+j)));
                _mm256_storeu_ps(cr+j,acc);
            } else {
                for (int64_t jj=j;jj<n1;++jj) {
                    float s=0.0f;
                    for (int64_t k=0;k<K;++k) s += ar[k]*B[k*N+jj];
                    cr[jj]=s;
                }
            }
        }
    }
}

void gemm_bt(const float* A, const float* B, float* C,
             int64_t M, int64_t K, int64_t N,
             int64_t r0, int64_t r1, int64_t n0, int64_t n1) {
    for (int64_t i=r0;i<r1;++i) {
        const float* ar=A+i*K;
        float* cr=C+i*N;
        for (int64_t j=n0;j<n1;++j) dot8(ar,B+j*K,K,&cr[j]);
    }
}

torch::Tensor sgemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cpu() && B.device().is_cpu(), "sgemm: CPU tensors required");
    TORCH_CHECK(A.scalar_type()==torch::kFloat32 && B.scalar_type()==torch::kFloat32,
                "sgemm: float32 tensors required");
    TORCH_CHECK(A.dim()==2 && B.dim()==2 && A.is_contiguous() && B.is_contiguous(),
                "sgemm: contiguous 2-D tensors required");
    TORCH_CHECK(A.size(1)==B.size(0), "sgemm: incompatible dimensions");
    const int64_t M=A.size(0), K=A.size(1), N=B.size(1);
    auto C=torch::empty({M,N},A.options());
    if (!M || !N) return C;
    const int64_t NB=64, blocks=(N+NB-1)/NB;
    const float* ap=A.data_ptr<float>(); const float* bp=B.data_ptr<float>(); float* cp=C.data_ptr<float>();
    at::parallel_for(0,M*blocks,1,[&](int64_t begin,int64_t end) {
        for (int64_t task=begin;task<end;++task) {
            int64_t i=task/blocks, nb=task%blocks;
            gemm(ap,bp,cp,M,K,N,i,i+1,nb*NB,std::min(N,(nb+1)*NB));
        }
    });
    return C;
}

torch::Tensor sgemm_bt(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cpu() && B.device().is_cpu(), "sgemm_bt: CPU tensors required");
    TORCH_CHECK(A.scalar_type()==torch::kFloat32 && B.scalar_type()==torch::kFloat32,
                "sgemm_bt: float32 tensors required");
    TORCH_CHECK(A.dim()==2 && B.dim()==2 && A.is_contiguous() && B.is_contiguous(),
                "sgemm_bt: contiguous 2-D tensors required");
    TORCH_CHECK(A.size(1)==B.size(1), "sgemm_bt: incompatible dimensions");
    const int64_t M=A.size(0), K=A.size(1), N=B.size(0);
    auto C=torch::empty({M,N},A.options());
    if (!M || !N) return C;
    const int64_t NB=32, blocks=(N+NB-1)/NB;
    const float* ap=A.data_ptr<float>(); const float* bp=B.data_ptr<float>(); float* cp=C.data_ptr<float>();
    at::parallel_for(0,M*blocks,1,[&](int64_t begin,int64_t end) {
        for (int64_t task=begin;task<end;++task) {
            int64_t i=task/blocks, nb=task%blocks;
            gemm_bt(ap,bp,cp,M,K,N,i,i+1,nb*NB,std::min(N,(nb+1)*NB));
        }
    });
    return C;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("sgemm",&sgemm,"Ivy Bridge AVX1 FP32 GEMM");
    m.def("sgemm_bt",&sgemm_bt,"Ivy Bridge AVX1 FP32 GEMM with RHS transposed");
}
