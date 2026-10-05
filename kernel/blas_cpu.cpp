#include <ATen/Parallel.h>
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <torch/extension.h>

// Small Ivy Bridge BLAS subset for SmaulNative.
// AVX1 only: Ivy Bridge has no AVX2/FMA, so the kernels use separate
// multiply/add instructions and keep the hot loops cache-friendly.

namespace {

inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline void gemm_row8(const float* a, const float* b, float* c,
                      int64_t k, int64_t n, int64_t j0, int64_t j1) {
    for (int64_t j = j0; j < j1; j += 8) {
        if (j + 8 <= j1) {
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            int64_t p = 0;

            // Four independent accumulators hide AVX multiply/add latency.
            for (; p + 4 <= k; p += 4) {
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_set1_ps(a[p]),
                                                     _mm256_loadu_ps(b + p*n + j)));
                a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_set1_ps(a[p+1]),
                                                     _mm256_loadu_ps(b + (p+1)*n + j)));
                a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_set1_ps(a[p+2]),
                                                     _mm256_loadu_ps(b + (p+2)*n + j)));
                a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_set1_ps(a[p+3]),
                                                     _mm256_loadu_ps(b + (p+3)*n + j)));
            }

            __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1),
                                        _mm256_add_ps(a2, a3));
            for (; p < k; ++p) {
                acc = _mm256_add_ps(acc, _mm256_mul_ps(
                    _mm256_set1_ps(a[p]), _mm256_loadu_ps(b + p*n + j)));
            }
            _mm256_storeu_ps(c + j, acc);
        } else {
            for (int64_t j2 = j; j2 < j1; ++j2) {
                float s = 0.0f;
                for (int64_t p = 0; p < k; ++p)
                    s += a[p] * b[p*n + j2];
                c[j2] = s;
            }
        }
    }
}

// A 4-output kernel for A @ B^T. Each accumulator computes one output
// dot product while the same A vector loads are reused across four rows of B.
inline void gemm_bt_row4(const float* a, const float* b, float* c,
                         int64_t k, int64_t n0, int64_t n1) {
    for (int64_t j = n0; j < n1; j += 4) {
        int64_t count = std::min<int64_t>(4, n1 - j);
        if (count == 4) {
            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps();
            __m256 s3 = _mm256_setzero_ps();
            int64_t p = 0;

            for (; p + 8 <= k; p += 8) {
                __m256 av = _mm256_loadu_ps(a + p);
                s0 = _mm256_add_ps(s0, _mm256_mul_ps(av, _mm256_loadu_ps(b + (j+0)*k + p)));
                s1 = _mm256_add_ps(s1, _mm256_mul_ps(av, _mm256_loadu_ps(b + (j+1)*k + p)));
                s2 = _mm256_add_ps(s2, _mm256_mul_ps(av, _mm256_loadu_ps(b + (j+2)*k + p)));
                s3 = _mm256_add_ps(s3, _mm256_mul_ps(av, _mm256_loadu_ps(b + (j+3)*k + p)));
            }

            float out0 = hsum8(s0);
            float out1 = hsum8(s1);
            float out2 = hsum8(s2);
            float out3 = hsum8(s3);
            for (; p < k; ++p) {
                const float av = a[p];
                out0 += av * b[(j+0)*k + p];
                out1 += av * b[(j+1)*k + p];
                out2 += av * b[(j+2)*k + p];
                out3 += av * b[(j+3)*k + p];
            }
            c[j+0] = out0;
            c[j+1] = out1;
            c[j+2] = out2;
            c[j+3] = out3;
        } else {
            for (int64_t q = 0; q < count; ++q) {
                float s = 0.0f;
                const float* br = b + (j+q)*k;
                for (int64_t p = 0; p < k; ++p)
                    s += a[p] * br[p];
                c[j+q] = s;
            }
        }
    }
}

torch::Tensor sgemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cpu() && B.device().is_cpu(), "sgemm: CPU tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "sgemm: float32 tensors required");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.is_contiguous() && B.is_contiguous(),
                "sgemm: contiguous 2-D tensors required");
    TORCH_CHECK(A.size(1) == B.size(0), "sgemm: incompatible dimensions");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    if (!M || !N) return C;

    const float* ap = A.data_ptr<float>();
    const float* bp = B.data_ptr<float>();
    float* cp = C.data_ptr<float>();

    // One coarse task per output row. The previous M*N-block task grid
    // created thousands of tiny tasks on a 2-thread Ivy Bridge CPU.
    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i)
            gemm_row8(ap + i*K, bp, cp + i*N, K, N, 0, N);
    });
    return C;
}

torch::Tensor sgemm_bt(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cpu() && B.device().is_cpu(), "sgemm_bt: CPU tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "sgemm_bt: float32 tensors required");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.is_contiguous() && B.is_contiguous(),
                "sgemm_bt: contiguous 2-D tensors required");
    TORCH_CHECK(A.size(1) == B.size(1), "sgemm_bt: incompatible dimensions");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(0);
    auto C = torch::empty({M, N}, A.options());
    if (!M || !N) return C;

    const float* ap = A.data_ptr<float>();
    const float* bp = B.data_ptr<float>();
    float* cp = C.data_ptr<float>();

    // B rows are already contiguous, so A @ B^T can stream both operands.
    // Process four output rows together to reuse each A vector load.
    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i)
            gemm_bt_row4(ap + i*K, bp, cp + i*N, K, 0, N);
    });
    return C;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm", &sgemm, "Ivy Bridge AVX1 FP32 GEMM");
    m.def("sgemm_bt", &sgemm_bt, "Ivy Bridge AVX1 FP32 GEMM with RHS transposed");
}
