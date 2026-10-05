#include <ATen/Parallel.h>
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <torch/extension.h>

// Small Ivy Bridge BLAS subset for SmaulNative.
// Ivy Bridge has AVX1 but no AVX2/FMA. The hot GEMM path therefore uses
// cache tiling plus a small SSE microkernel; this keeps register pressure low
// while reusing each B vector across two A rows.

namespace {

inline void kernel_2x4(const float* a0, const float* a1,
                       const float* b, float* c0, float* c1,
                       int64_t K, int64_t N, int64_t j,
                       bool accumulate) {
    __m128 c00 = accumulate ? _mm_loadu_ps(c0 + j) : _mm_setzero_ps();
    __m128 c01 = accumulate ? _mm_loadu_ps(c1 + j) : _mm_setzero_ps();
    __m128 c10 = _mm_setzero_ps();
    __m128 c11 = _mm_setzero_ps();

    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        __m128 bv = _mm_loadu_ps(b + k * N + j);
        c00 = _mm_add_ps(c00, _mm_mul_ps(_mm_set1_ps(a0[k]), bv));
        c01 = _mm_add_ps(c01, _mm_mul_ps(_mm_set1_ps(a1[k]), bv));

        bv = _mm_loadu_ps(b + (k + 1) * N + j);
        c00 = _mm_add_ps(c00, _mm_mul_ps(_mm_set1_ps(a0[k + 1]), bv));
        c01 = _mm_add_ps(c01, _mm_mul_ps(_mm_set1_ps(a1[k + 1]), bv));

        bv = _mm_loadu_ps(b + (k + 2) * N + j);
        c10 = _mm_add_ps(c10, _mm_mul_ps(_mm_set1_ps(a0[k + 2]), bv));
        c11 = _mm_add_ps(c11, _mm_mul_ps(_mm_set1_ps(a1[k + 2]), bv));

        bv = _mm_loadu_ps(b + (k + 3) * N + j);
        c10 = _mm_add_ps(c10, _mm_mul_ps(_mm_set1_ps(a0[k + 3]), bv));
        c11 = _mm_add_ps(c11, _mm_mul_ps(_mm_set1_ps(a1[k + 3]), bv));
    }

    __m128 s0 = _mm_add_ps(c00, c10);
    __m128 s1 = _mm_add_ps(c01, c11);
    _mm_storeu_ps(c0 + j, s0);
    _mm_storeu_ps(c1 + j, s1);

    if (k < K) {
        // K is normally a multiple of 4 in SmaulNative, but keep the kernel
        // correct for arbitrary dimensions.
        for (; k < K; ++k) {
            const float x0 = a0[k];
            const float x1 = a1[k];
            for (int q = 0; q < 4; ++q) {
                c0[j + q] += x0 * b[k * N + j + q];
                c1[j + q] += x1 * b[k * N + j + q];
            }
        }
    }
}

inline void kernel_2x4_range(const float* a0, const float* a1,
                             const float* b, float* c0, float* c1,
                             int64_t K, int64_t N, int64_t j0, int64_t j1,
                             bool accumulate) {
    int64_t j = j0;
    for (; j + 4 <= j1; j += 4)
        kernel_2x4(a0, a1, b, c0, c1, K, N, j, accumulate);

    // Scalar tail.
    for (; j < j1; ++j) {
        float s0 = accumulate ? c0[j] : 0.0f;
        float s1 = accumulate ? c1[j] : 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            s0 += a0[k] * b[k * N + j];
            s1 += a1[k] * b[k * N + j];
        }
        c0[j] = s0;
        c1[j] = s1;
    }
}

inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline void dot4_rows(const float* a, const float* b, float* c,
                      int64_t K, int64_t N, int64_t j0, int64_t j1) {
    for (int64_t j = j0; j < j1; j += 4) {
        const int64_t count = std::min<int64_t>(4, j1 - j);
        if (count == 4) {
            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps();
            __m256 s3 = _mm256_setzero_ps();
            int64_t k = 0;

            for (; k + 8 <= K; k += 8) {
                __m256 av = _mm256_loadu_ps(a + k);
                s0 = _mm256_add_ps(s0, _mm256_mul_ps(av, _mm256_loadu_ps(b + (j+0)*K + k)));
                s1 = _mm256_add_ps(s1, _mm256_mul_ps(av, _mm256_loadu_ps(b + (j+1)*K + k)));
                s2 = _mm256_add_ps(s2, _mm256_mul_ps(av, _mm256_loadu_ps(b + (j+2)*K + k)));
                s3 = _mm256_add_ps(s3, _mm256_mul_ps(av, _mm256_loadu_ps(b + (j+3)*K + k)));
            }

            float o0 = hsum8(s0);
            float o1 = hsum8(s1);
            float o2 = hsum8(s2);
            float o3 = hsum8(s3);
            for (; k < K; ++k) {
                const float x = a[k];
                o0 += x * b[(j+0)*K + k];
                o1 += x * b[(j+1)*K + k];
                o2 += x * b[(j+2)*K + k];
                o3 += x * b[(j+3)*K + k];
            }
            c[j+0] = o0;
            c[j+1] = o1;
            c[j+2] = o2;
            c[j+3] = o3;
        } else {
            for (int64_t q = 0; q < count; ++q) {
                float s = 0.0f;
                const float* br = b + (j + q) * K;
                for (int64_t k = 0; k < K; ++k)
                    s += a[k] * br[k];
                c[j + q] = s;
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

    // A 16x64 C tile is paired with a 128-column K panel. The K panel keeps
    // the active A/B working set small enough for Ivy Bridge caches while
    // the 2x4 microkernel reuses each B vector across two A rows.
    constexpr int64_t MB = 16;
    constexpr int64_t NB = 64;
    const int64_t mt = (M + MB - 1) / MB;

    at::parallel_for(0, mt, 1, [&](int64_t begin, int64_t end) {
        for (int64_t mb = begin; mb < end; ++mb) {
            const int64_t i0 = mb * MB;
            const int64_t i1 = std::min(M, i0 + MB);

            for (int64_t j0 = 0; j0 < N; j0 += NB) {
                const int64_t j1 = std::min(N, j0 + NB);
                int64_t i = i0;

                for (; i + 1 < i1; i += 2) {
                    const float* a0 = ap + i * K;
                    const float* a1 = ap + (i + 1) * K;
                    float* c0 = cp + i * N;
                    float* c1 = cp + (i + 1) * N;
                    constexpr int64_t KC = 128;
                    bool accumulate = false;
                    for (int64_t k0 = 0; k0 < K; k0 += KC) {
                        const int64_t kc = std::min(K, k0 + KC) - k0;
                        kernel_2x4_range(a0 + k0, a1 + k0, bp + k0 * N,
                                         c0, c1, kc, N, j0, j1, accumulate);
                        accumulate = true;
                    }
                }

                if (i < i1) {
                    const float* a = ap + i * K;
                    float* c = cp + i * N;
                    for (int64_t j = j0; j < j1; ++j) {
                        float s = 0.0f;
                        for (int64_t k = 0; k < K; ++k)
                            s += a[k] * bp[k * N + j];
                        c[j] = s;
                    }
                }
            }
        }
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

    // Each A row is independent. Use a coarse row tile rather than one task
    // per output element; the 4-output AVX kernel reuses A loads.
    constexpr int64_t MB = 16;
    const int64_t mt = (M + MB - 1) / MB;

    at::parallel_for(0, mt, 1, [&](int64_t begin, int64_t end) {
        for (int64_t mb = begin; mb < end; ++mb) {
            const int64_t i0 = mb * MB;
            const int64_t i1 = std::min(M, i0 + MB);
            for (int64_t i = i0; i < i1; ++i)
                dot4_rows(ap + i*K, bp, cp + i*N, K, N, 0, N);
        }
    });
    return C;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm", &sgemm, "Ivy Bridge AVX1 FP32 GEMM");
    m.def("sgemm_bt", &sgemm_bt, "Ivy Bridge AVX1 FP32 GEMM with RHS transposed");
}
