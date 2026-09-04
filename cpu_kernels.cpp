#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <immintrin.h>

void lion_step(torch::Tensor p, torch::Tensor g, torch::Tensor m, double lr, double b1, double b2, double wd) {
    TORCH_CHECK(p.device().is_cpu() && g.device().is_cpu() && m.device().is_cpu());
    TORCH_CHECK(p.dtype() == torch::kFloat32 && g.dtype() == torch::kFloat32 && m.dtype() == torch::kFloat32);
    TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous());
    TORCH_CHECK(p.numel() == g.numel() && p.numel() == m.numel());
    auto pp = p.data_ptr<float>();
    auto gg = g.data_ptr<float>();
    auto mm = m.data_ptr<float>();
    const auto n = p.numel();
    const float f_lr = (float)lr, f_b1 = (float)b1, f_b2 = (float)b2;
    const float f_wd = (float)(1.0 - lr * wd);
    at::parallel_for(0, n, 1 << 20, [&](int64_t begin, int64_t end) {
        int64_t i = begin;
        const __m256 vb1 = _mm256_set1_ps(f_b1);
        const __m256 v1_b1 = _mm256_set1_ps(1.0f - f_b1);
        const __m256 vb2 = _mm256_set1_ps(f_b2);
        const __m256 v1_b2 = _mm256_set1_ps(1.0f - f_b2);
        const __m256 vwd = _mm256_set1_ps(f_wd);
        const __m256 vlr = _mm256_set1_ps(f_lr);
        const __m256 vzero = _mm256_setzero_ps();
        const __m256 vone = _mm256_set1_ps(1.0f);
        for (; i + 8 <= end; i += 8) {
            __m256 m_val = _mm256_loadu_ps(mm + i);
            const __m256 g_val = _mm256_loadu_ps(gg + i);
            __m256 p_val = _mm256_loadu_ps(pp + i);
            const __m256 v = _mm256_add_ps(_mm256_mul_ps(vb1, m_val), _mm256_mul_ps(v1_b1, g_val));
            const __m256 gt = _mm256_cmp_ps(v, vzero, _CMP_GT_OQ);
            const __m256 lt = _mm256_cmp_ps(v, vzero, _CMP_LT_OQ);
            const __m256 sign = _mm256_sub_ps(_mm256_and_ps(gt, vone), _mm256_and_ps(lt, vone));
            p_val = _mm256_sub_ps(_mm256_mul_ps(p_val, vwd), _mm256_mul_ps(vlr, sign));
            m_val = _mm256_add_ps(_mm256_mul_ps(vb2, m_val), _mm256_mul_ps(v1_b2, g_val));
            _mm256_storeu_ps(pp + i, p_val);
            _mm256_storeu_ps(mm + i, m_val);
        }
        for (; i < end; ++i) {
            float v = f_b1 * mm[i] + (1.0f - f_b1) * gg[i];
            pp[i] *= f_wd;
            pp[i] -= f_lr * (v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f));
            mm[i] = f_b2 * mm[i] + (1.0f - f_b2) * gg[i];
        }
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lion_step", &lion_step, "Fused Lion update with AVX");
}
