#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>

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
    at::parallel_for(0, n, 32768, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            float v = f_b1 * mm[i] + (1.0f - f_b1) * gg[i];
            pp[i] *= f_wd;
            pp[i] -= f_lr * (v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f));
            mm[i] = f_b2 * mm[i] + (1.0f - f_b2) * gg[i];
        }
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lion_step", &lion_step, "Fused Lion update (CPU)");
}
