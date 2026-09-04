#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <immintrin.h>
#include <vector>

std::vector<torch::Tensor> wkv_forward(
    torch::Tensor state, torch::Tensor w, torch::Tensor k, torch::Tensor v,
    torch::Tensor kk, torch::Tensor a, torch::Tensor r) {
    TORCH_CHECK(state.device().is_cpu());
    TORCH_CHECK(state.dtype() == torch::kFloat32);
    TORCH_CHECK(w.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32 &&
                v.dtype() == torch::kFloat32 && kk.dtype() == torch::kFloat32 &&
                a.dtype() == torch::kFloat32 && r.dtype() == torch::kFloat32);
    TORCH_CHECK(state.dim() == 4 && w.dim() == 4 && k.dim() == 4 && v.dim() == 4 &&
                kk.dim() == 4 && a.dim() == 4 && r.dim() == 4);
    const int64_t B = state.size(0), H = state.size(1), N = state.size(2), T = w.size(1);
    TORCH_CHECK(state.size(2) == state.size(3));
    TORCH_CHECK(N <= 128, "native WKV head size must be <= 128");
    TORCH_CHECK(w.sizes() == k.sizes() && w.sizes() == v.sizes() && w.sizes() == kk.sizes() &&
                w.sizes() == a.sizes() && w.sizes() == r.sizes());
    TORCH_CHECK(w.size(2) == H && w.size(3) == N);
    auto out_state = state.contiguous().clone();
    auto y = torch::empty({B, T, H, N}, state.options());

    const float* wp = w.data_ptr<float>();
    const float* kp = k.data_ptr<float>();
    const float* vp = v.data_ptr<float>();
    const float* kkp = kk.data_ptr<float>();
    const float* ap = a.data_ptr<float>();
    const float* rp = r.data_ptr<float>();
    float* sp = out_state.data_ptr<float>();
    float* yp = y.data_ptr<float>();

    const int64_t BH = B * H;
    at::parallel_for(0, BH, 1, [&](int64_t bh0, int64_t bh1) {
        alignas(32) float su[128];
        for (int64_t bh = bh0; bh < bh1; ++bh) {
            const int64_t b = bh / H;
            const int64_t h = bh - b * H;
            const int64_t base = (b * T * H + h) * N;
            const int64_t sbase = bh * N * N;
            for (int64_t t = 0; t < T; ++t) {
                const int64_t off = base + t * H * N;
                const float* wt = wp + off;
                const float* kt = kp + off;
                const float* vt = vp + off;
                const float* kkt = kkp + off;
                const float* at = ap + off;
                const float* rt = rp + off;
                float* st = sp + sbase;
                float* yt = yp + off;

                for (int64_t i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    const float* row = st + i * N;
                    for (int64_t j = 0; j < N; ++j) sum += row[j] * (-kkt[j]);
                    su[i] = sum;
                }
                for (int64_t i = 0; i < N; ++i) {
                    float* row = st + i * N;
                    const float vi = vt[i];
                    for (int64_t j = 0; j < N; ++j) {
                        row[j] = row[j] * wt[j] + su[i] * (kkt[j] * at[j]) + vi * kt[j];
                    }
                    float out = 0.0f;
                    for (int64_t j = 0; j < N; ++j) out += row[j] * rt[j];
                    yt[i] = out;
                }
            }
        }
    });
    return {out_state, y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward, "WKV forward");
}
