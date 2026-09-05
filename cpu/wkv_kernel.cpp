#include <torch/extension.h>
#include <ATen/Parallel.h>
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

std::vector<torch::Tensor> wkv_backward(
    torch::Tensor state, torch::Tensor w, torch::Tensor k, torch::Tensor v,
    torch::Tensor kk, torch::Tensor a, torch::Tensor r,
    torch::Tensor grad_state, torch::Tensor grad_y) {
    TORCH_CHECK(state.device().is_cpu() && state.dtype() == torch::kFloat32);
    TORCH_CHECK(w.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32 &&
                v.dtype() == torch::kFloat32 && kk.dtype() == torch::kFloat32 &&
                a.dtype() == torch::kFloat32 && r.dtype() == torch::kFloat32);
    TORCH_CHECK(grad_state.dtype() == torch::kFloat32 && grad_y.dtype() == torch::kFloat32);
    const int64_t B = state.size(0), H = state.size(1), N = state.size(2), T = w.size(1);
    TORCH_CHECK(N == state.size(3) && N <= 128);
    TORCH_CHECK(w.size(0) == B && w.size(1) == T && w.size(2) == H && w.size(3) == N);

    auto gs0 = torch::zeros_like(state);
    auto gw = torch::zeros_like(w);
    auto gk = torch::zeros_like(k);
    auto gv = torch::zeros_like(v);
    auto gkk = torch::zeros_like(kk);
    auto ga = torch::zeros_like(a);
    auto gr = torch::zeros_like(r);

    const float* sp0 = state.data_ptr<float>();
    const float* wp = w.data_ptr<float>();
    const float* kp = k.data_ptr<float>();
    const float* vp = v.data_ptr<float>();
    const float* kkp = kk.data_ptr<float>();
    const float* ap = a.data_ptr<float>();
    const float* rp = r.data_ptr<float>();
    const float* gsp = grad_state.data_ptr<float>();
    const float* gyp = grad_y.data_ptr<float>();
    float* gs0p = gs0.data_ptr<float>();
    float* gwp = gw.data_ptr<float>();
    float* gkp = gk.data_ptr<float>();
    float* gvp = gv.data_ptr<float>();
    float* gkkp = gkk.data_ptr<float>();
    float* gap = ga.data_ptr<float>();
    float* grp = gr.data_ptr<float>();

    const int64_t BH = B * H;
    const int64_t state_stride = N * N;
    const int64_t token_stride = H * N;

    at::parallel_for(0, BH, 1, [&](int64_t bh0, int64_t bh1) {
        std::vector<float> hist((T + 1) * state_stride);
        std::vector<float> gnext(state_stride);
        std::vector<float> gcur(state_stride);
        std::vector<float> su(N);
        std::vector<float> c(N);
        std::vector<float> gsu(N);
        std::vector<float> gc(N);
        std::vector<float> gk_col(N, 0.0f);
        std::vector<float> gv_row(N, 0.0f);

        for (int64_t bh = bh0; bh < bh1; ++bh) {
            const int64_t b = bh / H;
            const int64_t h = bh - b * H;
            const int64_t sbase = bh * state_stride;
            const int64_t base = (b * T * H + h) * N;

            std::copy(sp0 + sbase, sp0 + sbase + state_stride, hist.data());

            for (int64_t t = 0; t < T; ++t) {
                const int64_t off = base + t * token_stride;
                const float* wt = wp + off;
                const float* kt = kp + off;
                const float* vt = vp + off;
                const float* kkt = kkp + off;
                const float* at = ap + off;
                const float* rt = rp + off;
                const float* prev = hist.data() + t * state_stride;
                float* next = hist.data() + (t + 1) * state_stride;

                for (int64_t i = 0; i < N; ++i) {
                    float s = 0.0f;
                    const float* row = prev + i * N;
                    for (int64_t j = 0; j < N; ++j) s -= row[j] * kkt[j];
                    su[i] = s;
                }
                for (int64_t j = 0; j < N; ++j) c[j] = kkt[j] * at[j];
                for (int64_t i = 0; i < N; ++i) {
                    const float* prow = prev + i * N;
                    float* nrow = next + i * N;
                    for (int64_t j = 0; j < N; ++j)
                        nrow[j] = prow[j] * wt[j] + su[i] * c[j] + vt[i] * kt[j];
                }
            }

            const float* gstate = gsp + sbase;
            std::copy(gstate, gstate + state_stride, gnext.begin());

            for (int64_t t = T - 1; t >= 0; --t) {
                const int64_t off = base + t * token_stride;
                const float* wt = wp + off;
                const float* kt = kp + off;
                const float* vt = vp + off;
                const float* kkt = kkp + off;
                const float* at = ap + off;
                const float* rt = rp + off;
                const float* gy = gyp + off;
                const float* prev = hist.data() + t * state_stride;
                const float* next = hist.data() + (t + 1) * state_stride;

                std::copy(gnext.begin(), gnext.end(), gcur.begin());
                for (int64_t i = 0; i < N; ++i)
                    for (int64_t j = 0; j < N; ++j)
                        gcur[i * N + j] += gy[i] * rt[j];

                float* grt = grp + off;
                for (int64_t j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int64_t i = 0; i < N; ++i) sum += gy[i] * next[i * N + j];
                    grt[j] += sum;
                }

                for (int64_t i = 0; i < N; ++i) {
                    float z = 0.0f;
                    for (int64_t j = 0; j < N; ++j) z += gcur[i * N + j] * c[j];
                    gsu[i] = z;
                }
                std::fill(gc.begin(), gc.end(), 0.0f);
                for (int64_t j = 0; j < N; ++j) {
                    float z = 0.0f;
                    for (int64_t i = 0; i < N; ++i) z += gcur[i * N + j] * su[i];
                    gc[j] = z;
                }

                float* gwt = gwp + off;
                float* gkt = gkp + off;
                float* gvt = gvp + off;
                float* gkkt = gkkp + off;
                float* gat = gap + off;
                for (int64_t j = 0; j < N; ++j) {
                    float sw = 0.0f, sk = 0.0f;
                    for (int64_t i = 0; i < N; ++i) {
                        const float gij = gcur[i * N + j];
                        sw += gij * prev[i * N + j];
                        sk += gij * vt[i];
                        gvt[i] += gij * kt[j];
                    }
                    gwt[j] += sw;
                    gkt[j] += sk;
                    gkkt[j] += gc[j] * at[j];
                    gat[j] += gc[j] * kkt[j];
                }

                float* gs = (t == 0) ? gs0p + sbase : gnext.data();
                for (int64_t i = 0; i < N; ++i) {
                    float gsu_i = gsu[i];
                    for (int64_t j = 0; j < N; ++j)
                        gs[i * N + j] = gcur[i * N + j] * wt[j] - gsu_i * kkt[j];
                }

                for (int64_t j = 0; j < N; ++j) {
                    float z = 0.0f;
                    for (int64_t i = 0; i < N; ++i) z -= gsu[i] * prev[i * N + j];
                    gkkt[j] += z;
                }
            }
        }
    });
    return {gs0, gw, gk, gv, gkk, ga, gr};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward, "WKV forward");
    m.def("wkv_backward", &wkv_backward, "WKV backward");
}
