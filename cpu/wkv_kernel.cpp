#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <algorithm>
#include <vector>

static void check_inputs(const torch::Tensor& state, const torch::Tensor& w, const torch::Tensor& k,
                         const torch::Tensor& v, const torch::Tensor& kk, const torch::Tensor& a,
                         const torch::Tensor& r) {
  TORCH_CHECK(state.device().is_cpu(), "state must be CPU");
  TORCH_CHECK(state.dtype() == torch::kFloat32, "state must be float32");
  TORCH_CHECK(state.dim() == 4, "state must be [B,H,N,N]");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  for (const auto& t : {w, k, v, kk, a, r}) {
    TORCH_CHECK(t.device().is_cpu(), "WKV inputs must be CPU");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "WKV inputs must be float32");
    TORCH_CHECK(t.dim() == 4 && t.is_contiguous(), "WKV inputs must be contiguous 4D tensors");
  }
  const int64_t B = state.size(0), H = state.size(1), N = state.size(2), T = w.size(1);
  TORCH_CHECK(state.size(3) == N, "state must have square head dimensions");
  TORCH_CHECK(B > 0 && H > 0 && N > 0 && T > 0, "WKV dimensions must be positive");
  TORCH_CHECK(N <= 128, "native WKV head size must be <= 128");
  TORCH_CHECK(w.size(0) == B && w.size(1) == T && w.size(2) == H && w.size(3) == N,
              "WKV inputs have incompatible dimensions");
  TORCH_CHECK(k.sizes() == w.sizes() && v.sizes() == w.sizes() && kk.sizes() == w.sizes() &&
              a.sizes() == w.sizes() && r.sizes() == w.sizes(), "WKV input shapes must match");
}

std::vector<torch::Tensor> wkv_forward(torch::Tensor state, torch::Tensor w, torch::Tensor k, torch::Tensor v,
                                       torch::Tensor kk, torch::Tensor a, torch::Tensor r) {
  check_inputs(state, w, k, v, kk, a, r);
  const int64_t B = state.size(0), H = state.size(1), N = state.size(2), T = w.size(1);
  auto out_state = state.clone();
  auto y = torch::empty({B, T, H, N}, state.options());
  const float *wp = w.data_ptr<float>(), *kp = k.data_ptr<float>(), *vp = v.data_ptr<float>();
  const float *kkp = kk.data_ptr<float>(), *ap = a.data_ptr<float>(), *rp = r.data_ptr<float>();
  float *sp = out_state.data_ptr<float>(), *yp = y.data_ptr<float>();
  const int64_t BH = B * H;
  at::parallel_for(0, BH, 1, [&](int64_t bh0, int64_t bh1) {
    alignas(32) float su[128], c[128];
    for (int64_t bh = bh0; bh < bh1; ++bh) {
      const int64_t b = bh / H, h = bh - b * H;
      const int64_t base = (b * T * H + h) * N, sbase = bh * N * N;
      for (int64_t t = 0; t < T; ++t) {
        const int64_t off = base + t * H * N;
        const float *wt = wp + off, *kt = kp + off, *vt = vp + off, *kkt = kkp + off, *at = ap + off, *rt = rp + off;
        float *st = sp + sbase, *yt = yp + off;
        for (int64_t j = 0; j < N; ++j) c[j] = kkt[j] * at[j];
        for (int64_t i = 0; i < N; ++i) {
          float sum = 0.0f;
          const float *row = st + i * N;
#pragma GCC ivdep
          for (int64_t j = 0; j < N; ++j) sum -= row[j] * kkt[j];
          su[i] = sum;
        }
#pragma GCC ivdep
        for (int64_t i = 0; i < N; ++i) {
          float *row = st + i * N;
          const float vi = vt[i];
#pragma GCC ivdep
          for (int64_t j = 0; j < N; ++j) row[j] = row[j] * wt[j] + su[i] * c[j] + vi * kt[j];
          float out = 0.0f;
#pragma GCC ivdep
          for (int64_t j = 0; j < N; ++j) out += row[j] * rt[j];
          yt[i] = out;
        }
      }
    }
  });
  return {out_state, y};
}

std::vector<torch::Tensor> wkv_backward(torch::Tensor state, torch::Tensor w, torch::Tensor k, torch::Tensor v,
                                        torch::Tensor kk, torch::Tensor a, torch::Tensor r, torch::Tensor grad_state,
                                        torch::Tensor grad_y) {
  check_inputs(state, w, k, v, kk, a, r);
  TORCH_CHECK(grad_state.device().is_cpu() && grad_state.dtype() == torch::kFloat32 && grad_state.dim() == 4 && grad_state.is_contiguous(),
              "grad_state must be contiguous CPU float32 [B,H,N,N]");
  TORCH_CHECK(grad_y.device().is_cpu() && grad_y.dtype() == torch::kFloat32 && grad_y.dim() == 4 && grad_y.is_contiguous(),
              "grad_y must be contiguous CPU float32 [B,T,H,N]");
  TORCH_CHECK(grad_state.sizes() == state.sizes(), "grad_state shape must match state");
  TORCH_CHECK(grad_y.sizes() == w.sizes(), "grad_y shape must match WKV inputs");

  const int64_t B = state.size(0), H = state.size(1), N = state.size(2), T = w.size(1);
  const int64_t BH = B * H, state_stride = N * N, token_stride = H * N;
  const int64_t CHUNK = 64;
  const int64_t blocks = (T + CHUNK - 1) / CHUNK;
  auto gs0 = torch::zeros_like(state), gw = torch::zeros_like(w), gk = torch::zeros_like(k),
       gv = torch::zeros_like(v), gkk = torch::zeros_like(kk), ga = torch::zeros_like(a), gr = torch::zeros_like(r);

  const float *sp0 = state.data_ptr<float>(), *wp = w.data_ptr<float>(), *kp = k.data_ptr<float>(), *vp = v.data_ptr<float>();
  const float *kkp = kk.data_ptr<float>(), *ap = a.data_ptr<float>(), *rp = r.data_ptr<float>();
  const float *gsp = grad_state.data_ptr<float>(), *gyp = grad_y.data_ptr<float>();
  float *gs0p = gs0.data_ptr<float>(), *gwp = gw.data_ptr<float>(), *gkp = gk.data_ptr<float>(), *gvp = gv.data_ptr<float>();
  float *gkkp = gkk.data_ptr<float>(), *gap = ga.data_ptr<float>(), *grp = gr.data_ptr<float>();

  at::parallel_for(0, BH, 1, [&](int64_t bh0, int64_t bh1) {
    std::vector<float> checkpoints((blocks + 1) * state_stride);
    std::vector<float> hist((CHUNK + 1) * state_stride);
    std::vector<float> current(state_stride);
    std::vector<float> gnext(state_stride), gcur(state_stride), su(N), c(N), gsu(N), gc(N);

    for (int64_t bh = bh0; bh < bh1; ++bh) {
      const int64_t b = bh / H, h = bh - b * H;
      const int64_t sbase = bh * state_stride, base = (b * T * H + h) * N;
      std::copy(sp0 + sbase, sp0 + sbase + state_stride, current.data());

      for (int64_t block = 0; block < blocks; ++block) {
        const int64_t lo = block * CHUNK, hi = std::min(T, lo + CHUNK);
        std::copy(current.begin(), current.end(), checkpoints.data() + block * state_stride);
        for (int64_t t = lo; t < hi; ++t) {
          const int64_t off = base + t * token_stride;
          const float *wt = wp + off, *kt = kp + off, *vt = vp + off, *kkt = kkp + off, *at = ap + off;
          for (int64_t j = 0; j < N; ++j) c[j] = kkt[j] * at[j];
          for (int64_t i = 0; i < N; ++i) {
            float s = 0.0f;
            const float *row = current.data() + i * N;
#pragma GCC ivdep
            for (int64_t j = 0; j < N; ++j) s -= row[j] * kkt[j];
            su[i] = s;
          }
          for (int64_t i = 0; i < N; ++i) {
            float *row = current.data() + i * N;
#pragma GCC ivdep
            for (int64_t j = 0; j < N; ++j) row[j] = row[j] * wt[j] + su[i] * c[j] + vt[i] * kt[j];
          }
        }
      }
      std::copy(current.begin(), current.end(), checkpoints.data() + blocks * state_stride);
      std::copy(gsp + sbase, gsp + sbase + state_stride, gnext.begin());

      for (int64_t block = blocks - 1; block >= 0; --block) {
        const int64_t lo = block * CHUNK, hi = std::min(T, lo + CHUNK), len = hi - lo;
        std::copy(checkpoints.data() + block * state_stride, checkpoints.data() + (block + 1) * state_stride, hist.data());
        for (int64_t local = 0; local < len; ++local) {
          const int64_t t = lo + local, off = base + t * token_stride;
          const float *wt = wp + off, *kt = kp + off, *vt = vp + off, *kkt = kkp + off, *at = ap + off;
          const float *prev = hist.data() + local * state_stride;
          float *next = hist.data() + (local + 1) * state_stride;
          for (int64_t j = 0; j < N; ++j) c[j] = kkt[j] * at[j];
          for (int64_t i = 0; i < N; ++i) {
            float s = 0.0f;
            const float *prow = prev + i * N;
#pragma GCC ivdep
            for (int64_t j = 0; j < N; ++j) s -= prow[j] * kkt[j];
            su[i] = s;
          }
          for (int64_t i = 0; i < N; ++i) {
            const float *prow = prev + i * N;
            float *nrow = next + i * N;
#pragma GCC ivdep
            for (int64_t j = 0; j < N; ++j) nrow[j] = prow[j] * wt[j] + su[i] * c[j] + vt[i] * kt[j];
          }
        }

        for (int64_t local = len - 1; local >= 0; --local) {
          const int64_t t = lo + local, off = base + t * token_stride;
          const float *wt = wp + off, *kt = kp + off, *vt = vp + off, *kkt = kkp + off, *at = ap + off, *rt = rp + off, *gy = gyp + off;
          const float *prev = hist.data() + local * state_stride, *next = hist.data() + (local + 1) * state_stride;
          for (int64_t i = 0; i < N; ++i) {
            float s = 0.0f;
            const float *prow = prev + i * N;
#pragma GCC ivdep
            for (int64_t j = 0; j < N; ++j) s -= prow[j] * kkt[j];
            su[i] = s;
          }
          for (int64_t j = 0; j < N; ++j) c[j] = kkt[j] * at[j];
          std::copy(gnext.begin(), gnext.end(), gcur.begin());
          for (int64_t i = 0; i < N; ++i) {
#pragma GCC ivdep
            for (int64_t j = 0; j < N; ++j) gcur[i * N + j] += gy[i] * rt[j];
          }

          float *grt = grp + off;
          for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
#pragma GCC ivdep
            for (int64_t i = 0; i < N; ++i) sum += gy[i] * next[i * N + j];
            grt[j] += sum;
          }
          for (int64_t i = 0; i < N; ++i) {
            float z = 0.0f;
#pragma GCC ivdep
            for (int64_t j = 0; j < N; ++j) z += gcur[i * N + j] * c[j];
            gsu[i] = z;
          }
          std::fill(gc.begin(), gc.end(), 0.0f);
          for (int64_t j = 0; j < N; ++j) {
            float z = 0.0f;
#pragma GCC ivdep
            for (int64_t i = 0; i < N; ++i) z += gcur[i * N + j] * su[i];
            gc[j] = z;
          }

          float *gwt = gwp + off, *gkt = gkp + off, *gvt = gvp + off, *gkkt = gkkp + off, *gat = gap + off;
          for (int64_t j = 0; j < N; ++j) {
            float sw = 0.0f, sk = 0.0f;
#pragma GCC ivdep
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

          float *gs = (t == 0) ? gs0p + sbase : gnext.data();
          for (int64_t i = 0; i < N; ++i) {
            const float gsu_i = gsu[i];
#pragma GCC ivdep
            for (int64_t j = 0; j < N; ++j) gs[i * N + j] = gcur[i * N + j] * wt[j] - gsu_i * kkt[j];
          }
          for (int64_t j = 0; j < N; ++j) {
            float z = 0.0f;
#pragma GCC ivdep
            for (int64_t i = 0; i < N; ++i) z -= gsu[i] * prev[i * N + j];
            gkkt[j] += z;
          }
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
