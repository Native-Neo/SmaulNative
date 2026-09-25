#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <torch/extension.h>
#include <tuple>
#include <utility>
#include <vector>

// Native linear-attention recurrent kernels for SmaulLinear.
//
// Ivy Bridge safe: AVX1 + SSE4.2 only. No AVX2/AVX-512/VNNI/AMX.
// Compile with the same flags as fp8_cpu.cpp:
//   -O3 -mavx -mf16c -msse4.2 -mno-avx2 -mno-avx512f -ffp-contract=off
//
// The math mirrors smaul_linear._attn_reference exactly, in FP32:
//   kn_t = k_t / max(||k_t||, 1e-6),
//   S_t = S_{t-1} + kn_t (x) v_t,   z_t = z_{t-1} + kn_t,
//   num_t = q_t @ S_t,  den_t = max(q_t . z_t, eps),  y_t = num_t / den_t.
// K arrives raw (post-elu); normalization stays inside the op so the backward
// pass differentiates the true norm. No softmax, no QK^T materialization.
// State is O(D^2) per (batch, head) task; nothing of sequence-length size
// is ever stored.

static inline int64_t qkv_off(int64_t b, int64_t t, int64_t h,
                              int64_t T, int64_t H, int64_t D) {
  return ((b * T + t) * H + h) * D;
}

static inline float hsum8(__m256 v) {
  float buf[8];
  _mm256_storeu_ps(buf, v);
  return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
}

// Per-step key normalization: kn = k / max(||k||, 1e-6).
// Matches the old caller-side `k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)`
// exactly; keeping raw k visible lets the backward pass differentiate the norm.
static inline void normalize_key(const float* k, float* kn, int64_t D) {
  float n2 = 0.0f;
  for (int64_t i = 0; i < D; ++i) n2 += k[i] * k[i];
  const float n = std::sqrt(n2);
  const __m256 inv = _mm256_set1_ps(1.0f / (n < 1e-6f ? 1e-6f : n));
  int64_t i = 0;
  for (; i + 8 <= D; i += 8)
    _mm256_storeu_ps(kn + i, _mm256_mul_ps(_mm256_loadu_ps(k + i), inv));
  for (; i < D; ++i) kn[i] = k[i] * (1.0f / (n < 1e-6f ? 1e-6f : n));
}

static void attn_forward_task(const float* Q, const float* K, const float* V,
                              float* Y, float* DEN, int64_t T, int64_t H, int64_t D,
                              float eps, int64_t b, int64_t h) {
  std::vector<float> S((size_t)D * D, 0.0f), z(D, 0.0f), num(D), kn(D);
  const int64_t den_base = b * T * H + h;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t o = qkv_off(b, t, h, T, H, D);
    const float* qt = Q + o;
    const float* kt = K + o;
    const float* vt = V + o;
    float* yt = Y + o;
    normalize_key(kt, kn.data(), D);
    const float* k = kn.data();
    for (int64_t i = 0; i < D; ++i) {
      const float ki = k[i];
      z[i] += ki;
      float* Sr = S.data() + (size_t)i * D;
      const __m256 kv = _mm256_set1_ps(ki);
      int64_t j = 0;
      for (; j + 8 <= D; j += 8) {
        __m256 s = _mm256_loadu_ps(Sr + j);
        s = _mm256_add_ps(s, _mm256_mul_ps(kv, _mm256_loadu_ps(vt + j)));
        _mm256_storeu_ps(Sr + j, s);
      }
      for (; j < D; ++j) Sr[j] += ki * vt[j];
    }
    float den = 0.0f;
    for (int64_t j = 0; j < D; ++j) num[j] = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
      const __m256 qv = _mm256_set1_ps(qt[i]);
      const float* Sr = S.data() + (size_t)i * D;
      int64_t j = 0;
      for (; j + 8 <= D; j += 8) {
        __m256 n = _mm256_loadu_ps(num.data() + j);
        n = _mm256_add_ps(n, _mm256_mul_ps(qv, _mm256_loadu_ps(Sr + j)));
        _mm256_storeu_ps(num.data() + j, n);
      }
      for (; j < D; ++j) num[j] += qt[i] * Sr[j];
      den += qt[i] * z[i];
    }
    const float denc = den < eps ? eps : den;
    if (DEN) DEN[den_base + t * H] = denc;
    for (int64_t j = 0; j < D; ++j) yt[j] = num[j] / denc;
  }
}

std::pair<torch::Tensor, torch::Tensor> attn_forward(torch::Tensor Q, torch::Tensor K,
                                                     torch::Tensor V, double eps,
                                                     bool need_den) {
  TORCH_CHECK(Q.device().is_cpu() && K.device().is_cpu() && V.device().is_cpu());
  TORCH_CHECK(Q.dtype() == torch::kFloat32 && K.dtype() == torch::kFloat32 &&
              V.dtype() == torch::kFloat32);
  TORCH_CHECK(Q.dim() == 4 && K.sizes() == Q.sizes() && V.sizes() == Q.sizes());
  TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
  const int64_t B = Q.size(0), T = Q.size(1), H = Q.size(2), D = Q.size(3);
  auto Y = torch::empty_like(Q);
  torch::Tensor DEN;
  float* denp = nullptr;
  if (need_den) {
    DEN = torch::empty({B, T, H}, Q.options().dtype(torch::kFloat32));
    denp = DEN.data_ptr<float>();
  }
  const float* Qp = Q.data_ptr<float>();
  const float* Kp = K.data_ptr<float>();
  const float* Vp = V.data_ptr<float>();
  float* Yp = Y.data_ptr<float>();
  const float ef = (float)eps;
  at::parallel_for(0, B * H, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      attn_forward_task(Qp, Kp, Vp, Yp, denp, T, H, D, ef, task / H, task % H);
    }
  });
  return {Y, DEN};
}

// Backward, pass A (forward direction): recompute S/z/num/den/y per step and
// accumulate dQ. Reverse-mode adjoints: dnum = dy/den, dden = -(dy.y)/den,
// dq = S@dnum + dden*z.
static void attn_bwd_A_task(const float* Q, const float* K, const float* V,
                            const float* dY, float* dQ, int64_t T, int64_t H,
                            int64_t D, float eps, int64_t b, int64_t h) {
  std::vector<float> S((size_t)D * D, 0.0f), z(D, 0.0f), num(D), dnum(D), kn(D);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t o = qkv_off(b, t, h, T, H, D);
    const float* qt = Q + o;
    const float* kt = K + o;
    const float* vt = V + o;
    const float* dyt = dY + o;
    float* dqt = dQ + o;
    normalize_key(kt, kn.data(), D);
    const float* k = kn.data();
    for (int64_t i = 0; i < D; ++i) {
      const float ki = k[i];
      z[i] += ki;
      float* Sr = S.data() + (size_t)i * D;
      const __m256 kv = _mm256_set1_ps(ki);
      int64_t j = 0;
      for (; j + 8 <= D; j += 8) {
        __m256 s = _mm256_loadu_ps(Sr + j);
        s = _mm256_add_ps(s, _mm256_mul_ps(kv, _mm256_loadu_ps(vt + j)));
        _mm256_storeu_ps(Sr + j, s);
      }
      for (; j < D; ++j) Sr[j] += ki * vt[j];
    }
    float den = 0.0f;
    for (int64_t j = 0; j < D; ++j) num[j] = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
      const __m256 qv = _mm256_set1_ps(qt[i]);
      const float* Sr = S.data() + (size_t)i * D;
      int64_t j = 0;
      for (; j + 8 <= D; j += 8) {
        __m256 n = _mm256_loadu_ps(num.data() + j);
        n = _mm256_add_ps(n, _mm256_mul_ps(qv, _mm256_loadu_ps(Sr + j)));
        _mm256_storeu_ps(num.data() + j, n);
      }
      for (; j < D; ++j) num[j] += qt[i] * Sr[j];
      den += qt[i] * z[i];
    }
    const float denc = den < eps ? eps : den;
    const float inv = 1.0f / denc;
    float yd = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      const float yj = num[j] / denc;
      dnum[j] = dyt[j] * inv;
      yd += dyt[j] * yj;
    }
    const float dden = -yd * inv;
    for (int64_t i = 0; i < D; ++i) {
      __m256 acc = _mm256_setzero_ps();
      const float* Sr = S.data() + (size_t)i * D;
      int64_t j = 0;
      for (; j + 8 <= D; j += 8)
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(Sr + j),
                                               _mm256_loadu_ps(dnum.data() + j)));
      float s = hsum8(acc);
      for (; j < D; ++j) s += Sr[j] * dnum[j];
      dqt[i] = s + dden * z[i];
    }
  }
}

// Backward, pass B (reverse time): maintain the S/z adjoints Dm/dz and emit
// dK/dV. Per step: Dm += dnum (x) q, dz += dden*q, dk~ = Dm@v + dz,
// dv += Dm^T@k~, then the key-normalization backward k/||k||.
static void attn_bwd_B_task(const float* Q, const float* K, const float* V,
                            const float* dY, const float* Y, const float* DEN,
                            float* dK, float* dV, int64_t T, int64_t H,
                            int64_t D, int64_t b, int64_t h) {
  std::vector<float> Dm((size_t)D * D, 0.0f), dz(D, 0.0f), dnum(D), kn(D);
  const int64_t den_base = b * T * H + h;
  for (int64_t t = T - 1; t >= 0; --t) {
    const int64_t o = qkv_off(b, t, h, T, H, D);
    const float* qt = Q + o;
    const float* kt = K + o;
    const float* vt = V + o;
    const float* dyt = dY + o;
    const float* yt = Y + o;
    float* dkt = dK + o;
    float* dvt = dV + o;
    normalize_key(kt, kn.data(), D);
    const float* k = kn.data();
    const float den = DEN[den_base + t * H];
    const float inv = 1.0f / den;
    float yd = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      dnum[j] = dyt[j] * inv;
      yd += dyt[j] * yt[j];
    }
    const float dden = -yd * inv;
    for (int64_t i = 0; i < D; ++i) {
      const float qi = qt[i];
      float* Dr = Dm.data() + (size_t)i * D;
      const __m256 qv = _mm256_set1_ps(qi);
      int64_t j = 0;
      for (; j + 8 <= D; j += 8) {
        __m256 d = _mm256_loadu_ps(Dr + j);
        d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(dnum.data() + j), qv));
        _mm256_storeu_ps(Dr + j, d);
      }
      for (; j < D; ++j) Dr[j] += dnum[j] * qi;
      dz[i] += dden * qi;
    }
    for (int64_t i = 0; i < D; ++i) {
      const float* Dr = Dm.data() + (size_t)i * D;
      __m256 acc = _mm256_setzero_ps();
      const __m256 kv = _mm256_set1_ps(k[i]);
      int64_t j = 0;
      for (; j + 8 <= D; j += 8) {
        __m256 d = _mm256_loadu_ps(Dr + j);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, _mm256_loadu_ps(V + o + j)));
        __m256 dv = _mm256_loadu_ps(dvt + j);
        dv = _mm256_add_ps(dv, _mm256_mul_ps(d, kv));
        _mm256_storeu_ps(dvt + j, dv);
      }
      float dot = hsum8(acc);
      for (; j < D; ++j) {
        dot += Dr[j] * vt[j];
        dvt[j] += Dr[j] * k[i];
      }
      dkt[i] = dz[i] + dot;
    }
    // k/||k|| backward: dk = dk~/nc - k*(k.dk~)/nc^3 with nc = max(||k||, 1e-6).
    // (Matches torch autograd through the clamped normalization; the clamp
    // never binds in practice because elu+1 keys keep a non-tiny norm.)
    float n2 = 0.0f, kd = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
      n2 += kt[i] * kt[i];
      kd += kt[i] * dkt[i];
    }
    const float n = std::sqrt(n2);
    const float nc = n < 1e-6f ? 1e-6f : n;
    const float s = kd / (nc * nc * nc);
    const float invn = 1.0f / nc;
    for (int64_t i = 0; i < D; ++i) dkt[i] = dkt[i] * invn - kt[i] * s;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> attn_backward(
    torch::Tensor dY, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Y, torch::Tensor DEN, double eps) {
  TORCH_CHECK(dY.device().is_cpu() && Q.device().is_cpu() && K.device().is_cpu() &&
              V.device().is_cpu() && Y.device().is_cpu() && DEN.device().is_cpu());
  TORCH_CHECK(dY.dtype() == torch::kFloat32 && Q.dtype() == torch::kFloat32 &&
              K.dtype() == torch::kFloat32 && V.dtype() == torch::kFloat32 &&
              Y.dtype() == torch::kFloat32 && DEN.dtype() == torch::kFloat32);
  TORCH_CHECK(Q.dim() == 4 && K.sizes() == Q.sizes() && V.sizes() == Q.sizes() &&
              dY.sizes() == Q.sizes() && Y.sizes() == Q.sizes());
  TORCH_CHECK(DEN.dim() == 3 && DEN.size(0) == Q.size(0) && DEN.size(1) == Q.size(1) &&
              DEN.size(2) == Q.size(2));
  TORCH_CHECK(dY.is_contiguous() && Q.is_contiguous() && K.is_contiguous() &&
              V.is_contiguous() && Y.is_contiguous() && DEN.is_contiguous());
  const int64_t B = Q.size(0), T = Q.size(1), H = Q.size(2), D = Q.size(3);
  auto dQ = torch::empty_like(Q);
  auto dK = torch::zeros_like(K);
  auto dV = torch::zeros_like(V);
  const float* dYp = dY.data_ptr<float>();
  const float* Qp = Q.data_ptr<float>();
  const float* Kp = K.data_ptr<float>();
  const float* Vp = V.data_ptr<float>();
  const float* Yp = Y.data_ptr<float>();
  const float* DENp = DEN.data_ptr<float>();
  float* dQp = dQ.data_ptr<float>();
  float* dKp = dK.data_ptr<float>();
  float* dVp = dV.data_ptr<float>();
  const float ef = (float)eps;
  at::parallel_for(0, B * H, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t b = task / H, h = task % H;
      attn_bwd_A_task(Qp, Kp, Vp, dYp, dQp, T, H, D, ef, b, h);
      attn_bwd_B_task(Qp, Kp, Vp, dYp, Yp, DENp, dKp, dVp, T, H, D, b, h);
    }
  });
  return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_forward", &attn_forward, "linear-attention recurrent forward (AVX1)");
  m.def("attn_backward", &attn_backward, "linear-attention recurrent backward (AVX1)");
}
