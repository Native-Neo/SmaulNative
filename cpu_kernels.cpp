#include <ATen/Parallel.h>
#include <cmath>
#include <immintrin.h>
#include <torch/extension.h>

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

static inline float rqt_level(int code, int bits) {
  const int mbits = bits == 4 ? 1 : 2;
  const int ebits = bits == 4 ? 2 : 3;
  const int bias = (1 << (ebits - 1)) - 1;
  const int exp = (code >> mbits) & ((1 << ebits) - 1);
  const int mant = code & ((1 << mbits) - 1);
  const float sign = (code >> (bits - 1)) ? -1.0f : 1.0f;
  const float value = exp == 0 ? (mant / float(1 << mbits)) * std::ldexp(1.0f, 1 - bias) : (1.0f + mant / float(1 << mbits)) * std::ldexp(1.0f, exp - bias);
  return sign * value;
}

static inline uint8_t rqt_code(const uint8_t* packed, int index, int bits) {
  if (bits == 4) {
    const uint8_t p = packed[index >> 1];
    return (index & 1) ? (p & 15) : (p >> 4);
  }
  const int group = index >> 2;
  const int off = index & 3;
  const uint8_t* p = packed + group * 3;
  if (off == 0) return p[0] >> 2;
  if (off == 1) return ((p[0] & 3) << 4) | (p[1] >> 4);
  if (off == 2) return ((p[1] & 15) << 2) | (p[2] >> 6);
  return p[2] & 63;
}

static inline void rqt_set_code(uint8_t* packed, int index, int bits, uint8_t code) {
  if (bits == 4) {
    uint8_t& p = packed[index >> 1];
    if (index & 1) p = (p & 0xF0) | code;
    else p = (p & 0x0F) | (code << 4);
    return;
  }
  const int group = index >> 2;
  const int off = index & 3;
  uint8_t* p = packed + group * 3;
  if (off == 0) p[0] = (p[0] & 0x03) | (code << 2);
  else if (off == 1) { p[0] = (p[0] & 0xFC) | (code >> 4); p[1] = (p[1] & 0x0F) | ((code & 15) << 4); }
  else if (off == 2) { p[1] = (p[1] & 0xF0) | (code >> 2); p[2] = (p[2] & 0x3F) | ((code & 3) << 6); }
  else p[2] = (p[2] & 0xC0) | code;
}

void rqt_requant_step(torch::Tensor packed, torch::Tensor scale, torch::Tensor update, int64_t in_features, int64_t out_features, int64_t bits, double decay) {
  TORCH_CHECK(packed.device().is_cpu() && scale.device().is_cpu() && update.device().is_cpu());
  TORCH_CHECK(packed.dtype() == torch::kUInt8 && scale.dtype() == torch::kFloat32 && update.dtype() == torch::kFloat32);
  TORCH_CHECK(packed.is_contiguous() && scale.is_contiguous() && update.is_contiguous());
  TORCH_CHECK(bits == 4 || bits == 6);
  TORCH_CHECK(scale.numel() == out_features && update.numel() == out_features * in_features);
  const int64_t stride = bits == 4 ? (in_features + 1) / 2 : ((in_features + 3) / 4) * 3;
  TORCH_CHECK(packed.numel() == out_features * stride);
  auto pp = packed.data_ptr<uint8_t>();
  auto ss = scale.data_ptr<float>();
  auto uu = update.data_ptr<float>();
  const float decay_mul = (float)(1.0 - decay);
  const float max_level = rqt_level((1 << bits) - 1, bits);
  at::parallel_for(0, out_features, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      uint8_t* dst = pp + row * stride;
      const float old_scale = ss[row];
      float max_abs = 0.0f;
      for (int64_t col = 0; col < in_features; ++col) {
        const float value = rqt_level(rqt_code(dst, (int)col, (int)bits), (int)bits) * old_scale * decay_mul - uu[row * in_features + col];
        max_abs = std::max(max_abs, std::abs(value));
      }
      const float new_scale = std::max(max_abs, std::numeric_limits<float>::epsilon()) / max_level;
      ss[row] = new_scale;
      for (int64_t col = 0; col < in_features; ++col) {
        const float value = (rqt_level(rqt_code(dst, (int)col, (int)bits), (int)bits) * old_scale * decay_mul - uu[row * in_features + col]) / new_scale;
        int best = 0;
        float best_dist = std::abs(value - rqt_level(0, (int)bits));
        for (int code = 1; code < (1 << bits); ++code) {
          const float dist = std::abs(value - rqt_level(code, (int)bits));
          if (dist < best_dist) { best = code; best_dist = dist; }
        }
        rqt_set_code(dst, (int)col, (int)bits, (uint8_t)best);
      }
    }
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lion_step", &lion_step, "Fused Lion update with AVX");
  m.def("rqt_requant_step", &rqt_requant_step, "Fused packed RQT requantization");
}
