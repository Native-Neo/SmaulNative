#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <immintrin.h>
#include <limits>
#include <torch/extension.h>
#include <vector>

void lion_step(torch::Tensor p, torch::Tensor g, torch::Tensor m, double lr, double b1, double b2, double wd) {
  TORCH_CHECK(p.device().is_cpu() && g.device().is_cpu() && m.device().is_cpu());
  TORCH_CHECK(p.dtype() == torch::kFloat32 && g.dtype() == torch::kFloat32 && m.dtype() == torch::kFloat32);
  TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous());
  TORCH_CHECK(p.numel() == g.numel() && p.numel() == m.numel());
  auto pp = p.data_ptr<float>(); auto gg = g.data_ptr<float>(); auto mm = m.data_ptr<float>();
  const auto n = p.numel(); const float f_lr = (float)lr, f_b1 = (float)b1, f_b2 = (float)b2, f_wd = (float)(1.0 - lr * wd);
  at::parallel_for(0, n, 1 << 20, [&](int64_t begin, int64_t end) {
    int64_t i = begin;
    const __m256 vb1 = _mm256_set1_ps(f_b1), v1_b1 = _mm256_set1_ps(1.0f - f_b1), vb2 = _mm256_set1_ps(f_b2), v1_b2 = _mm256_set1_ps(1.0f - f_b2);
    const __m256 vwd = _mm256_set1_ps(f_wd), vlr = _mm256_set1_ps(f_lr), vzero = _mm256_setzero_ps(), vone = _mm256_set1_ps(1.0f);
    for (; i + 8 <= end; i += 8) {
      __m256 m_val = _mm256_loadu_ps(mm + i), g_val = _mm256_loadu_ps(gg + i), p_val = _mm256_loadu_ps(pp + i);
      const __m256 v = _mm256_add_ps(_mm256_mul_ps(vb1, m_val), _mm256_mul_ps(v1_b1, g_val));
      const __m256 gt = _mm256_cmp_ps(v, vzero, _CMP_GT_OQ), lt = _mm256_cmp_ps(v, vzero, _CMP_LT_OQ);
      const __m256 sign = _mm256_sub_ps(_mm256_and_ps(gt, vone), _mm256_and_ps(lt, vone));
      p_val = _mm256_sub_ps(_mm256_mul_ps(p_val, vwd), _mm256_mul_ps(vlr, sign));
      m_val = _mm256_add_ps(_mm256_mul_ps(vb2, m_val), _mm256_mul_ps(v1_b2, g_val));
      _mm256_storeu_ps(pp + i, p_val); _mm256_storeu_ps(mm + i, m_val);
    }
    for (; i < end; ++i) { float v = f_b1 * mm[i] + (1.0f - f_b1) * gg[i]; pp[i] *= f_wd; pp[i] -= f_lr * (v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f)); mm[i] = f_b2 * mm[i] + (1.0f - f_b2) * gg[i]; }
  });
}

static const std::array<float, 128>& rqt_level_table() {
  static const auto table = [] {
    std::array<float, 128> out{};
    for (int bits : {4, 6}) {
      const int mbits = bits == 4 ? 1 : 2, ebits = bits == 4 ? 2 : 3, bias = (1 << (ebits - 1)) - 1, base = bits == 4 ? 0 : 64;
      for (int code = 0; code < (1 << bits); ++code) {
        const int exp = (code >> mbits) & ((1 << ebits) - 1), mant = code & ((1 << mbits) - 1);
        const float sign = (code >> (bits - 1)) ? -1.0f : 1.0f;
        const float value = exp == 0 ? (mant / float(1 << mbits)) * std::ldexp(1.0f, 1 - bias) : (1.0f + mant / float(1 << mbits)) * std::ldexp(1.0f, exp - bias);
        out[base + code] = sign * value;
      }
    }
    return out;
  }();
  return table;
}

struct RQTOrderedLevels { std::array<float, 64> values{}; std::array<uint8_t, 64> codes{}; };

static const RQTOrderedLevels& rqt_ordered_levels(int bits) {
  static const auto fp4 = [] {
    RQTOrderedLevels out;
    const auto& levels = rqt_level_table();
    std::array<int, 16> order{};
    for (int i = 0; i < 16; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return levels[a] < levels[b]; });
    for (int i = 0; i < 16; ++i) { out.values[i] = levels[order[i]]; out.codes[i] = (uint8_t)order[i]; }
    return out;
  }();
  static const auto fp6 = [] {
    RQTOrderedLevels out;
    const auto& levels = rqt_level_table();
    std::array<int, 64> order{};
    for (int i = 0; i < 64; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return levels[64 + a] < levels[64 + b]; });
    for (int i = 0; i < 64; ++i) { out.values[i] = levels[64 + order[i]]; out.codes[i] = (uint8_t)order[i]; }
    return out;
  }();
  return bits == 4 ? fp4 : fp6;
}

static inline float rqt_fp8_level(int code) {
  const int exponent = (code >> 3) & 15, mantissa = code & 7;
  const float sign = (code & 128) ? -1.0f : 1.0f;
  if (exponent == 0) return sign * std::ldexp((float)mantissa, -9);
  if (exponent == 15 && mantissa == 7) return std::numeric_limits<float>::quiet_NaN();
  return sign * (1.0f + mantissa / 8.0f) * std::ldexp(1.0f, exponent - 7);
}

static inline float rqt_level(int code, int bits) {
  return bits == 8 ? rqt_fp8_level(code) : rqt_level_table()[(bits == 4 ? 0 : 64) + code];
}

static inline uint8_t rqt_code(const uint8_t* packed, int index, int bits) {
  if (bits == 4) { const uint8_t p = packed[index >> 1]; return (index & 1) ? p & 15 : p >> 4; }
  if (bits == 8) return packed[index];
  const uint8_t* p = packed + (index >> 2) * 3;
  switch (index & 3) { case 0: return p[0] >> 2; case 1: return ((p[0] & 3) << 4) | (p[1] >> 4); case 2: return ((p[1] & 15) << 2) | (p[2] >> 6); default: return p[2] & 63; }
}

static inline void rqt_set_code(uint8_t* packed, int index, int bits, uint8_t code) {
  if (bits == 4) { uint8_t& p = packed[index >> 1]; if (index & 1) p = (p & 0xF0) | code; else p = (p & 0x0F) | (code << 4); return; }
  if (bits == 8) { packed[index] = code; return; }
  uint8_t* p = packed + (index >> 2) * 3;
  switch (index & 3) { case 0: p[0] = (p[0] & 0x03) | (code << 2); break; case 1: p[0] = (p[0] & 0xFC) | (code >> 4); p[1] = (p[1] & 0x0F) | ((code & 15) << 4); break; case 2: p[1] = (p[1] & 0xF0) | (code >> 2); p[2] = (p[2] & 0x3F) | ((code & 3) << 6); break; default: p[2] = (p[2] & 0xC0) | code; }
}

static inline int rqt_nearest_code(float value, int bits) {
  if (bits == 8) {
    if (!std::isfinite(value)) return value < 0.0f ? 0xFE : 0x7E;
    int best = 0; float best_error = std::numeric_limits<float>::infinity();
    for (int code = 0; code < 256; ++code) {
      if ((code & 0x7F) == 0x7F) continue;
      const float error = std::abs(value - rqt_fp8_level(code));
      if (error < best_error) { best_error = error; best = code; }
    }
    return best;
  }
  const auto& levels = rqt_level_table();
  const int count = 1 << bits;
  const int half = count >> 1;
  const int base = bits == 4 ? 0 : 64;
  const bool negative = value < 0.0f;
  const float magnitude = std::abs(value);

  int lo = 0, hi = half;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (levels[base + mid] < magnitude) lo = mid + 1;
    else hi = mid;
  }

  int code;
  if (lo == 0) {
    code = 0;
  } else if (lo == half) {
    code = half - 1;
  } else {
    const float left = levels[base + lo - 1];
    const float right = levels[base + lo];
    code = std::abs(magnitude - left) <= std::abs(right - magnitude) ? lo - 1 : lo;
  }
  return negative ? code | half : code;
}

void rqt_lion_step(torch::Tensor packed, torch::Tensor scale, torch::Tensor grad, torch::Tensor avg,
                   int64_t in_features, int64_t out_features, int64_t bits,
                   double lr, double b1, double b2, double decay) {
  TORCH_CHECK(packed.device().is_cpu() && scale.device().is_cpu() && grad.device().is_cpu() && avg.device().is_cpu());
  TORCH_CHECK(packed.element_size() == 1 && scale.dtype() == torch::kFloat32 &&
              grad.dtype() == torch::kFloat32 && avg.dtype() == torch::kFloat32);
  TORCH_CHECK(packed.is_contiguous() && scale.is_contiguous() && grad.is_contiguous() && avg.is_contiguous());
  TORCH_CHECK(bits == 4 || bits == 6 || bits == 8);
  TORCH_CHECK(scale.numel() == (bits == 8 ? 0 : out_features) &&
              grad.numel() == out_features * in_features &&
              avg.numel() == out_features * in_features);
  const int64_t stride = bits == 4 ? (in_features + 1) / 2 : bits == 6 ? ((in_features + 3) / 4) * 3 : in_features;
  TORCH_CHECK(packed.numel() == out_features * stride);
  auto pp = static_cast<uint8_t*>(packed.data_ptr()); auto ss = scale.data_ptr<float>();
  auto gg = grad.data_ptr<float>(); auto mm = avg.data_ptr<float>();
  const float f_lr = (float)lr;
  const float decay_mul = (float)(1.0 - decay);
  const float max_level = std::abs(rqt_level((1 << (int)bits) - 1, (int)bits));
  const int ibits = (int)bits;

  at::parallel_for(0, out_features, 1, [&](int64_t begin, int64_t end) {
    std::vector<int8_t> signs((size_t)in_features);
    for (int64_t row = begin; row < end; ++row) {
      uint8_t* dst = pp + row * stride;
      const float* grow = gg + row * in_features;
      float* mrow = mm + row * in_features;
      const float old_scale = ibits == 8 ? 1.0f : ss[row];
      float max_abs = 0.0f;

      if (ibits == 8) {
        for (int64_t col = 0; col < in_features; ++col) {
          const float g = grow[col], old_m = mrow[col];
          const float mixed = old_m * (float)b1 + g * (float)(1.0 - b1);
          mrow[col] = mixed * (float)b2 + g * (float)(1.0 - b2);
          const float value = rqt_level(rqt_code(dst, (int)col, ibits), ibits) * decay_mul - f_lr * (mixed > 0.0f ? 1.0f : (mixed < 0.0f ? -1.0f : 0.0f));
          rqt_set_code(dst, (int)col, ibits, (uint8_t)rqt_nearest_code(value, ibits));
        }
        continue;
      }

      for (int64_t col = 0; col < in_features; ++col) {
        const float g = grow[col];
        const float old_m = mrow[col];
        float mixed = old_m * (float)b1;
        mixed += g * (float)(1.0 - b1);
        const int8_t sign = mixed > 0.0f ? 1 : (mixed < 0.0f ? -1 : 0);
        signs[(size_t)col] = sign;
        float second = mixed * (float)b2;
        second += g * (float)(1.0 - b2);
        mrow[col] = second;
        const float old_w = rqt_level(rqt_code(dst, (int)col, ibits), ibits) * old_scale;
        const float value = old_w * decay_mul - f_lr * sign;
        max_abs = std::max(max_abs, std::abs(value));
      }

      const float new_scale = std::max(max_abs, std::numeric_limits<float>::epsilon()) / max_level;
      ss[row] = new_scale;

      for (int64_t col = 0; col < in_features; ++col) {
        const float old_w = rqt_level(rqt_code(dst, (int)col, ibits), ibits) * old_scale;
        const float value = (old_w * decay_mul - f_lr * signs[(size_t)col]) / new_scale;
        rqt_set_code(dst, (int)col, ibits, (uint8_t)rqt_nearest_code(value, ibits));
      }
    }
  });
}

void rqt_requant_step(torch::Tensor packed, torch::Tensor scale, torch::Tensor update, int64_t in_features, int64_t out_features, int64_t bits, double decay) {
  TORCH_CHECK(packed.device().is_cpu() && scale.device().is_cpu() && update.device().is_cpu());
  TORCH_CHECK(packed.element_size() == 1 && scale.dtype() == torch::kFloat32 && update.dtype() == torch::kFloat32);
  TORCH_CHECK(packed.is_contiguous() && scale.is_contiguous() && update.is_contiguous()); TORCH_CHECK(bits == 4 || bits == 6 || bits == 8);
  const int64_t stride = bits == 4 ? (in_features + 1) / 2 : bits == 6 ? ((in_features + 3) / 4) * 3 : in_features;
  TORCH_CHECK(scale.numel() == (bits == 8 ? 0 : out_features) && update.numel() == out_features * in_features && packed.numel() == out_features * stride);
  auto pp = static_cast<uint8_t*>(packed.data_ptr()); auto ss = scale.data_ptr<float>(); auto uu = update.data_ptr<float>();
  const float decay_mul = (float)(1.0 - decay);
  const float max_level = std::abs(rqt_level((1 << (int)bits) - 1, (int)bits));
  const int ibits = (int)bits;
  at::parallel_for(0, out_features, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      uint8_t* dst = pp + row * stride; const float old_scale = ibits == 8 ? 1.0f : ss[row]; float max_abs = 0.0f;
      if (ibits == 8) {
        for (int64_t col = 0; col < in_features; ++col) {
          const float value = rqt_level(rqt_code(dst, (int)col, ibits), ibits) * decay_mul - uu[row * in_features + col];
          rqt_set_code(dst, (int)col, ibits, (uint8_t)rqt_nearest_code(value, ibits));
        }
        continue;
      }
      for (int64_t col = 0; col < in_features; ++col) {
        const float value = rqt_level(rqt_code(dst, (int)col, ibits), ibits) * old_scale * decay_mul - uu[row * in_features + col];
        max_abs = std::max(max_abs, std::abs(value));
      }
      const float new_scale = std::max(max_abs, std::numeric_limits<float>::epsilon()) / max_level; ss[row] = new_scale;
      for (int64_t col = 0; col < in_features; ++col) {
        const float value = (rqt_level(rqt_code(dst, (int)col, ibits), ibits) * old_scale * decay_mul - uu[row * in_features + col]) / new_scale;
        rqt_set_code(dst, (int)col, ibits, (uint8_t)rqt_nearest_code(value, ibits));
      }
    }
  });
}

torch::Tensor rqt_linear_forward(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, int64_t in_features, int64_t out_features, int64_t bits) {
  TORCH_CHECK(x.device().is_cpu() && packed.device().is_cpu() && scale.device().is_cpu());
  TORCH_CHECK(x.dtype() == torch::kFloat32 && packed.element_size() == 1 && scale.dtype() == torch::kFloat32);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == in_features && packed.is_contiguous() && scale.is_contiguous() && x.is_contiguous());
  TORCH_CHECK(bits == 4 || bits == 6 || bits == 8);
  TORCH_CHECK(scale.numel() == (bits == 8 ? 0 : out_features));
  auto out = torch::empty({x.size(0), out_features}, x.options());
  const auto rows = x.size(0), n = in_features, m = out_features;
  const auto* xx = x.data_ptr<float>(); const auto* pp = static_cast<const uint8_t*>(packed.data_ptr());
  const auto* ss = scale.data_ptr<float>(); auto* yy = out.data_ptr<float>();
  const int64_t stride = bits == 4 ? (n + 1) / 2 : bits == 6 ? ((n + 3) / 4) * 3 : n;
  const int ibits = (int)bits; const auto& levels = rqt_level_table(); const int base = bits == 4 ? 0 : 64;

  at::parallel_for(0, rows * m, 64, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t r = task / m, o = task - r * m;
      const uint8_t* row = pp + o * stride; const float s = ibits == 8 ? 1.0f : ss[o]; const float* level = levels.data() + base;
      const float* xr = xx + r * n;
      float acc = 0.0f;
      if (ibits == 8) {
        for (int64_t i = 0; i < n; ++i) acc += xr[i] * rqt_level(row[i], ibits);
        yy[r * m + o] = acc;
        continue;
      }
      float scaled_levels[64];
      if (ibits == 6) for (int code = 0; code < 64; ++code) scaled_levels[code] = level[code] * s;
      if (ibits == 4) {
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) {
          const uint8_t p0 = row[i >> 1], p1 = row[(i >> 1) + 1], p2 = row[(i >> 1) + 2], p3 = row[(i >> 1) + 3];
          acc += xr[i] * level[p0 >> 4] * s;
          acc += xr[i + 1] * level[p0 & 15] * s;
          acc += xr[i + 2] * level[p1 >> 4] * s;
          acc += xr[i + 3] * level[p1 & 15] * s;
          acc += xr[i + 4] * level[p2 >> 4] * s;
          acc += xr[i + 5] * level[p2 & 15] * s;
          acc += xr[i + 6] * level[p3 >> 4] * s;
          acc += xr[i + 7] * level[p3 & 15] * s;
        }
        for (; i < n; ++i) {
          const uint8_t p = row[i >> 1];
          acc += xr[i] * level[(i & 1) ? (p & 15) : (p >> 4)] * s;
        }
      } else {
        int64_t i = 0;
        for (; i + 4 <= n; i += 4) {
          const uint8_t* p = row + (i >> 2) * 3;
          const uint8_t c0 = p[0] >> 2;
          const uint8_t c1 = ((p[0] & 3) << 4) | (p[1] >> 4);
          const uint8_t c2 = ((p[1] & 15) << 2) | (p[2] >> 6);
          const uint8_t c3 = p[2] & 63;
          acc += xr[i] * scaled_levels[c0];
          acc += xr[i + 1] * scaled_levels[c1];
          acc += xr[i + 2] * scaled_levels[c2];
          acc += xr[i + 3] * scaled_levels[c3];
        }
        for (; i < n; ++i) {
          acc += xr[i] * scaled_levels[rqt_code(row, (int)i, ibits)];
        }
      }
      yy[r * m + o] = acc;
    }
  });
  return out;
}

torch::Tensor rqt_linear_backward_input(torch::Tensor grad, torch::Tensor packed, torch::Tensor scale, int64_t in_features, int64_t out_features, int64_t bits) {
  TORCH_CHECK(grad.device().is_cpu() && packed.device().is_cpu() && scale.device().is_cpu());
  TORCH_CHECK(grad.dtype() == torch::kFloat32 && packed.element_size() == 1 && scale.dtype() == torch::kFloat32);
  TORCH_CHECK(grad.dim() == 2 && grad.size(1) == out_features && grad.is_contiguous() && packed.is_contiguous() && scale.is_contiguous());
  TORCH_CHECK(bits == 4 || bits == 6 || bits == 8);
  TORCH_CHECK(scale.numel() == (bits == 8 ? 0 : out_features));
  auto out = torch::empty({grad.size(0), in_features}, grad.options());
  const auto rows = grad.size(0), n = in_features, m = out_features;
  const auto* gg = grad.data_ptr<float>(); const auto* pp = static_cast<const uint8_t*>(packed.data_ptr());
  const auto* ss = scale.data_ptr<float>(); auto* xx = out.data_ptr<float>();
  const int64_t stride = bits == 4 ? (n + 1) / 2 : bits == 6 ? ((n + 3) / 4) * 3 : n;
  const int ibits = (int)bits; const auto& levels = rqt_level_table(); const int base = bits == 4 ? 0 : 64;

  at::parallel_for(0, rows * ((n + 7) / 8), 16, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t r = task / ((n + 7) / 8), block = task - r * ((n + 7) / 8), i = block * 8;
      const float* gr = gg + r * m;
      const int64_t width = std::min<int64_t>(8, n - i);

      if (ibits == 8) {
        for (int64_t k = 0; k < width; ++k) {
          float acc = 0.0f;
          for (int64_t o = 0; o < m; ++o) acc += gr[o] * rqt_level(pp[o * stride + i + k], ibits);
          xx[r * n + i + k] = acc;
        }
        continue;
      }

      if (width == 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int64_t o = 0; o < m; ++o) {
          const uint8_t* row = pp + o * stride;
          float w[8];
          for (int k = 0; k < 8; ++k) w[k] = levels[base + rqt_code(row, (int)(i + k), ibits)] * ss[o];
          const __m256 weights = _mm256_set_ps(w[7], w[6], w[5], w[4], w[3], w[2], w[1], w[0]);
          const __m256 g = _mm256_set1_ps(gr[o]);
          acc = _mm256_add_ps(acc, _mm256_mul_ps(weights, g));
        }
        _mm256_storeu_ps(xx + r * n + i, acc);
      } else {
        for (int64_t k = 0; k < width; ++k) {
          float acc = 0.0f;
          for (int64_t o = 0; o < m; ++o) {
            const uint8_t* row = pp + o * stride;
            acc += gr[o] * levels[base + rqt_code(row, (int)(i + k), ibits)] * ss[o];
          }
          xx[r * n + i + k] = acc;
        }
      }
    }
  });
  return out;
}

torch::Tensor rqt_linear_backward_weight(torch::Tensor x, torch::Tensor grad, int64_t in_features, int64_t out_features) {
  TORCH_CHECK(x.device().is_cpu() && grad.device().is_cpu());
  TORCH_CHECK(x.dtype() == torch::kFloat32 && grad.dtype() == torch::kFloat32 && x.dim() == 2 && grad.dim() == 2);
  TORCH_CHECK(x.size(0) == grad.size(0) && x.size(1) == in_features && grad.size(1) == out_features && x.is_contiguous() && grad.is_contiguous());
  return torch::mm(grad.transpose(0, 1), x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lion_step", &lion_step, "Fused Lion update with AVX");
  m.def("rqt_requant_step", &rqt_requant_step, "Fused packed RQT requantization");
  m.def("rqt_lion_step", &rqt_lion_step, "Fused Lion update and packed RQT requantization");
  m.def("rqt_linear_forward", &rqt_linear_forward, "Packed RQT linear forward");
  m.def("rqt_linear_backward_input", &rqt_linear_backward_input, "Packed RQT linear input gradient");
  m.def("rqt_linear_backward_weight", &rqt_linear_backward_weight, "RQT linear weight gradient");
}
