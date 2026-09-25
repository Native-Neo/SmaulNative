#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <torch/extension.h>

// Ivy Bridge safe: AVX1 + SSE4.2 only. No AVX2/AVX-512/VNNI/AMX.
// Compile with: -mavx -mf16c -msse4.2 -mno-avx2 -mno-avx512f -O3
// E4M3 decode via 256-entry FP32 LUT; AVX used for FP32 accumulation.

namespace {
std::once_flag g_lut_once;
float g_lut_table[256];

void init_lut_table() {
  for (int c = 0; c < 256; ++c) {
    int e = (c >> 3) & 15, m = c & 7;
    float s = (c & 128) ? -1.0f : 1.0f;
    float v;
    if (e == 15 && m == 7) v = s * 448.0f;
    else if (e == 0) v = s * ldexpf((float)m, -9);
    else v = s * (1.0f + m / 8.0f) * ldexpf(1.0f, e - 7);
    g_lut_table[c] = v;
  }
}
} // namespace

static const float* fp8_lut() {
  std::call_once(g_lut_once, init_lut_table);
  return g_lut_table;
}

// Forward: block over rows (MR) x outputs (OB). Each [OB x K] weight tile is
// decoded ONCE into a transposed stack buffer shared by all MR rows.
static void fp8_forward_task(const float* xp, const uint8_t* wp, const float* sp, float* yp,
                             const float* lut, int64_t in_f, int64_t out_f, int64_t nt, int64_t tile,
                             int64_t r0, int64_t MR, int64_t ob0, int64_t OBR) {
  float acc[32][64];
  for (int64_t r = 0; r < MR; ++r)
    for (int64_t o = 0; o < OBR; ++o) acc[r][o] = 0.0f;
  float wt[64][64];
  for (int64_t kt = 0; kt < in_f; kt += 64) {
    const int64_t Kr = std::min<int64_t>(64, in_f - kt);
    if (MR == 32 && OBR == 64 && Kr == 64) {
      for (int64_t k2 = 0; k2 < 64; ++k2) {
        const int64_t st = (kt + k2) / tile;
        for (int64_t o2 = 0; o2 < 64; o2 += 8) {
          float wv[8];
          for (int j = 0; j < 8; ++j) {
            const int64_t o = ob0 + o2 + j;
            wv[j] = lut[wp[o * in_f + kt + k2]] * sp[o * nt + st];
          }
          _mm256_storeu_ps(wt[k2] + o2, _mm256_loadu_ps(wv));
        }
      }
      for (int64_t r = 0; r < 32; ++r) {
        const float* xr = xp + (r0 + r) * in_f + kt;
        for (int64_t k2 = 0; k2 < 64; ++k2) {
          const __m256 xv = _mm256_set1_ps(xr[k2]);
          for (int64_t o2 = 0; o2 < 64; o2 += 8) {
            __m256 av = _mm256_loadu_ps(acc[r] + o2);
            av = _mm256_add_ps(av, _mm256_mul_ps(xv, _mm256_loadu_ps(wt[k2] + o2)));
            _mm256_storeu_ps(acc[r] + o2, av);
          }
        }
      }
    } else {
      for (int64_t r = 0; r < MR; ++r) {
        const float* xr = xp + (r0 + r) * in_f + kt;
        for (int64_t k2 = 0; k2 < Kr; ++k2) {
          const float xv = xr[k2];
          const int64_t st = (kt + k2) / tile;
          for (int64_t o2 = 0; o2 < OBR; ++o2) {
            const int64_t o = ob0 + o2;
            acc[r][o2] += xv * lut[wp[o * in_f + kt + k2]] * sp[o * nt + st];
          }
        }
      }
    }
  }
  for (int64_t r = 0; r < MR; ++r)
    for (int64_t o = 0; o < OBR; ++o) yp[(r0 + r) * out_f + ob0 + o] = acc[r][o];
}

torch::Tensor fp8_forward(torch::Tensor x, torch::Tensor w, torch::Tensor s,
                          int64_t in_f, int64_t out_f, int64_t tile) {
  TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && s.device().is_cpu(),
              "fp8_forward: all tensors must be CPU");
  TORCH_CHECK(x.dtype() == torch::kFloat32 && w.dtype() == torch::kUInt8 && s.dtype() == torch::kFloat32,
              "fp8_forward: expected dtypes f32/u8/f32");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && s.is_contiguous(),
              "fp8_forward: all tensors must be contiguous");
  TORCH_CHECK(tile > 0, "fp8_forward: tile must be > 0, got ", tile);
  TORCH_CHECK(in_f >= 0 && out_f >= 0, "fp8_forward: in_f/out_f must be >= 0");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == in_f,
              "fp8_forward: x must be [rows, in_f], got dim=", x.dim());
  // w/s may be [out_f, in_f]/[out_f, nt] or flattened 1-D (compute.py passes
  // reshape(-1)); pointer math only needs contiguity + correct numel.
  TORCH_CHECK((w.dim() == 2 && w.size(0) == out_f && w.size(1) == in_f) ||
              (w.dim() == 1 && w.size(0) == out_f * in_f),
              "fp8_forward: w must be [out_f, in_f] or flattened [out_f*in_f]");
  const int64_t rows = x.size(0), nt = (in_f + tile - 1) / tile;
  TORCH_CHECK(w.numel() == out_f * in_f && s.numel() == out_f * nt,
              "fp8_forward: size mismatch w=", w.numel(), " s=", s.numel(),
              " expected ", out_f * in_f, " and ", out_f * nt);
  TORCH_CHECK((s.dim() == 2 && s.size(0) == out_f && s.size(1) == nt) ||
              (s.dim() == 1 && s.size(0) == out_f * nt),
              "fp8_forward: s must be [out_f, nt] or flattened");
  auto out = torch::empty({rows, out_f}, x.options());
  const float* xp = x.data_ptr<float>();
  const uint8_t* wp = w.data_ptr<uint8_t>();
  const float* sp = s.data_ptr<float>();
  float* yp = out.data_ptr<float>();
  const float* lut = fp8_lut();
  const int64_t MR = 32, OB = 64;
  const int64_t nmb = (rows + MR - 1) / MR, nob = (out_f + OB - 1) / OB;
  at::parallel_for(0, nmb * nob, 4, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t mb = task / nob, ob = task % nob;
      fp8_forward_task(xp, wp, sp, yp, lut, in_f, out_f, nt, tile,
                       mb * MR, std::min<int64_t>(MR, rows - mb * MR),
                       ob * OB, std::min<int64_t>(OB, out_f - ob * OB));
    }
  });
  return out;
}

// Backward-input: block over rows (RB) x input-cols (IB). Each [OBlock x IB]
// weight sub-tile is decoded ONCE and reused across all RB rows
// (previously re-decoded per row: rows x fewer gathers now).
static void fp8_backward_task(const float* gp, const uint8_t* wp, const float* sp, float* dxp,
                              const float* lut, int64_t in_f, int64_t out_f, int64_t nt, int64_t tile,
                              int64_t r0, int64_t RB, int64_t ib0, int64_t IBR) {
  float acc[16][8];
  for (int64_t r = 0; r < RB; ++r)
    for (int64_t k = 0; k < IBR; ++k) acc[r][k] = 0.0f;
  float wsub[64][8];
  for (int64_t ob = 0; ob < out_f; ob += 64) {
    const int64_t OBR = std::min<int64_t>(64, out_f - ob);
    if (RB == 16 && IBR == 8 && OBR == 64) {
      for (int64_t o2 = 0; o2 < 64; ++o2) {
        const int64_t o = ob + o2;
        for (int64_t k2 = 0; k2 < 8; ++k2)
          wsub[o2][k2] = lut[wp[o * in_f + ib0 + k2]] * sp[o * nt + (ib0 + k2) / tile];
      }
      for (int64_t r = 0; r < 16; ++r) {
        const float* gr = gp + (r0 + r) * out_f + ob;
        __m256 av = _mm256_loadu_ps(acc[r]);
        for (int64_t o2 = 0; o2 < 64; ++o2) {
          const __m256 gv = _mm256_set1_ps(gr[o2]);
          av = _mm256_add_ps(av, _mm256_mul_ps(gv, _mm256_loadu_ps(wsub[o2])));
        }
        _mm256_storeu_ps(acc[r], av);
      }
    } else {
      for (int64_t o2 = 0; o2 < OBR; ++o2) {
        const int64_t o = ob + o2;
        for (int64_t r = 0; r < RB; ++r) {
          const float gv = gp[(r0 + r) * out_f + o];
          for (int64_t k2 = 0; k2 < IBR; ++k2)
            acc[r][k2] += gv * lut[wp[o * in_f + ib0 + k2]] * sp[o * nt + (ib0 + k2) / tile];
        }
      }
    }
  }
  for (int64_t r = 0; r < RB; ++r)
    for (int64_t k = 0; k < IBR; ++k) dxp[(r0 + r) * in_f + ib0 + k] = acc[r][k];
}

torch::Tensor fp8_backward_input(torch::Tensor g, torch::Tensor w, torch::Tensor s,
                                 int64_t in_f, int64_t out_f, int64_t tile) {
  TORCH_CHECK(g.device().is_cpu() && w.device().is_cpu() && s.device().is_cpu(),
              "fp8_backward_input: all tensors must be CPU");
  TORCH_CHECK(g.dtype() == torch::kFloat32 && w.dtype() == torch::kUInt8 && s.dtype() == torch::kFloat32,
              "fp8_backward_input: expected dtypes f32/u8/f32");
  TORCH_CHECK(g.is_contiguous() && w.is_contiguous() && s.is_contiguous(),
              "fp8_backward_input: all tensors must be contiguous");
  TORCH_CHECK(tile > 0, "fp8_backward_input: tile must be > 0, got ", tile);
  TORCH_CHECK(in_f >= 0 && out_f >= 0, "fp8_backward_input: in_f/out_f must be >= 0");
  TORCH_CHECK(g.dim() == 2 && g.size(1) == out_f,
              "fp8_backward_input: g must be [rows, out_f], got dim=", g.dim());
  TORCH_CHECK((w.dim() == 2 && w.size(0) == out_f && w.size(1) == in_f) ||
              (w.dim() == 1 && w.size(0) == out_f * in_f),
              "fp8_backward_input: w must be [out_f, in_f] or flattened");
  const int64_t rows = g.size(0);
  const int64_t nt = (in_f + tile - 1) / tile;
  TORCH_CHECK(w.numel() == out_f * in_f && s.numel() == out_f * nt,
              "fp8_backward_input: size mismatch w=", w.numel(), " s=", s.numel(),
              " expected ", out_f * in_f, " and ", out_f * nt);
  TORCH_CHECK((s.dim() == 2 && s.size(0) == out_f && s.size(1) == nt) ||
              (s.dim() == 1 && s.size(0) == out_f * nt),
              "fp8_backward_input: s must be [out_f, nt] or flattened");
  auto out = torch::empty({rows, in_f}, g.options());
  const float* gp = g.data_ptr<float>();
  const uint8_t* wp = w.data_ptr<uint8_t>();
  const float* sp = s.data_ptr<float>();
  float* xp = out.data_ptr<float>();
  const float* lut = fp8_lut();
  const int64_t RB = 16, IB = 8;
  const int64_t nrb = (rows + RB - 1) / RB, nib = (in_f + IB - 1) / IB;
  at::parallel_for(0, nrb * nib, 4, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t rb = task / nib, ib = task % nib;
      fp8_backward_task(gp, wp, sp, xp, lut, in_f, out_f, nt, tile,
                        rb * RB, std::min<int64_t>(RB, rows - rb * RB),
                        ib * IB, std::min<int64_t>(IB, in_f - ib * IB));
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_forward", &fp8_forward, "tiled E4M3 forward, AVX1 accumulate");
  m.def("fp8_backward_input", &fp8_backward_input, "tiled E4M3 backward-input");
}
