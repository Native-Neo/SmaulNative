#include <ATen/Parallel.h>
#include <algorithm>
#include <cstdint>
#include <torch/extension.h>

// Fused d/dvalues for the Rawr sparse projection (SparseLinear._SparseLinearFn).
//
//   grad_v[o, m] = sum_r dout[r, o] * x[r, cols[o, m]]
//
// i.e. for every nonzero of the (out_f x in_f, K-per-row) sparsity pattern,
// one dot product of length `rows`. Note there is no `values` factor: out[o]
// = sum_m x[cols[o, m]] * values[o, m], so d out[o] / d values[o, m] is
// x[cols[o, m]] itself.
//
// Why this is a separate kernel: the pure-torch form has to materialise the
// gathered activation, which is rows*out_f*K elements -- 209M fp32 (836 MiB)
// for an 8000x512 head at rows=512, K=51. That gather is written and then
// read back twice (multiply, then reduce), so the whole product runs at
// memory-bandwidth speed: measured 1004 ms. This kernel never materialises
// it. It reads dout once (rows*out_f*4 = 16 MiB), reads x (rows*in_f*4, which
// is 1 MiB and stays cache-resident) and writes grad_v (out_f*K*4 = 1.6 MiB)
// -- ~19 MiB of traffic instead of ~3.3 GiB, which is the whole win.
//
// Both operands are indexed [*, rows] so the inner product is contiguous:
//   dOutT is dout transposed to [out_f, rows]
//   xT     is x transposed to [in_f, rows]
// Consecutive m reuse the same dOutT row, and consecutive rows of the task
// reuse xT, so both stay in L1/L2.
//
// Ivy Bridge safe: AVX1 + SSE4.2 only, no AVX2/AVX-512/FMA (same flags as
// attn_cpu.cpp and fp8_cpu.cpp). The exact math is the scalar path; the AVX
// path differs only in FP32 summation order, the same caveat the FP8 kernels
// document.

namespace {

inline float dot_avx(const float* a, const float* b, int64_t n) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                          _mm256_loadu_ps(b + i)));
  float buf[8];
  _mm256_storeu_ps(buf, acc);
  float s = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline float dot_scalar(const float* a, const float* b, int64_t n) {
  float s = 0.0f;
  for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline bool cpu_has_avx() {
#if defined(__x86_64__) || defined(__i386__)
  return __builtin_cpu_supports("avx");
#else
  return false;
#endif
}

}  // namespace

// rows_per_task bounds how many output rows one task owns, so the working set
// is dOutT rows (rows floats each) plus grad_v for those rows.
static void grad_v_task(const float* dOutT, const float* xT, const int64_t* cols,
                        float* gradV, int64_t rows, int64_t in_f, int64_t K,
                        int64_t o0, int64_t o1) {
  const bool avx = cpu_has_avx();
  for (int64_t o = o0; o < o1; ++o) {
    const float* dr = dOutT + o * rows;
    float* gv = gradV + o * K;
    const int64_t* crow = cols + o * K;
    for (int64_t m = 0; m < K; ++m) {
      const int64_t j = crow[m];
      TORCH_CHECK(j >= 0 && j < in_f, "cols index out of range in sparse_grad_v");
      const float* xr = xT + j * rows;
      gv[m] = avx ? dot_avx(dr, xr, rows) : dot_scalar(dr, xr, rows);
    }
  }
}

// grad_v[o, m] = dot(dOutT[o, :], xT[cols[o, m], :])
torch::Tensor sparse_grad_v(torch::Tensor dOutT, torch::Tensor xT,
                            torch::Tensor cols, torch::Tensor out) {
  TORCH_CHECK(dOutT.device().is_cpu() && xT.device().is_cpu() &&
                  cols.device().is_cpu() && out.device().is_cpu(),
              "sparse_grad_v: all tensors must be CPU");
  TORCH_CHECK(dOutT.scalar_type() == torch::kFloat32 &&
                  xT.scalar_type() == torch::kFloat32,
              "sparse_grad_v: dense operands must be float32");
  TORCH_CHECK(cols.scalar_type() == torch::kInt64, "sparse_grad_v: cols must be int64");
  TORCH_CHECK(dOutT.dim() == 2 && xT.dim() == 2, "sparse_grad_v: dense operands must be 2-D");
  TORCH_CHECK(cols.dim() == 2 && out.dim() == 2, "sparse_grad_v: cols/out must be 2-D");
  TORCH_CHECK(cols.is_contiguous() && out.is_contiguous() &&
                  dOutT.is_contiguous() && xT.is_contiguous(),
              "sparse_grad_v: all tensors must be contiguous");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "sparse_grad_v: out must be float32");
  TORCH_CHECK(cols.sizes() == out.sizes(), "sparse_grad_v: cols and out must agree");
  const int64_t out_f = dOutT.size(0), rows = dOutT.size(1);
  const int64_t in_f = xT.size(0), K = cols.size(1);
  TORCH_CHECK(xT.size(1) == rows, "sparse_grad_v: xT rows must match dOutT rows");
  TORCH_CHECK(cols.size(0) == out_f, "sparse_grad_v: cols rows must match dOutT rows");
  if (out_f == 0 || rows == 0 || K == 0) return out;

  const float* dp = dOutT.data_ptr<float>();
  const float* xp = xT.data_ptr<float>();
  const int64_t* cp = cols.data_ptr<int64_t>();
  float* gp = out.data_ptr<float>();
  // One task per output row is too fine; batch so each task owns a run of rows
  // and the dOutT working set stays in cache.
  const int64_t per_task = std::max<int64_t>(1, std::min<int64_t>(out_f, 8));
  const int64_t ntask = (out_f + per_task - 1) / per_task;
  at::parallel_for(0, ntask, 1, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      const int64_t o0 = t * per_task;
      grad_v_task(dp, xp, cp, gp, rows, in_f, K, o0,
                  std::min<int64_t>(out_f, o0 + per_task));
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_grad_v", &sparse_grad_v,
        "fused d/dvalues for SparseLinear (AVX1)");
}
