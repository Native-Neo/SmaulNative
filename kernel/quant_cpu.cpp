// Tiled E4M3 quantizer for the RQT requant path.
//
// Reproduces kernel/fp8_tile.quantize_tiles exactly -- same codes, same
// scales -- because the codes ARE the persistent FP8 weights: any difference
// here silently changes the model rather than raising. The reference
// implementation is:
//
//   amax_t = max over the tile of |w|, ignoring non-finite entries, min 1e-12
//   sc_t   = max(amax_t / 448, 1e-12)
//   n      = clamp(nan_to_num(w / sc_t, nan=0), -448, 448)
//   code   = order[lower_bound(bounds, n)]        # round-to-nearest midpoint
//   code[n == 0] = 0
//
// where bounds are the midpoints of the 256 E4M3 codebook values in sorted
// order. The torch version spent 0.79 ms of its 1.94 ms on a 64x512 block in
// aten::bucketize alone, plus more in the amax/scale/clamp passes; here the
// whole block is one pass with a 255-entry binary search per element and no
// intermediate tensors.
//
// The zero-pad to a whole number of tiles is not materialised: padded entries
// are zeros, and 0 can never raise a max of absolute values, so skipping them
// is exactly equivalent.

#include <ATen/Parallel.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <torch/extension.h>
#include <vector>

namespace {

constexpr float kE4M3Max = 448.0f;
constexpr int kCodeCount = 256;
constexpr int kNumBounds = kCodeCount - 1;

// The sorted codebook permutation and its midpoints are supplied by the
// caller, from the same cached torch tables the reference implementation uses
// (kernel.fp8_tile._tables). They are deliberately NOT recomputed here: E4M3
// maps TWO codes to +448 (126 and 127) and two to -448 (254, 255), so the
// codebook has duplicate values and the tie order inside torch.argsort is not
// something a std::sort is guaranteed to reproduce. Re-deriving it picked a
// different code for every weight that saturated to exactly 448 -- invisible
// in the loss, but it would have silently rewritten the stored weights.
// Passing the tables makes the match exact by construction.
inline uint8_t encode(const uint8_t* order, const float* bounds, float n) {
  if (n == 0.0f) return 0;                       // mirrors code[nf == 0] = 0
  // first index with bounds[i] >= n, i.e. torch.bucketize(right=False)
  const int i = static_cast<int>(
      std::lower_bound(bounds, bounds + kNumBounds, n) - bounds);
  return order[i < 0 ? 0 : (i > 255 ? 255 : i)];
}

inline float finite_abs(float v) {
  return std::isfinite(v) ? std::fabs(v) : 0.0f;
}

}  // namespace

// Returns (codes uint8 [out_f, nt*tile], scales float32 [out_f, nt],
// non-finite input count). `order` is uint8[256] and `bounds` float32[255],
// both taken from kernel.fp8_tile._tables.
std::tuple<torch::Tensor, torch::Tensor, int64_t> fp8_quantize_tiles(
    torch::Tensor w32, int64_t tile, torch::Tensor order,
    torch::Tensor bounds) {
  TORCH_CHECK(w32.device().is_cpu(), "fp8_quantize_tiles: w32 must be CPU");
  TORCH_CHECK(w32.scalar_type() == torch::kFloat32,
              "fp8_quantize_tiles: w32 must be float32");
  TORCH_CHECK(w32.dim() == 2, "fp8_quantize_tiles: w32 must be 2-D");
  TORCH_CHECK(w32.is_contiguous(), "fp8_quantize_tiles: w32 must be contiguous");
  TORCH_CHECK(tile > 0, "fp8_quantize_tiles: tile must be positive, got ", tile);
  const int64_t out_f = w32.size(0), in_f = w32.size(1);
  TORCH_CHECK(out_f > 0 && in_f > 0, "fp8_quantize_tiles: empty input");
  TORCH_CHECK(order.numel() == kCodeCount && order.scalar_type() == torch::kUInt8 &&
                  order.is_contiguous(),
              "fp8_quantize_tiles: order must be contiguous uint8[256]");
  TORCH_CHECK(bounds.numel() == kNumBounds && bounds.scalar_type() == torch::kFloat32 &&
                  bounds.is_contiguous(),
              "fp8_quantize_tiles: bounds must be contiguous float32[255]");
  const uint8_t* ordp = order.data_ptr<uint8_t>();
  const float* bndp = bounds.data_ptr<float>();
  const int64_t nt = (in_f + tile - 1) / tile;

  auto codes = torch::empty({out_f, nt * tile}, torch::TensorOptions()
      .dtype(torch::kUInt8).device(w32.device()));
  auto scales = torch::empty({out_f, nt}, torch::TensorOptions()
      .dtype(torch::kFloat32).device(w32.device()));
  const float* wp = w32.data_ptr<float>();
  uint8_t* cp = codes.data_ptr<uint8_t>();
  float* sp = scales.data_ptr<float>();
  // The non-finite count is a diagnostic, but it is the one that decides
  // whether the caller warns about saturated weights -- i.e. the one that is
  // supposed to surface divergence. So it has to be exact.
  //
  // It was a plain int64 incremented from inside at::parallel_for, which is a
  // data race: two workers incrementing it concurrently lose updates, and the
  // count comes back short. It did not reproduce in 600 trials at 1/2/4
  // threads with maximal contention (an all-NaN 512x512 input, where every
  // iteration increments), so this is undefined behaviour that has not yet
  // bitten -- which is exactly why it is worth closing rather than watching.
  //
  // Count into a task-local and fold in once per task: the atomic is then paid
  // per chunk, not per element. The chunk count is what bounds that, and
  // at::parallel_for sizes chunks from the thread count, not from out_f.
  std::atomic<int64_t> nonfinite{0};
  at::parallel_for(0, out_f, 1, [&](int64_t begin, int64_t end) {
    int64_t nonfinite_local = 0;
    for (int64_t o = begin; o < end; ++o) {
      const float* wrow = wp + o * in_f;
      uint8_t* crow = cp + o * nt * tile;
      for (int64_t t = 0; t < nt; ++t) {
        const int64_t k0 = t * tile;
        const int64_t k1 = std::min(in_f, k0 + tile);
        float amax = 0.0f;
        for (int64_t k = k0; k < k1; ++k) {
          const float w = wrow[k];
          if (!std::isfinite(w)) ++nonfinite_local;
          const float a = finite_abs(w);
          if (a > amax) amax = a;
        }
        if (amax < 1e-12f) amax = 1e-12f;      // clamp_min
        const float sc = std::fmax(amax / kE4M3Max, 1e-12f);
        sp[o * nt + t] = sc;
        for (int64_t k = k0; k < k1; ++k) {
          float n = wrow[k] / sc;
          if (std::isnan(n)) n = 0.0f;         // nan_to_num(nan=0)
          n = std::fmin(std::fmax(n, -kE4M3Max), kE4M3Max);
          crow[k] = encode(ordp, bndp, n);
        }
        for (int64_t k = k1; k < k0 + tile; ++k) crow[k] = 0;  // pad
      }
    }
    nonfinite.fetch_add(nonfinite_local, std::memory_order_relaxed);
  });
  return {codes, scales, nonfinite.load(std::memory_order_relaxed)};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_quantize_tiles", &fp8_quantize_tiles,
        "tiled E4M3 quantizer matching quantize_tiles (AVX1)");
}
