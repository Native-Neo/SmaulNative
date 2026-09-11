#include <torch/extension.h>
#include <ATen/Parallel.h>

namespace {

inline float decode(uint8_t byte, int index, int bits) {
    const int shift = (8 - bits) - index * bits;
    const int code = (byte >> shift) & ((1 << bits) - 1);
    if (bits == 2) {
        static constexpr float levels[] = {-1.0f, 0.0f, 1.0f, 0.0f};
        return levels[code];
    }
    static constexpr float levels[] = {
        -2.0f, -1.0f, -0.5f, -0.25f,
        0.0f, 0.25f, 0.5f, 1.0f,
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    return levels[code];
}

}

torch::Tensor packed_linear(torch::Tensor x, torch::Tensor packed,
                            torch::Tensor scale, int64_t bits,
                            int64_t out_features, int64_t in_features) {
    TORCH_CHECK(x.device().is_cpu(), "x must be on CPU");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(packed.device().is_cpu(), "packed must be on CPU");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scale.device().is_cpu(), "scale must be on CPU");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.size(1) == in_features, "input feature mismatch");
    TORCH_CHECK(scale.numel() == in_features, "scale size mismatch");

    x = x.contiguous();
    packed = packed.contiguous();
    scale = scale.reshape({-1}).contiguous();

    const auto batch = x.size(0);
    auto out = torch::empty({batch, out_features}, x.options());
    const int per_byte = 8 / bits;
    const int row_bytes = (in_features + per_byte - 1) / per_byte;
    TORCH_CHECK(packed.numel() >= out_features * row_bytes,
                "packed weight is too small");

    const float* xp = x.data_ptr<float>();
    const uint8_t* wp = packed.data_ptr<uint8_t>();
    const float* sp = scale.data_ptr<float>();
    float* yp = out.data_ptr<float>();

    at::parallel_for(0, batch * out_features, 1, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            const int64_t n = index / out_features;
            const int64_t o = index % out_features;
            const float* xr = xp + n * in_features;
            const uint8_t* wr = wp + o * row_bytes;
            float sum = 0.0f;
            for (int64_t k = 0; k < in_features; ++k) {
                const int slot = static_cast<int>(k % per_byte);
                sum += xr[k] * decode(wr[k / per_byte], slot, static_cast<int>(bits)) * sp[k];
            }
            yp[index] = sum;
        }
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_linear", &packed_linear, "Packed FP2/FP4 linear");
}
