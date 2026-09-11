#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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

inline float quantize(float value, float scale, int bits) {
    if (bits == 2) {
        const float half = scale * 0.5f;
        if (value <= -half) return -scale;
        if (value <= half) return 0.0f;
        return scale;
    }

    const float t1 = scale * -1.5f;
    const float t2 = scale * -0.75f;
    const float t3 = scale * -0.375f;
    const float t4 = scale * -0.125f;
    const float t5 = scale * 0.125f;
    const float t6 = scale * 0.375f;
    const float t7 = scale * 0.75f;
    const float t8 = scale * 1.5f;
    if (value <= t1) return scale * -2.0f;
    if (value <= t2) return scale * -1.0f;
    if (value <= t3) return scale * -0.5f;
    if (value <= t4) return scale * -0.25f;
    if (value <= t5) return 0.0f;
    if (value <= t6) return scale * 0.25f;
    if (value <= t7) return scale * 0.5f;
    if (value <= t8) return scale;
    return scale * 2.0f;
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

            if (bits == 2) {
                const int64_t full_bytes = in_features >> 2;
                int64_t k = 0;
                for (int64_t b = 0; b < full_bytes; ++b) {
                    const uint8_t byte = wr[b];
                    sum += xr[k] * decode(byte, 0, 2) * sp[k];
                    sum += xr[k + 1] * decode(byte, 1, 2) * sp[k + 1];
                    sum += xr[k + 2] * decode(byte, 2, 2) * sp[k + 2];
                    sum += xr[k + 3] * decode(byte, 3, 2) * sp[k + 3];
                    k += 4;
                }
                for (; k < in_features; ++k)
                    sum += xr[k] * decode(wr[k >> 2], static_cast<int>(k & 3), 2) * sp[k];
            } else {
                const int64_t full_bytes = in_features >> 1;
                int64_t k = 0;
                for (int64_t b = 0; b < full_bytes; ++b) {
                    const uint8_t byte = wr[b];
                    sum += xr[k] * decode(byte, 0, 4) * sp[k];
                    sum += xr[k + 1] * decode(byte, 1, 4) * sp[k + 1];
                    k += 2;
                }
                for (; k < in_features; ++k)
                    sum += xr[k] * decode(wr[k >> 1], static_cast<int>(k & 1), 4) * sp[k];
            }
            yp[index] = sum;
        }
    });
    return out;
}

torch::Tensor qat_linear(torch::Tensor x, torch::Tensor weight, int64_t bits) {
    TORCH_CHECK(x.device().is_cpu(), "x must be on CPU");
    TORCH_CHECK(weight.device().is_cpu(), "weight must be on CPU");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x and weight must be 2D");
    TORCH_CHECK(x.size(1) == weight.size(1), "input feature mismatch");

    x = x.contiguous();
    weight = weight.contiguous();

    const int64_t batch = x.size(0);
    const int64_t out_features = weight.size(0);
    const int64_t in_features = weight.size(1);
    const float eps = std::numeric_limits<float>::epsilon();
    std::vector<float> scale(in_features, eps);
    const float* wp = weight.data_ptr<float>();

    constexpr int64_t block = 32;
    at::parallel_for(0, in_features, block, [&](int64_t begin, int64_t end) {
        for (int64_t k0 = begin; k0 < end; k0 += block) {
            const int64_t k1 = std::min(k0 + block, end);
            for (int64_t o = 0; o < out_features; ++o) {
                const float* wr = wp + o * in_features + k0;
                for (int64_t k = k0; k < k1; ++k)
                    scale[k] = std::max(scale[k], std::abs(wr[k - k0]));
            }
            for (int64_t k = k0; k < k1; ++k)
                scale[k] = std::max(scale[k], eps);
        }
    });

    std::vector<float> qweight(static_cast<size_t>(out_features) * in_features);
    at::parallel_for(0, out_features, 1, [&](int64_t begin, int64_t end) {
        for (int64_t o = begin; o < end; ++o) {
            const float* wr = wp + o * in_features;
            float* qr = qweight.data() + o * in_features;
            for (int64_t k = 0; k < in_features; ++k)
                qr[k] = quantize(wr[k], scale[k], static_cast<int>(bits));
        }
    });

    auto out = torch::empty({batch, out_features}, x.options());
    const float* xp = x.data_ptr<float>();
    float* yp = out.data_ptr<float>();

    at::parallel_for(0, batch * out_features, 1, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            const int64_t n = index / out_features;
            const int64_t o = index % out_features;
            const float* xr = xp + n * in_features;
            const float* qr = qweight.data() + o * in_features;
            float sum = 0.0f;
            for (int64_t k = 0; k < in_features; ++k)
                sum += xr[k] * qr[k];
            yp[index] = sum;
        }
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_linear", &packed_linear, "Packed FP2/FP4 linear");
    m.def("qat_linear", &qat_linear, "Fused FP2/FP4 QAT linear");
}
