#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

inline float quantize2(float value, float scale) {
    const float half = scale * 0.5f;
    if (value <= -half) return -scale;
    if (value <= half) return 0.0f;
    return scale;
}

inline float quantize4(float value, float scale) {
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

constexpr float fp4_levels[16] = {
    -2.0f, -1.0f, -0.5f, -0.25f,
    0.0f, 0.25f, 0.5f, 1.0f,
    2.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f
};

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

    auto scaled_x = torch::empty_like(x);
    float* sxp = scaled_x.data_ptr<float>();
    at::parallel_for(0, batch * in_features, 256, [&](int64_t begin, int64_t end) {
        int64_t k = begin % in_features;
        for (int64_t index = begin; index < end; ++index) {
            sxp[index] = xp[index] * sp[k];
            if (++k == in_features) k = 0;
        }
    });

    if (bits == 2) {
        const int64_t full_bytes = in_features >> 2;
        at::parallel_for(0, batch * out_features, 64, [&](int64_t begin, int64_t end) {
            int64_t n = begin / out_features;
            int64_t o = begin - n * out_features;
            for (int64_t index = begin; index < end; ++index) {
                const float* xr = sxp + n * in_features;
                const uint8_t* wr = wp + o * row_bytes;
                float sum = 0.0f;
                int64_t k = 0;
                for (int64_t b = 0; b < full_bytes; ++b) {
                    const uint8_t byte = wr[b];
                    const int c0 = byte >> 6;
                    const int c1 = (byte >> 4) & 3;
                    const int c2 = (byte >> 2) & 3;
                    const int c3 = byte & 3;
                    if (c0 == 0) sum -= xr[k]; else if (c0 == 2) sum += xr[k];
                    if (c1 == 0) sum -= xr[k + 1]; else if (c1 == 2) sum += xr[k + 1];
                    if (c2 == 0) sum -= xr[k + 2]; else if (c2 == 2) sum += xr[k + 2];
                    if (c3 == 0) sum -= xr[k + 3]; else if (c3 == 2) sum += xr[k + 3];
                    k += 4;
                }
                for (; k < in_features; ++k) {
                    const int code = (wr[k >> 2] >> (6 - (k & 3) * 2)) & 3;
                    if (code == 0) sum -= xr[k]; else if (code == 2) sum += xr[k];
                }
                yp[index] = sum;
                if (++o == out_features) {
                    o = 0;
                    ++n;
                }
            }
        });
    } else {
        const int64_t full_bytes = in_features >> 1;
        at::parallel_for(0, batch * out_features, 64, [&](int64_t begin, int64_t end) {
            int64_t n = begin / out_features;
            int64_t o = begin - n * out_features;
            for (int64_t index = begin; index < end; ++index) {
                const float* xr = sxp + n * in_features;
                const uint8_t* wr = wp + o * row_bytes;
                float sum = 0.0f;
                int64_t k = 0;
                for (int64_t b = 0; b < full_bytes; ++b) {
                    const uint8_t byte = wr[b];
                    sum += xr[k] * fp4_levels[byte >> 4];
                    sum += xr[k + 1] * fp4_levels[byte & 15];
                    k += 2;
                }
                for (; k < in_features; ++k) {
                    const int code = (wr[k >> 1] >> (4 - (k & 1) * 4)) & 15;
                    sum += xr[k] * fp4_levels[code];
                }
                yp[index] = sum;
                if (++o == out_features) {
                    o = 0;
                    ++n;
                }
            }
        });
    }
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
            float local[32];
            for (int64_t k = k0; k < k1; ++k)
                local[k - k0] = eps;
            for (int64_t o = 0; o < out_features; ++o) {
                const float* wr = wp + o * in_features + k0;
                for (int64_t k = k0; k < k1; ++k)
                    local[k - k0] = std::max(local[k - k0], std::abs(wr[k - k0]));
            }
            for (int64_t k = k0; k < k1; ++k)
                scale[k] = local[k - k0];
        }
    });

    auto qweight = torch::empty_like(weight);
    float* qwp = qweight.data_ptr<float>();
    if (bits == 2) {
        at::parallel_for(0, out_features, 64, [&](int64_t begin, int64_t end) {
            for (int64_t o = begin; o < end; ++o) {
                const float* wr = wp + o * in_features;
                float* qr = qwp + o * in_features;
                for (int64_t k = 0; k < in_features; ++k)
                    qr[k] = quantize2(wr[k], scale[k]);
            }
        });
    } else {
        at::parallel_for(0, out_features, 64, [&](int64_t begin, int64_t end) {
            for (int64_t o = begin; o < end; ++o) {
                const float* wr = wp + o * in_features;
                float* qr = qwp + o * in_features;
                for (int64_t k = 0; k < in_features; ++k)
                    qr[k] = quantize4(wr[k], scale[k]);
            }
        });
    }

    auto out = torch::empty({batch, out_features}, x.options());
    const float* xp = x.data_ptr<float>();
    float* yp = out.data_ptr<float>();

    at::parallel_for(0, batch * out_features, 64, [&](int64_t begin, int64_t end) {
        int64_t n = begin / out_features;
        int64_t o = begin - n * out_features;
        for (int64_t index = begin; index < end; ++index) {
            const float* xr = xp + n * in_features;
            const float* qr = qwp + o * in_features;
            float sum = 0.0f;
            for (int64_t k = 0; k < in_features; ++k)
                sum += xr[k] * qr[k];
            yp[index] = sum;
            if (++o == out_features) {
                o = 0;
                ++n;
            }
        }
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_linear", &packed_linear, "Packed FP2/FP4 linear");
    m.def("qat_linear", &qat_linear, "Fused FP2/FP4 QAT linear");
}
