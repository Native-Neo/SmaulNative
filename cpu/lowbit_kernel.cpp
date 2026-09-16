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

    constexpr float fp2_levels[4] = {
        -1.0f, 0.0f, 1.0f, 0.0f
    };

    constexpr float fp4_levels[16] = {
        -2.0f, -1.0f, -0.5f, -0.25f,
        0.0f, 0.25f, 0.5f, 1.0f,
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };

}

torch::Tensor packed_linear(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scale,
    int64_t bits,
    int64_t out_features,
    int64_t in_features
) {
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

    const int64_t batch = x.size(0);
    const int per_byte = 8 / bits;
    const int64_t row_bytes =
        (in_features + per_byte - 1) / per_byte;

    TORCH_CHECK(
        packed.numel() >= out_features * row_bytes,
        "packed weight is too small"
    );

    auto out = torch::empty(
        {batch, out_features},
        x.options()
    );

    const float* xp = x.data_ptr<float>();
    const uint8_t* wp = packed.data_ptr<uint8_t>();
    const float* sp = scale.data_ptr<float>();
    float* yp = out.data_ptr<float>();

    constexpr int64_t tile = 4;
    const int64_t out_tiles =
        (out_features + tile - 1) / tile;

    if (bits == 2) {
        const int64_t full_bytes = in_features >> 2;

        at::parallel_for(
            0,
            batch * out_tiles,
            1,
            [&](int64_t begin, int64_t end) {
                float sums[tile];
                int64_t n = begin / out_tiles;
                int64_t tile_index = begin - n * out_tiles;
                int64_t ob = tile_index * tile;

                for (int64_t task = begin; task < end; ++task) {
                    const int64_t count =
                        std::min(tile, out_features - ob);

                    const float* xr =
                        xp + n * in_features;

                    const uint8_t* wr[tile];

                    for (int64_t j = 0; j < count; ++j)
                        wr[j] =
                            wp + (ob + j) * row_bytes;

                    for (int64_t j = 0; j < count; ++j)
                        sums[j] = 0.0f;

                    int64_t k = 0;

                    for (int64_t b = 0; b < full_bytes; ++b, k += 4) {
                        const float p0 = xr[k] * sp[k];
                        const float p1 = xr[k + 1] * sp[k + 1];
                        const float p2 = xr[k + 2] * sp[k + 2];
                        const float p3 = xr[k + 3] * sp[k + 3];

                        for (int64_t j = 0; j < count; ++j) {
                            const uint8_t byte = wr[j][b];
                            const int c0 = byte >> 6;
                            const int c1 = (byte >> 4) & 3;
                            const int c2 = (byte >> 2) & 3;
                            const int c3 = byte & 3;

                            sums[j] += p0 * fp2_levels[c0];
                            sums[j] += p1 * fp2_levels[c1];
                            sums[j] += p2 * fp2_levels[c2];
                            sums[j] += p3 * fp2_levels[c3];
                        }
                    }

                    for (; k < in_features; ++k) {
                        const float p = xr[k] * sp[k];
                        const int shift = 6 - (k & 3) * 2;

                        for (int64_t j = 0; j < count; ++j) {
                            const int code =
                                (wr[j][k >> 2] >> shift) & 3;
                            sums[j] += p * fp2_levels[code];
                        }
                    }

                    for (int64_t j = 0; j < count; ++j)
                        yp[n * out_features + ob + j] = sums[j];

                    ++tile_index;
                    ob += tile;
                    if (tile_index == out_tiles) {
                        tile_index = 0;
                        ob = 0;
                        ++n;
                    }
                }
            }
        );
    } else {
        const int64_t full_bytes = in_features >> 1;

        at::parallel_for(
            0,
            batch * out_tiles,
            1,
            [&](int64_t begin, int64_t end) {
                float sums[tile];
                int64_t n = begin / out_tiles;
                int64_t tile_index = begin - n * out_tiles;
                int64_t ob = tile_index * tile;

                for (int64_t task = begin; task < end; ++task) {
                    const int64_t count =
                        std::min(tile, out_features - ob);

                    const float* xr =
                        xp + n * in_features;

                    const uint8_t* wr[tile];

                    for (int64_t j = 0; j < count; ++j)
                        wr[j] =
                            wp + (ob + j) * row_bytes;

                    for (int64_t j = 0; j < count; ++j)
                        sums[j] = 0.0f;

                    int64_t k = 0;

                    for (int64_t b = 0; b < full_bytes; ++b, k += 2) {
                        const float p0 = xr[k] * sp[k];
                        const float p1 = xr[k + 1] * sp[k + 1];

                        for (int64_t j = 0; j < count; ++j) {
                            const uint8_t byte = wr[j][b];
                            sums[j] += p0 * fp4_levels[byte >> 4];
                            sums[j] += p1 * fp4_levels[byte & 15];
                        }
                    }

                    for (; k < in_features; ++k) {
                        const float p = xr[k] * sp[k];
                        const int shift = 4 - (k & 1) * 4;

                        for (int64_t j = 0; j < count; ++j) {
                            const int code =
                                (wr[j][k >> 1] >> shift) & 15;
                            sums[j] += p * fp4_levels[code];
                        }
                    }

                    for (int64_t j = 0; j < count; ++j)
                        yp[n * out_features + ob + j] = sums[j];

                    ++tile_index;
                    ob += tile;
                    if (tile_index == out_tiles) {
                        tile_index = 0;
                        ob = 0;
                        ++n;
                    }
                }
            }
        );
    }

    return out;
}

torch::Tensor qat_linear(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t bits
) {
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

    at::parallel_for(
        0,
        in_features,
        block,
        [&](int64_t begin, int64_t end) {
            for (int64_t k0 = begin; k0 < end; k0 += block) {
                const int64_t k1 = std::min(k0 + block, end);
                float local[32];

                for (int64_t k = k0; k < k1; ++k)
                    local[k - k0] = eps;

                for (int64_t o = 0; o < out_features; ++o) {
                    const float* wr = wp + o * in_features + k0;

                    for (int64_t k = k0; k < k1; ++k) {
                        local[k - k0] = std::max(
                            local[k - k0],
                            std::abs(wr[k - k0])
                        );
                    }
                }

                for (int64_t k = k0; k < k1; ++k)
                    scale[k] = local[k - k0];
            }
        }
    );

    auto out = torch::empty({batch, out_features}, x.options());
    const float* xp = x.data_ptr<float>();
    float* yp = out.data_ptr<float>();

    if (bits == 2) {
        at::parallel_for(
            0,
            out_features,
            64,
            [&](int64_t begin, int64_t end) {
                std::vector<float> sums(batch, 0.0f);

                for (int64_t o = begin; o < end; ++o) {
                    const float* wr = wp + o * in_features;
                    std::fill(sums.begin(), sums.end(), 0.0f);
                    int64_t k = 0;

                    for (; k + 7 < in_features; k += 8) {
                        const float q0 = quantize2(wr[k], scale[k]);
                        const float q1 = quantize2(wr[k + 1], scale[k + 1]);
                        const float q2 = quantize2(wr[k + 2], scale[k + 2]);
                        const float q3 = quantize2(wr[k + 3], scale[k + 3]);
                        const float q4 = quantize2(wr[k + 4], scale[k + 4]);
                        const float q5 = quantize2(wr[k + 5], scale[k + 5]);
                        const float q6 = quantize2(wr[k + 6], scale[k + 6]);
                        const float q7 = quantize2(wr[k + 7], scale[k + 7]);

                        for (int64_t n = 0; n < batch; ++n) {
                            const float* xr = xp + n * in_features + k;
                            sums[n] += xr[0] * q0 + xr[1] * q1 + xr[2] * q2 + xr[3] * q3;
                            sums[n] += xr[4] * q4 + xr[5] * q5 + xr[6] * q6 + xr[7] * q7;
                        }
                    }

                    for (; k < in_features; ++k) {
                        const float q = quantize2(wr[k], scale[k]);
                        for (int64_t n = 0; n < batch; ++n)
                            sums[n] += xp[n * in_features + k] * q;
                    }

                    for (int64_t n = 0; n < batch; ++n)
                        yp[n * out_features + o] = sums[n];
                }
            }
        );
    } else {
        at::parallel_for(
            0,
            out_features,
            64,
            [&](int64_t begin, int64_t end) {
                std::vector<float> sums(batch, 0.0f);

                for (int64_t o = begin; o < end; ++o) {
                    const float* wr = wp + o * in_features;
                    std::fill(sums.begin(), sums.end(), 0.0f);
                    int64_t k = 0;

                    for (; k + 7 < in_features; k += 8) {
                        const float q0 = quantize4(wr[k], scale[k]);
                        const float q1 = quantize4(wr[k + 1], scale[k + 1]);
                        const float q2 = quantize4(wr[k + 2], scale[k + 2]);
                        const float q3 = quantize4(wr[k + 3], scale[k + 3]);
                        const float q4 = quantize4(wr[k + 4], scale[k + 4]);
                        const float q5 = quantize4(wr[k + 5], scale[k + 5]);
                        const float q6 = quantize4(wr[k + 6], scale[k + 6]);
                        const float q7 = quantize4(wr[k + 7], scale[k + 7]);

                        for (int64_t n = 0; n < batch; ++n) {
                            const float* xr = xp + n * in_features + k;
                            sums[n] += xr[0] * q0 + xr[1] * q1 + xr[2] * q2 + xr[3] * q3;
                            sums[n] += xr[4] * q4 + xr[5] * q5 + xr[6] * q6 + xr[7] * q7;
                        }
                    }

                    for (; k < in_features; ++k) {
                        const float q = quantize4(wr[k], scale[k]);
                        for (int64_t n = 0; n < batch; ++n)
                            sums[n] += xp[n * in_features + k] * q;
                    }

                    for (int64_t n = 0; n < batch; ++n)
                        yp[n * out_features + o] = sums[n];
                }
            }
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_linear", &packed_linear, "Packed FP2/FP4 linear");
    m.def("qat_linear", &qat_linear, "Fused FP2/FP4 QAT linear");
}