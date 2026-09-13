#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float fp2_level(int code) {
    return code == 0 ? -1.0f : code == 2 ? 1.0f : 0.0f;
}

__device__ __forceinline__ float fp4_level(int code) {
    switch (code) {
        case 0: return -2.0f;
        case 1: return -1.0f;
        case 2: return -0.5f;
        case 3: return -0.25f;
        case 5: return 0.25f;
        case 6: return 0.5f;
        case 7: return 1.0f;
        case 8: return 2.0f;
        default: return 0.0f;
    }
}

__global__ void packed_linear_kernel(
    const float* x,
    const uint8_t* packed,
    const float* scale,
    float* out,
    int64_t batch,
    int64_t out_features,
    int64_t in_features,
    int bits
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * out_features;
    if (index >= total) return;

    const int64_t n = index / out_features;
    const int64_t o = index - n * out_features;
    const int per_byte = 8 / bits;
    const int64_t row_bytes = (in_features + per_byte - 1) / per_byte;
    const uint8_t* row = packed + o * row_bytes;
    const float* xr = x + n * in_features;

    float sum = 0.0f;
    if (bits == 2) {
        for (int64_t k = 0; k < in_features; ++k) {
            const int shift = 6 - static_cast<int>(k & 3) * 2;
            const int code = (row[k >> 2] >> shift) & 3;
            sum += xr[k] * scale[k] * fp2_level(code);
        }
    } else {
        for (int64_t k = 0; k < in_features; ++k) {
            const int shift = 4 - static_cast<int>(k & 1) * 4;
            const int code = (row[k >> 1] >> shift) & 15;
            sum += xr[k] * scale[k] * fp4_level(code);
        }
    }
    out[index] = sum;
}

} // namespace

torch::Tensor packed_linear(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scale,
    int64_t bits,
    int64_t out_features,
    int64_t in_features
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP");
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA/HIP");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA/HIP");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.size(1) == in_features, "input feature mismatch");
    TORCH_CHECK(scale.numel() == in_features, "scale size mismatch");
    TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");

    x = x.contiguous();
    packed = packed.contiguous();
    scale = scale.reshape({-1}).contiguous();

    const int per_byte = 8 / bits;
    const int64_t row_bytes = (in_features + per_byte - 1) / per_byte;
    TORCH_CHECK(packed.numel() >= out_features * row_bytes, "packed weight is too small");

    auto out = torch::empty({x.size(0), out_features}, x.options());
    const int64_t total = x.size(0) * out_features;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getDefaultCUDAStream();
    packed_linear_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), packed.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        out.data_ptr<float>(), x.size(0), out_features, in_features, static_cast<int>(bits)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor qat_linear(torch::Tensor x, torch::Tensor weight, int64_t bits) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA/HIP");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x and weight must be 2D");
    TORCH_CHECK(x.size(1) == weight.size(1), "input feature mismatch");
    return torch::matmul(x, weight.transpose(0, 1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_linear", &packed_linear, "Packed FP2/FP4 linear");
    m.def("qat_linear", &qat_linear, "QAT linear fallback");
}
