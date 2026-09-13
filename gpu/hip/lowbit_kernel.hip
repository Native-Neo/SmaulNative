#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;

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

__device__ __forceinline__ float decode_level(const uint8_t* row, int64_t k, int bits) {
    if (bits == 2) {
        const int shift = 6 - static_cast<int>(k & 3) * 2;
        return fp2_level((row[k >> 2] >> shift) & 3);
    }
    const int shift = 4 - static_cast<int>(k & 1) * 4;
    return fp4_level((row[k >> 1] >> shift) & 15);
}

__global__ void packed_linear_kernel(
    const float* x, const uint8_t* packed, const float* scale, float* out,
    int64_t batch, int64_t out_features, int64_t in_features, int bits
) {
    const int64_t block = static_cast<int64_t>(blockIdx.x);
    const int64_t outputs_per_batch = (out_features + kThreads - 1) / kThreads;
    const int64_t n = block / outputs_per_batch;
    const int64_t output_base = (block - n * outputs_per_batch) * kThreads;
    const int o = output_base + threadIdx.x;
    if (n >= batch) return;

    const int per_byte = 8 / bits;
    const int64_t row_bytes = (in_features + per_byte - 1) / per_byte;
    const float* xr = x + n * in_features;
    const uint8_t* packed_rows = packed + output_base * row_bytes;
    float* out_row = out + n * out_features;
    const uint8_t* row = packed_rows + static_cast<int64_t>(threadIdx.x) * row_bytes;

    float sum = 0.0f;
    __shared__ float xscaled[kThreads];

    for (int64_t base = 0; base < in_features; base += kThreads) {
        const int64_t k = base + threadIdx.x;
        xscaled[threadIdx.x] = k < in_features ? xr[k] * scale[k] : 0.0f;
        __syncthreads();

        if (o < out_features) {
            const int64_t limit = min<int64_t>(kThreads, in_features - base);
            for (int64_t j = 0; j < limit; ++j)
                sum += xscaled[j] * decode_level(row, base + j, bits);
        }
        __syncthreads();
    }

    if (o < out_features)
        out_row[o] = sum;
}

__global__ void packed_linear_transpose_kernel(
    const float* go, const uint8_t* packed, const float* scale, float* grad_x,
    int64_t batch, int64_t out_features, int64_t in_features, int bits
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * in_features;
    if (index >= total) return;

    const int64_t n = index / in_features;
    const int64_t k = index - n * in_features;
    const int per_byte = 8 / bits;
    const int64_t row_bytes = (in_features + per_byte - 1) / per_byte;
    float sum = 0.0f;
    const float s = scale[k];
    for (int64_t o = 0; o < out_features; ++o) {
        const uint8_t* row = packed + o * row_bytes;
        sum += go[n * out_features + o] * s * decode_level(row, k, bits);
    }
    grad_x[index] = sum;
}

}

torch::Tensor packed_linear(
    torch::Tensor x, torch::Tensor packed, torch::Tensor scale,
    int64_t bits, int64_t out_features, int64_t in_features
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP");
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA/HIP");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA/HIP");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == in_features, "input shape mismatch");
    TORCH_CHECK(scale.numel() == in_features, "scale size mismatch");
    TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");

    x = x.contiguous();
    packed = packed.contiguous();
    scale = scale.reshape({-1}).contiguous();
    const int per_byte = 8 / bits;
    const int64_t row_bytes = (in_features + per_byte - 1) / per_byte;
    TORCH_CHECK(packed.numel() >= out_features * row_bytes, "packed weight is too small");

    auto out = torch::empty({x.size(0), out_features}, x.options());
    const int64_t output_tiles = (out_features + kThreads - 1) / kThreads;
    const int64_t blocks64 = x.size(0) * output_tiles;
    TORCH_CHECK(blocks64 <= static_cast<int64_t>(2147483647), "tensor too large for HIP grid");
    const int blocks = static_cast<int>(blocks64);
    auto stream = at::cuda::getCurrentCUDAStream();
    packed_linear_kernel<<<blocks, kThreads, 0, stream>>>(
        x.data_ptr<float>(), packed.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        out.data_ptr<float>(), x.size(0), out_features, in_features, static_cast<int>(bits)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor packed_linear_transpose(
    torch::Tensor go, torch::Tensor packed, torch::Tensor scale,
    int64_t bits, int64_t out_features, int64_t in_features
) {
    TORCH_CHECK(go.is_cuda() && packed.is_cuda() && scale.is_cuda(), "tensors must be CUDA/HIP");
    TORCH_CHECK(go.scalar_type() == torch::kFloat32, "grad_output must be float32");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8 && scale.scalar_type() == torch::kFloat32, "invalid weight types");
    TORCH_CHECK(go.dim() == 2 && go.size(1) == out_features, "grad_output shape mismatch");
    TORCH_CHECK(scale.numel() == in_features, "scale size mismatch");
    TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");

    go = go.contiguous();
    packed = packed.contiguous();
    scale = scale.reshape({-1}).contiguous();
    const int per_byte = 8 / bits;
    const int64_t row_bytes = (in_features + per_byte - 1) / per_byte;
    TORCH_CHECK(packed.numel() >= out_features * row_bytes, "packed weight is too small");

    auto grad_x = torch::empty({go.size(0), in_features}, go.options());
    const int64_t total = go.size(0) * in_features;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    packed_linear_transpose_kernel<<<blocks, threads, 0, stream>>>(
        go.data_ptr<float>(), packed.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        grad_x.data_ptr<float>(), go.size(0), out_features, in_features, static_cast<int>(bits)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}

torch::Tensor qat_linear(torch::Tensor x, torch::Tensor weight, int64_t bits) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "tensors must be CUDA/HIP");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32, "tensors must be float32");
    TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && x.size(1) == weight.size(1), "shape mismatch");
    return torch::matmul(x, weight.transpose(0, 1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_linear", &packed_linear, "Packed FP2/FP4 linear");
    m.def("packed_linear_transpose", &packed_linear_transpose, "Packed FP2/FP4 transpose linear");
    m.def("qat_linear", &qat_linear, "QAT linear fallback");
}
