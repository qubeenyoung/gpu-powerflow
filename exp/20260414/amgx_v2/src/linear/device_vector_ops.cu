#include "device_vector_ops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

constexpr int32_t kBlockSize = 256;

__global__ void negate_kernel(int32_t n,
                              const double* __restrict__ input,
                              double* __restrict__ output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = -input[i];
    }
}

__global__ void reduce_absmax_kernel(int32_t n,
                                     const double* __restrict__ input,
                                     double* __restrict__ partial)
{
    __shared__ double values[kBlockSize];
    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;
    values[tid] = (i < n) ? fabs(input[i]) : 0.0;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] = fmax(values[tid], values[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = values[0];
    }
}

void ensure_size(DeviceBuffer<double>& buffer, std::size_t count)
{
    if (buffer.size() != count) {
        buffer.resize(count);
    }
}

}  // namespace

void DeviceVectorOps::negate(int32_t n,
                             const double* input_device,
                             double* output_device)
{
    if (n <= 0 || input_device == nullptr || output_device == nullptr) {
        throw std::runtime_error("DeviceVectorOps::negate received invalid input");
    }
    const int32_t grid = (n + kBlockSize - 1) / kBlockSize;
    negate_kernel<<<grid, kBlockSize>>>(n, input_device, output_device);
    CUDA_CHECK(cudaGetLastError());
}

double DeviceVectorOps::norm_inf(int32_t n, const double* input_device)
{
    if (n <= 0 || input_device == nullptr) {
        throw std::runtime_error("DeviceVectorOps::norm_inf received invalid input");
    }

    int32_t current = (n + kBlockSize - 1) / kBlockSize;
    ensure_size(d_partial_, static_cast<std::size_t>(current));
    reduce_absmax_kernel<<<current, kBlockSize>>>(n, input_device, d_partial_.data());
    CUDA_CHECK(cudaGetLastError());

    bool in_partial = true;
    while (current > 1) {
        const int32_t next = (current + kBlockSize - 1) / kBlockSize;
        if (in_partial) {
            ensure_size(d_scratch_, static_cast<std::size_t>(next));
            reduce_absmax_kernel<<<next, kBlockSize>>>(
                current, d_partial_.data(), d_scratch_.data());
        } else {
            ensure_size(d_partial_, static_cast<std::size_t>(next));
            reduce_absmax_kernel<<<next, kBlockSize>>>(
                current, d_scratch_.data(), d_partial_.data());
        }
        CUDA_CHECK(cudaGetLastError());
        current = next;
        in_partial = !in_partial;
    }

    double result = 0.0;
    if (in_partial) {
        d_partial_.copyTo(&result, 1);
    } else {
        d_scratch_.copyTo(&result, 1);
    }
    return result;
}

}  // namespace exp_20260414::amgx_v2
