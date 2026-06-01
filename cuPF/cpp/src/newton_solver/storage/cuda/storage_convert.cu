// ---------------------------------------------------------------------------
// storage_convert.cu
//
// Device kernels + launchers for storage upload/download conversion. These
// replace O(batch*n_bus) host-side cast/trig loops with one bulk H2D/D2H plus
// a device kernel, which matters at large batch (host loops dominated transfer
// time). See storage_convert.hpp.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "storage_convert.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>

namespace {

constexpr int32_t kConvertBlock = 256;

template <typename StorageT>
__global__ void seed_state_kernel(const double* __restrict__ v0,
                                  StorageT* __restrict__ v_re,
                                  StorageT* __restrict__ v_im,
                                  StorageT* __restrict__ va,
                                  StorageT* __restrict__ vm,
                                  int32_t count)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double re = v0[2 * i];
    const double im = v0[2 * i + 1];
    v_re[i] = static_cast<StorageT>(re);
    v_im[i] = static_cast<StorageT>(im);
    // Polar form in double precision, then cast (matches host std::arg/std::abs).
    va[i] = static_cast<StorageT>(atan2(im, re));
    vm[i] = static_cast<StorageT>(hypot(re, im));
}

template <typename StorageT>
__global__ void split_complex_kernel(const double* __restrict__ src,
                                     StorageT* __restrict__ dst_re,
                                     StorageT* __restrict__ dst_im,
                                     int32_t count)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst_re[i] = static_cast<StorageT>(src[2 * i]);
    dst_im[i] = static_cast<StorageT>(src[2 * i + 1]);
}

template <typename StorageT>
__global__ void pack_complex_kernel(const StorageT* __restrict__ re,
                                    const StorageT* __restrict__ im,
                                    double* __restrict__ dst,
                                    int32_t count)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst[2 * i] = static_cast<double>(re[i]);
    dst[2 * i + 1] = static_cast<double>(im[i]);
}

int32_t grid_for(int32_t count)
{
    return (count + kConvertBlock - 1) / kConvertBlock;
}

}  // namespace


template <typename StorageT>
void launch_seed_state_from_v0(const double* v0_interleaved,
                               StorageT* v_re, StorageT* v_im,
                               StorageT* va, StorageT* vm,
                               int32_t count)
{
    if (count <= 0) return;
    seed_state_kernel<StorageT><<<grid_for(count), kConvertBlock, 0, cupf_current_cuda_stream()>>>(
        v0_interleaved, v_re, v_im, va, vm, count);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

template <typename StorageT>
void launch_split_complex(const double* src_interleaved,
                          StorageT* dst_re, StorageT* dst_im,
                          int32_t count)
{
    if (count <= 0) return;
    split_complex_kernel<StorageT><<<grid_for(count), kConvertBlock, 0, cupf_current_cuda_stream()>>>(
        src_interleaved, dst_re, dst_im, count);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

template <typename StorageT>
void launch_pack_complex_to_double(const StorageT* re, const StorageT* im,
                                   double* dst_interleaved,
                                   int32_t count)
{
    if (count <= 0) return;
    pack_complex_kernel<StorageT><<<grid_for(count), kConvertBlock, 0, cupf_current_cuda_stream()>>>(
        re, im, dst_interleaved, count);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

// Explicit instantiations for the storage scalar types (float: FP32/Mixed J/dx
// vs double: Mixed state). FP32 storage uses float; Mixed uses double state.
template void launch_seed_state_from_v0<float>(const double*, float*, float*, float*, float*, int32_t);
template void launch_seed_state_from_v0<double>(const double*, double*, double*, double*, double*, int32_t);
template void launch_split_complex<float>(const double*, float*, float*, int32_t);
template void launch_split_complex<double>(const double*, double*, double*, int32_t);
template void launch_pack_complex_to_double<float>(const float*, const float*, double*, int32_t);
template void launch_pack_complex_to_double<double>(const double*, const double*, double*, int32_t);

#endif  // CUPF_WITH_CUDA
