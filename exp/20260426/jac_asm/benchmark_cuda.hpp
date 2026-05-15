#pragma once

#include "benchmark_support.hpp"
#include "common/data_types.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace exp20260426::jac_asm_bench {

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    int32_t count = 0;

    DeviceBuffer() = default;

    explicit DeviceBuffer(const std::vector<T>& values)
    {
        assign(values);
    }

    explicit DeviceBuffer(int32_t n)
    {
        allocate(n);
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    void allocate(int32_t n)
    {
        count = n;
        if (count == 0) {
            return;
        }

        void* raw = nullptr;
        cudaMalloc(&raw, sizeof(T) * count);
        ptr = (T*)raw;
    }

    void assign(const std::vector<T>& values)
    {
        allocate(values.size());
        if (count > 0) {
            cudaMemcpy(ptr, values.data(), sizeof(T) * count, cudaMemcpyHostToDevice);
        }
    }
};

inline void checkCuda(cudaError_t err, const char* where)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

inline void checkCuda(cudaError_t err, const std::string& where)
{
    checkCuda(err, where.c_str());
}

inline YbusCsr makeDeviceCsr(const CaseData& data,
                             const DeviceBuffer<int32_t>& row_ptr,
                             const DeviceBuffer<int32_t>& col,
                             const DeviceBuffer<float>& y_re,
                             const DeviceBuffer<float>& y_im)
{
    YbusCsr ybus;
    ybus.n_bus = data.n_bus;
    ybus.n_edges = data.n_edges;
    ybus.row_ptr = row_ptr.ptr;
    ybus.col = col.ptr;
    ybus.real = y_re.ptr;
    ybus.imag = y_im.ptr;
    return ybus;
}

inline YbusCoo makeDeviceCoo(const CaseData& data,
                             const DeviceBuffer<int32_t>& row,
                             const DeviceBuffer<int32_t>& col,
                             const DeviceBuffer<float>& y_re,
                             const DeviceBuffer<float>& y_im)
{
    YbusCoo ybus;
    ybus.n_bus = data.n_bus;
    ybus.n_edges = data.n_edges;
    ybus.row = row.ptr;
    ybus.col = col.ptr;
    ybus.real = y_re.ptr;
    ybus.imag = y_im.ptr;
    return ybus;
}

inline float elapsedKernelMs(cudaEvent_t begin, cudaEvent_t end, int32_t iters)
{
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
    return ms / iters;
}

}  // namespace exp20260426::jac_asm_bench
