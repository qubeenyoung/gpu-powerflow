#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#ifndef CUITER_CUDA_CHECK
#define CUITER_CUDA_CHECK(call)                                                            \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +     \
                                     std::to_string(__LINE__) + " - " +                   \
                                     cudaGetErrorString(err__));                           \
        }                                                                                  \
    } while (0)
#endif

#ifndef CUITER_CUBLAS_CHECK
#define CUITER_CUBLAS_CHECK(call)                                                          \
    do {                                                                                   \
        cublasStatus_t status__ = (call);                                                  \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                           \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":" +   \
                                     std::to_string(__LINE__) + " - status=" +            \
                                     std::to_string(static_cast<int>(status__)));           \
        }                                                                                  \
    } while (0)
#endif

#ifndef CUITER_CUSOLVER_CHECK
#define CUITER_CUSOLVER_CHECK(call)                                                        \
    do {                                                                                   \
        cusolverStatus_t status__ = (call);                                                \
        if (status__ != CUSOLVER_STATUS_SUCCESS) {                                         \
            throw std::runtime_error(std::string("cuSOLVER error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - status=" +            \
                                     std::to_string(static_cast<int>(status__)));           \
        }                                                                                  \
    } while (0)
#endif

namespace cuiter {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count)
    {
        resize(count);
    }

    ~DeviceBuffer()
    {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_)
        , count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void resize(std::size_t count)
    {
        if (count == count_) {
            return;
        }
        reset();
        if (count == 0) {
            return;
        }
        CUITER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        count_ = count;
    }

    void assign(const T* src, std::size_t count, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
    {
        resize(count);
        if (count == 0 || src == nullptr) {
            return;
        }
        CUITER_CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), kind));
    }

    void copy_to(T* dst, std::size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) const
    {
        if (dst == nullptr || count == 0) {
            return;
        }
        if (count > count_) {
            throw std::runtime_error("DeviceBuffer::copy_to count exceeds allocation");
        }
        CUITER_CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), kind));
    }

    void memset_zero()
    {
        if (ptr_ != nullptr && count_ > 0) {
            CUITER_CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
        }
    }

    void reset()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    T* data() const
    {
        return ptr_;
    }

    std::size_t size() const
    {
        return count_;
    }

    bool empty() const
    {
        return count_ == 0;
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class CudaEventTimer {
public:
    CudaEventTimer()
    {
        CUITER_CUDA_CHECK(cudaEventCreate(&start_));
        CUITER_CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaEventTimer()
    {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start(cudaStream_t stream = nullptr)
    {
        CUITER_CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    double stop(cudaStream_t stream = nullptr)
    {
        CUITER_CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUITER_CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0f;
        CUITER_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return static_cast<double>(milliseconds) * 1.0e-3;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

}  // namespace cuiter
