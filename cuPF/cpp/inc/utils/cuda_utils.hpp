#pragma once

#ifdef CUPF_WITH_CUDA

#include <cuda_runtime.h>
#include <cusparse.h>

#ifdef CUPF_ENABLE_CUDSS
  #include <cudss.h>
#endif

#include <cstddef>
#include <stdexcept>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                         \
        if (err__ != cudaSuccess) {                                                         \
            throw std::runtime_error(                                                       \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err__));                                         \
        }                                                                                   \
    } while (0)
#endif

#ifndef CUSPARSE_CHECK
#define CUSPARSE_CHECK(call)                                                                \
    do {                                                                                    \
        cusparseStatus_t status__ = (call);                                                 \
        if (status__ != CUSPARSE_STATUS_SUCCESS) {                                          \
            throw std::runtime_error(                                                       \
                std::string("cuSPARSE error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - status=" + std::to_string(static_cast<int>(status__)));                 \
        }                                                                                   \
    } while (0)
#endif

#ifdef CUPF_ENABLE_CUDSS
  #ifndef CUDSS_CHECK
    #define CUDSS_CHECK(call)                                                               \
        do {                                                                                \
            cudssStatus_t status__ = (call);                                                \
            if (status__ != CUDSS_STATUS_SUCCESS) {                                         \
                throw std::runtime_error(                                                   \
                    std::string("cuDSS error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                    " - status=" + std::to_string(static_cast<int>(status__)));             \
            }                                                                               \
        } while (0)
  #endif
#endif

inline void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

inline cudaStream_t& cupf_active_cuda_stream_ref()
{
    static thread_local cudaStream_t stream = nullptr;
    return stream;
}

inline cudaStream_t cupf_current_cuda_stream()
{
    return cupf_active_cuda_stream_ref();
}

class ScopedCudaStream {
public:
    explicit ScopedCudaStream(cudaStream_t stream)
        : previous_(cupf_active_cuda_stream_ref())
    {
        cupf_active_cuda_stream_ref() = stream;
    }

    ~ScopedCudaStream()
    {
        cupf_active_cuda_stream_ref() = previous_;
    }

    ScopedCudaStream(const ScopedCudaStream&) = delete;
    ScopedCudaStream& operator=(const ScopedCudaStream&) = delete;

private:
    cudaStream_t previous_ = nullptr;
};

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
        free();
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
            free();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void resize(std::size_t count)
    {
        free();
        if (count == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        count_ = count;
    }

    void assign(const T* src, std::size_t count, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
    {
        if (count != count_ || ptr_ == nullptr) {
            resize(count);
        }
        if (count == 0 || src == nullptr) {
            return;
        }
        cudaStream_t stream = cupf_current_cuda_stream();
        if (stream != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(ptr_, src, count * sizeof(T), kind, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), kind));
        }
    }

    void assignAsync(const T* src,
                     std::size_t count,
                     cudaMemcpyKind kind = cudaMemcpyHostToDevice,
                     cudaStream_t stream = cupf_current_cuda_stream())
    {
        if (count != count_ || ptr_ == nullptr) {
            resize(count);
        }
        if (count == 0 || src == nullptr) {
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(ptr_, src, count * sizeof(T), kind, stream));
    }

    void copyTo(T* dst, std::size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) const
    {
        if (dst == nullptr || count == 0) {
            return;
        }
        if (count > count_) {
            throw std::runtime_error("DeviceBuffer::copyTo count exceeds allocation");
        }
        cudaStream_t stream = cupf_current_cuda_stream();
        if (stream != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, count * sizeof(T), kind, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), kind));
        }
    }

    void copyToAsync(T* dst,
                     std::size_t count,
                     cudaMemcpyKind kind = cudaMemcpyDeviceToHost,
                     cudaStream_t stream = cupf_current_cuda_stream()) const
    {
        if (dst == nullptr || count == 0) {
            return;
        }
        if (count > count_) {
            throw std::runtime_error("DeviceBuffer::copyToAsync count exceeds allocation");
        }
        CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, count * sizeof(T), kind, stream));
    }

    void memsetZero()
    {
        if (ptr_ != nullptr && count_ > 0) {
            CUDA_CHECK(cudaMemsetAsync(ptr_, 0, count_ * sizeof(T), cupf_current_cuda_stream()));
        }
    }

    void free()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
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

#endif  // CUPF_WITH_CUDA
