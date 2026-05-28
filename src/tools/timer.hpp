#pragma once

#include <chrono>

#if defined(PROFILE_ENABLE_CUDA_TIMER)
#include <cuda_runtime.h>
#endif

#include <string>

namespace sparse_direct::timer {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

TimePoint now() noexcept;
double elapsed_ms(TimePoint start, TimePoint stop) noexcept;

class Stopwatch {
public:
    Stopwatch() noexcept;

    void reset() noexcept;
    double elapsed_ms() const noexcept;

private:
    TimePoint start_;
};

class Scope {
public:
    Scope(std::string name, bool use_cuda_timer);
    ~Scope() noexcept;

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(Scope&&) = delete;

private:
    std::string name_;
    bool use_cuda_timer_ = false;

    TimePoint cpu_start_;

#if defined(PROFILE_ENABLE_CUDA_TIMER)
    cudaEvent_t cuda_start_ = nullptr;
    cudaEvent_t cuda_stop_ = nullptr;
    bool cuda_active_ = false;
#endif
};

void mark(const std::string& name);

}  // namespace sparse_direct::timer
