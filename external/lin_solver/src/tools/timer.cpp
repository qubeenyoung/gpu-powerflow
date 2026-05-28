#include "tools/timer.hpp"

#if defined(PROFILE_ENABLE_TIMER) || defined(PROFILE_ENABLE_CUDA_TIMER)
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#endif

#if defined(PROFILE_ENABLE_CUDA_TIMER)
#include <cuda_runtime.h>
#endif

namespace sparse_direct::timer {
namespace {

#if defined(PROFILE_ENABLE_TIMER) || defined(PROFILE_ENABLE_CUDA_TIMER)
std::mutex& log_mutex()
{
    static std::mutex mutex;
    return mutex;
}
#endif

#if defined(PROFILE_ENABLE_CUDA_TIMER)
bool start_cuda_timer(cudaEvent_t& start, cudaEvent_t& stop) noexcept
{
    if (cudaEventCreate(&start) != cudaSuccess) {
        start = nullptr;
        stop = nullptr;
        return false;
    }
    if (cudaEventCreate(&stop) != cudaSuccess) {
        cudaEventDestroy(start);
        start = nullptr;
        stop = nullptr;
        return false;
    }
    if (cudaEventRecord(start, 0) != cudaSuccess) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        start = nullptr;
        stop = nullptr;
        return false;
    }
    return true;
}

bool record_cuda_timer_stop(cudaEvent_t stop) noexcept
{
    return stop != nullptr && cudaEventRecord(stop, 0) == cudaSuccess;
}

bool read_cuda_timer_elapsed(cudaEvent_t start, cudaEvent_t stop, float& elapsed_ms) noexcept
{
    return start != nullptr &&
           stop != nullptr &&
           cudaEventSynchronize(stop) == cudaSuccess &&
           cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess &&
           std::isfinite(elapsed_ms);
}

void destroy_cuda_timer(cudaEvent_t& start, cudaEvent_t& stop) noexcept
{
    if (stop != nullptr) {
        cudaEventDestroy(stop);
        stop = nullptr;
    }
    if (start != nullptr) {
        cudaEventDestroy(start);
        start = nullptr;
    }
}
#endif

}  // namespace

TimePoint now() noexcept
{
    return Clock::now();
}

double elapsed_ms(TimePoint start, TimePoint stop) noexcept
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

Stopwatch::Stopwatch() noexcept
    : start_(now())
{
}

void Stopwatch::reset() noexcept
{
    start_ = now();
}

double Stopwatch::elapsed_ms() const noexcept
{
    return timer::elapsed_ms(start_, now());
}

Scope::Scope(std::string name, bool use_cuda_timer)
    : name_(std::move(name)),
      use_cuda_timer_(use_cuda_timer)
      ,
      cpu_start_(Clock::now())
{
#if defined(PROFILE_ENABLE_CUDA_TIMER)
    if (use_cuda_timer_) {
        cuda_active_ = start_cuda_timer(cuda_start_, cuda_stop_);
    }
#else
    (void)use_cuda_timer_;
#endif
}

Scope::~Scope() noexcept
{
    const auto cpu_stop = Clock::now();

#if defined(PROFILE_ENABLE_CUDA_TIMER)
    const bool cuda_stop_recorded = use_cuda_timer_ && cuda_active_ && record_cuda_timer_stop(cuda_stop_);
#endif

#if defined(PROFILE_ENABLE_CUDA_TIMER)
    float cuda_elapsed_ms = 0.0F;
    const bool has_cuda_elapsed =
        use_cuda_timer_ && cuda_stop_recorded && read_cuda_timer_elapsed(cuda_start_, cuda_stop_, cuda_elapsed_ms);
    destroy_cuda_timer(cuda_start_, cuda_stop_);
#endif

#if defined(PROFILE_ENABLE_TIMER) || defined(PROFILE_ENABLE_CUDA_TIMER)
    const double cpu_elapsed_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start_).count();

    bool should_log = false;
#if defined(PROFILE_ENABLE_TIMER)
    should_log = true;
#endif
#if defined(PROFILE_ENABLE_CUDA_TIMER)
    should_log = should_log || use_cuda_timer_;
#endif

    if (!should_log) {
        return;
    }

    std::lock_guard<std::mutex> lock(log_mutex());
    std::cerr << std::setprecision(6) << "[timer] " << name_;
    if (should_log) {
#if defined(PROFILE_ENABLE_TIMER)
        std::cerr << " cpu_ms=" << cpu_elapsed_ms;
#endif
    }
#if defined(PROFILE_ENABLE_CUDA_TIMER)
    if (use_cuda_timer_) {
        if (has_cuda_elapsed) {
            std::cerr << " cuda_ms=" << cuda_elapsed_ms;
        } else {
            std::cerr << " cuda_ms=unavailable";
        }
    }
#endif
    std::cerr << "\n";
#endif
}

void mark(const std::string& name)
{
#if defined(PROFILE_ENABLE_TIMER) || defined(PROFILE_ENABLE_CUDA_TIMER)
    std::lock_guard<std::mutex> lock(log_mutex());
    std::cerr << "[mark] " << name << "\n";
#else
    (void)name;
#endif
}

}  // namespace sparse_direct::timer
