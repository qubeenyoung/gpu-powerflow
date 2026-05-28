#pragma once

#include <string>

#include "tools/nvtx_profiler.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::profile {

class Scope {
public:
    explicit Scope(std::string name);
    Scope(std::string name, bool use_cuda_timer);
    Scope(std::string name, std::string category);
    Scope(std::string name, std::string category, bool use_cuda_timer);

private:
    std::string name_;
    timer::Scope timer_scope_;
    nvtx_profiler::Range nvtx_range_;
};

void mark(const std::string& name);

}  // namespace sparse_direct::profile

#define PROFILE_CONCAT_IMPL(lhs, rhs) lhs##rhs
#define PROFILE_CONCAT(lhs, rhs) PROFILE_CONCAT_IMPL(lhs, rhs)

#if defined(PROFILE_ENABLE_TIMER) || defined(PROFILE_ENABLE_CUDA_TIMER) || defined(PROFILE_ENABLE_NVTX)
#define PROFILE_SCOPE(name) \
    ::sparse_direct::profile::Scope PROFILE_CONCAT(profile_scope_, __LINE__)(name)
#define PROFILE_CUDA_SCOPE(name) \
    ::sparse_direct::profile::Scope PROFILE_CONCAT(profile_cuda_scope_, __LINE__)(name, true)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__func__)
#define PROFILE_CUDA_FUNCTION() PROFILE_CUDA_SCOPE(__func__)
#define PROFILE_MARK(name) ::sparse_direct::profile::mark(name)
#else
#define PROFILE_SCOPE(name) ((void)0)
#define PROFILE_CUDA_SCOPE(name) ((void)0)
#define PROFILE_FUNCTION() ((void)0)
#define PROFILE_CUDA_FUNCTION() ((void)0)
#define PROFILE_MARK(name) ((void)0)
#endif
