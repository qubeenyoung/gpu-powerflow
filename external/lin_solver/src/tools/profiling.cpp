#include "tools/profiling.hpp"

namespace sparse_direct::profile {

Scope::Scope(std::string name)
    : Scope(std::move(name), {}, false)
{
}

Scope::Scope(std::string name, bool use_cuda_timer)
    : Scope(std::move(name), {}, use_cuda_timer)
{
}

Scope::Scope(std::string name, std::string category)
    : Scope(std::move(name), std::move(category), false)
{
}

Scope::Scope(std::string name, std::string category, bool use_cuda_timer)
    : name_(std::move(name)),
      timer_scope_(name_, use_cuda_timer),
      nvtx_range_(name_, category)
{
}

void mark(const std::string& name)
{
    timer::mark(name);
    nvtx_profiler::mark(name);
}

}  // namespace sparse_direct::profile
