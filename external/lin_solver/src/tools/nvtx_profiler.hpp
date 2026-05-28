#pragma once

#include <string>

namespace sparse_direct::nvtx_profiler {

class Range {
public:
    Range(const std::string& name, const std::string& category);
    ~Range() noexcept;

    Range(const Range&) = delete;
    Range& operator=(const Range&) = delete;
    Range(Range&&) = delete;
    Range& operator=(Range&&) = delete;
};

void mark(const std::string& name);

}  // namespace sparse_direct::nvtx_profiler
