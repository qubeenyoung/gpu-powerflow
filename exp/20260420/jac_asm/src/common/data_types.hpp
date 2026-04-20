#pragma once

#include <cstdint>

namespace exp20260420 {

struct YbusGraph {
    int32_t n_bus = 0;
    int32_t n_edges = 0;

    const int32_t* row = nullptr;
    const int32_t* col = nullptr;
    const int32_t* row_ptr = nullptr;

    const float* real = nullptr;
    const float* imag = nullptr;
};

}  // namespace exp20260420

using exp20260420::YbusGraph;
