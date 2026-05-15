#pragma once

#include <cstdint>

namespace exp20260426 {

// CSR is the natural input for a vertex-based traversal: the row bus is the
// vertex being processed, so no per-nonzero row array is needed.
struct YbusCsr {
    // Matrix size and number of stored complex admittance entries.
    int32_t n_bus = 0;
    int32_t n_edges = 0;

    // CSR topology: row_ptr[bus]..row_ptr[bus + 1] indexes the neighbors.
    const int32_t* row_ptr = nullptr;
    const int32_t* col = nullptr;

    // Complex Ybus values aligned with the CSR column index array.
    const float* real = nullptr;
    const float* imag = nullptr;
};

// COO is the explicit edge-parallel input. The extra row[k] array is the
// materialized CSR->edge mapping this experiment wants to account for.
struct YbusCoo {
    // Matrix size and number of stored complex admittance entries.
    int32_t n_bus = 0;
    int32_t n_edges = 0;

    // COO topology: row[k] and col[k] identify the bus pair for edge k.
    const int32_t* row = nullptr;
    const int32_t* col = nullptr;

    // Complex Ybus values aligned with the COO edge index.
    const float* real = nullptr;
    const float* imag = nullptr;
};

}  // namespace exp20260426

using exp20260426::YbusCoo;
using exp20260426::YbusCsr;
