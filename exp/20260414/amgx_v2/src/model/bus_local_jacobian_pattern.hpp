#pragma once

#include "linear/amgx_preconditioner.hpp"
#include "model/bus_local_index.hpp"

#include <cstdint>
#include <vector>

namespace exp_20260414::amgx_v2 {

struct BusLocalJacobianPattern {
    int32_t dim = 0;
    int32_t nnz = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;

    CsrMatrixView host_view(const std::vector<double>& values) const;
};

BusLocalJacobianPattern build_bus_local_jacobian_pattern(
    const BusLocalIndex& index,
    const std::vector<int32_t>& ybus_row_ptr,
    const std::vector<int32_t>& ybus_col_idx);

}  // namespace exp_20260414::amgx_v2
