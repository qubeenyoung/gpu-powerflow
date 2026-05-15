#pragma once

#include "data_types.hpp"

#include <cstdint>
#include <vector>

namespace exp20260426::newton_solver {

// Compact CSR pattern of the reduced Newton Jacobian.
struct JacobianPattern {
    // Matrix dimension and number of stored Jacobian coefficients.
    int32_t dim = 0;
    int32_t nnz = 0;
    // CSR structure used by the solver-side value array.
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
};

// Logical bus ordering for the reduced variables and equations.
struct JacobianIndex {
    // n_pvpq covers angle rows/columns; n_pq covers voltage-magnitude rows/cols.
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
    int32_t dim = 0;
    // pvpq stores the active buses in Jacobian row order: [PV..., PQ...].
    std::vector<int32_t> pvpq;
    // Reverse maps use -1 for buses that are absent from the reduced system.
    std::vector<int32_t> bus_to_pvpq;
    std::vector<int32_t> bus_to_pq;
};

// This map is common, not edge-specific. Both kernels need to know where each
// Ybus nonzero contributes inside the compact Jacobian value array.
struct JacobianMap {
    // Off-diagonal maps are indexed by Ybus nonzero k.
    std::vector<int32_t> offdiagJ11;
    std::vector<int32_t> offdiagJ12;
    std::vector<int32_t> offdiagJ21;
    std::vector<int32_t> offdiagJ22;

    // Diagonal maps are indexed by bus id because each bus owns one diagonal.
    std::vector<int32_t> diagJ11;
    std::vector<int32_t> diagJ12;
    std::vector<int32_t> diagJ21;
    std::vector<int32_t> diagJ22;
};

// Symbolic build product shared by vertex- and edge-parallel fill kernels.
struct JacobianBuild {
    JacobianPattern pattern;
    JacobianIndex index;
    JacobianMap map;
};

// Builds the reduced Jacobian pattern and all coefficient-slot lookup tables.
JacobianBuild buildJacobian(const YbusCsr& ybus,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            std::vector<int32_t>* edge_row = nullptr);

}  // namespace exp20260426::newton_solver

using exp20260426::newton_solver::JacobianBuild;
using exp20260426::newton_solver::JacobianIndex;
using exp20260426::newton_solver::JacobianMap;
using exp20260426::newton_solver::JacobianPattern;
using exp20260426::newton_solver::buildJacobian;

using BusIndexMap = exp20260426::newton_solver::JacobianIndex;
