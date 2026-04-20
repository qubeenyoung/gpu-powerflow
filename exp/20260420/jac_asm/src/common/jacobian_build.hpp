#pragma once

#include "data_types.hpp"

#include <cstdint>
#include <vector>

namespace exp20260420::newton_solver {

struct JacobianPattern {
    int32_t dim = 0;
    int32_t nnz = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
};

struct JacobianIndex {
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
    int32_t dim = 0;
    std::vector<int32_t> pvpq;
    std::vector<int32_t> bus_to_pvpq;
    std::vector<int32_t> bus_to_pq;
};

struct JacobianMap {
    std::vector<int32_t> offdiagJ11;
    std::vector<int32_t> offdiagJ12;
    std::vector<int32_t> offdiagJ21;
    std::vector<int32_t> offdiagJ22;

    std::vector<int32_t> diagJ11;
    std::vector<int32_t> diagJ12;
    std::vector<int32_t> diagJ21;
    std::vector<int32_t> diagJ22;
};

struct JacobianBuild {
    JacobianPattern pattern;
    JacobianIndex index;
    JacobianMap map;
};

JacobianBuild buildJacobian(const YbusGraph& ybus,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq);

}  // namespace exp20260420::newton_solver

using exp20260420::newton_solver::JacobianBuild;
using exp20260420::newton_solver::JacobianIndex;
using exp20260420::newton_solver::JacobianMap;
using exp20260420::newton_solver::JacobianPattern;
using exp20260420::newton_solver::buildJacobian;

using BusIndexMap = exp20260420::newton_solver::JacobianIndex;
