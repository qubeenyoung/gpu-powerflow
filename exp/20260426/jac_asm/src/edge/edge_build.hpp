#pragma once

#include "data_types.hpp"
#include "jacobian_build.hpp"

#include <cstdint>
#include <vector>

namespace exp20260426::newton_solver {

// Edge-parallel kernels need an explicit row id for each Ybus nonzero.
struct EdgeYbusMap {
    std::vector<int32_t> row;
};

// Edge build result pairs the common Jacobian metadata with edge-only topology.
struct EdgeJacobianBuild {
    JacobianBuild jacobian;
    EdgeYbusMap edge_map;
};

// Edge kernels launch by nonzero index k, so they need row[k].  This is the
// only edge-only preprocessing step in this experiment.
EdgeYbusMap buildEdgeYbusMap(const YbusCsr& ybus);

// Same symbolic build as buildJacobian(), but row[k] is materialized during the
// Ybus->Jacobian map pass.
EdgeJacobianBuild buildJacobianWithEdgeYbusMap(const YbusCsr& ybus,
                                               const int32_t* pv,
                                               int32_t n_pv,
                                               const int32_t* pq,
                                               int32_t n_pq);

}  // namespace exp20260426::newton_solver

using exp20260426::newton_solver::EdgeJacobianBuild;
using exp20260426::newton_solver::EdgeYbusMap;
using exp20260426::newton_solver::buildJacobianWithEdgeYbusMap;
using exp20260426::newton_solver::buildEdgeYbusMap;
