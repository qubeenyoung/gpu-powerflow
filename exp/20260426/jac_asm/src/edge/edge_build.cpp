#include "edge_build.hpp"

namespace exp20260426::newton_solver {

EdgeYbusMap buildEdgeYbusMap(const YbusCsr& ybus)
{
    EdgeYbusMap map;
    // One row id is materialized for each CSR nonzero so edge kernels can launch
    // directly by k without reconstructing the owning bus.
    map.row.resize(ybus.n_edges);

    // Walk each CSR row and stamp its bus id onto every edge in that row.
    for (int32_t bus = 0; bus < ybus.n_bus; ++bus) {
        for (int32_t k = ybus.row_ptr[bus]; k < ybus.row_ptr[bus + 1]; ++k) {
            map.row[k] = bus;
        }
    }

    return map;
}

EdgeJacobianBuild buildJacobianWithEdgeYbusMap(const YbusCsr& ybus,
                                               const int32_t* pv,
                                               int32_t n_pv,
                                               const int32_t* pq,
                                               int32_t n_pq)
{
    EdgeJacobianBuild build;
    // Reuse the common symbolic builder and ask it to fill row[k] in the same
    // Ybus pass that creates the coefficient-slot maps.
    build.jacobian = buildJacobian(ybus, pv, n_pv, pq, n_pq, &build.edge_map.row);
    return build;
}

}  // namespace exp20260426::newton_solver
