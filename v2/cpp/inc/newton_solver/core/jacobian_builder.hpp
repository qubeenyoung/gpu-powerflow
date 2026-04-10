#pragma once

#include "jacobian_types.hpp"
#include "newton_solver_types.hpp"


// ---------------------------------------------------------------------------
// JacobianBuilder: performs the one-time sparsity analysis before the NR loop.
//
// Usage:
//   JacobianBuilder builder(JacobianBuilderType::EdgeBased);
//   auto result = builder.analyze(ybus, pv, n_pv, pq, n_pq);
//   backend.analyze(ybus, result.maps, result.J, n_bus);
//   // ... NR loop calls backend.updateJacobian() each iteration
// ---------------------------------------------------------------------------
class JacobianBuilder {
public:
    explicit JacobianBuilder(JacobianBuilderType type = JacobianBuilderType::EdgeBased);

    struct Result {
        JacobianMaps      maps;
        JacobianStructure J;    // CSR sparsity pattern; values filled each NR iter
    };

    // Analyze Ybus sparsity → return mapping tables + Jacobian structure.
    // pv/pq are global bus indices. Call once before the first solve().
    Result analyze(const YbusView& ybus,
                   const int32_t* pv, int32_t n_pv,
                   const int32_t* pq, int32_t n_pq);

private:
    JacobianBuilderType type_;

    Result analyzeEdgeBased(const YbusView& ybus,
                            const int32_t* pv, int32_t n_pv,
                            const int32_t* pq, int32_t n_pq);

    Result analyzeVertexBased(const YbusView& ybus,
                              const int32_t* pv, int32_t n_pv,
                              const int32_t* pq, int32_t n_pq);
};
