#pragma once

#include "jacobian_types.hpp"
#include "newton_solver_types.hpp"


// ---------------------------------------------------------------------------
// JacobianBuilder: one-time sparsity analysis before the NR loop.
//
// Usage:
//   JacobianBuilder builder(JacobianBuilderType::EdgeBased);
//   auto [maps, J] = builder.analyze(ybus, pv, n_pv, pq, n_pq);
//   // maps and J are then passed into StorageFactory / PlanBuilder
// ---------------------------------------------------------------------------
class JacobianBuilder {
public:
    explicit JacobianBuilder(JacobianBuilderType type = JacobianBuilderType::EdgeBased);

    struct Result {
        JacobianMaps      maps;
        JacobianStructure J;    // CSR sparsity; values filled each NR iteration
    };

    // Analyze Ybus sparsity → return mapping tables + Jacobian CSR structure.
    // pv/pq are global bus indices. Call once before the first solve().
    Result analyze(const YbusView& ybus,
                   const int32_t*  pv,   int32_t n_pv,
                   const int32_t*  pq,   int32_t n_pq);

private:
    JacobianBuilderType type_;

    Result analyzeEdgeBased(const YbusView& ybus,
                            const int32_t*  pv, int32_t n_pv,
                            const int32_t*  pq, int32_t n_pq);

    Result analyzeVertexBased(const YbusView& ybus,
                              const int32_t*  pv, int32_t n_pv,
                              const int32_t*  pq, int32_t n_pq);
};
