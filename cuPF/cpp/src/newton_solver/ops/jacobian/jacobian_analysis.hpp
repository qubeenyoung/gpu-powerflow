#pragma once

#include "newton_solver/core/newton_solver_types.hpp"

#include <cstdint>
#include <vector>


// ---------------------------------------------------------------------------
// JacobianIndexing: PV/PQ bus indexing shared by Jacobian analysis steps.
// ---------------------------------------------------------------------------
struct JacobianIndexing {
    int32_t n_bus = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;

    // pvpq = pv || pq
    std::vector<int32_t> pvpq;

    // Bus number -> Jacobian row/column index. -1 means "not in this block".
    std::vector<int32_t> row_pvpq;
    std::vector<int32_t> row_pq;
    std::vector<int32_t> col_pvpq;
    std::vector<int32_t> col_pq;
};


// ---------------------------------------------------------------------------
// JacobianPattern: backend-agnostic CSR sparsity pattern of the Jacobian.
//
// Values are filled every NR iteration; this object owns only row/column
// structure.
// ---------------------------------------------------------------------------
struct JacobianPattern {
    std::vector<int32_t> row_ptr;  // size = dim + 1
    std::vector<int32_t> col_idx;  // size = nnz, sorted within each row
    int32_t dim = 0;               // rows == cols
    int32_t nnz = 0;
};


// ---------------------------------------------------------------------------
// JacobianScatterMap: Ybus entry/bus -> Jacobian value-position map.
//
// For each non-zero entry k in Ybus (CSR order), mapJ**[k] gives the index
// into JacobianPattern values where that entry contributes. Diagonal
// contributions are handled separately via diagJ**.
// ---------------------------------------------------------------------------
struct JacobianScatterMap {
    // Off-diagonal/direct: Ybus entry k (CSR order) -> position in J values.
    std::vector<int32_t> mapJ11, mapJ12, mapJ21, mapJ22;

    // Diagonal: bus i -> position on J diagonal for each quadrant.
    std::vector<int32_t> diagJ11, diagJ12, diagJ21, diagJ22;

    // pvpq = pv || pq
    std::vector<int32_t> pvpq;

    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
};


JacobianIndexing make_jacobian_indexing(
    int32_t n_bus,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq
);


class JacobianPatternGenerator {
public:
    JacobianPattern generate(const YbusView& ybus,
                             const JacobianIndexing& indexing) const;
};


class JacobianMapBuilder {
public:
    JacobianScatterMap build(const YbusView& ybus,
                             const JacobianIndexing& indexing,
                             const JacobianPattern& pattern) const;
};
