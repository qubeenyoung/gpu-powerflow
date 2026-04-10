#pragma once

#include <cstdint>
#include <vector>


// ---------------------------------------------------------------------------
// JacobianBuilderType: selects which algorithm builds the sparsity maps.
//
// EdgeBased   — iterates over Ybus non-zeros, one map entry per edge.
//               This is the v1 approach and the current default.
//
// VertexBased — iterates over buses, aggregates neighbor contributions.
//               Will require a different analyze() signature when added.
// ---------------------------------------------------------------------------
enum class JacobianBuilderType {
    EdgeBased,
    VertexBased,
};


// ---------------------------------------------------------------------------
// JacobianMaps: sparsity mapping tables produced by JacobianBuilder::analyze().
//
// The Jacobian is a 2×2 block matrix:
//
//   J = [ J11  J12 ] = [ dP/dθ    dP/d|V| ]    rows: pvpq, cols: pvpq
//       [ J21  J22 ]   [ dQ/dθ    dQ/d|V| ]    rows: pq,   cols: pq
//
// For each non-zero entry k in Ybus (CSR order), mapJ**[k] gives the index
// into J.values where that entry contributes. The diagonal contribution of
// each bus is handled separately via diagJ**.
//
// Indices are in CSR position space (matching JacobianStructure).
// Built once in analyze() and reused every NR iteration.
// ---------------------------------------------------------------------------
struct JacobianMaps {
    // Which execution strategy these maps were built for.
    // VertexBased currently reuses the same CSR-indexed maps as EdgeBased,
    // but the CUDA backend switches to a row-owner fill kernel.
    JacobianBuilderType builder_type = JacobianBuilderType::EdgeBased;

    // Off-diagonal: Ybus entry k (CSR order) → position in J value array
    std::vector<int32_t> mapJ11, mapJ12, mapJ21, mapJ22;

    // Diagonal: bus i → position on J diagonal for each quadrant
    std::vector<int32_t> diagJ11, diagJ12, diagJ21, diagJ22;

    // pvpq = pv ∪ pq (concatenated in that order: pv[0..n_pv), pq[0..n_pq))
    std::vector<int32_t> pvpq;

    int32_t n_pvpq = 0;
    int32_t n_pq   = 0;

    // Jacobian dimension: dimF = n_pvpq + n_pq  (= n_pv + 2*n_pq)
};


// ---------------------------------------------------------------------------
// JacobianStructure: backend-agnostic CSR representation of the Jacobian
// sparsity pattern produced by JacobianBuilder::analyze().
//
// CPU backend converts this to Eigen CSC internally.
// CUDA backend uploads it to GPU directly — no Eigen dependency in the interface.
// ---------------------------------------------------------------------------
struct JacobianStructure {
    std::vector<int32_t> row_ptr;  // size = dim + 1
    std::vector<int32_t> col_idx;  // size = nnz, sorted within each row
    int32_t dim = 0;               // number of rows == cols
    int32_t nnz = 0;
};
