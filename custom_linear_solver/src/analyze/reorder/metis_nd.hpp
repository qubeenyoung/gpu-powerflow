#pragma once

#include <vector>

// METIS nested-dissection fill-reducing ordering.
namespace custom_linear_solver::reordering {

// Fill `perm` (size n) with a METIS ND ordering of the symmetric pattern
// (A + Aᵀ, diagonal excluded) of the n×n matrix given in CSC/CSR-pattern form
// (col_ptr/row_idx; the pattern is symmetric in structure either way).
// Falls back to the natural order if METIS cannot order the graph.
// Returns false only on invalid input.
// parallel=true uses parallel nested dissection (recurse separator halves across cores,
// ~-40% on analyze wall for large Jacobians, fill ~= serial so factor stays competitive).
// NOTE: METIS's
// RNG is a thread-unsafe global -> the parallel ordering is NON-deterministic run-to-run
// (within a run / NR loop the ordering is computed once and fixed). Production enables it
// for the A win; reproducible-benchmark callers leave it false (serial).
bool metis_nd(int n, const int* col_ptr, const int* row_idx, std::vector<int>& perm,
              bool parallel = false, std::vector<int>* sym_col_ptr = nullptr,
              std::vector<int>* sym_row_idx = nullptr, int seed = 42);

// ND ordering from a prebuilt symmetric graph (xadj size n+1, adjncy = directed unique
// off-diagonal edges, neighbors sorted+deduped per vertex) — e.g. one built on the GPU by
// matrix::build_symmetric_graph_device. The inputs may be moved from (consumed) when the
// METIS idx_t is 32-bit. Same ordering as metis_nd given the same graph.
bool metis_nd_from_graph(int n, std::vector<int>& xadj, std::vector<int>& adjncy,
                         std::vector<int>& perm, bool parallel, int seed = 42);

// exp_260612: GPU/TC-objective nested dissection from a prebuilt symmetric graph. Owns the
// recursion + per-split objective + stopping granularity (reusing METIS_ComputeVertexSeparator as
// the bisection primitive); optimizes a GPU critical-path cost (TC-discounted separator front +
// imbalance penalty) instead of fill. Knobs via env: CLS_GPU_ND_CAND/_LEAF/_LAMBDA/_TC_G. Same
// perm convention as metis_nd_from_graph (drop-in). Inputs may be moved from.
bool gpu_nd_from_graph(int n, std::vector<int>& xadj, std::vector<int>& adjncy,
                       std::vector<int>& perm, int seed = 42);

// exp_260612 Stage 2: ELECTRICAL-weighted ND from the Jacobian CSR (host arrays: rowptr/colidx/vals).
// Builds a |J_ij|-weighted symmetric graph and, for the top CLS_GPU_ND_EW_DEPTH levels, bisects by
// edge-weighted METIS partitioning (cuts electrically weak tie-lines — the power-grid separators
// METIS_NodeND cannot target since it has no edge weights), deriving + FM-refining a vertex
// separator; below that depth it falls back to gpu_nd. Same perm convention. Knobs: CLS_GPU_ND_EW_DEPTH.
bool gpu_nd_weighted_from_graph(int n, const int* rowptr, const int* colidx, const double* vals,
                                std::vector<int>& perm, int seed = 42);

}  // namespace custom_linear_solver::reordering
