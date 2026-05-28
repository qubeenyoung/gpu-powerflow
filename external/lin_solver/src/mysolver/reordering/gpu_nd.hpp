#pragma once

namespace mysolver::reordering {

// cy201: GPU nested-dissection vertex separator (the core of an opt-in GPU-ND ordering, env
// GPU_ND). Fills part[v] in {0,1,2} (half0, half1, separator) for the CSR graph (xadj,adjncy,
// idx_t==int). Reuses the par_nd_rec framework (induce/recurse/assemble) -- only the separator
// computation moves to GPU. Returns true on a valid bisection; false -> caller falls back to METIS.
// First version: GPU level-synchronous BFS + median bisection (quality iterates over cycles).
bool gpu_nd_separator(int n, const int* xadj, const int* adjncy, int nnz, int* part);

}  // namespace mysolver::reordering
