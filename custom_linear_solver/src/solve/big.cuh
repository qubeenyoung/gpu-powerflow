#pragma once

// Internal — included only by multifrontal.cu (single TU).
//
// Solve kernels for the regular (block-per-front) path. One block per (front, batch); the
// substitution is driven by one warp inside the block, the CB update is parallelized across
// all threads. The whole front lives in global memory — solve doesn't have a shared-resident
// variant because the work per front is much lighter than factor.

#include <cuda_runtime.h>

#include "solve/primitives.cuh"

namespace custom_linear_solver {
namespace {

// Forward solve level (L y = b). T = front type AND solve working-vector type.
template <typename T>
__global__ void solve_fwd(int lbegin, int lend, const int* __restrict__ plcols,
                          const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                          const int* __restrict__ ncols, const int* __restrict__ front_rows,
                          const T* frontB, T* yB, long front_total, int n)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const T* front = frontB + (long)blockIdx.y * front_total;
    T* y = yB + (long)blockIdx.y * n;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    __shared__ T sh_piv[MF_MAX_NC];

    // Phase 1 — panel substitution (one warp).
    if (t < 32) fwd_substitute<T>(F, fsz, nc, fr, y, sh_piv, /*lane=*/t);
    __syncthreads();
    // Phase 2 — CB rows update across all threads.
    fwd_cb_update<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t, nt);
}

// Backward solve level (U x = y). T = front type AND solve working-vector type.
template <typename T>
__global__ void solve_bwd(int lbegin, int lend, const int* __restrict__ plcols,
                          const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                          const int* __restrict__ ncols, const int* __restrict__ front_rows,
                          const T* frontB, T* yB, long front_total, int n)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const T* front = frontB + (long)blockIdx.y * front_total;
    T* y = yB + (long)blockIdx.y * n;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    const int cb = fsz - nc;

    // Caller-provided dynamic shared holds the x cache (cb entries); rhs is a static shared array.
    extern __shared__ unsigned char xsh_raw[];
    T* xsh = reinterpret_cast<T*>(xsh_raw);
    __shared__ T rhs[MF_MAX_NC];

    // Phase 1 — load rhs[] and x cache from y.
    bwd_load_rhs_and_x<T>(y, fr, nc, cb, rhs, xsh, t, nt);
    __syncthreads();
    // Phase 2 — CB contribution to rhs. Threads with t < nc each compute one rhs entry.
    bwd_cb_subtract<T>(F, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
    __syncthreads();
    // Phase 3 — panel substitution (one warp), writes x back to y[fr[0..nc)].
    if (t < 32) bwd_substitute<T>(F, fsz, nc, fr, y, rhs, /*lane=*/t);
}

}  // namespace
}  // namespace custom_linear_solver
