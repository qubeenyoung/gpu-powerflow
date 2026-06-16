#pragma once

// SOLVE — kernel entry points (__global__).
//
// Internal — included into the factor/solve driver TUs (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// Two kernel families × two directions:
//
//   tier    | block        | front location  | kernels (forward / backward)
//   --------+--------------+-----------------+------------------------------------
//   small   | 8 warps      | global (L1)     | solve_fwd_small<T>, solve_bwd_small<T>
//   regular | 64-256 thr   | global          | solve_fwd<T>, solve_bwd<T>
//
// The regular kernels are NOT tier-split into mid/big like factor — the solve work per front
// is much lighter than factor (no rank-nc GEMM; just substitution + CB row update), so a single
// block-per-front kernel with caller-tuned thread count covers all max_fsz > kTinyFrontMax
// (i.e. the small / big / large factor tiers all share one solve regular kernel).
//
// Each kernel is a thin orchestrator composing the device building blocks in phases.cuh:
//   forward:  fwd_substitute  → sync  → fwd_cb_update
//   backward: bwd_load_rhs_and_x → sync → bwd_cb_subtract → sync → bwd_substitute

#include <cuda_runtime.h>

#include "solve/phases.cuh"

namespace custom_linear_solver {
namespace {

// =======================================================================================
//  SMALL tier — one WARP per (front, batch), 8 warps per block
// =======================================================================================
//
// The bottom etree levels (tens of thousands of tiny fronts) dominate the batched solve the
// same way they dominate factor. Block-per-front would launch one 32-thread block per
// (front, batch) → block-launch / scheduling overhead + poor SM packing. solve_fwd_small /
// solve_bwd_small pack W warps per block (one (front, batch) per warp) and use __syncwarp.

// sub_group_size = sub-group lane count (8 / 16 / 32). One sub-group of sub_group_size lanes owns one (front, batch);
// fronts_per_warp = 32/sub_group_size fronts pack per warp (same tiny-front-packing idea as factor_tiny). sub_group_size=32 is the
// classic one-warp-per-front form. The dispatcher picks sub_group_size from the level's max_fsz.
template <typename T, int sub_group_size>
__global__ void solve_fwd_small(int lbegin, int level_size, int B, int slab,
                                const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ front_rows,
                                const T* frontB, T* yB, long front_total, int n)
{
    constexpr int fronts_per_warp = 32 / sub_group_size;
    extern __shared__ unsigned char fsm_raw[];
    T* slabs = reinterpret_cast<T*>(fsm_raw);
    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sg = lane / sub_group_size, sl = lane % sub_group_size;
    const unsigned mask = (sub_group_size == 32) ? 0xffffffffu : (((1u << sub_group_size) - 1u) << (sg * sub_group_size));
    const int warps_per_blk = blockDim.x >> 5;
    const int slot = (blockIdx.x * warps_per_blk + warp_in_blk) * fronts_per_warp + sg;
    if (slot >= level_size * B) return;
    const int fl = slot % level_size, bb = slot / level_size;
    const T* front = frontB + (long)bb * front_total;
    T* y = yB + (long)bb * n;
    const int p = plcols[lbegin + fl];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;
    T* sh_piv = slabs + (long)(warp_in_blk * fronts_per_warp + sg) * slab;

    fwd_substitute<T, sub_group_size>(F, fsz, nc, fr, y, sh_piv, sl, mask);
    __syncwarp(mask);
    fwd_cb_update<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, sl, /*nt=*/sub_group_size);
}

template <typename T, int sub_group_size>
__global__ void solve_bwd_small(int lbegin, int level_size, int B, int slab,
                                const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ front_rows,
                                const T* frontB, T* yB, long front_total, int n)
{
    constexpr int fronts_per_warp = 32 / sub_group_size;
    extern __shared__ unsigned char bsm_raw[];
    T* slabs = reinterpret_cast<T*>(bsm_raw);
    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sg = lane / sub_group_size, sl = lane % sub_group_size;
    const unsigned mask = (sub_group_size == 32) ? 0xffffffffu : (((1u << sub_group_size) - 1u) << (sg * sub_group_size));
    const int warps_per_blk = blockDim.x >> 5;
    const int slot = (blockIdx.x * warps_per_blk + warp_in_blk) * fronts_per_warp + sg;
    if (slot >= level_size * B) return;
    const int fl = slot % level_size, bb = slot / level_size;
    const T* front = frontB + (long)bb * front_total;
    T* y = yB + (long)bb * n;
    const int p = plcols[lbegin + fl];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int cb = fsz - nc;
    T* rhs = slabs + (long)(warp_in_blk * fronts_per_warp + sg) * slab;   // [0, nc)
    T* xsh = rhs + kMaxPivotColumns;                                 // [0, cb)

    bwd_load_rhs_and_x<T>(y, fr, nc, cb, rhs, xsh, sl, /*nt=*/sub_group_size);
    __syncwarp(mask);
    bwd_cb_subtract<T, sub_group_size>(F, fsz, nc, cb, xsh, rhs, sl, /*width=*/sub_group_size, mask);
    __syncwarp(mask);
    bwd_substitute<T, sub_group_size>(F, fsz, nc, fr, y, rhs, sl, mask);
}

// =======================================================================================
//  REGULAR tier — block per (front, batch); thread count caller-chosen (64 / 128 / 256)
// =======================================================================================
//
// One block per (front, batch). The substitution (Phase 1 of fwd, Phase 3 of bwd) is driven
// by one warp inside the block — the recurrence in k is inherently sequential and a single
// warp is fast enough. The CB update / CB subtract (which IS parallelizable) is spread across
// all `nt` block threads. The whole front lives in global memory — solve doesn't have a
// shared-resident variant because the work per front is much lighter than factor.

template <typename T, int NC>
__device__ __forceinline__ void fwd_substitute_fixed_regular(const T* F, int fsz, const int* fr,
                                                             T* y, T* sh_piv, int t)
{
    if constexpr (NC <= 8) {
        if (t < 8) fwd_substitute_fixed<T, NC, 8>(F, fsz, fr, y, sh_piv, t, 0x000000ffu);
    } else if constexpr (NC <= 16) {
        if (t < 16) fwd_substitute_fixed<T, NC, 16>(F, fsz, fr, y, sh_piv, t, 0x0000ffffu);
    } else {
        if (t < 32) fwd_substitute_fixed<T, NC>(F, fsz, fr, y, sh_piv, t);
    }
}

template <typename T, int NC>
__device__ __forceinline__ void bwd_substitute_fixed_regular(const T* F, int fsz, const int* fr,
                                                             T* y, const T* rhs, int t)
{
    if constexpr (NC <= 8) {
        if (t < 8) bwd_substitute_fixed<T, NC, 8>(F, fsz, fr, y, rhs, t, 0x000000ffu);
    } else if constexpr (NC <= 16) {
        if (t < 16) bwd_substitute_fixed<T, NC, 16>(F, fsz, fr, y, rhs, t, 0x0000ffffu);
    } else {
        if (t < 32) bwd_substitute_fixed<T, NC>(F, fsz, fr, y, rhs, t);
    }
}

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
    __shared__ T sh_piv[kMaxPivotColumns];

    // Phase 1 — panel substitution (one warp).
    if (t < 32) fwd_substitute<T>(F, fsz, nc, fr, y, sh_piv, /*lane=*/t);
    __syncthreads();
    // Phase 2 — CB rows update across all threads.
    fwd_cb_update<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t, nt);
}

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

    // Caller-provided dynamic shared holds the x cache (cb entries); rhs is a static shared.
    extern __shared__ unsigned char xsh_raw[];
    T* xsh = reinterpret_cast<T*>(xsh_raw);
    __shared__ T rhs[kMaxPivotColumns];

    // Phase 1 — load rhs[] and x cache from y.
    bwd_load_rhs_and_x<T>(y, fr, nc, cb, rhs, xsh, t, nt);
    __syncthreads();
    // Phase 2 — CB contribution to rhs. Threads with t < nc each compute one rhs entry.
    bwd_cb_subtract<T>(F, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
    __syncthreads();
    // Phase 3 — panel substitution (one warp), writes x back to y[fr[0..nc)].
    if (t < 32) bwd_substitute<T>(F, fsz, nc, fr, y, rhs, /*lane=*/t);
}

template <typename T, int NC>
__global__ void solve_fwd_fixed_nc(int lbegin, int lend, const int* __restrict__ plcols,
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
    __shared__ T sh_piv[kMaxPivotColumns];

    if (nc == NC) {
        fwd_substitute_fixed_regular<T, NC>(F, fsz, fr, y, sh_piv, t);
        __syncthreads();
        fwd_cb_update_fixed<T, NC>(F, fsz, /*cb=*/fsz - NC, fr, y, sh_piv, t, nt);
    } else {
        if (t < 32) fwd_substitute<T>(F, fsz, nc, fr, y, sh_piv, /*lane=*/t);
        __syncthreads();
        fwd_cb_update<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t, nt);
    }
}

template <typename T, int NC>
__global__ void solve_fwd_exact_nc(int lbegin, int lend, const int* __restrict__ plcols,
                                   const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                                   const int* __restrict__ front_rows,
                                   const T* frontB, T* yB, long front_total, int n)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const T* front = frontB + (long)blockIdx.y * front_total;
    T* y = yB + (long)blockIdx.y * n;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    __shared__ T sh_piv[kMaxPivotColumns];

    fwd_substitute_fixed_regular<T, NC>(F, fsz, fr, y, sh_piv, t);
    __syncthreads();
    fwd_cb_update_fixed<T, NC>(F, fsz, /*cb=*/fsz - NC, fr, y, sh_piv, t, nt);
}

template <typename T, int NC>
__global__ void solve_bwd_fixed_nc(int lbegin, int lend, const int* __restrict__ plcols,
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

    extern __shared__ unsigned char xsh_raw[];
    T* xsh = reinterpret_cast<T*>(xsh_raw);
    __shared__ T rhs[kMaxPivotColumns];

    if (nc == NC) {
        bwd_load_rhs_and_x_fixed<T, NC>(y, fr, fsz - NC, rhs, xsh, t, nt);
        __syncthreads();
        bwd_cb_subtract_fixed<T, NC>(F, fsz, cb, xsh, rhs, t, /*width=*/nt);
        __syncthreads();
        bwd_substitute_fixed_regular<T, NC>(F, fsz, fr, y, rhs, t);
    } else {
        bwd_load_rhs_and_x<T>(y, fr, nc, cb, rhs, xsh, t, nt);
        __syncthreads();
        bwd_cb_subtract<T>(F, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
        __syncthreads();
        if (t < 32) bwd_substitute<T>(F, fsz, nc, fr, y, rhs, /*lane=*/t);
    }
}

template <typename T, int NC>
__global__ void solve_bwd_exact_nc(int lbegin, int lend, const int* __restrict__ plcols,
                                   const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                                   const int* __restrict__ front_rows,
                                   const T* frontB, T* yB, long front_total, int n)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const T* front = frontB + (long)blockIdx.y * front_total;
    T* y = yB + (long)blockIdx.y * n;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    const int cb = fsz - NC;

    extern __shared__ unsigned char xsh_raw[];
    T* xsh = reinterpret_cast<T*>(xsh_raw);
    __shared__ T rhs[kMaxPivotColumns];

    bwd_load_rhs_and_x_fixed<T, NC>(y, fr, cb, rhs, xsh, t, nt);
    __syncthreads();
    bwd_cb_subtract_fixed<T, NC>(F, fsz, cb, xsh, rhs, t, /*width=*/nt);
    __syncthreads();
    bwd_substitute_fixed_regular<T, NC>(F, fsz, fr, y, rhs, t);
}

// =======================================================================================
//  SPINE tier — one block per batch walks the cnt=1 chain end-to-end
// =======================================================================================

template <typename T>
__global__ void solve_fwd_spine(int n_spine,
                                const int* __restrict__ spine_panels,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ front_rows,
                                const T* frontB, T* yB, long front_total, int n)
{
    const T* front = frontB + (long)blockIdx.y * front_total;
    T* y = yB + (long)blockIdx.y * n;
    const int t = threadIdx.x, nt = blockDim.x;
    __shared__ T sh_piv[kMaxPivotColumns];

    for (int idx = 0; idx < n_spine; ++idx) {
        const int p = spine_panels[idx];
        const int s = front_ptr[p];
        const int fsz = front_ptr[p + 1] - s;
        const int nc = ncols[p];
        const T* F = front + front_off[p];
        const int* fr = front_rows + s;

        if (t < 32) fwd_substitute<T>(F, fsz, nc, fr, y, sh_piv, /*lane=*/t);
        __syncthreads();
        fwd_cb_update_plain<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t, nt);
        __syncthreads();
    }
}

template <typename T>
__global__ void solve_bwd_spine(int n_spine,
                                const int* __restrict__ spine_panels,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ front_rows,
                                const T* frontB, T* yB, long front_total, int n)
{
    const T* front = frontB + (long)blockIdx.y * front_total;
    T* y = yB + (long)blockIdx.y * n;
    const int t = threadIdx.x, nt = blockDim.x;
    __shared__ T rhs[kMaxPivotColumns];
    extern __shared__ unsigned char xsh_raw[];
    T* xsh = reinterpret_cast<T*>(xsh_raw);

    for (int idx = n_spine - 1; idx >= 0; --idx) {
        const int p = spine_panels[idx];
        const int s = front_ptr[p];
        const int fsz = front_ptr[p + 1] - s;
        const int nc = ncols[p];
        const int cb = fsz - nc;
        const T* F = front + front_off[p];
        const int* fr = front_rows + s;

        bwd_load_rhs_and_x<T>(y, fr, nc, cb, rhs, xsh, t, nt);
        __syncthreads();
        bwd_cb_subtract<T>(F, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
        __syncthreads();
        if (t < 32) bwd_substitute<T>(F, fsz, nc, fr, y, rhs, /*lane=*/t);
        __syncthreads();
    }
}

template <typename T>
__global__ void solve_spine(int n_spine,
                            const int* __restrict__ spine_panels,
                            const int* __restrict__ front_off,
                            const int* __restrict__ front_ptr,
                            const int* __restrict__ ncols,
                            const int* __restrict__ front_rows,
                            const T* frontB, T* yB, long front_total, int n)
{
    const T* front = frontB + (long)blockIdx.y * front_total;
    T* y = yB + (long)blockIdx.y * n;
    const int t = threadIdx.x, nt = blockDim.x;
    __shared__ T sh_piv[kMaxPivotColumns];
    __shared__ T rhs[kMaxPivotColumns];
    extern __shared__ unsigned char xsh_raw[];
    T* xsh = reinterpret_cast<T*>(xsh_raw);

    for (int idx = 0; idx < n_spine; ++idx) {
        const int p = spine_panels[idx];
        const int s = front_ptr[p];
        const int fsz = front_ptr[p + 1] - s;
        const int nc = ncols[p];
        const T* F = front + front_off[p];
        const int* fr = front_rows + s;

        if (t < 32) fwd_substitute<T>(F, fsz, nc, fr, y, sh_piv, /*lane=*/t);
        __syncthreads();
        fwd_cb_update_plain<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t, nt);
        __syncthreads();
    }

    for (int idx = n_spine - 1; idx >= 0; --idx) {
        const int p = spine_panels[idx];
        const int s = front_ptr[p];
        const int fsz = front_ptr[p + 1] - s;
        const int nc = ncols[p];
        const int cb = fsz - nc;
        const T* F = front + front_off[p];
        const int* fr = front_rows + s;

        bwd_load_rhs_and_x<T>(y, fr, nc, cb, rhs, xsh, t, nt);
        __syncthreads();
        bwd_cb_subtract<T>(F, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
        __syncthreads();
        if (t < 32) bwd_substitute<T>(F, fsz, nc, fr, y, rhs, /*lane=*/t);
        __syncthreads();
    }
}

}  // namespace
}  // namespace custom_linear_solver
