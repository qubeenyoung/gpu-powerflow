#pragma once

// ABLATION — small tier the MAGMA/STRUMPACK way (operation-separated, global-resident, no fusion).
//
// factor_small (small.cuh) is our optimized small-tier kernel: it stages the WHOLE front into shared
// once and runs all phases fused in a SINGLE kernel (panel LU + U-solve + trailing + extend-add).
// This ablation reproduces instead the prior-art structure — exactly how STRUMPACK drives a level
// via MAGMA: each phase is a SEPARATE kernel launch over the level's fronts, operating directly on
// the front in GLOBAL memory, so the front data round-trips global between phases and there is no
// cross-phase fusion. extend-add is its own pass (as STRUMPACK keeps it). One block per front, same
// grid as factor_small — only the *mapping* differs, so the result is bit-identical and the timing
// gap isolates the contribution of factor_small's fused whole-front-shared design.
//
// Enable with env  CLS_SMALL_ABLATION=1  (read once in dispatch_factor_small). Default off.

#include <cstdlib>
#include "factorize/front_ops.cuh"   // lu_panel_factor / u_panel_solve_fewsync / trailing_update_scalar / extend_add

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// --- Phase kernels: one block per (front, batch), all reading/writing F in GLOBAL (no shared) ---

template <typename T>
__global__ void abl_small_lu(int lbegin, int lend, const int* __restrict__ plc,
                             const int* __restrict__ foff, const int* __restrict__ fptr,
                             const int* __restrict__ ncols, T* frontB, long ftot, int* sing,
                             bool sp, double pt, double ps)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* F = frontB + (long)blockIdx.y * ftot;
    const int p = plc[idx];
    const int fsz = fptr[p + 1] - fptr[p];
    const int nc = ncols[p];
    lu_panel_factor<T>(F + foff[p], fsz, nc, threadIdx.x, blockDim.x, sing, sp, pt, ps);  // Phase 1
}

template <typename T>
__global__ void abl_small_panel(int lbegin, int lend, const int* __restrict__ plc,
                                const int* __restrict__ foff, const int* __restrict__ fptr,
                                const int* __restrict__ ncols, T* frontB, long ftot)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* F = frontB + (long)blockIdx.y * ftot;
    const int p = plc[idx];
    const int fsz = fptr[p + 1] - fptr[p];
    const int nc = ncols[p];
    u_panel_solve_fewsync<T>(F + foff[p], fsz, nc, fsz - nc, threadIdx.x, blockDim.x);     // Phase 2
}

template <typename T>
__global__ void abl_small_trail(int lbegin, int lend, const int* __restrict__ plc,
                                const int* __restrict__ foff, const int* __restrict__ fptr,
                                const int* __restrict__ ncols, T* frontB, long ftot)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* F = frontB + (long)blockIdx.y * ftot;
    const int p = plc[idx];
    const int fsz = fptr[p + 1] - fptr[p];
    const int nc = ncols[p];
    trailing_update_scalar<T>(F + foff[p], fsz, nc, fsz - nc, threadIdx.x, blockDim.x);    // Phase 3
}

template <typename T>
__global__ void abl_small_extend(int lbegin, int lend, const int* __restrict__ plc,
                                 const int* __restrict__ foff, const int* __restrict__ fptr,
                                 const int* __restrict__ ncols, const int* __restrict__ panel_parent,
                                 const int* __restrict__ asm_ptr, const int* __restrict__ asm_local,
                                 T* frontB, long ftot, int do_extend)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plc[idx];
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    T* F = frontB + (long)blockIdx.y * ftot;
    const int fsz = fptr[p + 1] - fptr[p];
    const int nc = ncols[p];
    T* Fp = F + foff[par];
    const int pfsz = fptr[par + 1] - fptr[par];
    extend_add<T, T>(Fp, pfsz, F + foff[p], fsz, nc, fsz - nc, asm_local, asm_ptr[p],
                     threadIdx.x, blockDim.x);                                             // extend-add (separate)
}

// Returns true if the small-tier ablation is enabled (CLS_SMALL_ABLATION=1), read once.
inline bool small_ablation_enabled()
{
    static const bool on = [] {
        const char* s = std::getenv("CLS_SMALL_ABLATION");
        return s && std::atoi(s) != 0;
    }();
    return on;
}

// Same prior-art ablation applied to the TINY tier (CLS_TINY_ABLATION=1). The abl_* kernels are
// front-size agnostic, so routing tiny fronts through them reproduces the MAGMA/STRUMPACK mapping
// (one block per front, op-separated, global) — i.e. it removes BOTH the sub-group packing and the
// fused shared-resident pipeline that make factor_tiny fast. The gap is the tiny kernel's effect.
inline bool tiny_ablation_enabled()
{
    static const bool on = [] {
        const char* s = std::getenv("CLS_TINY_ABLATION");
        return s && std::atoi(s) != 0;
    }();
    return on;
}

// STRUMPACK-style operation-separated dispatch of the small tier (4 graph-ordered launches).
template <typename T>
static void dispatch_small_ablation(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                    int b, int e, const int* d_plc, T* frontB, int do_extend)
{
    const dim3 grid(e - b, st.batch_count);
    constexpr int thr = 256;
    abl_small_lu<T><<<grid, thr, 0, stream>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, frontB, plan.front_total, st.d_sing,
        st.static_pivoting, st.pivot_threshold, st.pivot_shift);
    abl_small_panel<T><<<grid, thr, 0, stream>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, frontB, plan.front_total);
    abl_small_trail<T><<<grid, thr, 0, stream>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, frontB, plan.front_total);
    abl_small_extend<T><<<grid, thr, 0, stream>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local,
        frontB, plan.front_total, do_extend);
}

}  // namespace
}  // namespace custom_linear_solver
