#pragma once

// Internal — included only by multifrontal.cu (single translation unit; build uses
// CUDA_SEPARABLE_COMPILATION OFF, so launched kernels must be defined in the launching TU).
//
// Dense per-front building blocks shared by the factor kernels (factor_big, factor_mid,
// factor_mid_fp32, factor_small). Each helper is __forceinline__ so callers get the same code
// as if the bodies were inlined directly.
//
// Convention: F is row-major with leading dimension fsz. Each helper operates per (front, batch)
// block: t = threadIdx.x, nt = blockDim.x. nc = pivot columns, uc = fsz - nc = contribution block.
//
// A front layout:
//
//        cols 0..nc-1 (pivot)   cols nc..fsz-1 (trailing)
//      +--------------------+---------------------+
//      |        L_pp        |         U           |   rows 0..nc-1   (panel)
//      +--------------------+---------------------+
//      |        L           |         C           |   rows nc..fsz-1 (contribution)
//      +--------------------+---------------------+
//
// Phases of the per-front factor:
//   1. lu_panel_factor       — rank-nc LU on the nc-wide panel
//   2. u_panel_solve         — triangular solve for the U strip
//   3. trailing_update_*     — rank-nc trailing update on C (scalar fallback or WMMA)
//   4. extend_add            — atomicAdd C into the parent front (rows mapped by asm_local)

#include <cuda_runtime.h>

namespace custom_linear_solver {
namespace {

// Small-front fused no-pivot LU (fsz <= 48): nc rank-1 passes over the FULL trailing block.
template <typename T>
__device__ __forceinline__ void lu_small_front(T* F, int fsz, int nc, int t, int nt, int* sing)
{
    for (int k = 0; k < nc; ++k) {
        T piv = F[(long)k * fsz + k];
        if (piv == T(0)) { if (t == 0) *sing = 1; piv = T(1); }
        for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
        __syncthreads();
        const int m = fsz - k - 1;
        for (int e = t; e < m * m; e += nt) {
            const int ii = k + 1 + e / m, jj = k + 1 + e % m;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        __syncthreads();
    }
}

// Phase 1 — rank-nc LU on the nc-wide panel (full front height). Each k divides column k below
// the diagonal then does the rank-1 update restricted to the panel columns (jj < nc). The
// trailing (jj >= nc) columns are handled by u_panel_solve + the trailing update.
template <typename T>
__device__ __forceinline__ void lu_panel_factor(T* F, int fsz, int nc, int t, int nt, int* sing)
{
    for (int k = 0; k < nc; ++k) {
        T piv = F[(long)k * fsz + k];
        if (piv == T(0)) { if (t == 0) *sing = 1; piv = T(1); }
        for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
        __syncthreads();
        const int pc = nc - 1 - k;
        for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
            const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        if (pc > 0) __syncthreads();
    }
}

// Phase 2 — U-panel triangular solve for the trailing columns (rows 0..nc-1, cols nc..fsz-1).
template <typename T>
__device__ __forceinline__ void u_panel_solve(T* F, int fsz, int nc, int uc, int t, int nt)
{
    for (int k = 1; k < nc; ++k) {
        for (int e = t; e < uc; e += nt) {
            const int jj = nc + e;
            T v = F[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
            F[(long)k * fsz + jj] = v;
        }
        __syncthreads();
    }
}

// Phase 3 — scalar rank-nc trailing update C(uc x uc) -= L(uc x nc) * U(nc x uc). The TC kernel
// substitutes an FP16 WMMA GEMM here; this is its fallback for fronts too big for the tile.
template <typename T>
__device__ __forceinline__ void trailing_update_scalar(T* F, int fsz, int nc, int uc, int t,
                                                       int nt)
{
    for (int e = t; e < uc * uc; e += nt) {
        const int ii = nc + e / uc, jj = nc + e % uc;
        T acc = T(0);
        for (int k = 0; k < nc; ++k) acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        F[(long)ii * fsz + jj] -= acc;
    }
}

// Phase 4 — scatter this front's contribution block C into the parent front (atomicAdd). The
// destination rows/cols within the parent are looked up in asm_local[abase + 0..uc).
template <typename DstT, typename SrcT>
__device__ __forceinline__ void extend_add(DstT* Fp, int pfsz, const SrcT* Fsrc, int fsz, int nc,
                                           int uc, const int* asm_local, int abase, int t, int nt)
{
    for (int e = t; e < uc * uc; e += nt) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  static_cast<DstT>(Fsrc[(long)(nc + a) * fsz + (nc + b)]));
    }
}

// Shared → global writeback of the factored L/U the solve reads back: the nc pivot-column rows
// (pivot + U panel) and the L block of the trailing rows. The uc x uc contribution block stays
// in shared for the extend-add and is never read again from global.
template <typename MT, typename WT>
__device__ __forceinline__ void writeback_factored(MT* M, const WT* W, int fsz, int nc, int uc,
                                                   int t, int nt)
{
    for (long e = t; e < (long)nc * fsz; e += nt) M[e] = static_cast<MT>(W[e]);
    for (int e = t; e < uc * nc; e += nt) {
        const long id2 = (long)(nc + e / nc) * fsz + (e % nc);
        M[id2] = static_cast<MT>(W[id2]);
    }
}

// ---------------------------------------------------------------------------------------
// factorize_front — phases 1–3 of the per-front factor in their canonical order.
//
// This is the "main factor body" called by every tier-specific kernel (factor_small,
// factor_mid<T>, factor_mid_tc, factor_big<T>, factor_big_tc). Each tier wraps it with its
// own stage-in / writeback / extend-add specific to where the front lives (global vs.
// shared) and which trailing variant to use.
//
// Caller supplies the trailing update as a functor `trailing()`, called only when
// fsz > 48. Small fronts use lu_small_front, which fuses panel LU and trailing into one
// rank-1 sweep over the full block.
template <typename T, typename TrailingFn>
__device__ __forceinline__ void factorize_front(T* F, int fsz, int nc, int uc, int t, int nt,
                                                int* sing, TrailingFn&& trailing)
{
    if (fsz <= 48) {
        lu_small_front<T>(F, fsz, nc, t, nt, sing);   // phase 1+3 fused
    } else {
        lu_panel_factor<T>(F, fsz, nc, t, nt, sing);  // phase 1: rank-nc panel LU
        u_panel_solve<T>(F, fsz, nc, uc, t, nt);      // phase 2: U strip triangular solve
        trailing();                                    // phase 3: caller-chosen trailing
    }
}

}  // namespace
}  // namespace custom_linear_solver
