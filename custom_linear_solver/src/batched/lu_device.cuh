#pragma once

// Internal — included only by batched/multifrontal_batched.cu (single TU; the build uses
// CUDA_SEPARABLE_COMPILATION OFF, so every batched kernel must be defined in the TU that
// launches it). Shared dense-LU building blocks for the three batched factor kernels
// (mf_factor_extend_level_b / mf_factor_extend_mixed_b / mf_factor_extend_mixed_tc_b). Each
// helper is __forceinline__ so the generated code is identical to the previously inlined bodies.
// All helpers operate per (front, batch) block: t = threadIdx.x, nt = blockDim.x, the front F is
// row-major with leading dimension fsz, nc = pivot columns, uc = fsz - nc = contribution block.

#include <cuda_runtime.h>

namespace custom_linear_solver::batched {
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

// Phase 1: factor the nc-wide panel (full front height). Each k divides column k below the
// diagonal then does the rank-1 update RESTRICTED to the panel columns (jj < nc). The trailing
// (jj >= nc) columns are handled by u_panel_solve + the trailing update.
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

// Phase 2: U-panel triangular solve for the trailing columns (rows 0..nc-1, cols nc..fsz-1).
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

// Phase 3: scalar rank-nc trailing update C(uc x uc) -= L(uc x nc) * U(nc x uc). (The TC kernel
// substitutes an FP16 WMMA GEMM here; this is its fallback for fronts too big for the tile.)
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

// Extend-add: scatter this front's contribution block into the parent front (atomicAdd). SrcT is
// the working-front element type (FT, or float for Mixed/TC), DstT the parent front type (FT, or
// double for the Mixed/TC master); the value is cast SrcT -> DstT on the way in.
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

// Mixed/TC master -> working copy of the whole front (DST = float working, SRC = double master).
template <typename DST, typename SRC>
__device__ __forceinline__ void cast_copy(DST* dst, const SRC* src, long n, int t, int nt)
{
    for (long e = t; e < n; e += nt) dst[e] = static_cast<DST>(src[e]);
}

// Mixed/TC working -> master writeback of the factored entries: the nc pivot-column rows (L + the
// U panel) plus the L block of the trailing rows (the CB stays in working for the extend-add).
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

}  // namespace
}  // namespace custom_linear_solver::batched
