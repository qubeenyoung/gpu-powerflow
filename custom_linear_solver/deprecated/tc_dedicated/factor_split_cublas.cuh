#pragma once

// Phase Σ.6 — split big-front factor into 3 phases so cuBLAS can do the trailing GEMM.
//   Phase A:  per (front, batch) custom kernel: LU + U-solve in global. NO trailing, NO extend-add.
//             After this returns, F has factored pivot + L panel + U panel, and the trailing block
//             C still holds the original (post-extend-add-from-children) values.
//   cuBLAS:   per panel, cublasSgemmStridedBatched (batchCount=B) does C -= L * U on global.
//   Phase B:  per (front, batch) custom kernel: extend-add the (now updated) C panel into the
//             parent front using atomicAdds.
//
// Drop-in for `mf_factor_extend_tc32_b` when CLS_USE_CUBLAS=1. Same precision (FP32 inputs,
// FP32 accumulate), no WMMA conversion overhead, but per-panel cuBLAS launch cost.

#include <cuda_runtime.h>

#include "batched/lu_device.cuh"

namespace custom_linear_solver::tc {
namespace {

// Phase A — LU + U-solve only. Mirrors the first half of `mf_factor_extend_tc32_b`. Front F
// stays in global; LU is done in-place. Used when CLS_USE_CUBLAS=1 so that the trailing GEMM
// can be replaced by a cuBLAS call between phase A and phase B.
__global__ void mf_factor_big_phaseA_b(int lbegin, int lend, const int* __restrict__ plcols,
                                       const int* __restrict__ front_off,
                                       const int* __restrict__ front_ptr,
                                       const int* __restrict__ ncols, float* frontB,
                                       long front_total, int* sing)
{
    using namespace custom_linear_solver::batched;
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;

    if (fsz <= 48) {
        // Small front: do everything in this kernel (cuBLAS overhead would dominate).
        // Callers should not route fsz<=48 here, but guard anyway.
        lu_small_front<float>(F, fsz, nc, t, nt, sing);
        return;
    }
    lu_panel_factor<float>(F, fsz, nc, t, nt, sing);
    u_panel_solve<float>(F, fsz, nc, uc, t, nt);
    // Trailing update is deferred to cuBLAS by the caller.
}

// Phase A for MID-front (fsz fits in shared). Stages front in shared, runs LU + U-solve,
// writes back L+U+pivot to global. C panel in global is LEFT UNTOUCHED (still holds the
// post-extend-add input values from children); the cuBLAS grouped trailing then computes
// C -= L*U on global, reading L/U/C from global memory.
__global__ void mf_factor_mid_phaseA_b(int lbegin, int lend,
                                       const int* __restrict__ plcols,
                                       const int* __restrict__ front_off,
                                       const int* __restrict__ front_ptr,
                                       const int* __restrict__ ncols, float* frontB,
                                       long front_total, int* sing, int fsz_cap)
{
    using namespace custom_linear_solver::batched;
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;

    extern __shared__ char smem_midA[];
    float* Fs = reinterpret_cast<float*>(smem_midA);

    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    if (fsz <= 48) {
        // lu_small_front fuses LU + U-solve + trailing on Fs. The C panel is now UPDATED in
        // Fs but cuBLAS will skip this panel (m=0 set at setup) -- so we must write the
        // ENTIRE front (including the updated C panel) back to F, otherwise phase B reads
        // the stale pre-factor C from global.
        lu_small_front<float>(Fs, fsz, nc, t, nt, sing);
        __syncthreads();
        for (long e = t; e < (long)fsz * fsz; e += nt) F[e] = Fs[e];
    } else {
        // For big-enough fronts cuBLAS handles the trailing; only L/U/pivot need writeback,
        // and F's C panel is left intact (= the post-extend-add input that cuBLAS will
        // subtract L*U from).
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
        __syncthreads();
        writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);
    }
    (void)fsz_cap;
}

// Build the cuBLAS grouped-batched pointer arrays on-device from per-panel within-front
// offsets (U_off, L_off, C_off), the per-panel arena base offset (h_front_off), the front
// base (d_frontBf), and front_total (the per-system arena stride). This replaces a 3*P*B*8
// byte H2D copy with a 3*P*4 + P*4 byte upload (offsets) + a single kernel — for USA P=74k
// B=64 that's 891 KB upload + ~1 ms kernel vs 114 MB upload + ~55 ms PCIe.
__global__ void mf_build_cublas_ptrs_b(int P, int B, long front_total,
                                       const int* __restrict__ front_off,
                                       const int* __restrict__ within_U,
                                       const int* __restrict__ within_L,
                                       const int* __restrict__ within_C,
                                       float* base,
                                       float** Aptrs, float** Bptrs, float** Cptrs)
{
    const int i = blockIdx.x;  // panel-position index (0..P-1)
    if (i >= P) return;
    const int b = blockIdx.y * blockDim.x + threadIdx.x;  // batch index
    if (b >= B) return;
    float* F = base + (long)b * front_total + front_off[i];
    const long slot = (long)i * B + b;
    Aptrs[slot] = F + within_U[i];
    Bptrs[slot] = F + within_L[i];
    Cptrs[slot] = F + within_C[i];
}

// Init pivot storage to PANEL-LOCAL identity. Per panel p of size ncols[p] starting at
// pivot_offset[p], writes pivots[pivot_offset[p] + k] = k for k = 0..ncols[p]-1, replicated
// across B systems. Runs at the start of factor so solve sees identity for any panel whose
// factor kernel did NOT write pivots (small_warp / mid_tiled / extend_tc32 paths).
__global__ void mf_init_pivots_identity_b(int* pivotsB, const int* pivot_offset,
                                          const int* ncols, int num_panels, int total_pivots,
                                          int B)
{
    const int p = blockIdx.x;
    if (p >= num_panels) return;
    const int bb = blockIdx.y;
    if (bb >= B) return;
    const int nc = ncols[p];
    const int off = pivot_offset[p];
    int* dst = pivotsB + (long)bb * total_pivots + off;
    for (int k = threadIdx.x; k < nc; k += blockDim.x) dst[k] = k;
}

// Phase A MID with within-panel partial pivoting (Phase Σ.8). Writes per-panel pivots[nc]
// to d_pivotsB at offset (system * total_pivots + pivot_offset[p]). Layout otherwise
// identical to mf_factor_mid_phaseA_b.
__global__ void mf_factor_mid_phaseA_pp_b(int lbegin, int lend,
                                          const int* __restrict__ plcols,
                                          const int* __restrict__ front_off,
                                          const int* __restrict__ front_ptr,
                                          const int* __restrict__ ncols, float* frontB,
                                          long front_total, int* sing, int fsz_cap,
                                          int* pivotsB, const int* pivot_offset, int total_pivots)
{
    using namespace custom_linear_solver::batched;
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int b_sys = blockIdx.y;
    float* front = frontB + (long)b_sys * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    int* pivots_p = pivotsB + (long)b_sys * total_pivots + pivot_offset[p];

    extern __shared__ char smem_midA_pp[];
    float* Fs = reinterpret_cast<float*>(smem_midA_pp);

    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    if (fsz <= 48) {
        lu_small_front_pp<float>(Fs, fsz, nc, t, nt, sing, pivots_p);
        __syncthreads();
        for (long e = t; e < (long)fsz * fsz; e += nt) F[e] = Fs[e];
    } else {
        lu_panel_factor_pp<float>(Fs, fsz, nc, t, nt, sing, pivots_p);
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
        __syncthreads();
        writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);
    }
    (void)fsz_cap;
}

// Phase A BIG with within-panel partial pivoting. Operates on F (global) directly.
__global__ void mf_factor_big_phaseA_pp_b(int lbegin, int lend,
                                          const int* __restrict__ plcols,
                                          const int* __restrict__ front_off,
                                          const int* __restrict__ front_ptr,
                                          const int* __restrict__ ncols, float* frontB,
                                          long front_total, int* sing,
                                          int* pivotsB, const int* pivot_offset, int total_pivots)
{
    using namespace custom_linear_solver::batched;
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int b_sys = blockIdx.y;
    float* front = frontB + (long)b_sys * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    int* pivots_p = pivotsB + (long)b_sys * total_pivots + pivot_offset[p];

    if (fsz <= 48) {
        lu_small_front_pp<float>(F, fsz, nc, t, nt, sing, pivots_p);
        return;
    }
    lu_panel_factor_pp<float>(F, fsz, nc, t, nt, sing, pivots_p);
    u_panel_solve<float>(F, fsz, nc, uc, t, nt);
}

// Phase B — extend-add only. Reads the (now factored + trailing-updated) F and atomically
// adds the C panel into the parent. Mirrors the tail of `mf_factor_extend_tc32_b`.
__global__ void mf_factor_big_phaseB_b(int lbegin, int lend, const int* __restrict__ plcols,
                                       const int* __restrict__ front_off,
                                       const int* __restrict__ front_ptr,
                                       const int* __restrict__ ncols,
                                       const int* __restrict__ panel_parent,
                                       const int* __restrict__ asm_ptr,
                                       const int* __restrict__ asm_local, float* frontB,
                                       long front_total, int do_extend)
{
    using namespace custom_linear_solver::batched;
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const int uc = fsz - nc;
    const float* F = front + front_off[p];
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    const int t = threadIdx.x, nt = blockDim.x;
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

}  // namespace
}  // namespace custom_linear_solver::tc
