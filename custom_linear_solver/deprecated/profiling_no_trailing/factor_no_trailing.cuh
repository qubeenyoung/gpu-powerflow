#pragma once

// Σ.16-PROFILE — "no trailing" clones of the 4 dominant FP32 factor kernels for direct
// trailing-GEMM time measurement. Run kernel A (original, with trailing), measure time T_A.
// Run kernel B (no trailing), measure time T_B. trailing_wall_time = T_A − T_B.
//
// These kernels produce a WRONG factor (skipped the rank-nc trailing update). Use only with
// CLS_PROFILE_NO_TRAILING=1 for measurement — never in production paths.
//
// Cloned kernels:
//   1. mf_factor_mid_tiled_NT_b      — clone of tc::mf_factor_mid_tiled_b
//   2. mf_factor_mid_tc32_NT_b       — clone of batched::mf_factor_mid_tc32_b<false>
//   3. mf_factor_extend_level_NT_b   — clone of batched::mf_factor_extend_level_b<float>
//   4. mf_factor_small_warp_NT_b     — clone of batched::mf_factor_small_warp_b<float>
//      (uses lu_small_front_no_trailing which skips the per-step rank-1 update)

#include <cuda_runtime.h>
#include "batched/lu_device.cuh"

namespace custom_linear_solver::tc {
namespace {

// ---- (1) mid_tiled (TC + FP32 path's default for max_fsz >= 48) --------------------------
__global__ void mf_factor_mid_tiled_NT_b(int lbegin, int lend,
                                          const int* __restrict__ plcols,
                                          const int* __restrict__ front_off,
                                          const int* __restrict__ front_ptr,
                                          const int* __restrict__ ncols,
                                          const int* __restrict__ panel_parent,
                                          const int* __restrict__ asm_ptr,
                                          const int* __restrict__ asm_local, float* frontB,
                                          long front_total, int* sing, int do_extend,
                                          int fsz_cap, int level_max_nc, int level_max_uc)
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

    extern __shared__ char smem_mid_NT[];
    float* Fs = reinterpret_cast<float*>(smem_mid_NT);
    // (sh_L, sh_U regions also allocated but not used here since trailing is skipped)

    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    if (fsz <= 48) {
        lu_small_front_no_trailing<float>(Fs, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
        // SKIPPED: trailing_update_staged
    }
    __syncthreads();
    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
    (void)level_max_nc; (void)level_max_uc;
}

}  // namespace
}  // namespace custom_linear_solver::tc


namespace custom_linear_solver::batched {
namespace {

// ---- (2) mid_tc32_b<false> (FP32 fallback for max_fsz < 48 or shared overflow) ------------
// Mirrors mf_factor_mid_tc32_b<false> from factor_tc.cuh but with trailing skipped.
__global__ void mf_factor_mid_tc32_NT_b(int lbegin, int lend, const int* __restrict__ plcols,
                                         const int* __restrict__ front_off,
                                         const int* __restrict__ front_ptr,
                                         const int* __restrict__ ncols,
                                         const int* __restrict__ panel_parent,
                                         const int* __restrict__ asm_ptr,
                                         const int* __restrict__ asm_local, float* frontB,
                                         long front_total, int* sing, int do_extend, int ucp_max,
                                         int fsz_cap)
{
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
    extern __shared__ char smem_tc32_NT[];
    float* Fs = reinterpret_cast<float*>(smem_tc32_NT);
    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();
    if (fsz <= 48) {
        lu_small_front_no_trailing<float>(Fs, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
        // SKIPPED: trailing
    }
    __syncthreads();
    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
    (void)ucp_max; (void)fsz_cap;
}

// ---- (3) extend_level<float> (FP32 BIG path, USA only) -----------------------------------
// Mirrors mf_factor_extend_level_b<float> from factor_kernels.cuh but trailing skipped.
__global__ void mf_factor_extend_level_NT_b(int lbegin, int lend, const int* __restrict__ plcols,
                                             const int* __restrict__ front_off,
                                             const int* __restrict__ front_ptr,
                                             const int* __restrict__ ncols,
                                             const int* __restrict__ panel_parent,
                                             const int* __restrict__ asm_ptr,
                                             const int* __restrict__ asm_local, float* frontB,
                                             long front_total, int* sing, int do_extend)
{
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
        lu_small_front_no_trailing<float>(F, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(F, fsz, nc, t, nt, sing);
        u_panel_solve<float>(F, fsz, nc, uc, t, nt);
        // SKIPPED: trailing_update_scalar
    }
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// ---- (4) small_warp<float> -----------------------------------------------------------------
// Warp-per-front small kernel. Trailing is interleaved inside lu_small_front; we use
// lu_small_front_no_trailing.
__global__ void mf_factor_small_warp_NT_b(int lbegin, int level_size, int B, int fsz2cap,
                                           const int* __restrict__ plcols,
                                           const int* __restrict__ front_off,
                                           const int* __restrict__ front_ptr,
                                           const int* __restrict__ ncols,
                                           const int* __restrict__ panel_parent,
                                           const int* __restrict__ asm_ptr,
                                           const int* __restrict__ asm_local, float* frontB,
                                           long front_total, int* sing, int do_extend)
{
    constexpr int WARPS_PER_BLOCK = 8;
    extern __shared__ unsigned char fwsm_NT[];
    float* fs_all = reinterpret_cast<float*>(fwsm_NT);
    const int warp_in_blk = threadIdx.x >> 5;
    const int wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wg >= level_size * B) return;
    const int fl = wg % level_size;
    const int bb = wg / level_size;
    float* front = frontB + (long)bb * front_total;
    const int p = plcols[lbegin + fl];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const int uc = fsz - nc;
    float* F = front + front_off[p];
    float* Fs = fs_all + (long)warp_in_blk * fsz2cap;
    for (int e = lane; e < fsz * fsz; e += 32) Fs[e] = F[e];
    __syncwarp();
    // Inline of lu_small_front_no_trailing for the warp-level case
    for (int k = 0; k < nc; ++k) {
        float piv = Fs[k * fsz + k];
        if (piv == 0.0f) { if (lane == 0) *sing = 1; piv = 1.0f; }
        for (int i = k + 1 + lane; i < fsz; i += 32) Fs[i * fsz + k] /= piv;
        __syncwarp();
        // SKIPPED: rank-1 trailing update
    }
    __syncwarp();
    for (int e = lane; e < fsz * fsz; e += 32) F[e] = Fs[e];
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncwarp();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    // Warp-level extend_add (atomicAdd to parent)
    for (int e = lane; e < uc * uc; e += 32) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}

}  // namespace
}  // namespace custom_linear_solver::batched
