#pragma once

// Internal — included only by multifrontal.cu (single TU).
//
// MID-tier (32 < fsz <= WARP_MID_THRESH) warp-per-front factor kernel. This extends the
// proven `factor_small` pattern (small.cuh) from fsz<=32 to fsz<=64 (configurable).
//
// Why:
//   factor_mid<float> at B=64 stalls 41% on CTA barrier (`__syncthreads`) per ncu — per-front
//   sync count is ~3·nc+5 (`docs/03-optimization-notes/09`). The block-per-front kernel pays a
//   full 8-warp `__syncthreads` (~6-10 cyc avg) on every panel-LU sub-step. For fsz that still
//   permits 32-lane parallelism, replacing those with `__syncwarp` (~1 cyc) eliminates the
//   bulk of barrier stall.
//
// How:
//   One WARP per (front, batch), `warps_per_block` warps per block (dispatcher computes
//   warps_per_block from the level's max_fsz and the 96 KB shared budget). Each warp stages
//   its front into per-warp shared, runs `lu_small_warp` (warp-parallel fused no-pivot LU,
//   `__syncwarp` between panel-LU passes), writes back factored L/U, and extend-adds from
//   shared. The CB stays in shared until extend-add, so no global memory round-trip.
//
// Why fsz<=64 (default):
//   per-warp shared = fsz²·sizeof(T). At fsz=64 fp32, per-warp = 16 KB; budget 96 KB → 6 warps
//   per block (vs 8 at fsz<=48 where per-warp = 9 KB). The fused panel-LU+trailing inside
//   `lu_small_warp` does ≈ nc·(fsz-nc)² FMAs/pass; for power-grid (nc~8), this is comparable
//   to the split form's `lu_panel_factor + u_panel_solve + trailing_update_scalar` total work
//   while saving 2/3 of the syncs. For fsz>64, fused work grows quadratically and split form
//   wins — that's where T4.2 (sub-block) applies, not here.

#include <cuda_runtime.h>
#include <cstdio>

#include "factorize/primitives.cuh"  // writeback_factored
#include "factorize/small.cuh"        // lu_small_warp (used for fsz<=32 only)

namespace custom_linear_solver {
namespace {

// Warp-parallel fused no-pivot LU for a MID-sized front (32 < fsz <= ~96), `lane` in [0,32).
//
// Difference from lu_small_warp (small.cuh): when fsz is not a multiple of 32, column k's
// entries span MULTIPLE lanes' shared writes during stage-in and the divide step. The rank-1
// update then reads `F[ii*fsz+k]` from another lane's recently-written entry. `__syncwarp()`
// alone provides execution-order convergence but compiler-level reordering of shared loads /
// stores within a warp can still hide the dependency. We explicitly mark shared memory loads
// as volatile around the cross-lane read to force the compiler to re-issue them after the
// preceding stores. This restores correctness for arbitrary fsz at the cost of one extra
// shared load per FMA in the rank-1 update.
//
// `Fs` here is __shared__ memory; the volatile-cast pattern is a standard CUDA idiom for
// inter-lane synchronization without a heavyweight memory barrier. Performance impact at
// fsz=48..80 is small because the inner load is L1-resident shared memory.
template <typename FT>
__device__ __forceinline__ void lu_mid_warp(FT* F, int fsz, int nc, int lane, int* sing)
{
    for (int k = 0; k < nc; ++k) {
        FT piv = F[(long)k * fsz + k];
        if (piv == FT(0)) {
            if (lane == 0) *sing = 1;
            piv = FT(1);
        }
        for (int i = k + 1 + lane; i < fsz; i += 32) F[(long)i * fsz + k] /= piv;
        __syncwarp();
        const int m = fsz - k - 1;
        for (int e = lane; e < m * m; e += 32) {
            const int ii = k + 1 + e / m, jj = k + 1 + e % m;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        __syncwarp();
    }
}

// One warp per (front, batch); blockDim = warps_per_block * 32. Identical to factor_small in
// flow — only the per-warp shared footprint is larger because fsz_cap can be up to 64.
template <typename FT>
__global__ void factor_mid_warp(int lbegin, int level_size, int B, int fsz2cap,
                                const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ panel_parent,
                                const int* __restrict__ asm_ptr,
                                const int* __restrict__ asm_local, FT* frontB,
                                long front_total, int* sing, int do_extend)
{
    extern __shared__ unsigned char smem_mw_raw[];
    FT* smem_mw = reinterpret_cast<FT*>(smem_mw_raw);
    const int warp_in_blk = threadIdx.x >> 5;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_global >= level_size * B) return;
    const int fl = warp_global % level_size;
    const int bb = warp_global / level_size;
    FT* front = frontB + (long)bb * front_total;
    const int p = plcols[lbegin + fl];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    FT* Fs = smem_mw + (long)warp_in_blk * fsz2cap;

    // T4.3: Asynchronous stage-in via cp.async (Ampere+); falls back to synchronous on sm<80.
    // Note: __pipeline operations are scoped to the thread, not the warp. After all 32 lanes
    // issue their async copies, __pipeline_wait_prior(0) waits on this lane's copies and
    // __syncwarp() ensures all lanes have completed their waits.
    stage_in_async<FT>(Fs, F, fsz2, lane, 32);
    __syncwarp();
#ifdef CLS_MID_WARP_DEBUG
    if (lane == 0 && bb == 0)
        printf("mid_warp dispatch p=%d fsz=%d nc=%d fsz2cap=%d warp_in_blk=%d block=%d\n",
               p, fsz, nc, fsz2cap, warp_in_blk, blockIdx.x);
#endif
    lu_mid_warp<FT>(Fs, fsz, nc, lane, sing);
    __syncwarp();
    // Write back factored L/U (pivot row + L block); CB stays in shared for extend-add.
    writeback_factored<FT, FT>(F, Fs, fsz, nc, uc, lane, 32);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    FT* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = lane; e < uc * uc; e += 32) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}

}  // namespace
}  // namespace custom_linear_solver
