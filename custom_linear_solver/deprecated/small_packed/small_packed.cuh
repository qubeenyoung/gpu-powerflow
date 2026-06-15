#pragma once

// Internal — included only by multifrontal.cu (single TU).
//
// MULTI-FRONT-PER-WARP packed factor kernel for the SMALL tier (T-pack experiment).
//
// Motivation:
//   factor_small (one warp per (front, batch)) wastes 26+ of 32 lanes when fsz≤6 — median
//   fsz=6, fsz²=36 entries × nc=2 outer loop. analyze.cu sorts plcols within each level by
//   (nc, fsz, parent, p) so fronts with identical (nc, fsz) become contiguous; the dispatcher
//   then groups N adjacent same-(nc, fsz) fronts into a single warp.
//
// Lane mapping (per warp, 32 lanes, FRONTS_PER_WARP = N fronts):
//   lanes_per_front L = 32 / N
//   sub_warp_idx     = lane / L   ∈ [0, N)         — which front in the group
//   sub_lane         = lane % L   ∈ [0, L)         — which entry of that front
//
// Pack policy (host-chosen, matches kernel template instantiation):
//   fsz ≤ 3 :   N = 4 (L = 8)   — fits even very small fronts trivially
//   fsz = 4 :   N = 2 (L = 16)  — fsz²=16, stage-in 1 iter/lane
//   fsz = 5 :   N = 2 (L = 16)  — fsz²=25, stage-in 2 iter/lane
//   fsz ∈ [6, 7]: N = 2 (L = 16) — fsz²=36..49, stage-in 3 iter/lane
//   fsz ≥ 8 :   N = 1 (L = 32)  — original layout (still wins by removing outer loop overhead)
//
// nc=2 unrolled fast path:
//   ~75-88% of small fronts (case8387, USA) have nc=2. The outer loop is unrolled to two
//   k-steps with explicit instructions; that saves the loop epilogue/predicate overhead.
//   For nc≠2, the kernel takes the generic loop (rare in small tier).

#include <cuda_runtime.h>

#include "factorize/primitives.cuh"  // writeback_factored

namespace custom_linear_solver {
namespace {

// Pack-rank chosen at dispatch time. Codegen one specialization per (FT, FRONTS_PER_WARP).
template <typename FT, int FRONTS_PER_WARP>
__global__ void factor_small_packed(int group_off, int group_cnt, int B, int fsz, int nc,
                                     int fsz2cap,
                                     const int* __restrict__ plcols,
                                     const int* __restrict__ front_off,
                                     const int* __restrict__ front_ptr,
                                     const int* __restrict__ panel_parent,
                                     const int* __restrict__ asm_ptr,
                                     const int* __restrict__ asm_local,
                                     FT* frontB, long front_total, int* sing, int do_extend)
{
    static_assert(FRONTS_PER_WARP >= 1 && FRONTS_PER_WARP <= 8 &&
                  ((FRONTS_PER_WARP & (FRONTS_PER_WARP - 1)) == 0),
                  "FRONTS_PER_WARP must be a power of 2 in [1, 8]");
    constexpr int LANES_PER_FRONT = 32 / FRONTS_PER_WARP;

    extern __shared__ unsigned char smem_sp_raw[];
    FT* smem_sp = reinterpret_cast<FT*>(smem_sp_raw);

    const int warp_in_blk = threadIdx.x >> 5;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int sub_warp = lane / LANES_PER_FRONT;
    const int sub_lane = lane % LANES_PER_FRONT;

    // Per-warp work index: FRONTS_PER_WARP (front, batch) pairs per warp.
    const long total_fronts = (long)group_cnt * B;
    const long packed_idx = (long)warp_global * FRONTS_PER_WARP + sub_warp;
    if (packed_idx >= total_fronts) return;

    const int fl = (int)(packed_idx % group_cnt);
    const int bb = (int)(packed_idx / group_cnt);
    FT* front = frontB + (long)bb * front_total;
    const int p = plcols[group_off + fl];
    FT* F = front + front_off[p];

    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;

    // Per-sub-warp shared region. Each block hosts (warps_per_block * FRONTS_PER_WARP) sub-warps.
    const int region_idx = warp_in_blk * FRONTS_PER_WARP + sub_warp;
    FT* Fs = smem_sp + (long)region_idx * fsz2cap;

    // Stage-in: each sub-lane handles entries (sub_lane, sub_lane + LANES_PER_FRONT, ...).
    #pragma unroll 4
    for (int e = sub_lane; e < fsz2; e += LANES_PER_FRONT) Fs[e] = F[e];
    __syncwarp();

    // ----- Fused right-looking LU (mirrors lu_small_warp), with sub-warp lanes -----
    // For nc=2 we unroll the two outer iterations; sub-lane condition guards inactive lanes.
    if (nc == 2) {
        // k = 0
        {
            const FT piv = Fs[0];
            const FT piv_safe = (piv == FT(0)) ? FT(1) : piv;
            if (piv == FT(0) && sub_lane == 0) *sing = 1;
            const int m = fsz - 1;
            // Divide column 0: F[i*fsz+0] /= piv, for i in [1, fsz)
            #pragma unroll 2
            for (int i = 1 + sub_lane; i < fsz; i += LANES_PER_FRONT) {
                Fs[(long)i * fsz] /= piv_safe;
            }
            __syncwarp();
            // Rank-1 update on full m×m sub-block: F[ii][jj] -= F[ii][0] * F[0][jj]
            const int mm = m * m;
            #pragma unroll 2
            for (int e = sub_lane; e < mm; e += LANES_PER_FRONT) {
                const int ii = 1 + e / m;
                const int jj = 1 + e % m;
                Fs[(long)ii * fsz + jj] -= Fs[(long)ii * fsz] * Fs[(long)jj];
            }
            __syncwarp();
        }
        // k = 1
        {
            const FT piv = Fs[(long)fsz + 1];
            const FT piv_safe = (piv == FT(0)) ? FT(1) : piv;
            if (piv == FT(0) && sub_lane == 0) *sing = 1;
            const int m = fsz - 2;
            #pragma unroll 2
            for (int i = 2 + sub_lane; i < fsz; i += LANES_PER_FRONT) {
                Fs[(long)i * fsz + 1] /= piv_safe;
            }
            __syncwarp();
            const int mm = m * m;
            #pragma unroll 2
            for (int e = sub_lane; e < mm; e += LANES_PER_FRONT) {
                const int ii = 2 + e / m;
                const int jj = 2 + e % m;
                Fs[(long)ii * fsz + jj] -= Fs[(long)ii * fsz + 1] * Fs[(long)fsz + jj];
            }
            __syncwarp();
        }
    } else {
        // Generic nc loop (rare in small tier, used for safety).
        for (int k = 0; k < nc; ++k) {
            const FT piv = Fs[(long)k * fsz + k];
            const FT piv_safe = (piv == FT(0)) ? FT(1) : piv;
            if (piv == FT(0) && sub_lane == 0) *sing = 1;
            for (int i = k + 1 + sub_lane; i < fsz; i += LANES_PER_FRONT) {
                Fs[(long)i * fsz + k] /= piv_safe;
            }
            __syncwarp();
            const int m = fsz - k - 1;
            const int mm = m * m;
            for (int e = sub_lane; e < mm; e += LANES_PER_FRONT) {
                const int ii = k + 1 + e / m;
                const int jj = k + 1 + e % m;
                Fs[(long)ii * fsz + jj] -=
                    Fs[(long)ii * fsz + k] * Fs[(long)k * fsz + jj];
            }
            __syncwarp();
        }
    }

    // Writeback factored L/U (rows 0..nc-1 + L block); CB stays in Fs for extend-add.
    // First nc rows: nc * fsz entries.
    for (int e = sub_lane; e < nc * fsz; e += LANES_PER_FRONT) {
        F[e] = Fs[e];
    }
    // L block (rows nc..fsz-1, cols 0..nc-1): uc * nc entries.
    for (int e = sub_lane; e < uc * nc; e += LANES_PER_FRONT) {
        const long id2 = (long)(nc + e / nc) * fsz + (e % nc);
        F[id2] = Fs[id2];
    }

    // Extend-add CB → parent (atomicAdd). Each sub-warp scatters its front's CB.
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    FT* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = sub_lane; e < uc * uc; e += LANES_PER_FRONT) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}

}  // namespace
}  // namespace custom_linear_solver
