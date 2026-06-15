#pragma once
// ============================================================================================
// DEPRECATED (2026-06-11) — big-tier TF32 trailing split across thread blocks.
//
// Reference snapshot only — NOT compiled. Lived behind `#ifdef EXP_260611_BIG_SPLIT` in
// src/factorize/big.cuh. Removed from production because the win is narrow and fragile (see
// README.md): usa B=1 factorize 1.14× / B=4 1.10×, but 70K regresses (~0.95×) and every B≥16
// loses. The 2-kernel structure (panel -> tiled trailing) pays a global L/U round-trip + a 2nd
// launch per level; only the largest under-utilized single-system case wins. A clean fix needs a
// cooperative single kernel (grid.sync), which conflicts with the whole-iteration CUDA graph.
//
// Two trailing variants were tried; v2 (2D tile) is the one that produced the 1.14×:
//   v1 factor_big_trail_split_tf32 — each block stages the FULL L/U and does a block-strided
//      subset of M-tiles (redundant staging; needed the trailing_update_tf32_tc block_m/m_split
//      params, since reverted on the production function). Net slower.
//   v2 factor_big_trail_tile_tf32 — each block owns one BM×BN output tile and stages only its
//      L-rows + U-cols (~8 KB shared -> ~9-10 blocks/SM). This is the version to revive.
//
// ncu (v2 tile kernel, usa): occupancy limit 1 -> 9-10 block/SM, barrier stall 5.34 -> 0.04,
// shared 33 KB -> 8.5 KB. The occupancy lever works; the launch/round-trip overhead is what loses.
//
// Correctness note (bug found + fixed before deprecation): the non-TC path (uc>256 / nc>32) must
// run scalar trailing AND the extend-add drain to the parent — dropping the drain corrupted any
// case with a uc>256 front (70K, relres 0.30). The `if (!tc) { ... extend_add ... }` block below
// is the fixed form.
//
// Pairs with src/factorize/big.cuh::dispatch_factor_big (the gate snippet at the bottom).
// ============================================================================================

// --- panel kernel: Phase 1 (LU) + Phase 2 (U-solve), one block/front, writes L/U to global F ---
__global__ void factor_big_panel_tf32(int lbegin, int lend, const int* __restrict__ plcols,
        const int* __restrict__ front_off, const int* __restrict__ front_ptr,
        const int* __restrict__ ncols, float* frontB, long front_total, int* sing)
{
    const int idx = lbegin + blockIdx.x; if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p]; const int nc = ncols[p];
    float* F = front + front_off[p]; const int t = threadIdx.x, nt = blockDim.x; const int uc = fsz - nc;
    lu_panel_factor<float>(F, fsz, nc, t, nt, sing);   // Phase 1
    u_panel_solve<float>(F, fsz, nc, uc, t, nt);       // Phase 2
}

// --- v2: 2D-tiled trailing. One block computes one BM×BN output tile of S -= L·U, staging only
//     that tile's L rows (BM×nc) + U cols (nc×BN) -> ~8 KB shared -> many blocks/SM. ---
constexpr int EXP_BM = 64;   // output tile rows
constexpr int EXP_BN = 64;   // output tile cols

template <bool FuseExtend>
__device__ __forceinline__ void trailing_tile_tf32(float* F, int fsz, int nc, int uc,
        int r0, int c0, int bm, int cn, float* Lsh, float* Ush, int t, int nt,
        float* Fp, int pfsz, const int* asm_local, int abase)
{
    const int bm_pad = ((bm + 15) / 16) * 16;
    const int cn_pad = ((cn + 7) / 8) * 8;
    const int nc_pad = ((nc + 7) / 8) * 8;
    const int us = cn_pad + 4;                       // Ush column stride (+4 anti bank-conflict)
    for (int e = t; e < bm_pad * nc_pad; e += nt) {
        const int i = e / nc_pad, k = e % nc_pad;
        Lsh[e] = (i < bm && k < nc) ? F[(long)(nc + r0 + i) * fsz + k] : 0.0f;
    }
    for (int e = t; e < nc_pad * cn_pad; e += nt) {
        const int k = e / cn_pad, j = e % cn_pad;
        Ush[k * us + j] = (k < nc && j < cn) ? F[(long)k * fsz + (nc + c0 + j)] : 0.0f;
    }
    __syncthreads();
    const int m_tiles = bm_pad / 16, n_tiles = cn_pad / 8, k_tiles = nc_pad / 8;
    const int warp = t >> 5, nwarp = nt >> 5, lane = t & 31;
    const int laneR = lane >> 2, laneC = (lane & 3) * 2;
    auto store_c = [&](int r, int col, float c) {
        if (r >= bm || col >= cn) return;
        const long off = (long)(nc + r0 + r) * fsz + (nc + c0 + col);
        if constexpr (FuseExtend)
            atomicAdd(&Fp[(long)asm_local[abase + r0 + r] * pfsz + asm_local[abase + c0 + col]], F[off] - c);
        else
            F[off] -= c;
    };
    for (int ti = warp; ti < m_tiles; ti += nwarp) {
        const int r_top = ti * 16 + laneR, r_bot = r_top + 8;
        for (int tj8 = 0; tj8 < n_tiles; ++tj8) {
            float c0a = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (int kc = 0; kc < k_tiles; ++kc) {
                const float* A_top = &Lsh[(ti * 16 + laneR + 0) * nc_pad + kc * 8 + laneC];
                const float* A_bot = &Lsh[(ti * 16 + laneR + 8) * nc_pad + kc * 8 + laneC];
#ifdef CLS_TF32_OZAKI_TC2
                const Tf32Pair a0 = tf32_ozaki_pair(A_top[0]); const Tf32Pair a1 = tf32_ozaki_pair(A_bot[0]);
                const Tf32Pair a2 = tf32_ozaki_pair(A_top[1]); const Tf32Pair a3 = tf32_ozaki_pair(A_bot[1]);
                const Tf32Pair b0 = tf32_ozaki_pair(Ush[(kc * 8 + laneC + 0) * us + tj8 * 8 + laneR]);
                const Tf32Pair b1 = tf32_ozaki_pair(Ush[(kc * 8 + laneC + 1) * us + tj8 * 8 + laneR]);
                CLS_MMA_TF32_OZAKI2(c0a, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
                const unsigned a0 = __float_as_uint(A_top[0]); const unsigned a1 = __float_as_uint(A_bot[0]);
                const unsigned a2 = __float_as_uint(A_top[1]); const unsigned a3 = __float_as_uint(A_bot[1]);
                const unsigned b0 = __float_as_uint(Ush[(kc * 8 + laneC + 0) * us + tj8 * 8 + laneR]);
                const unsigned b1 = __float_as_uint(Ush[(kc * 8 + laneC + 1) * us + tj8 * 8 + laneR]);
                CLS_MMA_TF32_M16N8K8(c0a, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
            }
            const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
            store_c(r_top, col0, c0a); store_c(r_top, col1, c1);
            store_c(r_bot, col0, c2);  store_c(r_bot, col1, c3);
        }
    }
}

__global__ void factor_big_trail_tile_tf32(int lbegin, int lend, const int* __restrict__ plcols,
        const int* __restrict__ front_off, const int* __restrict__ front_ptr,
        const int* __restrict__ ncols, const int* __restrict__ panel_parent,
        const int* __restrict__ asm_ptr, const int* __restrict__ asm_local,
        float* frontB, long front_total, int do_extend, int n_tiles_max, int nc_pad)
{
    const int idx = lbegin + blockIdx.x; if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p]; const int nc = ncols[p];
    float* F = front + front_off[p]; const int t = threadIdx.x, nt = blockDim.x; const int uc = fsz - nc;
    const int par = panel_parent[p];
    const bool extend_ok = (par >= 0 && do_extend);
    float* Fp = (par >= 0) ? (front + front_off[par]) : nullptr;
    const int pfsz = (par >= 0) ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = asm_ptr[p];
    const bool tc = (nc <= 32 && uc <= 256);
    if (!tc) {   // uc>256 or nc>32: scalar trailing in one block + extend-add drain (BUG FIX — must drain)
        if (blockIdx.z == 0) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
            if (extend_ok) { __syncthreads(); extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt); }
        }
        return;
    }
    const int mt = blockIdx.z / n_tiles_max, ntile = blockIdx.z % n_tiles_max;
    const int r0 = mt * EXP_BM, c0 = ntile * EXP_BN;
    if (r0 >= uc || c0 >= uc) return;                 // tile outside this front
    const int bm = min(EXP_BM, uc - r0), cn = min(EXP_BN, uc - c0);
    extern __shared__ char smem_tile[];
    float* Lsh = reinterpret_cast<float*>(smem_tile);
    float* Ush = Lsh + EXP_BM * nc_pad;               // Lsh: EXP_BM × nc_pad, Ush after
    if (extend_ok)
        trailing_tile_tf32<true>(F, fsz, nc, uc, r0, c0, bm, cn, Lsh, Ush, t, nt, Fp, pfsz, asm_local, abase);
    else
        trailing_tile_tf32<false>(F, fsz, nc, uc, r0, c0, bm, cn, Lsh, Ush, t, nt, nullptr, 0, nullptr, 0);
}

// --- dispatch snippet (was inside dispatch_factor_big, Precision::TF32 branch, before the fused
//     factor_big launch). split_fill default 32; only under-filled + max_uc≤256 levels split. ---
#if 0  // reference only
    static const int split_fill = []{ const char* s = std::getenv("CLS_EXP_SPLIT_FILL"); return s ? atoi(s) : 32; }();
    if ((long)(e - b) * st.batch_count < split_fill && max_uc <= 256) {
        factor_big_panel_tf32<<<grid, bigT, 0, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            st.d_front_batch_f, plan.front_total, st.d_sing);
        const int m_tiles_max = (max_uc + EXP_BM - 1) / EXP_BM;
        const int n_tiles_max = (max_uc + EXP_BN - 1) / EXP_BN;
        const size_t sh_tile = (size_t)(EXP_BM * nc_pad_max + nc_pad_max * (EXP_BN + 4)) * sizeof(float);
        constexpr int tileT = 128;
        dim3 tgrid(e - b, st.batch_count, m_tiles_max * n_tiles_max);
        factor_big_trail_tile_tf32<<<tgrid, tileT, sh_tile, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
            plan.front_total, do_extend, n_tiles_max, nc_pad_max);
        return;
    }
#endif
