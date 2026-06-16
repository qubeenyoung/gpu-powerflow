#pragma once

// FACTORIZE — BIG tier (front still fits the per-precision opt-in shared budget, but whole-front
// staging would starve occupancy under batch). Only the L/U PANELS live in shared; the bulky
// contribution block stays in global and the extend-add is fused into the parent. Kernel +
// panel-resident TC trailing + dispatch, all here. At B==1 the tier delegates to the SMALL-tier
// whole-front kernel (see dispatch_factor_big).

#include "factorize/front_ops.cuh"
#include "factorize/small.cuh"   // dispatch_factor_small — B==1 delegation

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// =======================================================================================
//  PANEL-RESIDENT BIG KERNEL   (exp_260612 structural — break the whole-front shared cap)
// =======================================================================================
//
// The baseline SMALL-tier factor_small stages the WHOLE front (fsz²) into shared, so big-tier fronts are
// shared-limited to 1–2 blocks/SM → 8–16% occupancy → the memory-bound kernel only reaches ~35–40%
// of peak DRAM (ncu). The work is there (waves≫1 at B=64); the mapping starves bandwidth.
//
// Here only the L/U PANELS live in shared — the contribution block (uc², the bulk) stays in global:
//   Lpan = left panel rows[0,fsz) × cols[0,nc)  (fsz·nc, stride nc) — pivot block + L strip
//   Upan = U panel    rows[0,nc)  × cols[nc,fsz) (nc·uc,  stride uc)
// Shared shrinks from fsz² to nc(fsz+uc) ≈ 3× smaller → ~3–4× more blocks/SM → occupancy/bandwidth.
// Phase 1/2 (iterative panel LU + sync-free U-solve) run in shared; Phase 3 (single-pass trailing)
// reads Lpan/Upan from shared and the assembled CB from global, and fuses the extend-add straight
// into the parent (atomicAdd) — so global CB traffic is the same one pass as the baseline's stage-in.
// TF32 tensor-core trailing for the panel-resident kernel. Reads L (uc×nc) and U (nc×uc) DIRECTLY
// from the shared panels (Lpan stride nc, Upan stride uc) — no extra padded staging, so the panel
// kernel's small-shared / high-occupancy advantage is preserved — and the store path computes
// Schur = assembled_CB(global) − L·U and fuses the extend-add into the parent. m16n8k8 mma, K=nc_pad.
__device__ __forceinline__ void trailing_panel_tf32_tc(const float* __restrict__ Lpan,
                                                       const float* __restrict__ Upan,
                                                       float* F, int fsz, int nc, int uc,
                                                       float* Fp, int pfsz,
                                                       const int* __restrict__ asm_local, int abase,
                                                       bool extend_ok, int t, int nt)
{
    if (uc <= 0 || nc <= 0) return;
    const int DP = ((uc + 15) / 16) * 16;
    const int nc_pad = ((nc + 7) / 8) * 8;
    const int m_tiles = DP / 16;
    const int n_tiles = DP / 8;
    const int k_tiles = nc_pad / 8;
    const int warp = t >> 5, nwarp = nt >> 5;
    const int lane = t & 31;
    const int laneR = lane >> 2;
    const int laneC = (lane & 3) * 2;

    auto load_l = [&](int r, int k) {     // L = Lpan rows [nc,fsz) × cols [0,nc)
        return (r < uc && k < nc) ? Lpan[(long)(nc + r) * nc + k] : 0.0f;
    };
    auto load_u = [&](int k, int col) {   // U = Upan rows [0,nc) × cols [0,uc)
        return (k < nc && col < uc) ? Upan[(long)k * uc + col] : 0.0f;
    };
    auto store_c = [&](int r, int col, float c) {
        if (r >= uc || col >= uc) return;
        const long off = (long)(nc + r) * fsz + (nc + col);
        const float schur = F[off] - c;   // assembled CB minus the L·U product
        if (extend_ok)
            atomicAdd(&Fp[(long)asm_local[abase + r] * pfsz + asm_local[abase + col]], schur);
        else
            F[off] = schur;
    };

    constexpr int NTJ8_MAX = 18;          // uc up to CLS_TC_UC_CAP → n_tiles ≤ ~18
    for (int ti = warp; ti < m_tiles; ti += nwarp) {
        const int r_top = ti * 16 + laneR;
        const int r_bot = r_top + 8;
        float c[NTJ8_MAX][4];
        #pragma unroll
        for (int j = 0; j < NTJ8_MAX; ++j) { c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; }
        for (int kc = 0; kc < k_tiles; ++kc) {
            const int k = kc * 8 + laneC;
#ifdef CLS_TF32_OZAKI_TC2
            const Tf32Pair a0 = tf32_ozaki_pair(load_l(r_top, k + 0));
            const Tf32Pair a1 = tf32_ozaki_pair(load_l(r_bot, k + 0));
            const Tf32Pair a2 = tf32_ozaki_pair(load_l(r_top, k + 1));
            const Tf32Pair a3 = tf32_ozaki_pair(load_l(r_bot, k + 1));
#else
            const unsigned a0 = __float_as_uint(load_l(r_top, k + 0));
            const unsigned a1 = __float_as_uint(load_l(r_bot, k + 0));
            const unsigned a2 = __float_as_uint(load_l(r_top, k + 1));
            const unsigned a3 = __float_as_uint(load_l(r_bot, k + 1));
#endif
            #pragma unroll
            for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                if (tj8 >= n_tiles) break;
                const int col = tj8 * 8 + laneR;
#ifdef CLS_TF32_OZAKI_TC2
                const Tf32Pair b0 = tf32_ozaki_pair(load_u(k + 0, col));
                const Tf32Pair b1 = tf32_ozaki_pair(load_u(k + 1, col));
                CLS_MMA_TF32_OZAKI2(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3], a0, a1, a2, a3, b0, b1);
#else
                const unsigned b0 = __float_as_uint(load_u(k + 0, col));
                const unsigned b1 = __float_as_uint(load_u(k + 1, col));
                CLS_MMA_TF32_M16N8K8(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3], a0, a1, a2, a3, b0, b1);
#endif
            }
        }
        #pragma unroll
        for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
            if (tj8 >= n_tiles) break;
            const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
            store_c(r_top, col0, c[tj8][0]);
            store_c(r_top, col1, c[tj8][1]);
            store_c(r_bot, col0, c[tj8][2]);
            store_c(r_bot, col1, c[tj8][3]);
        }
    }
}

template <typename T, bool UseTC = false>
__global__ void factor_big(int lbegin, int lend,
                                 const int* __restrict__ plcols,
                                 const int* __restrict__ front_off,
                                 const int* __restrict__ front_ptr,
                                 const int* __restrict__ ncols,
                                 const int* __restrict__ panel_parent,
                                 const int* __restrict__ asm_ptr,
                                 const int* __restrict__ asm_local,
                                 T* frontB, long front_total, int* sing,
                                 int do_extend, int fsz_cap, int nc_cap,
                                 bool static_pivoting, double pivot_threshold,
                                 double pivot_shift)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    const int t = threadIdx.x, nt = blockDim.x;
    T* F = frontB + (long)blockIdx.y * front_total + front_off[p];

    extern __shared__ char smem_big_panel[];
    T* Lpan = reinterpret_cast<T*>(smem_big_panel);     // fsz×nc, stride nc
    T* Upan = Lpan + (long)fsz_cap * nc_cap;            // nc×uc, stride uc

    // Stage the panels from global F (the CB block, rows≥nc cols≥nc, is left in global).
    for (int e = t; e < fsz * nc; e += nt) {
        const int i = e / nc, k = e % nc;
        Lpan[i * nc + k] = F[(long)i * fsz + k];
    }
    for (int e = t; e < nc * uc; e += nt) {
        const int k = e / uc, j = e % uc;
        Upan[k * uc + j] = F[(long)k * fsz + (nc + j)];
    }
    __syncthreads();

    // Phase 1 — panel LU on Lpan (row-fused, 1 barrier/pivot; uniform indexing, no row split).
    for (int k = 0; k < nc; ++k) {
        const long diag = (long)k * nc + k;
        T piv = guarded_pivot(Lpan[diag], static_pivoting, pivot_threshold, pivot_shift,
                              sing, t == 0);
        if (t == 0) Lpan[diag] = piv;
        const T inv = T(1) / piv;
        for (int i = k + 1 + t; i < fsz; i += nt) {
            const T lik = Lpan[(long)i * nc + k] * inv;
            Lpan[(long)i * nc + k] = lik;
            for (int jj = k + 1; jj < nc; ++jj)
                Lpan[(long)i * nc + jj] -= lik * Lpan[(long)k * nc + jj];
        }
        __syncthreads();
    }

    // Phase 2 — U-solve on Upan (sync-free: each thread owns a column, nc barriers → 1).
    for (int e = t; e < uc; e += nt) {
        for (int k = 1; k < nc; ++k) {
            T v = Upan[(long)k * uc + e];
            for (int i = 0; i < k; ++i) v -= Lpan[(long)k * nc + i] * Upan[(long)i * uc + e];
            Upan[(long)k * uc + e] = v;
        }
    }
    __syncthreads();

    // Writeback L/U panels to global (the solve reads these); CB region of F is untouched (the
    // parent receives this front's CB via the fused extend below).
    for (int e = t; e < fsz * nc; e += nt) {
        const int i = e / nc, k = e % nc;
        F[(long)i * fsz + k] = Lpan[i * nc + k];
    }
    for (int e = t; e < nc * uc; e += nt) {
        const int k = e / uc, j = e % uc;
        F[(long)k * fsz + (nc + j)] = Upan[k * uc + j];
    }

    // Phase 3 — trailing CB[i,j] = assembled - Σ_k L[i,k]·U[k,j], fused extend-add into the parent.
    const int par = panel_parent[p];
    const bool extend_ok = (par >= 0 && do_extend);
    T* Fp = extend_ok ? (frontB + (long)blockIdx.y * front_total + front_off[par]) : nullptr;
    const int pfsz = extend_ok ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = asm_ptr[p];
    if constexpr (UseTC) {   // TF32 tensor-core trailing (T == float); reads the shared panels direct
        trailing_panel_tf32_tc(reinterpret_cast<const float*>(Lpan), reinterpret_cast<const float*>(Upan),
                               reinterpret_cast<float*>(F), fsz, nc, uc,
                               reinterpret_cast<float*>(Fp), pfsz, asm_local, abase, extend_ok, t, nt);
    } else {
        for (long e = t; e < (long)uc * uc; e += nt) {
            const int i = e / uc, j = e % uc;
            T acc = T(0);
            for (int k = 0; k < nc; ++k) acc += Lpan[(long)(nc + i) * nc + k] * Upan[(long)k * uc + j];
            const T schur = F[(long)(nc + i) * fsz + (nc + j)] - acc;
            if (extend_ok)
                atomicAdd(&Fp[(long)asm_local[abase + i] * pfsz + asm_local[abase + j]], schur);
            else
                F[(long)(nc + i) * fsz + (nc + j)] = schur;
        }
    }
}

// BIG tier — panel-resident shared kernel (factor_big). Only the L/U panels live in shared; the
// bulky contribution block stays in global and the extend-add is fused into the parent. shared shrinks
// from fsz² to nc(fsz+uc) → ~3–4× more blocks/SM → recovers DRAM bandwidth on the big-tier fronts the
// whole-front kernel starves (storyline §4-(3) — the largest measured batch lever). The tier
// (kSmallFrontMax < fsz <= whole_front_shared_max) is exactly the size range where the whole-front
// kernel's shared footprint drops occupancy *under batch*.
//
// Batch-regime split (the only non-size condition, and a clean B==1 boundary — not an occupancy-fill
// heuristic): panel-residency's whole benefit is the extra blocks/SM that batch occupancy provides, so
// at B=1 it has nothing to exploit and its extra global-CB pass is not repaid. At B=1 the lever is the
// opposite one — whole-front + TF32 tensor-core trailing shortens the per-front critical path — so the
// big tier delegates to the whole-front kernel there (which keeps the TF32 TC path alive for B=1, and
// whole-front fits shared across this tier). B>1 takes the panel-resident bandwidth path.
static void dispatch_factor_big(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                int b, int e, const int* d_plc, const int* h_plc,
                                const FrontRangeCaps& caps)
{
    if (st.batch_count == 1) {
        dispatch_factor_small(plan, st, stream, b, e, d_plc, h_plc, caps);
        return;
    }
    const auto& [max_fsz, max_uc, level_max_nc, level_max_uc] = caps;
    const Precision precision = st.precision;
    const int B = st.batch_count;
    const int level_size = e - b;
    constexpr int do_extend = kFactorDoExtend;
    const int fsz_cap = max_fsz;
    const int nc_cap = level_max_nc, uc_cap = level_max_uc;
    (void)max_uc; (void)h_plc;
    dim3 grid(level_size, B);
    const long blocks_total = (long)level_size * B;
    const long sms = factor_num_sms();
    const int threads = (blocks_total < sms) ? 512 : (blocks_total < 4 * sms ? 256 : 128);
    const size_t ebytes = (precision == Precision::FP64) ? sizeof(double) : sizeof(float);
    const size_t pshared = (size_t)(fsz_cap * nc_cap + nc_cap * uc_cap) * ebytes;
    // The trailing follows the precision mode, like every other tier: TF32 -> TF32 mma (Ozaki when
    // built) so the TF32 path runs in the big tier too; FP32 -> FP32 scalar; FP64 -> double. Note the
    // panel kernel recovers occupancy, so the TF32 mma here has no idle headroom and runs measurably
    // slower than the FP32 scalar trailing — that is the accepted cost of a consistent TF32 path; use
    // --precision fp32 for the fast FP32-scalar big tier.
    if (precision == Precision::FP64) {
        cudaFuncSetAttribute(factor_big<double>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pshared);
        factor_big<double><<<grid, threads, pshared, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
            plan.front_total, st.d_sing, do_extend, fsz_cap, nc_cap,
            st.static_pivoting, st.pivot_threshold, st.pivot_shift);
    } else if (precision == Precision::TF32) {   // float front + TF32 mma TC trailing (Ozaki when built)
        cudaFuncSetAttribute(factor_big<float, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pshared);
        factor_big<float, true><<<grid, threads, pshared, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
            plan.front_total, st.d_sing, do_extend, fsz_cap, nc_cap,
            st.static_pivoting, st.pivot_threshold, st.pivot_shift);
    } else {  // FP32 — FP32 scalar trailing
        cudaFuncSetAttribute(factor_big<float, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pshared);
        factor_big<float, false><<<grid, threads, pshared, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
            plan.front_total, st.d_sing, do_extend, fsz_cap, nc_cap,
            st.static_pivoting, st.pivot_threshold, st.pivot_shift);
    }
}

}  // namespace
}  // namespace custom_linear_solver
