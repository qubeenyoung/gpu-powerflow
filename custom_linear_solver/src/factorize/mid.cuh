#pragma once

// FACTORIZE — MID tier (front fits the per-precision opt-in shared budget). The whole front is
// staged into shared and factorized in place; TF32 uses the blocked Tensor-Core trailing.
// Kernel + blocked-TC helpers + dispatch, all here.

#include "factorize/front_ops.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// =======================================================================================
//  STAGE-IN  —  global → shared
// =======================================================================================
//
// Copies the whole fsz×fsz front from global to dynamic shared so the four phases below
// touch only shared memory. On Ampere+ each element is issued through cp.async via
// __pipeline_memcpy_async, then committed and waited as one batch; on pre-Ampere the helper
// falls back to a synchronous strided copy. The async path lets a thread queue many loads
// before stalling, giving the SM something else to schedule while the loads are in flight.
template <typename T>
__device__ __forceinline__ void stage_in_async(T* __restrict__ Fs, const T* __restrict__ F,
                                                int fsz2, int t, int nt)
{
#if __CUDA_ARCH__ >= 800
    constexpr size_t SHAPE = sizeof(T);
    for (int e = t; e < fsz2; e += nt) {
        __pipeline_memcpy_async(&Fs[e], &F[e], SHAPE);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
#else
    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
#endif
}

__device__ __forceinline__ void block_update_tf32_tc(float* F, int fsz,
                                                                    int row0, int col0,
                                                                    int dim, int k0, int kb,
                                                                    int t, int nt)
{
    if (dim <= 0 || kb <= 0) return;
    const int DP = ((dim + 15) / 16) * 16;
    const int nc_pad = ((kb + 7) / 8) * 8;
    const int m_tiles = DP / 16;
    const int n_tiles  = DP / 8;
    const int k_tiles   = nc_pad / 8;
    const int warp  = t >> 5;
    const int nwarp = nt >> 5;
    const int lane  = t & 31;
    const int laneR = lane >> 2;
    const int laneC = (lane & 3) * 2;

    auto load_l = [&](int r, int k) {
        return (r < dim && k < kb) ? F[(long)(row0 + r) * fsz + (k0 + k)] : 0.0f;
    };
    auto load_u = [&](int k, int col) {
        return (k < kb && col < dim) ? F[(long)(k0 + k) * fsz + (col0 + col)] : 0.0f;
    };
    auto store_c = [&](int r, int col, float c) {
        if (r < dim && col < dim) F[(long)(row0 + r) * fsz + (col0 + col)] -= c;
    };

    constexpr int NTJ8_MAX = 16;
    if (n_tiles <= NTJ8_MAX) {
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
                    CLS_MMA_TF32_OZAKI2(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3],
                                        a0, a1, a2, a3, b0, b1);
#else
                    const unsigned b0 = __float_as_uint(load_u(k + 0, col));
                    const unsigned b1 = __float_as_uint(load_u(k + 1, col));
                    CLS_MMA_TF32_M16N8K8(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3],
                                         a0, a1, a2, a3, b0, b1);
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
        return;
    }

    for (int ti = warp; ti < m_tiles; ti += nwarp) {
        const int r_top = ti * 16 + laneR;
        const int r_bot = r_top + 8;
        for (int tj8 = 0; tj8 < n_tiles; ++tj8) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (int kc = 0; kc < k_tiles; ++kc) {
                const int k = kc * 8 + laneC;
#ifdef CLS_TF32_OZAKI_TC2
                const Tf32Pair a0 = tf32_ozaki_pair(load_l(r_top, k + 0));
                const Tf32Pair a1 = tf32_ozaki_pair(load_l(r_bot, k + 0));
                const Tf32Pair a2 = tf32_ozaki_pair(load_l(r_top, k + 1));
                const Tf32Pair a3 = tf32_ozaki_pair(load_l(r_bot, k + 1));
                const int col = tj8 * 8 + laneR;
                const Tf32Pair b0 = tf32_ozaki_pair(load_u(k + 0, col));
                const Tf32Pair b1 = tf32_ozaki_pair(load_u(k + 1, col));
                CLS_MMA_TF32_OZAKI2(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
                const unsigned a0 = __float_as_uint(load_l(r_top, k + 0));
                const unsigned a1 = __float_as_uint(load_l(r_bot, k + 0));
                const unsigned a2 = __float_as_uint(load_l(r_top, k + 1));
                const unsigned a3 = __float_as_uint(load_l(r_bot, k + 1));
                const int col = tj8 * 8 + laneR;
                const unsigned b0 = __float_as_uint(load_u(k + 0, col));
                const unsigned b1 = __float_as_uint(load_u(k + 1, col));
                CLS_MMA_TF32_M16N8K8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
            }
            const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
            store_c(r_top, col0, c0);
            store_c(r_top, col1, c1);
            store_c(r_bot, col0, c2);
            store_c(r_bot, col1, c3);
        }
    }
}

__device__ __forceinline__ void factorize_front_blocked_tf32(float* F, int fsz, int nc,
                                                             int t, int nt, int* sing)
{
    constexpr int BK = 8;
    for (int k0 = 0; k0 < nc; k0 += BK) {
        const int kb = (k0 + BK <= nc) ? BK : (nc - k0);
        const int next = k0 + kb;

        // Factor the current diagonal block and L panel below it. Only the current block's
        // panel columns are updated here; the right-looking TC update below handles the rest.
        for (int kk = 0; kk < kb; ++kk) {
            const int k = k0 + kk;
            float piv = F[(long)k * fsz + k];
            if (piv == 0.0f) { if (t == 0) *sing = 1; piv = 1.0f; }
            const float inv_piv = 1.0f / piv;
            for (int i = k + 1 + t; i < fsz; i += nt) {
                const float lik = F[(long)i * fsz + k] * inv_piv;
                F[(long)i * fsz + k] = lik;
                for (int jj = k + 1; jj < next; ++jj) {
                    F[(long)i * fsz + jj] -= lik * F[(long)k * fsz + jj];
                }
            }
            __syncthreads();
        }

        // Solve the block row U over all remaining columns, including both the rest of the
        // panel and the contribution block. Row-by-row: row r depends on rows < r in this pivot
        // block, so a __syncthreads() follows each row.
        for (int kk = 0; kk < kb; ++kk) {
            const int row = k0 + kk;
            for (int j = next + t; j < fsz; j += nt) {
                float v = F[(long)row * fsz + j];
                for (int i = k0; i < row; ++i) v -= F[(long)row * fsz + i] * F[(long)i * fsz + j];
                F[(long)row * fsz + j] = v;
            }
            __syncthreads();
        }

        const int dim = fsz - next;
        if (dim > 0) {
            block_update_tf32_tc(F, fsz, next, next, dim, k0, kb, t, nt);
            __syncthreads();
        }
    }
}

// (blocked TC kernels — baked in, formerly #ifdef CLS_BIG_TF32_BLOCKED_TC)
// Shared-resident blocked mid kernel — the unified mid-tier kernel for every precision.
// The full front is staged into shared (fits the 99 KiB opt-in budget for fsz below the
// per-precision shared limit) and factorized in place. For UseTC (T = float, TF32 precision)
// eligible shapes use the blocked TF32 Tensor-Core trailing; otherwise (FP64/FP32, or
// TC-ineligible shapes) the same shared-resident blocked structure runs scalar trailing.
// Supersedes factor_mid / factor_mid_tf32_ptx / factor_big_shared_tf32_blocked.
template <typename T, bool UseTC>
__global__ void factor_mid_blocked(int lbegin, int lend,
                                   const int* __restrict__ plcols,
                                   const int* __restrict__ front_off,
                                   const int* __restrict__ front_ptr,
                                   const int* __restrict__ ncols,
                                   const int* __restrict__ panel_parent,
                                   const int* __restrict__ asm_ptr,
                                   const int* __restrict__ asm_local,
                                   T* frontB, long front_total, int* sing,
                                   int do_extend, int fsz_cap, int rb_width)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;

    extern __shared__ char smem_mid_blocked[];
    T* Fs = reinterpret_cast<T*>(smem_mid_blocked);
    stage_in_async<T>(Fs, F, fsz2, t, nt);
    __syncthreads();

    // Factorize the front in place: TF32-eligible shapes take the blocked Tensor-Core path; all
    // other shapes / precisions run the scalar phases (Phase 1 panel LU + Phase 2 U-solve +
    // Phase 3 trailing, with the fsz<=kFusedSmallFrontMax fused Phase 1+3 fast path).
    bool did_tc = false;
    if constexpr (UseTC) {
        if (fsz > CLS_TC_FSZ_MIN && uc >= CLS_TC_UC_MIN && nc >= CLS_TC_NC_MIN && nc <= kTensorCorePivotColumnCap && uc <= CLS_TC_UC_CAP) {
            factorize_front_blocked_tf32(Fs, fsz, nc, t, nt, sing);
            did_tc = true;
        }
    }
    if (!did_tc) {
        if (fsz <= kFusedSmallFrontMax) {
            lu_small_front<T>(Fs, fsz, nc, t, nt, sing);              // Phase 1 + Phase 3 fused
        } else {
            lu_panel_factor<T>(Fs, fsz, nc, t, nt, sing);            // Phase 1
            u_panel_solve_fewsync<T>(Fs, fsz, nc, uc, t, nt);        // Phase 2 (barrier-cut default)
            trailing_update<T>(Fs, fsz, nc, uc, t, nt, rb_width);    // Phase 3 (register-blocked)
        }
    }

    writeback_factored<T, T>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<T, T>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
    (void)fsz_cap;
}

// =======================================================================================
//  PANEL-RESIDENT MID KERNEL   (exp_260612 structural — break the whole-front shared cap)
// =======================================================================================
//
// The baseline factor_mid_blocked stages the WHOLE front (fsz²) into shared, so big-mid fronts are
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
            const unsigned a0 = __float_as_uint(load_l(r_top, k + 0));
            const unsigned a1 = __float_as_uint(load_l(r_bot, k + 0));
            const unsigned a2 = __float_as_uint(load_l(r_top, k + 1));
            const unsigned a3 = __float_as_uint(load_l(r_bot, k + 1));
            #pragma unroll
            for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                if (tj8 >= n_tiles) break;
                const int col = tj8 * 8 + laneR;
                const unsigned b0 = __float_as_uint(load_u(k + 0, col));
                const unsigned b1 = __float_as_uint(load_u(k + 1, col));
                CLS_MMA_TF32_M16N8K8(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3], a0, a1, a2, a3, b0, b1);
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
__global__ void factor_mid_panel(int lbegin, int lend,
                                 const int* __restrict__ plcols,
                                 const int* __restrict__ front_off,
                                 const int* __restrict__ front_ptr,
                                 const int* __restrict__ ncols,
                                 const int* __restrict__ panel_parent,
                                 const int* __restrict__ asm_ptr,
                                 const int* __restrict__ asm_local,
                                 T* frontB, long front_total, int* sing,
                                 int do_extend, int fsz_cap, int nc_cap)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    const int t = threadIdx.x, nt = blockDim.x;
    T* F = frontB + (long)blockIdx.y * front_total + front_off[p];

    extern __shared__ char smem_mid_panel[];
    T* Lpan = reinterpret_cast<T*>(smem_mid_panel);     // fsz×nc, stride nc
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
        T piv = Lpan[k * nc + k];
        if (piv == T(0)) { if (t == 0) *sing = 1; piv = T(1); }
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

// MID tier: shared-resident kernel, one block per (front, batch). Returns false when the chosen
// shared layout overflows the per-block budget (e.g. FP64 with a large fsz_cap), so the caller
// falls through to the big tier on the same range.
// MID tier — the unified shared-resident blocked kernel for every precision. "Mid" = the front
// fits the per-precision opt-in shared budget (fsz below ~159 for FP32/TF32, ~111 for FP64); the
// dispatch returns false for larger fronts so the caller falls through to the big tier. Thread
// count follows the swept occupancy heuristic (underfilled levels parallelise each front more;
// saturated levels pack more blocks per SM).
static bool dispatch_factor_mid(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                int b, int e, const int* d_plc, const int* h_plc,
                                const FrontRangeCaps& caps)
{
    const auto& [max_fsz, max_uc, level_max_nc, level_max_uc] = caps;
    const Precision precision = st.precision;
    const int B = st.batch_count;
    const int level_size = e - b;
    constexpr int do_extend = kFactorDoExtend;
    const int fsz_cap = max_fsz;
    // exp_260612: register-blocked trailing width. CLS_TRAIL_RB={0 scalar,2,4,8}, default 4.
    static const int rb_width = [] {
        const char* s = std::getenv("CLS_TRAIL_RB");
        return s ? std::atoi(s) : 4;
    }();
    (void)max_uc; (void)level_max_nc; (void)level_max_uc; (void)h_plc;
    dim3 grid(level_size, B);

    // exp_260612 STRUCTURAL: panel-resident kernel — only L/U panels in shared (CB stays global),
    // shrinking shared from fsz² to nc(fsz+uc) → ~3–4× more blocks/SM → saturates DRAM on the
    // filled (waves≫1) big-mid launches that the whole-front kernel starves at 8–16% occupancy.
    // fp32 scatter only; env-gated CLS_MID_PANEL while validating.
    static const int use_panel = [] {
        const char* s = std::getenv("CLS_MID_PANEL"); return s ? std::atoi(s) : 1;   // default on
    }();
    // Panel kernel wins once the whole-front kernel is shared-limited (big fronts) AND occupancy-bound
    // (enough blocks to fill the SMs). The benefit scales with how starved the whole-front kernel is =
    // front size, so two tiers: BIG fronts (fsz≥112, ≥2·SMs blocks) help even at moderate block count;
    // MEDIUM fronts (fsz≥64) only help under heavy occupancy pressure (≥8·SMs blocks), else the extra
    // global-CB pass isn't repaid (measured: ACTIVSg25k B=16 regresses without the stronger gate).
    // Applies to every precision now (FP64/TF32 too — FP64 is even more shared-starved). gather off.
    static const int panel_min_big = [] {
        const char* s = std::getenv("CLS_MID_PANEL_MIN"); return s ? std::atoi(s) : 112;
    }();
    static const int panel_min_med = [] {
        const char* s = std::getenv("CLS_MID_PANEL_MED"); return s ? std::atoi(s) : 64;
    }();
    static const int panel_med_blk = [] {
        const char* s = std::getenv("CLS_MID_PANEL_MED_BLK"); return s ? std::atoi(s) : 16;
    }();
    static const int panel_tc = [] {   // TF32 panel: TC trailing (1) vs scalar (0, default)
        const char* s = std::getenv("CLS_MID_PANEL_TC"); return s ? std::atoi(s) : 0;
    }();
    const long blocks_total = (long)level_size * B;
    const long sms = factor_num_sms();
    const bool panel_big = fsz_cap >= panel_min_big && blocks_total >= 2L * sms;
    const bool panel_med = fsz_cap >= panel_min_med && blocks_total >= (long)panel_med_blk * sms;
    const bool panel_eligible = use_panel && (panel_big || panel_med);
    if (panel_eligible) {
        const int nc_cap = level_max_nc, uc_cap = level_max_uc;
        const size_t ebytes = (precision == Precision::FP64) ? sizeof(double) : sizeof(float);
        const size_t pshared = (size_t)(fsz_cap * nc_cap + nc_cap * uc_cap) * ebytes;
        if (pshared <= kDynamicSharedMemoryOptInBytes) {
            const int threads = (blocks_total < sms) ? 512 : (blocks_total < 4 * sms ? 256 : 128);
            if (precision == Precision::FP64) {
                cudaFuncSetAttribute(factor_mid_panel<double>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pshared);
                factor_mid_panel<double><<<grid, threads, pshared, stream>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
                    plan.front_total, st.d_sing, do_extend, fsz_cap, nc_cap);
            } else if (precision == Precision::TF32 && panel_tc) {   // float front + TF32 mma TC trailing
                cudaFuncSetAttribute(factor_mid_panel<float, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pshared);
                factor_mid_panel<float, true><<<grid, threads, pshared, stream>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
                    plan.front_total, st.d_sing, do_extend, fsz_cap, nc_cap);
            } else {  // FP32, or TF32 with scalar trailing (default — TC measured a net regression)
                cudaFuncSetAttribute(factor_mid_panel<float, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pshared);
                factor_mid_panel<float, false><<<grid, threads, pshared, stream>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
                    plan.front_total, st.d_sing, do_extend, fsz_cap, nc_cap);
            }
            return true;
        }
    }

    const size_t element_bytes = (precision == Precision::FP64) ? sizeof(double) : sizeof(float);
    const size_t shared_bytes = (size_t)fsz_cap * fsz_cap * element_bytes;   // whole front in shared
    if (shared_bytes > kDynamicSharedMemoryOptInBytes) return false;         // too big → big tier

    const long blocks = blocks_total;   // (sms already computed above for the panel gate)
    // NB: bumping the under-fill case 512->1024 threads was measured neutral (each front is
    // dependency-latency-bound on the sequential panel-LU chain, not warp-throughput-bound; extra
    // warps don't hide a single front's internal critical path). Kept at 512. (exp_260612)
    const int threads = (blocks < sms) ? 512 : (blocks < 4 * sms ? 256 : 128);
    // exp_260612 batch note: at high grid (B>=16) a large front is shared-memory-limited to 1
    // block/SM, so the narrow-block branch (128) wastes warp slots; forcing 256 measured ~+5% at
    // B=64 but tf32 (the real batch lever, +9-15%) subsumes it. Left as heuristic (see doc 05).

    if (precision == Precision::FP64) {
        factor_mid_blocked<double, false><<<grid, threads, shared_bytes, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
            plan.front_total, st.d_sing, do_extend, fsz_cap, rb_width);
    } else if (precision == Precision::TF32) {
        factor_mid_blocked<float, true><<<grid, threads, shared_bytes, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
            plan.front_total, st.d_sing, do_extend, fsz_cap, rb_width);
    } else {  // FP32
        factor_mid_blocked<float, false><<<grid, threads, shared_bytes, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
            plan.front_total, st.d_sing, do_extend, fsz_cap, rb_width);
    }
    return true;
}

}  // namespace
}  // namespace custom_linear_solver
