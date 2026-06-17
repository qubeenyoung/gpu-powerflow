#pragma once

// FACTORIZE — TINY tier (max_fsz <= kTinyFrontMax = warp). One sub-group of lanes owns a whole front;
// sub-groups pack into warps. Kernel + launch + dispatch, all here.

#include "factorize/front_ops.cuh"
#include "factorize/small_ablation.cuh"   // CLS_TINY_ABLATION=1 → MAGMA/STRUMPACK-style ablation

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// SG = sub-group lane count (8 / 16 / 32). One sub-group of SG lanes owns one front; SG=32 is
// the classic one-warp-per-front form. `sl` is the lane within the sub-group (0..SG-1) and
// `mask` the sub-group's active-lane mask. Tiny fronts (fsz ≤ SG) keep all SG lanes busy
// instead of idling 32−fsz of a warp, and packing 32/SG fronts per warp raises memory-level
// parallelism on this latency-bound tier (see factorize/schedule.cuh).
template <typename FT, int SG>
__device__ __forceinline__ void lu_tiny_warp(FT* F, int fsz, int nc, int sl, unsigned mask,
                                              int* sing, bool static_pivoting,
                                              double pivot_threshold, double pivot_shift)
{
    for (int k = 0; k < nc; ++k) {
        const long diag = (long)k * fsz + k;
        FT piv = guarded_pivot(F[diag], static_pivoting, pivot_threshold, pivot_shift,
                               sing, sl == 0);
        if (sl == 0) F[diag] = piv;
        for (int i = k + 1 + sl; i < fsz; i += SG) F[(long)i * fsz + k] /= piv;
        __syncwarp(mask);
        const int m = fsz - k - 1;
        for (int e = sl; e < m * m; e += SG) {
            const int ii = k + 1 + e / m, jj = k + 1 + e % m;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        __syncwarp(mask);
    }
}

// =======================================================================================
//  TINY tier  —  one warp per (front, batch); W warps per block
// =======================================================================================
//
// Used when the level's max_fsz ≤ kTinyFrontMax (see factorize/schedule.cuh).
//
// At the leaves of the elimination tree the fronts are tiny (fsz ≲ 30, nc ≲ 8) but
// numerous. A 256-thread block per front would leave most threads idle and pay a full
// __syncthreads on each rank-1 step. Instead each warp factors a whole front independently
// using __syncwarp, and W warps are packed into one block to amortise launch cost.
//
// Per-warp flow:
//   1. copy fsz×fsz front from global F into per-warp shared scratch Fs (synchronous load
//      — at this size cp.async's commit/wait overhead exceeds the latency saving).
//   2. lu_tiny_warp: fused Phase-1 panel LU + Phase-3 trailing on Fs, lane-parallel with
//      __syncwarp.
//   3. writeback Fs → global F (factored L | U panel only; the uc×uc CB stays in Fs).
//   4. extend_add: scatter the CB straight from shared into the parent front via atomicAdd.
// sub_group_size = sub-group lane count (8 / 16 / 32). One sub-group of sub_group_size lanes owns one (front, batch);
// fronts_per_warp = 32/sub_group_size sub-groups (fronts) pack per warp, kTinyTierWarpsPerBlock warps per block. sub_group_size=32 is the
// classic one-warp-per-front form. The dispatcher picks sub_group_size from the level's max_fsz so the
// tiny fronts (fsz ≤ 16) keep all sub_group_size lanes busy and expose fronts_per_warp independent fronts' memory
// traffic per warp (latency hiding on this memory-latency-bound tier).
template <typename FrontType, int sub_group_size>
__global__ void factor_tiny(int lbegin, int level_size, int B, int front_area,
                              const int* __restrict__ plcols,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ panel_parent,
                              const int* __restrict__ asm_ptr,
                              const int* __restrict__ asm_local, FrontType* frontB,
                              long front_total, int* sing, int do_extend,
                              bool static_pivoting, double pivot_threshold, double pivot_shift)
{
    constexpr int fronts_per_warp = 32 / sub_group_size;                          // fronts per warp
    extern __shared__ unsigned char smem_sw_raw[];
    FrontType* smem_sw = reinterpret_cast<FrontType*>(smem_sw_raw);

    // Sub-group identity: which front this group of sub_group_size lanes owns, and the lane within it.
    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sg = lane / sub_group_size;                             // sub-group id in warp (0..fronts_per_warp-1)
    const int sl = lane % sub_group_size;                             // lane within sub-group (0..sub_group_size-1)
    const unsigned mask = (sub_group_size == 32) ? 0xffffffffu : (((1u << sub_group_size) - 1u) << (sg * sub_group_size));

    const int warps_per_blk = blockDim.x >> 5;
    const int warp_global = blockIdx.x * warps_per_blk + warp_in_blk;
    const int slot = warp_global * fronts_per_warp + sg;              // global (front, batch) index
    if (slot >= level_size * B) return;                   // whole sub-group exits together
    const int front_local = slot % level_size;
    const int batch_idx = slot / level_size;

    // Locate the front buffer for (batch batch_idx, panel p).
    FrontType* front = frontB + (long)batch_idx * front_total;
    const int p = plcols[lbegin + front_local];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    FrontType* F = front + front_off[p];

    // Per-sub-group shared scratch (slot front_area reserved per sub-group).
    FrontType* Fs = smem_sw + (long)(warp_in_blk * fronts_per_warp + sg) * front_area;

    // 1. global F → per-sub-group Fs.
    for (int e = sl; e < fsz2; e += sub_group_size) Fs[e] = F[e];
    __syncwarp(mask);

    // 2. fused panel LU + trailing on Fs.
    lu_tiny_warp<FrontType, sub_group_size>(Fs, fsz, nc, sl, mask, sing,
                                             static_pivoting, pivot_threshold, pivot_shift);
    __syncwarp(mask);

    // 3. writeback factored panel.
    writeback_factored<FrontType, FrontType>(F, Fs, fsz, nc, uc, sl, sub_group_size);

    // 4. CB extend-add into parent front (skip for roots / when disabled).
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    FrontType* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = sl; e < uc * uc; e += sub_group_size) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}

// Pick the tiny-tier sub-group size: sub_group_size ∈ {8,16,32}; packing (sub_group_size<32) only applies once the
// packed grid still fills the GPU (mirrors solve/dispatch.cuh solve_tiny_sg).
static int factor_tiny_sg(int max_fsz, long warps_unpacked)
{
#ifdef CLS_SMALL_SG32_ONLY
    (void)max_fsz; (void)warps_unpacked;
    return 32;
#else
    int sg = (max_fsz <= 8) ? 8 : (max_fsz <= 16 ? 16 : 32);
    if (warps_unpacked / (32 / sg) < factor_warp_fill()) sg = 32;
    return sg;
#endif
}

// Launch helper: instantiates factor_tiny for a concrete (front type, sub-group size). Keeps
// the runtime sub_group_size switch in issue_factor_level_range compact.
template <typename FrontType, int sub_group_size>
static inline void launch_factor_tiny(int num_blocks, int threads_per_block, size_t shared_bytes, cudaStream_t stream,
                                       int b, int level_size, int B, int front_area,
                                       const MultifrontalPlan& plan, const int* d_plc,
                                       FrontType* frontB, int* sing, int do_extend,
                                       bool static_pivoting, double pivot_threshold,
                                       double pivot_shift)
{
    factor_tiny<FrontType, sub_group_size><<<num_blocks, threads_per_block, shared_bytes, stream>>>(
        b, level_size, B, front_area, d_plc, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, frontB,
        plan.front_total, sing, do_extend, static_pivoting, pivot_threshold, pivot_shift);
}

// Launch the tiny kernel with (front type fixed, sub_group_size resolved at the call site).
template <typename FrontType>
static inline void launch_factor_tiny_t(int sub_group_size, int num_blocks, int threads_per_block, size_t shared_bytes, cudaStream_t stream,
                                         int b, int level_size, int B, int front_area,
                                         const MultifrontalPlan& plan, const int* d_plc,
                                         FrontType* frontB, int* sing, int do_extend,
                                         bool static_pivoting, double pivot_threshold,
                                         double pivot_shift)
{
    if (sub_group_size == 8)       launch_factor_tiny<FrontType, 8>(num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area, plan, d_plc, frontB, sing, do_extend, static_pivoting, pivot_threshold, pivot_shift);
    else if (sub_group_size == 16) launch_factor_tiny<FrontType, 16>(num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area, plan, d_plc, frontB, sing, do_extend, static_pivoting, pivot_threshold, pivot_shift);
    else               launch_factor_tiny<FrontType, 32>(num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area, plan, d_plc, frontB, sing, do_extend, static_pivoting, pivot_threshold, pivot_shift);
}

// TINY tier: sub-group-packed kernel. One sub-group of sub_group_size lanes owns one front; 32/sub_group_size fronts
// pack per warp, sub_group_size chosen by max_fsz so the dominant tiny fronts keep all sub_group_size lanes busy. The
// factor_tiny_sg gate keeps the classic one-warp-per-front form (sub_group_size=32) until the packed grid
// fills the GPU. (Within-kernel launch-config only — the tier itself is fixed by front size.)
static void dispatch_factor_tiny(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                 int b, int e, const int* d_plc, const FrontRangeCaps& caps)
{
    const Precision precision = st.precision;
    const int B = st.batch_count;
    const int level_size = e - b;
    constexpr int do_extend = kFactorDoExtend;

    // A/B ablation: process tiny fronts the MAGMA/STRUMPACK way (one block per front, op-separated,
    // global) — removes the sub-group packing + fused shared pipeline. (env CLS_TINY_ABLATION=1)
    if (tiny_ablation_enabled()) {
        if (precision == Precision::FP64)
            dispatch_small_ablation<double>(plan, st, stream, b, e, d_plc, st.d_front_batch, do_extend);
        else
            dispatch_small_ablation<float>(plan, st, stream, b, e, d_plc, st.d_front_batch_f, do_extend);
        return;
    }

    const long warps_unpacked = (long)level_size * B;            // unpacked (one-warp-per-front) count
    const int sub_group_size = factor_tiny_sg(caps.max_fsz, warps_unpacked), fronts_per_warp = kWarpSize / sub_group_size;
    const int threads_per_block = kTinyTierWarpsPerBlock * kWarpSize;
    const int num_blocks = (int)(((warps_unpacked + fronts_per_warp - 1) / fronts_per_warp + kTinyTierWarpsPerBlock - 1) / kTinyTierWarpsPerBlock);
    const int front_area = caps.max_fsz * caps.max_fsz;
    const size_t element_bytes = (precision == Precision::FP64) ? sizeof(double) : sizeof(float);
    const size_t shared_bytes = (size_t)kTinyTierWarpsPerBlock * fronts_per_warp * front_area * element_bytes;
    if (precision == Precision::FP64)
        launch_factor_tiny_t<double>(sub_group_size, num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area,
                                      plan, d_plc, st.d_front_batch, st.d_sing, do_extend,
                                      st.static_pivoting, st.pivot_threshold, st.pivot_shift);
    else
        launch_factor_tiny_t<float>(sub_group_size, num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area,
                                     plan, d_plc, st.d_front_batch_f, st.d_sing, do_extend,
                                     st.static_pivoting, st.pivot_threshold, st.pivot_shift);
}

}  // namespace
}  // namespace custom_linear_solver
