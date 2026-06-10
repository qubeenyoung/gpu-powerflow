#pragma once

// FACTORIZE — per-tier __global__ kernel entry points.
//
// Internal — included only by numeric_engine.cu (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// Per-front canonical flow (all kernels):
//
//     ┌─────────────────────────────────────────────────────────────────────────┐
//     │  stage_in_async       global F → shared Fs (mid only — big runs on F)  │
//     │  factorize_front      Phase 1 panel LU + Phase 2 U-solve + Phase 3 GEMM │
//     │  writeback_factored   shared Fs → global F (mid only)                  │
//     │  extend_add           uc×uc CB → parent front via atomicAdd            │
//     └─────────────────────────────────────────────────────────────────────────┘
//
// Phase 3 (trailing GEMM C ← C − L·U) varies per kernel; the rest of the body is identical
// modulo the front element type. See phases.cuh for the building blocks.
//
// Tier / precision matrix:
//
//   tier   | block  | front | kernel                  | Phase-3 trailing
//   -------+--------+-------+-------------------------+-------------------------------------
//   small  | 8 warp | smem  | factor_small<T, sub_group_size>     | lu_small_warp (fused Phase 1+3)
//   mid    | 64..256| smem  | factor_mid<T>           | trailing_update_staged<T>
//          | 64..256| smem  | factor_mid_fp16_ptx     | trailing_update_mma_fp16_ptx
//   big    | 1024   | gmem  | factor_big<T>/_staged   | trailing_update_scalar / _staged<T>
//          | 512    | gmem  | factor_big_fp16_ptx     | trailing_update_mma_fp16_ptx
//          | 512    | gmem  | factor_big_tf32_ptx     | trailing_update_mma_tf32_ptx
//
// The mid tier defaults to staged-scalar except for eligible FP16 fronts. The FP16 mid Tensor-Core
// kernel keeps scalar fallback for small/skinny fronts, so the TC overhead is paid only where the
// trailing block has enough work to amortise staging.
//
// Why a separate kernel symbol per variant:
//   * The <T> templates parameterise over the front element type (double / float).
//   * The big Tensor-Core variants differ in shared-memory layout: the FP16 PTX path stages
//     __half panels, the TF32 PTX path stages float panels. Neither needs a Csc fragment-readback
//     scratch — the inline-asm mma.sync writes the accumulator straight to per-lane registers.
//   * The big PTX kernels (factor_big_fp16_ptx, factor_big_tf32_ptx) carry
//     __launch_bounds__(512, 2) and a 512-thread block so two blocks stay resident per SM on
//     sm_86 — neither can be selected inside a 1024-thread kernel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "factorize/phases.cuh"

namespace custom_linear_solver {
namespace {

// =======================================================================================
//  SMALL tier  —  one warp per (front, batch); W warps per block
// =======================================================================================
//
// Used when the level's max_fsz ≤ kSmallFrontMax (see factorize/dispatch.cuh).
//
// At the leaves of the elimination tree the fronts are tiny (fsz ≲ 30, nc ≲ 8) but
// numerous. A 256-thread block per front would leave most threads idle and pay a full
// __syncthreads on each rank-1 step. Instead each warp factors a whole front independently
// using __syncwarp, and W warps are packed into one block to amortise launch cost.
//
// Per-warp flow:
//   1. copy fsz×fsz front from global F into per-warp shared scratch Fs (synchronous load
//      — at this size cp.async's commit/wait overhead exceeds the latency saving).
//   2. lu_small_warp: fused Phase-1 panel LU + Phase-3 trailing on Fs, lane-parallel with
//      __syncwarp.
//   3. writeback Fs → global F (factored L | U panel only; the uc×uc CB stays in Fs).
//   4. extend_add: scatter the CB straight from shared into the parent front via atomicAdd.
// sub_group_size = sub-group lane count (8 / 16 / 32). One sub-group of sub_group_size lanes owns one (front, batch);
// fronts_per_warp = 32/sub_group_size sub-groups (fronts) pack per warp, kSmallTierWarpsPerBlock warps per block. sub_group_size=32 is the
// classic one-warp-per-front form. The dispatcher picks sub_group_size from the level's max_fsz so the
// tiny fronts (fsz ≤ 16) keep all sub_group_size lanes busy and expose fronts_per_warp independent fronts' memory
// traffic per warp (latency hiding on this memory-latency-bound tier).
template <typename FrontType, int sub_group_size>
__global__ void factor_small(int lbegin, int level_size, int B, int front_area,
                              const int* __restrict__ plcols,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ panel_parent,
                              const int* __restrict__ asm_ptr,
                              const int* __restrict__ asm_local, FrontType* frontB,
                              long front_total, int* sing, int do_extend)
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
    const int fl = slot % level_size;
    const int bb = slot / level_size;

    // Locate the front buffer for (batch bb, panel p).
    FrontType* front = frontB + (long)bb * front_total;
    const int p = plcols[lbegin + fl];
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
    lu_small_warp<FrontType, sub_group_size>(Fs, fsz, nc, sl, mask, sing);
    __syncwarp(mask);

    // 3. writeback factored panel.
    writeback_factored<FrontType, FrontType>(F, Fs, fsz, nc, uc, sl, sub_group_size);

    // 4. CB extend-add into parent front (skip for roots / when disabled).
    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    FrontType* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = sl; e < uc * uc; e += sub_group_size) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}

#ifdef CLS_SMALL_TF32_TC
__global__ void factor_small_tf32_tc(int lbegin, int level_size, int B, int front_area,
                                     const int* __restrict__ plcols,
                                     const int* __restrict__ front_off,
                                     const int* __restrict__ front_ptr,
                                     const int* __restrict__ ncols,
                                     const int* __restrict__ panel_parent,
                                     const int* __restrict__ asm_ptr,
                                     const int* __restrict__ asm_local,
                                     float* frontB, long front_total, int* sing, int do_extend)
{
    extern __shared__ unsigned char smem_small_tf32_raw[];
    float* smem = reinterpret_cast<float*>(smem_small_tf32_raw);

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_blk = blockDim.x >> 5;
    const int slot = blockIdx.x * warps_per_blk + warp_in_blk;
    if (slot >= level_size * B) return;

    const int fl = slot % level_size;
    const int bb = slot / level_size;
    float* front = frontB + (long)bb * front_total;
    const int p = plcols[lbegin + fl];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    float* F = front + front_off[p];
    float* Fs = smem + (long)warp_in_blk * front_area;
    constexpr unsigned mask = 0xffffffffu;

    for (int e = lane; e < fsz2; e += kWarpSize) Fs[e] = F[e];
    __syncwarp(mask);

    const bool use_tc = (fsz > 16 && uc >= 16 && nc >= 4 && nc <= 32);
    if (use_tc) {
        lu_panel_factor_warp(Fs, fsz, nc, lane, mask, sing);
#ifdef CLS_TF32_COLUMN_USOLVE
        u_panel_solve_warp_column_owned(Fs, fsz, nc, uc, lane, mask);
#else
        u_panel_solve_warp(Fs, fsz, nc, uc, lane, mask);
#endif
        trailing_update_mma_tf32_warp_shared(Fs, fsz, nc, uc, lane);
    } else {
        lu_small_warp<float, 32>(Fs, fsz, nc, lane, mask, sing);
    }
    __syncwarp(mask);

    writeback_factored<float, float>(F, Fs, fsz, nc, uc, lane, kWarpSize);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = lane; e < uc * uc; e += kWarpSize) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}
#endif

// =======================================================================================
//  MID tier  —  one block per (front, batch); whole front staged into shared
// =======================================================================================
//
// Used when kSmallFrontMax < max_fsz ≤ kMidFrontMax and the shared-memory budget fits
// (≤ 96 KB / block on sm_86). The 256-thread block stages the entire fsz × fsz front into
// dynamic shared once and runs all four phases against shared, avoiding O(nc) re-reads
// from global. Falls through to the big tier when shared is too tight (e.g. FP64 with
// fsz > ~88).
//
// Per-block flow (all four mid kernels):
//   1. stage_in_async copies F → Fs (cp.async on Ampere+, sync fallback otherwise).
//   2. factorize_front runs Phase 1 (panel LU), Phase 2 (U-solve), Phase 3 (trailing GEMM,
//      kernel-specific).
//   3. writeback_factored copies the factored L / U panel back to F (CB stays in Fs).
//   4. extend_add scatters Fs's uc × uc CB to the parent front via atomicAdd.
//
// Variants differ only in step 2's trailing GEMM and the shared-memory layout it requires.

// FP64 / FP32 staged-scalar trailing.
// Shared layout:  Fs[fsz_cap²] | sh_L[level_max_uc · level_max_nc] | sh_U[level_max_nc · level_max_uc].
template <typename T>
__global__ void factor_mid(int lbegin, int lend, const int* __restrict__ plcols,
                           const int* __restrict__ front_off,
                           const int* __restrict__ front_ptr,
                           const int* __restrict__ ncols,
                           const int* __restrict__ panel_parent,
                           const int* __restrict__ asm_ptr,
                           const int* __restrict__ asm_local, T* frontB,
                           long front_total, int* sing, int do_extend, int fsz_cap,
                           int level_max_nc, int level_max_uc)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;

    // (batch, panel) → front pointer.
    T* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;

    // Shared layout — Fs first (largest, alignment), then the L / U staging panels.
    extern __shared__ char smem_mid[];
    T* Fs   = reinterpret_cast<T*>(smem_mid);
    T* sh_L = Fs + (long)fsz_cap * fsz_cap;
    T* sh_U = sh_L + (long)level_max_uc * level_max_nc;

    stage_in_async<T>(Fs, F, fsz2, t, nt);
    __syncthreads();
    factorize_front<T>(Fs, fsz, nc, uc, t, nt, sing,
        [&] { trailing_update_staged<T>(Fs, fsz, nc, uc, t, nt, sh_L, sh_U); });
    __syncthreads();
    writeback_factored<T, T>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<T, T>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
}

// FP16 PTX mid trailing on a shared-resident front. The front stays as FP32 in shared memory;
// only the L/U panels are converted to __half before mma.sync. Non-root TC fronts drain the
// accumulator directly into the parent front, because the mid-tier C block is only consumed by
// extend-add and is not written back. Fronts that are too small or too skinny for Tensor Cores
// keep the scalar staged path inside the same kernel.
__global__ void factor_mid_fp16_ptx(int lbegin, int lend, const int* __restrict__ plcols,
                                    const int* __restrict__ front_off,
                                    const int* __restrict__ front_ptr,
                                    const int* __restrict__ ncols,
                                    const int* __restrict__ panel_parent,
                                    const int* __restrict__ asm_ptr,
                                    const int* __restrict__ asm_local, float* frontB,
                                    long front_total, int* sing, int do_extend, int fsz_cap,
                                    int level_max_nc, int level_max_uc, int ucp_max,
                                    int kp_max)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;

    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    const int par = panel_parent[p];
#ifdef CLS_FP16_BLOCKED_SHARED_TC
    const bool use_tc = (fsz > 48 && uc >= 32 && nc >= 8 && nc <= 32 && uc <= 256);
#else
    const bool use_tc = (fsz > 48 && uc >= 32 && nc >= 10 && nc <= 32 && uc <= 256);
#endif
    const bool use_fused_trailing =
        (use_tc && par >= 0 && do_extend && extend_add_allowed_for_uc(uc));
    float* Fp = use_fused_trailing ? (front + front_off[par]) : nullptr;
    const int pfsz = use_fused_trailing ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = use_fused_trailing ? asm_ptr[p] : 0;

    extern __shared__ char smem_mid_fp16_ptx[];
    float* Fs   = reinterpret_cast<float*>(smem_mid_fp16_ptx);
#ifndef CLS_FP16_BLOCKED_SHARED_TC
    __half* Lh  = reinterpret_cast<__half*>(Fs + (long)fsz_cap * fsz_cap);
    __half* Uh  = Lh + (long)ucp_max * kp_max;
#else
    (void)level_max_nc;
    (void)level_max_uc;
    (void)ucp_max;
    (void)kp_max;
    (void)Fp;
    (void)pfsz;
    (void)abase;
#endif

    stage_in_async<float>(Fs, F, fsz2, t, nt);
    __syncthreads();

#ifdef CLS_FP16_BLOCKED_SHARED_TC
    if (use_tc) {
        factorize_front_blocked_fp16(Fs, fsz, nc, t, nt, sing);
    } else {
        factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing,
            [&] { trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt); });
    }
#else
    factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing, [&] {
        if (use_fused_trailing) {
            trailing_update_mma_fp16_ptx<true>(Fs, fsz, nc, uc, Lh, Uh, t, nt, Fp, pfsz,
                                               asm_local, abase);
        } else if (use_tc) {
            trailing_update_mma_fp16_ptx<false>(Fs, fsz, nc, uc, Lh, Uh, t, nt);
        } else {
            trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt);
        }
    });
#endif

    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* parent_front = front + front_off[par];
    const int parent_fsz = front_ptr[par + 1] - front_ptr[par];
    const int asm_base = asm_ptr[p];
    extend_add<float, float>(parent_front, parent_fsz, Fs, fsz, nc, uc, asm_local, asm_base,
                             t, nt);
}

#ifdef CLS_MID_TF32_TC
// TF32 PTX mid trailing on a shared-resident FP32 front. This mirrors the FP16 mid TC path
// but keeps L/U staging in float and lets mma.sync's .tf32 ABI truncate multiplicands.
__global__ void factor_mid_tf32_ptx(int lbegin, int lend, const int* __restrict__ plcols,
                                    const int* __restrict__ front_off,
                                    const int* __restrict__ front_ptr,
                                    const int* __restrict__ ncols,
                                    const int* __restrict__ panel_parent,
                                    const int* __restrict__ asm_ptr,
                                    const int* __restrict__ asm_local, float* frontB,
                                    long front_total, int* sing, int do_extend, int fsz_cap,
                                    int ucp_max, int kp_max, int direct_shared_mode)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;

    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    const int par = panel_parent[p];
    const bool use_tc =
#if defined(CLS_MID_TF32_LOW_TC) && defined(CLS_MID_TF32_FORCE_BLOCKED)
        ((fsz > 48 && uc >= 32) ||
         (fsz > 16 && uc >= 16)) &&
        nc >= 4 && nc <= 32 && uc <= 256;
#elif defined(CLS_MID_TF32_LOW_TC)
        ((fsz > 48 && uc >= 32) ||
         (direct_shared_mode && fsz > 16 && uc >= 16)) &&
        nc >= 4 && nc <= 32 && uc <= 256;
#else
        (fsz > 48 && uc >= 32 && nc >= 8 && nc <= 32 && uc <= 256);
#endif

	extern __shared__ char smem_mid_tf32_ptx[];
	float* Fs  = reinterpret_cast<float*>(smem_mid_tf32_ptx);
#if !(defined(CLS_TF32_OZAKI_TC2) && defined(CLS_TF32_OZAKI_STAGE_DIRECT))
	(void)ucp_max;
	(void)kp_max;
#endif

	stage_in_async<float>(Fs, F, fsz2, t, nt);
	__syncthreads();

    const bool direct_shared_tc =
#ifdef CLS_MID_TF32_DIRECT_SHARED
        true;
#else
        (direct_shared_mode != 0);
	#endif
	    bool extend_fused = false;
#if defined(CLS_TF32_OZAKI_TC2) && defined(CLS_TF32_OZAKI_STAGE_DIRECT)
	    const bool use_ozaki_stage = direct_shared_tc && fsz_cap <= kMidSplitFrontMax;
	    unsigned* ozaki_Lh = reinterpret_cast<unsigned*>(Fs + (long)fsz_cap * fsz_cap);
	    unsigned* ozaki_Lt = ozaki_Lh + (long)ucp_max * kp_max;
	    unsigned* ozaki_Uh = ozaki_Lt + (long)ucp_max * kp_max;
	    unsigned* ozaki_Ut = ozaki_Uh + (long)kp_max * ucp_max;
#endif
	#ifdef CLS_MID_TF32_DIRECT_FUSE_EXTEND
	    float* parent_front_fuse = nullptr;
	    int parent_fsz_fuse = 0;
    int asm_base_fuse = 0;
    const bool can_fuse_extend =
        use_tc && direct_shared_tc && par >= 0 && do_extend && extend_add_allowed_for_uc(uc);
    if (can_fuse_extend) {
        parent_front_fuse = front + front_off[par];
        parent_fsz_fuse = front_ptr[par + 1] - front_ptr[par];
        asm_base_fuse = asm_ptr[p];
    }
    auto direct_tf32_trailing = [&] {
        if (can_fuse_extend) {
            trailing_update_mma_tf32_direct_shared<true>(
	                Fs, fsz, nc, uc, t, nt, parent_front_fuse, parent_fsz_fuse,
	                asm_local, asm_base_fuse);
	            extend_fused = true;
	        } else {
#if defined(CLS_TF32_OZAKI_TC2) && defined(CLS_TF32_OZAKI_STAGE_DIRECT)
	            if (use_ozaki_stage) {
	                trailing_update_mma_tf32_direct_shared_staged_ozaki(
	                    Fs, fsz, nc, uc, t, nt, ozaki_Lh, ozaki_Lt, ozaki_Uh, ozaki_Ut);
	            } else
#endif
	            trailing_update_mma_tf32_direct_shared(Fs, fsz, nc, uc, t, nt);
	        }
	    };
	#else
#if !(defined(CLS_TF32_OZAKI_TC2) && defined(CLS_TF32_OZAKI_STAGE_DIRECT))
	    (void)direct_shared_tc;
#endif
	    auto direct_tf32_trailing = [&] {
#if defined(CLS_TF32_OZAKI_TC2) && defined(CLS_TF32_OZAKI_STAGE_DIRECT)
	        if (use_ozaki_stage) {
	            trailing_update_mma_tf32_direct_shared_staged_ozaki(
	                Fs, fsz, nc, uc, t, nt, ozaki_Lh, ozaki_Lt, ozaki_Uh, ozaki_Ut);
	        } else
#endif
	        trailing_update_mma_tf32_direct_shared(Fs, fsz, nc, uc, t, nt);
	    };
	#endif
    auto direct_tf32_factorize = [&] {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
#ifdef CLS_TF32_COLUMN_USOLVE
        u_panel_solve_column_owned<float>(Fs, fsz, nc, uc, t, nt);
#else
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
#endif
        direct_tf32_trailing();
    };

    if (use_tc) {
#ifdef CLS_MID_TF32_DIRECT_SHARED
        direct_tf32_factorize();
#else
        if (direct_shared_mode) {
            direct_tf32_factorize();
        } else {
            factorize_front_blocked_tf32(Fs, fsz, nc, t, nt, sing);
        }
#endif
    } else {
        factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing,
            [&] { trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt); });
    }

    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    if (par < 0 || !do_extend || extend_fused || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* parent_front = front + front_off[par];
    const int parent_fsz = front_ptr[par + 1] - front_ptr[par];
    const int asm_base = asm_ptr[p];
    extend_add<float, float>(parent_front, parent_fsz, Fs, fsz, nc, uc, asm_local, asm_base,
                             t, nt);
}
#endif

#ifdef CLS_CUBLAS_TF32_TRAILING
// MID phase A for the cuBLAS TF32 trailing path. The shared front runs Phase 1 (panel LU)
// and Phase 2 (U-solve), then writes only the factorized panel rows and L strip back to global.
// The C block remains as the post-child-extend input; cuBLAS later applies C -= L * U in global,
// and factor_big_extend scatters that updated C block into the parent.
__global__ void factor_mid_cublas_phaseA(int lbegin, int lend,
                                         const int* __restrict__ plcols,
                                         const int* __restrict__ front_off,
                                         const int* __restrict__ front_ptr,
                                         const int* __restrict__ ncols,
                                         float* frontB, long front_total, int* sing,
                                         int fsz_cap)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;

    extern __shared__ char smem_mid_cublas_phaseA[];
    float* Fs = reinterpret_cast<float*>(smem_mid_cublas_phaseA);

    stage_in_async<float>(Fs, F, fsz2, t, nt);
    __syncthreads();

    if (fsz <= 48) {
        // Guard only. The dispatcher gates the cuBLAS path to larger mid ranges.
        lu_small_front<float>(Fs, fsz, nc, t, nt, sing);
        __syncthreads();
        for (long e = t; e < (long)fsz * fsz; e += nt) F[e] = Fs[e];
    } else {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
#ifdef CLS_TF32_COLUMN_USOLVE
        u_panel_solve_column_owned<float>(Fs, fsz, nc, uc, t, nt);
#else
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
#endif
        __syncthreads();
        writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);
    }
    (void)fsz_cap;
}

// Build the per-range device pointer arrays consumed by cublasSgemmGroupedBatched. The host
// grouped metadata is ordered by the same q positions as plcols, so the cublas call can pass
// d_cublas_*ptrs + lbegin * B with group_count = lend - lbegin.
__global__ void build_cublas_trailing_ptrs(int lbegin, int lend, int B, long front_total,
                                           const int* __restrict__ plcols,
                                           const int* __restrict__ front_off,
                                           const int* __restrict__ front_ptr,
                                           const int* __restrict__ ncols,
                                           float* frontB,
                                           float** Aptrs, float** Bptrs, float** Cptrs)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = frontB + (long)b * front_total + front_off[p];
    const long slot = (long)idx * B + b;
    Aptrs[slot] = F + nc;                         // U: row 0, col nc
    Bptrs[slot] = F + (long)nc * fsz;             // L: row nc, col 0
    Cptrs[slot] = F + (long)nc * fsz + nc;        // C: row nc, col nc
}
#endif

// =======================================================================================
//  BIG tier  —  one block per (front, batch); front stays in global memory
// =======================================================================================
//
// Used when max_fsz > kMidFrontMax or the mid shared budget overflows. The front is
// too large to stage (e.g. 128² FP32 = 64 KB just for Fs), so all phases operate directly on
// the global front buffer F. The block is widened (1024 threads) to expose enough
// parallelism for the larger uc² · nc trailing GEMM.
//
// Per-block flow:
//   1. factorize_front runs Phases 1+2+3 on F directly (no stage-in).
//   2. extend_add scatters the uc × uc CB from F into the parent front via atomicAdd.
//
// Tensor-Core variants still allocate a small shared scratch — only for the WMMA / PTX
// staging panels (and the WMMA fragment Csc), not for the front itself. Scalar fallback
// kicks in if nc > 32 or uc > 256.

// FP64 / FP32 scalar trailing on the global front.
template <typename T>
__global__ void factor_big(int lbegin, int lend, const int* __restrict__ plcols,
                           const int* __restrict__ front_off,
                           const int* __restrict__ front_ptr,
                           const int* __restrict__ ncols,
                           const int* __restrict__ panel_parent,
                           const int* __restrict__ asm_ptr,
                           const int* __restrict__ asm_local, T* frontB,
                           long front_total, int* sing, int do_extend)
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

    factorize_front<T>(F, fsz, nc, uc, t, nt, sing,
        [&] { trailing_update_scalar<T>(F, fsz, nc, uc, t, nt); });

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<T, T>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// FP64 / FP32 big front with L/U-staged trailing. Identical to factor_big except the rank-nc
// trailing stages the L (uc x nc) and U (nc x uc) panels into shared once (trailing_update_staged)
// instead of re-reading them from the global front ~uc times per output. The front itself stays
// global-resident; only the L/U panels are staged. Shared is sized by (level_max_nc, level_max_uc)
// — the caller gates on the 96 KB budget and falls back to factor_big when it overflows. Measured
// factor_big -23%, end-to-end factor -11% (USA fp64 B=64) / -5.7% (25K) for the filled big tier.
template <typename T>
__global__ void factor_big_staged(int lbegin, int lend, const int* __restrict__ plcols,
                                  const int* __restrict__ front_off,
                                  const int* __restrict__ front_ptr,
                                  const int* __restrict__ ncols,
                                  const int* __restrict__ panel_parent,
                                  const int* __restrict__ asm_ptr,
                                  const int* __restrict__ asm_local, T* frontB,
                                  long front_total, int* sing, int do_extend,
                                  int level_max_nc, int level_max_uc)
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

    extern __shared__ char smem_big_staged[];
    T* sh_L = reinterpret_cast<T*>(smem_big_staged);
    T* sh_U = sh_L + (long)level_max_uc * level_max_nc;

    const int par = panel_parent[p];
#ifdef CLS_FUSE_FP32_TRAIL_EXTEND
    const bool use_fused_trailing =
        (par >= 0 && do_extend && extend_add_allowed_for_uc(uc) &&
         fsz > 48 && nc <= 32 && uc <= 256);
    if (use_fused_trailing) {
        T* Fpf = front + front_off[par];
        const int pfszf = front_ptr[par + 1] - front_ptr[par];
        const int abasef = asm_ptr[p];
        factorize_front<T>(F, fsz, nc, uc, t, nt, sing,
            [&] { trailing_update_staged<T, true>(F, fsz, nc, uc, t, nt, sh_L, sh_U, Fpf, pfszf, asm_local, abasef); });
        return;
    }
#endif
    factorize_front<T>(F, fsz, nc, uc, t, nt, sing,
        [&] { trailing_update_staged<T>(F, fsz, nc, uc, t, nt, sh_L, sh_U); });

    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<T, T>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// ----- Multi-block big-front split (B=1 / underfilled-level path) ----------------------------
//
// When a big level has too few fronts to fill the GPU (level_size × B < num_SMs, e.g. the deep
// levels of a single-system solve), the fused factor_big runs one block per front — a handful
// of busy SMs while the rest idle, with the FLOP-heavy trailing GEMM serialized onto those few
// blocks. These three kernels split the work so the trailing fans out across many blocks:
//
//   factor_big_panel<T>     Phase 1 (panel LU) + Phase 2 (U-solve), one block per front. Small
//                           fronts (fsz ≤ 48) take the fused lu_small_front (panel + trailing in
//                           one pass) so the trailing kernel can skip them.
//   factor_big_trailing_mb  Phase 3 (scalar trailing) split into blockIdx.z element tiles of
//                           `elems_per_block` C entries — grid (front, batch, tiles).
//   factor_big_extend<T>    Phase 4 (extend-add) into the parent front, one block per front.
//
// Each C output element is owned by exactly one tile and L/U are read-only, so the tiles are
// race-free; the kernel-launch boundaries order panel → trailing → extend within the stream.
template <typename T>
__global__ void factor_big_panel(int lbegin, int lend, const int* __restrict__ plcols,
                                 const int* __restrict__ front_off,
                                 const int* __restrict__ front_ptr,
                                 const int* __restrict__ ncols, T* frontB,
                                 long front_total, int* sing)
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
    if (fsz <= 48) {
        lu_small_front<T>(F, fsz, nc, t, nt, sing);   // Phase 1 + 3 fused; trailing kernel skips these
    } else {
        lu_panel_factor<T>(F, fsz, nc, t, nt, sing);  // Phase 1
        u_panel_solve<T>(F, fsz, nc, uc, t, nt);      // Phase 2 (trailing deferred to the MB kernel)
    }
}

template <typename T>
__global__ void factor_big_trailing_mb(int lbegin, int lend, const int* __restrict__ plcols,
                                       const int* __restrict__ front_off,
                                       const int* __restrict__ front_ptr,
                                       const int* __restrict__ ncols, T* frontB,
                                       long front_total, int elems_per_block)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    if (fsz <= 48) return;                               // already trailed by the fused panel kernel
    const long total = (long)uc * uc;
    const long base = (long)blockIdx.z * elems_per_block;
    if (base >= total) return;
    const long end = (base + elems_per_block < total) ? base + elems_per_block : total;
    T* F = frontB + (long)blockIdx.y * front_total + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    // Scalar trailing C[ii,jj] -= Σ_k L[ii,k]·U[k,jj] over this block's element slice.
    for (long ee = base + t; ee < end; ee += nt) {
        const int ii = nc + (int)(ee / uc), jj = nc + (int)(ee % uc);
        T acc = T(0);
        for (int k = 0; k < nc; ++k) acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        F[(long)ii * fsz + jj] -= acc;
    }
}

template <typename T>
__global__ void factor_big_extend(int lbegin, int lend, const int* __restrict__ plcols,
                                  const int* __restrict__ front_off,
                                  const int* __restrict__ front_ptr,
                                  const int* __restrict__ ncols,
                                  const int* __restrict__ panel_parent,
                                  const int* __restrict__ asm_ptr,
                                  const int* __restrict__ asm_local, T* frontB,
                                  long front_total)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int par = panel_parent[p];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    if (par < 0 || !extend_add_allowed_for_uc(uc)) return;
    T* front = frontB + (long)blockIdx.y * front_total;
    T* F = front + front_off[p];
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    const int t = threadIdx.x, nt = blockDim.x;
    extend_add<T, T>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// FP16 PTX trailing on the global front (Precision::FP16). 512-thread block with
// __launch_bounds__(512, 2) so two blocks stay resident per SM on sm_86. Shared scratch is
// only the __half L/U staging panels; the inline-asm mma writes the accumulator to per-lane
// registers, so no Csc readback scratch is needed.
__global__ void __launch_bounds__(512, 2)
                                factor_big_fp16_ptx(int lbegin, int lend, const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ panel_parent,
                                const int* __restrict__ asm_ptr,
                                const int* __restrict__ asm_local, float* frontB,
                                long front_total, int* sing, int do_extend, int ucp_max,
                                int kp_max)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;

#ifdef CLS_FUSE_FP16_TRAIL_EXTEND
    const int par = panel_parent[p];
    const bool use_fused_trailing =
        (par >= 0 && do_extend && extend_add_allowed_for_uc(uc) &&
         fsz > 48 && nc <= 32 && uc <= 256);
    float* Fp = use_fused_trailing ? (front + front_off[par]) : nullptr;
    const int pfsz = use_fused_trailing ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = use_fused_trailing ? asm_ptr[p] : 0;
    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_fp16_ptx[];
            __half* Lh = reinterpret_cast<__half*>(smem_big_fp16_ptx);
            __half* Uh = Lh + (long)ucp_max * kp_max;
            if (use_fused_trailing) {
                trailing_update_mma_fp16_ptx<true>(F, fsz, nc, uc, Lh, Uh, t, nt, Fp, pfsz,
                                                   asm_local, abase);
            } else {
                trailing_update_mma_fp16_ptx<false>(F, fsz, nc, uc, Lh, Uh, t, nt);
            }
        }
    });

    if (use_fused_trailing || par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    {
        float* parent_front = front + front_off[par];
        const int parent_fsz = front_ptr[par + 1] - front_ptr[par];
        const int asm_base = asm_ptr[p];
        extend_add<float, float>(parent_front, parent_fsz, F, fsz, nc, uc, asm_local, asm_base,
                                 t, nt);
    }
#else
    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_fp16_ptx[];
            __half* Lh = reinterpret_cast<__half*>(smem_big_fp16_ptx);
            __half* Uh = Lh + (long)ucp_max * kp_max;
            trailing_update_mma_fp16_ptx(F, fsz, nc, uc, Lh, Uh, t, nt);
        }
    });

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
#endif
}

// TF32 PTX trailing on the global front (Precision::TF32). The 512-thread block plus
// __launch_bounds__(512, 2) caps register usage so two blocks resident per SM is reachable
// on sm_86; the inline-asm mma.sync writes directly to per-lane accumulators so no Csc
// scratch is needed.
__global__ void __launch_bounds__(512, 2)
                                factor_big_tf32_ptx(int lbegin, int lend, const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ panel_parent,
                                const int* __restrict__ asm_ptr,
                                const int* __restrict__ asm_local, float* frontB,
                                long front_total, int* sing, int do_extend, int ucp_max,
                                int kp_max)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;

#ifdef CLS_FUSE_TF32_TRAIL_EXTEND
    const int par = panel_parent[p];
    const bool use_fused_trailing =
        (par >= 0 && do_extend && extend_add_allowed_for_uc(uc) &&
         fsz > 48 && nc <= 32 && uc <= 256);
    float* Fp = use_fused_trailing ? (front + front_off[par]) : nullptr;
    const int pfsz = use_fused_trailing ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = use_fused_trailing ? asm_ptr[p] : 0;
    auto tf32_big_trailing = [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_tf32_ptx[];
            float* Ltf = reinterpret_cast<float*>(smem_big_tf32_ptx);
            float* Utf = Ltf + (long)ucp_max * kp_max;
            if (use_fused_trailing) {
                trailing_update_mma_tf32_ptx<true>(F, fsz, nc, uc, Ltf, Utf, t, nt, Fp, pfsz,
                                                   asm_local, abase);
            } else {
                trailing_update_mma_tf32_ptx<false>(F, fsz, nc, uc, Ltf, Utf, t, nt);
            }
        }
    };
#ifdef CLS_TF32_GLOBAL_COLUMN_USOLVE
    lu_panel_factor<float>(F, fsz, nc, t, nt, sing);
    u_panel_solve_column_owned<float>(F, fsz, nc, uc, t, nt);
    tf32_big_trailing();
#else
    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] { tf32_big_trailing(); });
#endif

    if (use_fused_trailing || par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    {
        float* parent_front = front + front_off[par];
        const int parent_fsz = front_ptr[par + 1] - front_ptr[par];
        const int asm_base = asm_ptr[p];
        extend_add<float, float>(parent_front, parent_fsz, F, fsz, nc, uc, asm_local, asm_base,
                                 t, nt);
    }
#else
    auto tf32_big_trailing = [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_tf32_ptx[];
            float* Ltf = reinterpret_cast<float*>(smem_big_tf32_ptx);
            float* Utf = Ltf + (long)ucp_max * kp_max;
            trailing_update_mma_tf32_ptx(F, fsz, nc, uc, Ltf, Utf, t, nt);
        }
    };
#ifdef CLS_TF32_GLOBAL_COLUMN_USOLVE
    lu_panel_factor<float>(F, fsz, nc, t, nt, sing);
    u_panel_solve_column_owned<float>(F, fsz, nc, uc, t, nt);
    tf32_big_trailing();
#else
    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] { tf32_big_trailing(); });
#endif

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
#endif
}

#ifdef CLS_FP16_BLOCKED_SHARED_TC
// Big-low shared-resident blocked FP16 path. The front remains FP32 in shared memory; the
// right-looking block updates feed FP16 multiplicands to Tensor Cores with FP32 accumulation.
__global__ void factor_big_shared_fp16_blocked(int lbegin, int lend,
                                               const int* __restrict__ plcols,
                                               const int* __restrict__ front_off,
                                               const int* __restrict__ front_ptr,
                                               const int* __restrict__ ncols,
                                               const int* __restrict__ panel_parent,
                                               const int* __restrict__ asm_ptr,
                                               const int* __restrict__ asm_local,
                                               float* frontB, long front_total, int* sing,
                                               int do_extend, int fsz_cap)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    const bool use_tc = (fsz > 48 && uc >= 32 && nc >= 8 && nc <= 32 && uc <= 256);

    extern __shared__ char smem_big_shared_fp16[];
    float* Fs = reinterpret_cast<float*>(smem_big_shared_fp16);
    stage_in_async<float>(Fs, F, fsz2, t, nt);
    __syncthreads();

    if (use_tc) {
        factorize_front_blocked_fp16(Fs, fsz, nc, t, nt, sing);
    } else {
        factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing,
            [&] { trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt); });
    }

    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
    (void)fsz_cap;
}
#endif

#ifdef CLS_BIG_TF32_BLOCKED_TC
// Big-low shared-resident blocked TF32 path. It targets fsz<=159 ranges isolated by
// CLS_BIG_LOW_SPLIT, where the full dense front fits in the 99 KiB opt-in shared budget.
__global__ void factor_big_shared_tf32_blocked(int lbegin, int lend,
                                               const int* __restrict__ plcols,
                                               const int* __restrict__ front_off,
                                               const int* __restrict__ front_ptr,
                                               const int* __restrict__ ncols,
                                               const int* __restrict__ panel_parent,
                                               const int* __restrict__ asm_ptr,
                                               const int* __restrict__ asm_local,
                                               float* frontB, long front_total, int* sing,
                                               int do_extend, int fsz_cap)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    const bool use_tc = (fsz > 48 && uc >= 32 && nc >= 8 && nc <= 32 && uc <= 256);

    extern __shared__ char smem_big_shared_tf32[];
    float* Fs = reinterpret_cast<float*>(smem_big_shared_tf32);
    stage_in_async<float>(Fs, F, fsz2, t, nt);
    __syncthreads();

    if (use_tc) {
        factorize_front_blocked_tf32(Fs, fsz, nc, t, nt, sing);
    } else {
        factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing,
            [&] { trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt); });
    }

    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
    (void)fsz_cap;
}

// Big-front right-looking blocked TF32 path. Unlike factor_big_tf32_ptx, this uses Tensor Cores
// for the block update after each pivot block, so TC covers remaining panel columns as well as C.
__global__ void __launch_bounds__(512, 2)
                                factor_big_tf32_blocked_ptx(int lbegin, int lend,
                                const int* __restrict__ plcols,
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
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const bool use_tc = (fsz > 48 && uc >= 32 && nc >= 8 && nc <= 32 && uc <= 256);

    if (use_tc) {
        factorize_front_blocked_tf32(F, fsz, nc, t, nt, sing);
    } else {
        factorize_front<float>(F, fsz, nc, uc, t, nt, sing,
            [&] { trailing_update_scalar<float>(F, fsz, nc, uc, t, nt); });
    }

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}
#endif

}  // namespace
}  // namespace custom_linear_solver
