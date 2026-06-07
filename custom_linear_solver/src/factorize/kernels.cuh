#pragma once

// FACTORIZE — per-tier __global__ kernel entry points.
//
// Internal — included only by multifrontal.cu (single TU; CUDA_SEPARABLE_COMPILATION OFF).
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
//   small  | 8 warp | smem  | factor_small<T>         | lu_small_warp (fused Phase 1+3)
//   mid    | 64..256| smem  | factor_mid<T>           | trailing_update_staged<T>
//          |        |       | factor_mid_tc           | trailing_update_wmma_fp16
//          |        |       | factor_mid_tf32_wmma    | trailing_update_wmma_tf32
//          |        |       | factor_mid_tf32_ptx<K>  | trailing_update_mma_tf32{_k4}_ptx
//   big    | 1024   | gmem  | factor_big<T>           | trailing_update_scalar<T>
//          | 1024   | gmem  | factor_big_tc           | trailing_update_wmma_fp16
//          | 1024   | gmem  | factor_big_tf32_wmma    | trailing_update_wmma_tf32
//          | 512    | gmem  | factor_big_tf32_ptx     | trailing_update_mma_tf32_ptx
//
// Why a separate kernel symbol per variant:
//   * The <T> templates parameterise over the front element type (double / float).
//   * The Tensor-Core variants differ in shared-memory layout: FP16 stages __half panels,
//     TF32 stages float panels, and the PTX variants omit the Csc fragment-readback scratch
//     because the inline-asm mma.sync writes the accumulator straight to per-lane registers
//     (the WMMA wrappers cannot — wmma::store_matrix_sync is the only public path out of an
//     accumulator fragment, so they need Csc).
//   * factor_mid_tf32_ptx<K> templates the K=4 / K=8 mma shapes into a single kernel because
//     only the inline-asm form differs; smem layout and block geometry are identical.
//   * factor_big_tf32_ptx is split from factor_big_tf32_wmma not just for the trailing
//     function but because it carries __launch_bounds__(512, 2) and a different block size,
//     neither of which can be selected inside one kernel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "factorize/phases.cuh"

namespace custom_linear_solver {
namespace {

// =======================================================================================
//  SMALL tier  —  one warp per (front, batch); W warps per block
// =======================================================================================
//
// Used when the level's max_fsz ≤ SMALL_THRESH = 32 (see factorize/dispatch.cuh).
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
template <typename FT>
__global__ void factor_small(int lbegin, int level_size, int B, int fsz2cap,
                              const int* __restrict__ plcols,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ panel_parent,
                              const int* __restrict__ asm_ptr,
                              const int* __restrict__ asm_local, FT* frontB,
                              long front_total, int* sing, int do_extend)
{
    extern __shared__ unsigned char smem_sw_raw[];
    FT* smem_sw = reinterpret_cast<FT*>(smem_sw_raw);

    // Identify this warp's (front, batch) — one warp per (front in this level, batch index).
    const int warp_in_blk = threadIdx.x >> 5;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_global >= level_size * B) return;
    const int fl = warp_global % level_size;
    const int bb = warp_global / level_size;

    // Locate the front buffer for (batch bb, panel p).
    FT* front = frontB + (long)bb * front_total;
    const int p = plcols[lbegin + fl];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    FT* F = front + front_off[p];

    // Per-warp shared scratch for this front (slot fsz2cap reserved per warp).
    FT* Fs = smem_sw + (long)warp_in_blk * fsz2cap;

    // 1. global F → per-warp Fs.
    for (int e = lane; e < fsz2; e += 32) Fs[e] = F[e];
    __syncwarp();

    // 2. fused panel LU + trailing on Fs.
    lu_small_warp<FT>(Fs, fsz, nc, lane, sing);
    __syncwarp();

    // 3. writeback factored panel.
    writeback_factored<FT, FT>(F, Fs, fsz, nc, uc, lane, 32);

    // 4. CB extend-add into parent front (skip for roots / when disabled).
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

// =======================================================================================
//  MID tier  —  one block per (front, batch); whole front staged into shared
// =======================================================================================
//
// Used when SMALL_THRESH < max_fsz ≤ MID_THRESH = 128 and the shared-memory budget fits
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
    if (par < 0 || !do_extend) return;
    __syncthreads();
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<T, T>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
}

// FP16 WMMA trailing (Precision::FP16). Stages L / U into __half panels for
// wmma::mma_sync m16n16k16, accumulates in FP32, reads the accumulator back via the per-warp
// Csc smem scratch (wmma::store_matrix_sync is the only public way to read the fragment),
// and subtracts from Fs with bounds checks.
// Shared layout:  Lh[ucp_max · 32] (__half) | Uh[32 · ucp_max] (__half)
//               | Csc[(nt/32) · 256] (float)  | Fs[fsz_cap²] (float).
__global__ void factor_mid_tc(int lbegin, int lend, const int* __restrict__ plcols,
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
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;

    extern __shared__ char smem_mid_tc[];
    __half* Lh  = reinterpret_cast<__half*>(smem_mid_tc);
    __half* Uh  = Lh + (long)ucp_max * 32;
    float*  Csc = reinterpret_cast<float*>(Uh + (long)32 * ucp_max);
    float*  Fs  = Csc + (nt / 32) * 256;

    stage_in_async<float>(Fs, F, fsz2, t, nt);
    __syncthreads();
    factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing, [&] {
        // WMMA tile constraints: 16×16×16 needs nc ≤ 32, uc ≤ 256. Anything larger falls
        // back to the scalar dot-product trailing on Fs.
        if (nc <= 32 && uc <= 256) {
            trailing_update_wmma_fp16(Fs, fsz, nc, uc, Lh, Uh, Csc, ucp_max, t, nt);
        } else {
            trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt);
        }
    });
    __syncthreads();
    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
}

// TF32 WMMA trailing (Precision::TF32_WMMA). Same shape as factor_mid_tc but the staging
// panels hold FP32 (TF32 inputs are bit-aliased FP32), KP aligns to 8 not 16, and Csc is
// still required for the fragment readback.
// Shared layout:  Ltf[ucp_max · kp_max] | Utf[kp_max · ucp_max]
//               | Csc[(nt/32) · 256]    | Fs[fsz_cap²]   (all float).
__global__ void factor_mid_tf32_wmma(int lbegin, int lend, const int* __restrict__ plcols,
                                      const int* __restrict__ front_off,
                                      const int* __restrict__ front_ptr,
                                      const int* __restrict__ ncols,
                                      const int* __restrict__ panel_parent,
                                      const int* __restrict__ asm_ptr,
                                      const int* __restrict__ asm_local, float* frontB,
                                      long front_total, int* sing, int do_extend, int ucp_max,
                                      int kp_max, int fsz_cap)
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

    extern __shared__ char smem_mid_tf32_wmma[];
    float* Ltf = reinterpret_cast<float*>(smem_mid_tf32_wmma);
    float* Utf = Ltf + (long)ucp_max * kp_max;
    float* Csc = Utf + (long)kp_max * ucp_max;
    float* Fs  = Csc + (nt / 32) * 256;

    stage_in_async<float>(Fs, F, fsz2, t, nt);
    __syncthreads();
    factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing, [&] {
        if (nc <= 32 && uc <= 256) {
            trailing_update_wmma_tf32(Fs, fsz, nc, uc, Ltf, Utf, Csc, t, nt);
        } else {
            trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt);
        }
    });
    __syncthreads();
    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
}

// TF32 PTX trailing (Precision::TF32). Calls mma.sync via inline asm and subtracts from Fs
// directly from per-lane accumulator registers, so no Csc scratch is required.
//
//   K = 8 (factor_mid_tf32_ptx<8>) — mma.m16n8k8. Default branch of the per-level hybrid.
//   K = 4 (factor_mid_tf32_ptx<4>) — mma.m16n8k4. Picked by dispatch.cuh when a level's
//                                    max nc satisfies round_up(nc, 4) < round_up(nc, 8)
//                                    (i.e. nc % 8 ∈ {1..4}), halving K-padding waste.
//
// Shared layout:  Ltf[ucp_max · kp_max] | Utf[kp_max · ucp_max] | Fs[fsz_cap²]  (all float).
template <int K>
__global__ void factor_mid_tf32_ptx(int lbegin, int lend, const int* __restrict__ plcols,
                                    const int* __restrict__ front_off,
                                    const int* __restrict__ front_ptr,
                                    const int* __restrict__ ncols,
                                    const int* __restrict__ panel_parent,
                                    const int* __restrict__ asm_ptr,
                                    const int* __restrict__ asm_local, float* frontB,
                                    long front_total, int* sing, int do_extend, int ucp_max,
                                    int kp_max, int fsz_cap)
{
    static_assert(K == 4 || K == 8, "factor_mid_tf32_ptx: K must be 4 or 8");

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

    extern __shared__ char smem_mid_tf32_ptx[];
    float* Ltf = reinterpret_cast<float*>(smem_mid_tf32_ptx);
    float* Utf = Ltf + (long)ucp_max * kp_max;
    float* Fs  = Utf + (long)kp_max * ucp_max;

    stage_in_async<float>(Fs, F, fsz2, t, nt);
    __syncthreads();
    factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing, [&] {
        if (nc <= 32 && uc <= 256) {
            if constexpr (K == 4) {
                trailing_update_mma_tf32_k4_ptx(Fs, fsz, nc, uc, Ltf, Utf, t, nt);
            } else {
                trailing_update_mma_tf32_ptx(Fs, fsz, nc, uc, Ltf, Utf, t, nt);
            }
        } else {
            trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt);
        }
    });
    __syncthreads();
    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
}

// =======================================================================================
//  BIG tier  —  one block per (front, batch); front stays in global memory
// =======================================================================================
//
// Used when max_fsz > MID_THRESH = 128 or the mid shared budget overflows. The front is
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
    if (par < 0 || !do_extend) return;
    __syncthreads();
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<T, T>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// FP16 WMMA trailing on the global front (Precision::FP16). Shared scratch is for the
// __half L/U staging panels plus the Csc fragment readback only.
__global__ void factor_big_tc(int lbegin, int lend, const int* __restrict__ plcols,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ panel_parent,
                              const int* __restrict__ asm_ptr,
                              const int* __restrict__ asm_local, float* frontB,
                              long front_total, int* sing, int do_extend, int ucp_max)
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

    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_tc[];
            __half* Lh  = reinterpret_cast<__half*>(smem_big_tc);
            __half* Uh  = Lh + (long)ucp_max * 32;
            float*  Csc = reinterpret_cast<float*>(Uh + (long)32 * ucp_max);
            trailing_update_wmma_fp16(F, fsz, nc, uc, Lh, Uh, Csc, ucp_max, t, nt);
        }
    });

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// TF32 WMMA trailing on the global front (Precision::TF32_WMMA). Same Csc-via-wmma::store
// flow as the FP16 big variant, with float staging panels instead of __half.
__global__ void factor_big_tf32_wmma(int lbegin, int lend, const int* __restrict__ plcols,
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

    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_tf32_wmma[];
            float* Ltf = reinterpret_cast<float*>(smem_big_tf32_wmma);
            float* Utf = Ltf + (long)ucp_max * kp_max;
            float* Csc = Utf + (long)kp_max * ucp_max;
            trailing_update_wmma_tf32(F, fsz, nc, uc, Ltf, Utf, Csc, t, nt);
        }
    });

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
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

    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_tf32_ptx[];
            float* Ltf = reinterpret_cast<float*>(smem_big_tf32_ptx);
            float* Utf = Ltf + (long)ucp_max * kp_max;
            trailing_update_mma_tf32_ptx(F, fsz, nc, uc, Ltf, Utf, t, nt);
        }
    });

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

}  // namespace
}  // namespace custom_linear_solver
