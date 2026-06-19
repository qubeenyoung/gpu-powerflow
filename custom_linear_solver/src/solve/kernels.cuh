#pragma once

// SOLVE — kernel entry points (__global__).
//
// Internal — included into the factor/Solve driver TUs (single TU;
// CUDA_SEPARABLE_COMPILATION OFF).
//
// Two kernel families × two directions:
//
//   tier    | block        | front location  | kernels (forward / backward)
//   --------+--------------+-----------------+------------------------------------
//   small    | 8 warps      | global (L1)     | SolveFwdSmall<T>,
//   SolveBwdSmall<T> regular | 64-256 thr   | global          | SolveFwd<T>,
//   SolveBwd<T>
//
// The regular kernels are NOT tier-split into small/mid/big like factor — the
// Solve work per front is much lighter than factor (no rank-nc GEMM; just
// substitution + CB row update), so a single block-per-front kernel with
// caller-tuned thread count covers all max_fsz > kSmallFrontMax (i.e. the small
// / mid / big factor tiers all share one Solve regular kernel).
//
// Each kernel is a thin orchestrator composing the device building blocks in
// phases.cuh:
//   forward:  FwdSubstitute  → sync  → FwdCbUpdate
//   backward: BwdLoadRhsAndX → sync → BwdCbSubtract → sync →
//   BwdSubstitute

#include <cuda_runtime.h>

#include "solve/phases.cuh"

namespace custom_linear_solver {
namespace {

// =======================================================================================
//  SMALL tier — one WARP per (front, batch), 8 warps per block
// =======================================================================================
//
// The bottom Etree levels (tens of thousands of small fronts) dominate the
// batched Solve. Block-per-front would launch one 32-thread block per
// (front, batch): high launch/scheduling overhead and poor SM packing. These
// kernels pack W warps per block (one (front, batch) per warp) and sync with
// __syncwarp.

// sub_group_size = sub-group lane count (8 / 16 / 32). One sub-group owns one
// (front, batch); fronts_per_warp = 32/sub_group_size fronts pack per warp.
// sub_group_size=32 is the classic one-warp-per-front form. The dispatcher
// picks sub_group_size from the level's max_fsz.
template <typename T, int sub_group_size>
__global__ void SolveFwdSmall(int lbegin, int level_size, int B, int slab,
                              const int* __restrict__ plcols,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ front_rows,
                              const T* frontB, T* yB, long front_total, int n) {
  constexpr int fronts_per_warp = 32 / sub_group_size;
  extern __shared__ unsigned char fsm_raw[];
  T* slabs = reinterpret_cast<T*>(fsm_raw);
  const int warp_in_blk = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int sg = lane / sub_group_size, sl = lane % sub_group_size;
  const unsigned mask = (sub_group_size == 32) ? 0xffffffffu
                                               : (((1u << sub_group_size) - 1u)
                                                  << (sg * sub_group_size));
  const int warps_per_blk = blockDim.x >> 5;
  const int slot =
      (blockIdx.x * warps_per_blk + warp_in_blk) * fronts_per_warp + sg;
  if (slot >= level_size * B) return;
  const int fl = slot % level_size, bb = slot / level_size;
  const T* front = frontB + (long)bb * front_total;
  T* y = yB + (long)bb * n;
  const int p = plcols[lbegin + fl];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  T* sh_piv = slabs + (long)(warp_in_blk * fronts_per_warp + sg) * slab;

  // 1. Forward substitution: solve L_pp * sh_piv = b for this front's pivots.
  FwdSubstitute<T, sub_group_size>(F, fsz, nc, fr, y, sh_piv, sl, mask);
  __syncwarp(mask);

  // 2. Push the contribution-block rows into the parent RHS (atomicAdd).
  FwdCbUpdate<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, sl,
                 /*nt=*/sub_group_size);
}

template <typename T, int sub_group_size>
__global__ void SolveBwdSmall(int lbegin, int level_size, int B, int slab,
                              const int* __restrict__ plcols,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ front_rows,
                              const T* frontB, T* yB, long front_total, int n) {
  constexpr int fronts_per_warp = 32 / sub_group_size;
  extern __shared__ unsigned char bsm_raw[];
  T* slabs = reinterpret_cast<T*>(bsm_raw);
  const int warp_in_blk = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int sg = lane / sub_group_size, sl = lane % sub_group_size;
  const unsigned mask = (sub_group_size == 32) ? 0xffffffffu
                                               : (((1u << sub_group_size) - 1u)
                                                  << (sg * sub_group_size));
  const int warps_per_blk = blockDim.x >> 5;
  const int slot =
      (blockIdx.x * warps_per_blk + warp_in_blk) * fronts_per_warp + sg;
  if (slot >= level_size * B) return;
  const int fl = slot % level_size, bb = slot / level_size;
  const T* front = frontB + (long)bb * front_total;
  T* y = yB + (long)bb * n;
  const int p = plcols[lbegin + fl];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int cb = fsz - nc;
  T* rhs =
      slabs + (long)(warp_in_blk * fronts_per_warp + sg) * slab;  // [0, nc)
  T* xsh = rhs + kMaxPivotColumns;                                // [0, cb)

  // 1. Gather rhs[] and the x cache from y.
  BwdLoadRhsAndX<T>(y, fr, nc, cb, rhs, xsh, sl, /*nt=*/sub_group_size);
  __syncwarp(mask);

  // 2. Subtract the contribution-block product from rhs.
  BwdCbSubtract<T, sub_group_size>(F, fsz, nc, cb, xsh, rhs, sl,
                                   /*width=*/sub_group_size, mask);
  __syncwarp(mask);

  // 3. Back-substitute U_pp * x = rhs, writing x back to y.
  BwdSubstitute<T, sub_group_size>(F, fsz, nc, fr, y, rhs, sl, mask);
}

// =======================================================================================
//  REGULAR tier — block per (front, batch); thread count caller-chosen (64 /
//  128 / 256)
// =======================================================================================
//
// One block per (front, batch). The substitution (Phase 1 of fwd, Phase 3 of
// bwd) is driven by one warp inside the block — the recurrence in k is
// inherently sequential and a single warp is fast enough. The CB update / CB
// subtract (which IS parallelizable) is spread across all `nt` block threads.
// The whole front lives in global memory — Solve doesn't have a shared-resident
// variant because the work per front is much lighter than factor.

template <typename T, int NC>
__device__ __forceinline__ void FwdSubstituteFixedRegular(const T* F, int fsz,
                                                          const int* fr, T* y,
                                                          T* sh_piv, int t) {
  if constexpr (NC <= 8) {
    if (t < 8)
      FwdSubstituteFixed<T, NC, 8>(F, fsz, fr, y, sh_piv, t, 0x000000ffu);
  } else if constexpr (NC <= 16) {
    if (t < 16)
      FwdSubstituteFixed<T, NC, 16>(F, fsz, fr, y, sh_piv, t, 0x0000ffffu);
  } else {
    if (t < 32) FwdSubstituteFixed<T, NC>(F, fsz, fr, y, sh_piv, t);
  }
}

template <typename T, int NC>
__device__ __forceinline__ void BwdSubstituteFixedRegular(const T* F, int fsz,
                                                          const int* fr, T* y,
                                                          const T* rhs, int t) {
  if constexpr (NC <= 8) {
    if (t < 8) BwdSubstituteFixed<T, NC, 8>(F, fsz, fr, y, rhs, t, 0x000000ffu);
  } else if constexpr (NC <= 16) {
    if (t < 16)
      BwdSubstituteFixed<T, NC, 16>(F, fsz, fr, y, rhs, t, 0x0000ffffu);
  } else {
    if (t < 32) BwdSubstituteFixed<T, NC>(F, fsz, fr, y, rhs, t);
  }
}

template <typename T>
__global__ void SolveFwd(int lbegin, int lend, const int* __restrict__ plcols,
                         const int* __restrict__ front_off,
                         const int* __restrict__ front_ptr,
                         const int* __restrict__ ncols,
                         const int* __restrict__ front_rows, const T* frontB,
                         T* yB, long front_total, int n) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  __shared__ T sh_piv[kMaxPivotColumns];

  // Phase 1 — stage the L panel (rows [0,fsz) x cols [0,nc), compact stride nc)
  // into shared with a coalesced load; the compact load packs nc-contiguous
  // runs so both phases read bank-friendly shared instead of strided global.
  extern __shared__ unsigned char fwd_smem_raw[];
  T* Lsh = reinterpret_cast<T*>(fwd_smem_raw);
  for (int e = t; e < fsz * nc; e += nt)
    Lsh[e] = F[(long)(e / nc) * fsz + (e % nc)];
  __syncthreads();

  // Phase 2 — panel substitution (one warp).
  if (t < 32) FwdSubstituteSh<T>(Lsh, /*ld=*/nc, nc, fr, y, sh_piv, /*lane=*/t);
  __syncthreads();

  // Phase 3 — CB rows update across all threads.
  FwdCbUpdateSh<T>(Lsh, /*ld=*/nc, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t,
                   nt);
}

template <typename T>
__global__ void SolveBwd(int lbegin, int lend, const int* __restrict__ plcols,
                         const int* __restrict__ front_off,
                         const int* __restrict__ front_ptr,
                         const int* __restrict__ ncols,
                         const int* __restrict__ front_rows, const T* frontB,
                         T* yB, long front_total, int n) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  const int cb = fsz - nc;

  // Stage the top nc rows of F (the U pivot block + U panel; F[0, nc*fsz) is
  // contiguous, so the load is fully coalesced) into shared, then run both
  // U-using phases from shared. xsh (x cache, cb entries) follows it in dynamic
  // shared; rhs is a static shared.
  extern __shared__ unsigned char bwd_smem_raw[];
  T* Ush = reinterpret_cast<T*>(bwd_smem_raw);  // nc rows × fsz, stride fsz
  T* xsh = Ush + (long)nc * fsz;                // x cache (cb)
  __shared__ T rhs[kMaxPivotColumns];

  // Phase 1 — stage U rows (coalesced) + load rhs[] and x cache from y.
  for (int e = t; e < nc * fsz; e += nt) Ush[e] = F[e];
  BwdLoadRhsAndX<T>(y, fr, nc, cb, rhs, xsh, t, nt);
  __syncthreads();
  // Phase 2 — CB contribution to rhs. Threads with t < nc each compute one rhs
  // entry.
  BwdCbSubtract<T>(Ush, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
  __syncthreads();
  // Phase 3 — panel substitution (one warp), writes x back to y[fr[0..nc)].
  if (t < 32) BwdSubstitute<T>(Ush, fsz, nc, fr, y, rhs, /*lane=*/t);
}

// Non-staged fwd for large fronts whose L panel (fsz x nc) would exceed the
// default shared cap if staged. Reads F directly: staging is only a small-front
// coalescing win, and large fronts coalesce anyway. Same math as SolveFwd
// without the Lsh copy (shared = 0).
template <typename T>
__global__ void SolveFwdNostage(
    int lbegin, int lend, const int* __restrict__ plcols,
    const int* __restrict__ front_off, const int* __restrict__ front_ptr,
    const int* __restrict__ ncols, const int* __restrict__ front_rows,
    const T* frontB, T* yB, long front_total, int n) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  __shared__ T sh_piv[kMaxPivotColumns];

  // 1. Panel substitution (one warp).
  if (t < 32) FwdSubstitute<T>(F, fsz, nc, fr, y, sh_piv, /*lane=*/t);
  __syncthreads();

  // 2. CB rows update across all threads.
  FwdCbUpdate<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t, nt);
}

// Non-staged bwd counterpart (shared = x cache only, bounded by cb; the U-panel
// stays in global).
template <typename T>
__global__ void SolveBwdNostage(
    int lbegin, int lend, const int* __restrict__ plcols,
    const int* __restrict__ front_off, const int* __restrict__ front_ptr,
    const int* __restrict__ ncols, const int* __restrict__ front_rows,
    const T* frontB, T* yB, long front_total, int n) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  const int cb = fsz - nc;
  extern __shared__ unsigned char bwd_nostage_raw[];
  T* xsh = reinterpret_cast<T*>(bwd_nostage_raw);  // cb entries
  __shared__ T rhs[kMaxPivotColumns];

  // 1. Gather rhs[] and x cache from y.
  BwdLoadRhsAndX<T>(y, fr, nc, cb, rhs, xsh, t, nt);
  __syncthreads();

  // 2. CB contribution to rhs.
  BwdCbSubtract<T>(F, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
  __syncthreads();

  // 3. Panel substitution (one warp), writes x back to y.
  if (t < 32) BwdSubstitute<T>(F, fsz, nc, fr, y, rhs, /*lane=*/t);
}

template <typename T, int NC>
__global__ void SolveFwdFixedNc(
    int lbegin, int lend, const int* __restrict__ plcols,
    const int* __restrict__ front_off, const int* __restrict__ front_ptr,
    const int* __restrict__ ncols, const int* __restrict__ front_rows,
    const T* frontB, T* yB, long front_total, int n) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  __shared__ T sh_piv[kMaxPivotColumns];

  // Compile-time-unrolled path when this front's nc matches NC; otherwise the
  // generic runtime-nc path.
  if (nc == NC) {
    FwdSubstituteFixedRegular<T, NC>(F, fsz, fr, y, sh_piv, t);
    __syncthreads();
    FwdCbUpdateFixed<T, NC>(F, fsz, /*cb=*/fsz - NC, fr, y, sh_piv, t, nt);
  } else {
    if (t < 32) FwdSubstitute<T>(F, fsz, nc, fr, y, sh_piv, /*lane=*/t);
    __syncthreads();
    FwdCbUpdate<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t, nt);
  }
}

template <typename T, int NC>
__global__ void SolveFwdExactNc(int lbegin, int lend,
                                const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ front_rows,
                                const T* frontB, T* yB, long front_total,
                                int n) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  __shared__ T sh_piv[kMaxPivotColumns];

  FwdSubstituteFixedRegular<T, NC>(F, fsz, fr, y, sh_piv, t);
  __syncthreads();
  FwdCbUpdateFixed<T, NC>(F, fsz, /*cb=*/fsz - NC, fr, y, sh_piv, t, nt);
}

template <typename T, int NC>
__global__ void SolveBwdFixedNc(
    int lbegin, int lend, const int* __restrict__ plcols,
    const int* __restrict__ front_off, const int* __restrict__ front_ptr,
    const int* __restrict__ ncols, const int* __restrict__ front_rows,
    const T* frontB, T* yB, long front_total, int n) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  const int cb = fsz - nc;

  extern __shared__ unsigned char xsh_raw[];
  T* xsh = reinterpret_cast<T*>(xsh_raw);
  __shared__ T rhs[kMaxPivotColumns];

  // Compile-time-unrolled path when this front's nc matches NC; otherwise the
  // generic runtime-nc path.
  if (nc == NC) {
    BwdLoadRhsAndXFixed<T, NC>(y, fr, fsz - NC, rhs, xsh, t, nt);
    __syncthreads();
    BwdCbSubtractFixed<T, NC>(F, fsz, cb, xsh, rhs, t, /*width=*/nt);
    __syncthreads();
    BwdSubstituteFixedRegular<T, NC>(F, fsz, fr, y, rhs, t);
  } else {
    BwdLoadRhsAndX<T>(y, fr, nc, cb, rhs, xsh, t, nt);
    __syncthreads();
    BwdCbSubtract<T>(F, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
    __syncthreads();
    if (t < 32) BwdSubstitute<T>(F, fsz, nc, fr, y, rhs, /*lane=*/t);
  }
}

template <typename T, int NC>
__global__ void SolveBwdExactNc(int lbegin, int lend,
                                const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ front_rows,
                                const T* frontB, T* yB, long front_total,
                                int n) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const T* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  const int cb = fsz - NC;

  extern __shared__ unsigned char xsh_raw[];
  T* xsh = reinterpret_cast<T*>(xsh_raw);
  __shared__ T rhs[kMaxPivotColumns];

  BwdLoadRhsAndXFixed<T, NC>(y, fr, cb, rhs, xsh, t, nt);
  __syncthreads();
  BwdCbSubtractFixed<T, NC>(F, fsz, cb, xsh, rhs, t, /*width=*/nt);
  __syncthreads();
  BwdSubstituteFixedRegular<T, NC>(F, fsz, fr, y, rhs, t);
}

// =======================================================================================
//  SPINE tier — one block per batch walks the cnt=1 chain end-to-end
// =======================================================================================

template <typename T>
__global__ void SolveFwdSpine(int n_spine, const int* __restrict__ spine_panels,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ front_rows,
                              const T* frontB, T* yB, long front_total, int n) {
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int t = threadIdx.x, nt = blockDim.x;
  __shared__ T sh_piv[kMaxPivotColumns];

  for (int idx = 0; idx < n_spine; ++idx) {
    const int p = spine_panels[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;

    if (t < 32) FwdSubstitute<T>(F, fsz, nc, fr, y, sh_piv, /*lane=*/t);
    __syncthreads();
    FwdCbUpdatePlain<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t, nt);
    __syncthreads();
  }
}

template <typename T>
__global__ void SolveBwdSpine(int n_spine, const int* __restrict__ spine_panels,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ front_rows,
                              const T* frontB, T* yB, long front_total, int n) {
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int t = threadIdx.x, nt = blockDim.x;
  __shared__ T rhs[kMaxPivotColumns];
  extern __shared__ unsigned char xsh_raw[];
  T* xsh = reinterpret_cast<T*>(xsh_raw);

  for (int idx = n_spine - 1; idx >= 0; --idx) {
    const int p = spine_panels[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const int cb = fsz - nc;
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;

    BwdLoadRhsAndX<T>(y, fr, nc, cb, rhs, xsh, t, nt);
    __syncthreads();
    BwdCbSubtract<T>(F, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
    __syncthreads();
    if (t < 32) BwdSubstitute<T>(F, fsz, nc, fr, y, rhs, /*lane=*/t);
    __syncthreads();
  }
}

template <typename T>
__global__ void SolveSpine(int n_spine, const int* __restrict__ spine_panels,
                           const int* __restrict__ front_off,
                           const int* __restrict__ front_ptr,
                           const int* __restrict__ ncols,
                           const int* __restrict__ front_rows, const T* frontB,
                           T* yB, long front_total, int n) {
  const T* front = frontB + (long)blockIdx.y * front_total;
  T* y = yB + (long)blockIdx.y * n;
  const int t = threadIdx.x, nt = blockDim.x;
  __shared__ T sh_piv[kMaxPivotColumns];
  __shared__ T rhs[kMaxPivotColumns];
  extern __shared__ unsigned char xsh_raw[];
  T* xsh = reinterpret_cast<T*>(xsh_raw);

  // Forward pass: walk the spine chain leaves->root.
  for (int idx = 0; idx < n_spine; ++idx) {
    const int p = spine_panels[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;

    if (t < 32) FwdSubstitute<T>(F, fsz, nc, fr, y, sh_piv, /*lane=*/t);
    __syncthreads();
    FwdCbUpdatePlain<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, t, nt);
    __syncthreads();
  }

  // Backward pass: walk the spine chain root->leaves.
  for (int idx = n_spine - 1; idx >= 0; --idx) {
    const int p = spine_panels[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const int cb = fsz - nc;
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;

    BwdLoadRhsAndX<T>(y, fr, nc, cb, rhs, xsh, t, nt);
    __syncthreads();
    BwdCbSubtract<T>(F, fsz, nc, cb, xsh, rhs, t, /*width=*/nt);
    __syncthreads();
    if (t < 32) BwdSubstitute<T>(F, fsz, nc, fr, y, rhs, /*lane=*/t);
    __syncthreads();
  }
}

}  // namespace
}  // namespace custom_linear_solver
