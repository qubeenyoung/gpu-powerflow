#pragma once

// SOLVE — host-side level dispatch.
//
// Internal — included into the factor/Solve driver TUs (single TU;
// CUDA_SEPARABLE_COMPILATION OFF).
//
// Two layers of dispatch:
//
//   IssueSolveLevels        — outer driver. Forward pass leaves→root, then
//   backward
//                                pass root→leaves. Two execution modes:
//                                  - single stream: walks every level on one
//                                  stream.
//                                  - multi-stream: forks per-subtree work onto
//                                  K extra
//                                    streams (the same st.subtree_streams used
//                                    by factor) for the forward sweep, joins,
//                                    re-forks for the backward sweep, joins
//                                    again.
//
//   per-tier routing          — within each pass, the small tier (max_fsz ≤
//   kSmallFrontMax) goes to
//                                the warp-packed kernel (8 warps/block, one
//                                (front,batch) per warp); the mid / big tiers
//                                share the block-per-front kernel with thread
//                                count tuned to the level's max_fsz.

#include <cuda_runtime.h>

#include "internal/plan/front_range_caps.hpp"
#include "internal/runtime/state.hpp"
#include "solve/kernels.cuh"
#include "solve/single.cuh"  // B=1 single-system solve schedule

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

constexpr int kSolveSmallWarpsPerBlock = kSmallTierWarpsPerBlock;

// HW warp-slot count (SMs × warps/SM) — the GPU-fill quantity used by the
// small-tier sub-group gate (mirrors Factorize/front_ops.cuh
// FactorWarpFill()).
static long SolveWarpFill() {
  static const long v = [] {
    int dev = 0;
    cudaGetDevice(&dev);
    int sm = 1, tpm = 1536;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    return (long)sm * (tpm / 32);
  }();
  return v;
}

// Workload-based mapping gate, mirrors the Factorize schedule. The small tier's
// parallelism is (level front count × batch). When it saturates the GPU warp
// capacity, pack fronts into warps (one warp per front); when it under-fills
// (narrow levels) fall through to the block-per-front path (more threads/front)
// instead. fp32 fronts are lighter so they fill with fewer fronts → count fp32
// ×2.
static bool SolveSmallPacks(int level_size, int B, bool fp64) {
  const long work = (long)level_size * B;
  const long eff_work = fp64 ? work : work * 2;
  return eff_work >= SolveWarpFill();
}

// Small-tier sub-group size: sub_group_size ∈ {8,16,32}. The triangular
// recurrences require lanes for pivot columns, not every contribution row, so
// use max_nc rather than max_fsz when packing multiple fronts into one warp.
static int SolveSmallSg(int max_nc, long warps_unpacked, int B) {
  int sg = (max_nc <= 8) ? 8 : (max_nc <= 16 ? 16 : 32);
  const long packed_warps = (warps_unpacked + (32 / sg) - 1) / (32 / sg);
  if (B == 1 && packed_warps < SolveWarpFill() && packed_warps < 800) sg = 32;
  return sg;
}

static int SolveRegularThreads(int max_front_size, int level_size,
                               int level_max_nc, int B) {
  if (B == 1) {
    if (max_front_size > 96) return 512;
    if (max_front_size > 80 && level_size <= 128) return 512;
    if (max_front_size > 64 && level_size <= 512 && level_max_nc >= 16)
      return 512;
    if (max_front_size > 64 && level_size <= 512) return 256;
    if (max_front_size > 48 && level_size <= 512) return 256;
  }
  return max_front_size <= 64 ? 64 : (max_front_size <= 128 ? 128 : 256);
}

static bool SolveRangeAllNc(const MultifrontalPlan& plan, const int* h_plc,
                            int b, int e, int nc) {
  for (int q = b; q < e; ++q)
    if (plan.h_ncols[h_plc[q]] != nc) return false;
  return true;
}

static int SolveRegularFixedNcChoice(const MultifrontalPlan& plan,
                                     const int* h_plc, int b, int e,
                                     int level_max_nc, int B) {
  (void)plan;
  (void)h_plc;
  (void)b;
  (void)e;
  if (B != 1) return 0;
  int choice = 0;
  if (level_max_nc == 8)
    choice = 8;
  else if (level_max_nc == 10)
    choice = 10;
  else if (level_max_nc == 14)
    choice = 14;
  else if (level_max_nc == 16)
    choice = 16;
  else if (level_max_nc == 20)
    choice = 20;
  return choice;
}

// One launch of a sub-group-packed small Solve kernel (forward or backward),
// with the (front type, sub_group_size) resolved at the call site.
// SolveFwdSmall / SolveBwdSmall share a signature; `fwd` selects which,
// `slab` is the per-sub-group shared slab.
template <typename FrontType, int sub_group_size>
static inline void LaunchSolveSmall(bool fwd, int num_blocks,
                                    int threads_per_block, size_t shared_bytes,
                                    cudaStream_t ks, int b, int level_size,
                                    int B, int slab,
                                    const MultifrontalPlan& plan,
                                    const int* d_plc, FrontType* frontB,
                                    FrontType* yB) {
  if (fwd)
    SolveFwdSmall<FrontType, sub_group_size>
        <<<num_blocks, threads_per_block, shared_bytes, ks>>>(
            b, level_size, B, slab, d_plc, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, plan.d_front_rows, frontB, yB, plan.front_total,
            plan.num_rows);
  else
    SolveBwdSmall<FrontType, sub_group_size>
        <<<num_blocks, threads_per_block, shared_bytes, ks>>>(
            b, level_size, B, slab, d_plc, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, plan.d_front_rows, frontB, yB, plan.front_total,
            plan.num_rows);
}

template <typename FrontType>
static inline void LaunchSolveSmallT(bool fwd, int sub_group_size,
                                     int num_blocks, int threads_per_block,
                                     size_t shared_bytes, cudaStream_t ks,
                                     int b, int level_size, int B, int slab,
                                     const MultifrontalPlan& plan,
                                     const int* d_plc, FrontType* frontB,
                                     FrontType* yB) {
  if (sub_group_size == 8)
    LaunchSolveSmall<FrontType, 8>(fwd, num_blocks, threads_per_block,
                                   shared_bytes, ks, b, level_size, B, slab,
                                   plan, d_plc, frontB, yB);
  else if (sub_group_size == 16)
    LaunchSolveSmall<FrontType, 16>(fwd, num_blocks, threads_per_block,
                                    shared_bytes, ks, b, level_size, B, slab,
                                    plan, d_plc, frontB, yB);
  else
    LaunchSolveSmall<FrontType, 32>(fwd, num_blocks, threads_per_block,
                                    shared_bytes, ks, b, level_size, B, slab,
                                    plan, d_plc, frontB, yB);
}

template <typename FrontType>
static inline void LaunchSolveFwdDynamic(dim3 grid, int threads_per_block,
                                         size_t shared_bytes, cudaStream_t ks,
                                         int b, int e,
                                         const MultifrontalPlan& plan,
                                         const int* d_plc, FrontType* frontB,
                                         FrontType* yB) {
  SolveFwd<FrontType><<<grid, threads_per_block, shared_bytes, ks>>>(
      b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
      plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
}

// Non-staged fwd/bwd launches for large fronts (staged shared would exceed the
// default cap).
template <typename FrontType>
static inline void LaunchSolveFwdNostage(dim3 grid, int threads_per_block,
                                         cudaStream_t ks, int b, int e,
                                         const MultifrontalPlan& plan,
                                         const int* d_plc, FrontType* frontB,
                                         FrontType* yB) {
  SolveFwdNostage<FrontType><<<grid, threads_per_block, 0, ks>>>(
      b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
      plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
}

template <typename FrontType>
static inline void LaunchSolveBwdNostage(dim3 grid, int threads_per_block,
                                         size_t shared_bytes, cudaStream_t ks,
                                         int b, int e,
                                         const MultifrontalPlan& plan,
                                         const int* d_plc, FrontType* frontB,
                                         FrontType* yB) {
  SolveBwdNostage<FrontType><<<grid, threads_per_block, shared_bytes, ks>>>(
      b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
      plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
}

// Largest staged-panel shared we allow under the default (non-opt-in) cap;
// above this the dynamic Solve falls back to the non-staged kernel (reads F
// directly). Leaves room for the static sh_piv.
static constexpr size_t kSolveStageSharedCap = 46u * 1024u;

template <typename FrontType, int NC>
static inline void LaunchSolveFwdFixed(bool exact, dim3 grid,
                                       int threads_per_block, cudaStream_t ks,
                                       int b, int e,
                                       const MultifrontalPlan& plan,
                                       const int* d_plc, FrontType* frontB,
                                       FrontType* yB) {
  if (exact) {
    SolveFwdExactNc<FrontType, NC><<<grid, threads_per_block, 0, ks>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
        frontB, yB, plan.front_total, plan.num_rows);
  } else {
    SolveFwdFixedNc<FrontType, NC><<<grid, threads_per_block, 0, ks>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_front_rows, frontB, yB, plan.front_total, plan.num_rows);
  }
}

// Forward dispatch for one plcols (sub)range on a caller-chosen stream.
static void FwdLevel(const MultifrontalPlan& plan, State& st, cudaStream_t ks,
                     int b, int e, const int* d_plc, const int* h_plc) {
  if (e <= b) return;
  const int B = st.batch_count;
  const bool float_front = IsFp32Front(st.precision);
  const int element_bytes =
      float_front ? (int)sizeof(float) : (int)sizeof(double);
  const FrontRangeCaps metrics = ScanFrontRange(plan, h_plc, b, e);
  const int max_front_size = metrics.max_fsz;

  // Small tier: sub-group-packed warp kernel (fronts_per_warp =
  // kWarpSize/sub_group_size fronts per warp).
  if (ClassifyFrontTier(max_front_size, !float_front) == FrontTier::kSmall &&
      SolveSmallPacks(e - b, B, !float_front)) {
    const long warps_unpacked = (long)(e - b) * B;
    const int sub_group_size =
                  SolveSmallSg(metrics.level_max_nc, warps_unpacked, B),
              fronts_per_warp = kWarpSize / sub_group_size;
    const int threads_per_block = kSolveSmallWarpsPerBlock * kWarpSize;
    const int num_blocks =
        (int)(((warps_unpacked + fronts_per_warp - 1) / fronts_per_warp +
               kSolveSmallWarpsPerBlock - 1) /
              kSolveSmallWarpsPerBlock);
    const size_t shared_bytes = (size_t)kSolveSmallWarpsPerBlock *
                                fronts_per_warp * kMaxPivotColumns *
                                element_bytes;
    if (float_front)
      LaunchSolveSmallT<float>(/*fwd=*/true, sub_group_size, num_blocks,
                               threads_per_block, shared_bytes, ks, b, e - b, B,
                               kMaxPivotColumns, plan, d_plc,
                               st.d_front_batch_f, st.d_y_batch_f);
    else
      LaunchSolveSmallT<double>(/*fwd=*/true, sub_group_size, num_blocks,
                                threads_per_block, shared_bytes, ks, b, e - b,
                                B, kMaxPivotColumns, plan, d_plc,
                                st.d_front_batch, st.d_y_batch);
    return;
  }

  // Larger levels: one block per (front, batch), thread count tuned to max_fsz.
  const int threads_per_block =
      SolveRegularThreads(max_front_size, e - b, metrics.level_max_nc, B);
  const dim3 fg(e - b, B);
  // P3: dynamic shared for the staged L panel (rows × cols[0,nc), upper-bounded
  // by level maxes).
  const size_t fwd_panel_shared =
      (size_t)max_front_size * metrics.level_max_nc * element_bytes;
  const int fixed_nc =
      SolveRegularFixedNcChoice(plan, h_plc, b, e, metrics.level_max_nc, B);
  const bool use_nc8 = fixed_nc == 8;
  const bool use_nc10 = fixed_nc == 10;
  const bool use_nc14 = fixed_nc == 14;
  const bool use_nc16 = fixed_nc == 16;
  const bool use_nc20 = fixed_nc == 20;
  const bool exact_nc8 = use_nc8 && SolveRangeAllNc(plan, h_plc, b, e, 8);
  const bool exact_nc10 = use_nc10 && SolveRangeAllNc(plan, h_plc, b, e, 10);
  const bool exact_nc14 = use_nc14 && SolveRangeAllNc(plan, h_plc, b, e, 14);
  const bool exact_nc16 = false;
  const bool exact_nc20 = use_nc20 && SolveRangeAllNc(plan, h_plc, b, e, 20);
  if (float_front) {
    if (use_nc8)
      LaunchSolveFwdFixed<float, 8>(exact_nc8, fg, threads_per_block, ks, b, e,
                                    plan, d_plc, st.d_front_batch_f,
                                    st.d_y_batch_f);
    else if (use_nc10)
      LaunchSolveFwdFixed<float, 10>(exact_nc10, fg, threads_per_block, ks, b,
                                     e, plan, d_plc, st.d_front_batch_f,
                                     st.d_y_batch_f);
    else if (use_nc14)
      LaunchSolveFwdFixed<float, 14>(exact_nc14, fg, threads_per_block, ks, b,
                                     e, plan, d_plc, st.d_front_batch_f,
                                     st.d_y_batch_f);
    else if (use_nc16)
      LaunchSolveFwdFixed<float, 16>(exact_nc16, fg, threads_per_block, ks, b,
                                     e, plan, d_plc, st.d_front_batch_f,
                                     st.d_y_batch_f);
    else if (use_nc20)
      LaunchSolveFwdFixed<float, 20>(exact_nc20, fg, threads_per_block, ks, b,
                                     e, plan, d_plc, st.d_front_batch_f,
                                     st.d_y_batch_f);
    else if (fwd_panel_shared > kSolveStageSharedCap)
      LaunchSolveFwdNostage<float>(fg, threads_per_block, ks, b, e, plan, d_plc,
                                   st.d_front_batch_f, st.d_y_batch_f);
    else
      LaunchSolveFwdDynamic<float>(fg, threads_per_block, fwd_panel_shared, ks,
                                   b, e, plan, d_plc, st.d_front_batch_f,
                                   st.d_y_batch_f);
  } else {
    if (use_nc8)
      LaunchSolveFwdFixed<double, 8>(exact_nc8, fg, threads_per_block, ks, b, e,
                                     plan, d_plc, st.d_front_batch,
                                     st.d_y_batch);
    else if (use_nc10)
      LaunchSolveFwdFixed<double, 10>(exact_nc10, fg, threads_per_block, ks, b,
                                      e, plan, d_plc, st.d_front_batch,
                                      st.d_y_batch);
    else if (use_nc14)
      LaunchSolveFwdFixed<double, 14>(exact_nc14, fg, threads_per_block, ks, b,
                                      e, plan, d_plc, st.d_front_batch,
                                      st.d_y_batch);
    else if (use_nc16)
      LaunchSolveFwdFixed<double, 16>(exact_nc16, fg, threads_per_block, ks, b,
                                      e, plan, d_plc, st.d_front_batch,
                                      st.d_y_batch);
    else if (use_nc20)
      LaunchSolveFwdFixed<double, 20>(exact_nc20, fg, threads_per_block, ks, b,
                                      e, plan, d_plc, st.d_front_batch,
                                      st.d_y_batch);
    else if (fwd_panel_shared > kSolveStageSharedCap)
      LaunchSolveFwdNostage<double>(fg, threads_per_block, ks, b, e, plan,
                                    d_plc, st.d_front_batch, st.d_y_batch);
    else
      LaunchSolveFwdDynamic<double>(fg, threads_per_block, fwd_panel_shared, ks,
                                    b, e, plan, d_plc, st.d_front_batch,
                                    st.d_y_batch);
  }
}

// Backward dispatch for one plcols (sub)range on a caller-chosen stream.
static void BwdLevel(const MultifrontalPlan& plan, State& st, cudaStream_t ks,
                     int b, int e, const int* d_plc, const int* h_plc) {
  if (e <= b) return;
  const int B = st.batch_count;
  const bool float_front = IsFp32Front(st.precision);
  const int element_bytes =
      float_front ? (int)sizeof(float) : (int)sizeof(double);
  const FrontRangeCaps metrics = ScanFrontRange(plan, h_plc, b, e);
  const int max_front_size = metrics.max_fsz;
  const int max_contribution = metrics.level_max_uc;
  // P3: regular SolveBwd<T> stages the top-nc U rows (≤ max_fsz·level_max_nc)
  // plus the x cache.
  const size_t bwd_panel_shared =
      (size_t)(max_front_size * metrics.level_max_nc + max_contribution) *
      element_bytes;

  // Small tier: sub-group-packed warp kernel. slab = kMaxPivotColumns (rhs) +
  // max_contribution (x cache).
  if (ClassifyFrontTier(max_front_size, !float_front) == FrontTier::kSmall &&
      SolveSmallPacks(e - b, B, !float_front)) {
    const long warps_unpacked = (long)(e - b) * B;
    const int sub_group_size =
                  SolveSmallSg(metrics.level_max_nc, warps_unpacked, B),
              fronts_per_warp = kWarpSize / sub_group_size;
    const int threads_per_block = kSolveSmallWarpsPerBlock * kWarpSize;
    const int num_blocks =
        (int)(((warps_unpacked + fronts_per_warp - 1) / fronts_per_warp +
               kSolveSmallWarpsPerBlock - 1) /
              kSolveSmallWarpsPerBlock);
    const int slab = kMaxPivotColumns + max_contribution;
    const size_t shared_bytes = (size_t)kSolveSmallWarpsPerBlock *
                                fronts_per_warp * slab * element_bytes;
    if (float_front)
      LaunchSolveSmallT<float>(/*fwd=*/false, sub_group_size, num_blocks,
                               threads_per_block, shared_bytes, ks, b, e - b, B,
                               slab, plan, d_plc, st.d_front_batch_f,
                               st.d_y_batch_f);
    else
      LaunchSolveSmallT<double>(/*fwd=*/false, sub_group_size, num_blocks,
                                threads_per_block, shared_bytes, ks, b, e - b,
                                B, slab, plan, d_plc, st.d_front_batch,
                                st.d_y_batch);
    return;
  }

  // Larger levels: one block per (front, batch); max_contribution-sized shared
  // for the x cache.
  const int threads_per_block =
      SolveRegularThreads(max_front_size, e - b, metrics.level_max_nc, B);
  const dim3 bg(e - b, B);
  const int fixed_nc =
      SolveRegularFixedNcChoice(plan, h_plc, b, e, metrics.level_max_nc, B);
  const bool use_nc8 = fixed_nc == 8;
  const bool use_nc10 = fixed_nc == 10;
  const bool use_nc14 = fixed_nc == 14;
  const bool use_nc16 = fixed_nc == 16;
  const bool use_nc20 = fixed_nc == 20;
  const bool exact_nc8 = use_nc8 && SolveRangeAllNc(plan, h_plc, b, e, 8);
  const bool exact_nc10 = use_nc10 && SolveRangeAllNc(plan, h_plc, b, e, 10);
  const bool exact_nc14 = use_nc14 && SolveRangeAllNc(plan, h_plc, b, e, 14);
  const bool exact_nc16 = false;
  const bool exact_nc20 = use_nc20 && SolveRangeAllNc(plan, h_plc, b, e, 20);
  if (float_front) {
    if (use_nc8) {
      if (exact_nc8)
        SolveBwdExactNc<float, 8>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<float, 8>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch_f,
                     st.d_y_batch_f, plan.front_total, plan.num_rows);
    } else if (use_nc10) {
      if (exact_nc10)
        SolveBwdExactNc<float, 10>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<float, 10>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch_f,
                     st.d_y_batch_f, plan.front_total, plan.num_rows);
    } else if (use_nc14) {
      if (exact_nc14)
        SolveBwdExactNc<float, 14>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<float, 14>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch_f,
                     st.d_y_batch_f, plan.front_total, plan.num_rows);
    } else if (use_nc16) {
      if (exact_nc16)
        SolveBwdExactNc<float, 16>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<float, 16>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch_f,
                     st.d_y_batch_f, plan.front_total, plan.num_rows);
    } else if (use_nc20) {
      if (exact_nc20)
        SolveBwdExactNc<float, 20>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<float, 20>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(float),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch_f,
                     st.d_y_batch_f, plan.front_total, plan.num_rows);
    } else if (bwd_panel_shared > kSolveStageSharedCap)
      LaunchSolveBwdNostage<float>(
          bg, threads_per_block, (size_t)max_contribution * sizeof(float), ks,
          b, e, plan, d_plc, st.d_front_batch_f, st.d_y_batch_f);
    else
      SolveBwd<float><<<bg, threads_per_block, bwd_panel_shared, ks>>>(
          b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
          plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f,
          plan.front_total, plan.num_rows);
  } else {
    if (use_nc8) {
      if (exact_nc8)
        SolveBwdExactNc<double, 8>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch, st.d_y_batch,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<double, 8>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch,
                     st.d_y_batch, plan.front_total, plan.num_rows);
    } else if (use_nc10) {
      if (exact_nc10)
        SolveBwdExactNc<double, 10>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch, st.d_y_batch,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<double, 10>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch,
                     st.d_y_batch, plan.front_total, plan.num_rows);
    } else if (use_nc14) {
      if (exact_nc14)
        SolveBwdExactNc<double, 14>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch, st.d_y_batch,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<double, 14>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch,
                     st.d_y_batch, plan.front_total, plan.num_rows);
    } else if (use_nc16) {
      if (exact_nc16)
        SolveBwdExactNc<double, 16>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch, st.d_y_batch,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<double, 16>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch,
                     st.d_y_batch, plan.front_total, plan.num_rows);
    } else if (use_nc20) {
      if (exact_nc20)
        SolveBwdExactNc<double, 20>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_front_rows, st.d_front_batch, st.d_y_batch,
                     plan.front_total, plan.num_rows);
      else
        SolveBwdFixedNc<double, 20>
            <<<bg, threads_per_block, (size_t)max_contribution * sizeof(double),
               ks>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, plan.d_front_rows, st.d_front_batch,
                     st.d_y_batch, plan.front_total, plan.num_rows);
    } else if (bwd_panel_shared > kSolveStageSharedCap)
      LaunchSolveBwdNostage<double>(
          bg, threads_per_block, (size_t)max_contribution * sizeof(double), ks,
          b, e, plan, d_plc, st.d_front_batch, st.d_y_batch);
    else
      SolveBwd<double><<<bg, threads_per_block, bwd_panel_shared, ks>>>(
          b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
          plan.d_front_rows, st.d_front_batch, st.d_y_batch, plan.front_total,
          plan.num_rows);
  }
}

static int SpineMaxContribution(const MultifrontalPlan& plan) {
  int max_contribution = 1;
  for (int p : plan.h_spine_panels) {
    const int fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
    const int cb = fsz - plan.h_ncols[p];
    if (cb > max_contribution) max_contribution = cb;
  }
  return max_contribution;
}

static int SpineThreadsPerBlock(const MultifrontalPlan& plan, int B) {
  int max_front_size = 1;
  for (int p : plan.h_spine_panels) {
    const int fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
    if (fsz > max_front_size) max_front_size = fsz;
  }
  if (B == 1 && max_front_size <= 80) return 64;
  if (B == 1 && max_front_size > 96) return 512;
  return max_front_size <= 64 ? 64 : (max_front_size <= 128 ? 128 : 256);
}

static void FwdSpine(const MultifrontalPlan& plan, State& st, cudaStream_t ks) {
  const int n_spine = (int)plan.h_spine_panels.size();
  if (n_spine <= 0 || !plan.d_spine_panels) return;
  const int spine_threads = SpineThreadsPerBlock(plan, st.batch_count);
  const dim3 grid(1, st.batch_count);
  if (IsFp32Front(st.precision))
    SolveFwdSpine<float><<<grid, spine_threads, 0, ks>>>(
        n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f,
        plan.front_total, plan.num_rows);
  else
    SolveFwdSpine<double><<<grid, spine_threads, 0, ks>>>(
        n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, plan.d_front_rows, st.d_front_batch, st.d_y_batch,
        plan.front_total, plan.num_rows);
}

static void BwdSpine(const MultifrontalPlan& plan, State& st, cudaStream_t ks) {
  const int n_spine = (int)plan.h_spine_panels.size();
  if (n_spine <= 0 || !plan.d_spine_panels) return;
  const int spine_threads = SpineThreadsPerBlock(plan, st.batch_count);
  const int max_contribution = SpineMaxContribution(plan);
  const dim3 grid(1, st.batch_count);
  if (IsFp32Front(st.precision))
    SolveBwdSpine<float>
        <<<grid, spine_threads, (size_t)max_contribution * sizeof(float), ks>>>(
            n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f,
            plan.front_total, plan.num_rows);
  else
    SolveBwdSpine<double><<<grid, spine_threads,
                            (size_t)max_contribution * sizeof(double), ks>>>(
        n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, plan.d_front_rows, st.d_front_batch, st.d_y_batch,
        plan.front_total, plan.num_rows);
}

static void SolveSpineChain(const MultifrontalPlan& plan, State& st,
                            cudaStream_t ks) {
  const int n_spine = (int)plan.h_spine_panels.size();
  if (n_spine <= 0 || !plan.d_spine_panels) return;
  const int spine_threads = SpineThreadsPerBlock(plan, st.batch_count);
  if (spine_threads > 128) {
    FwdSpine(plan, st, ks);
    BwdSpine(plan, st, ks);
    return;
  }
  const int max_contribution = SpineMaxContribution(plan);
  const dim3 grid(1, st.batch_count);
  if (IsFp32Front(st.precision))
    SolveSpine<float>
        <<<grid, spine_threads, (size_t)max_contribution * sizeof(float), ks>>>(
            n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, plan.d_front_rows, st.d_front_batch_f, st.d_y_batch_f,
            plan.front_total, plan.num_rows);
  else
    SolveSpine<double><<<grid, spine_threads,
                         (size_t)max_contribution * sizeof(double), ks>>>(
        n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, plan.d_front_rows, st.d_front_batch, st.d_y_batch,
        plan.front_total, plan.num_rows);
}

// Deterministic per-tier split of one (level- or cell-) range, for either
// sweep. Each non-empty tier sub-range dispatches to its kernel (small ->
// warp-packed; mid/big -> block-per-front with thread count by max_fsz). Same
// range = independent fronts -> order-free.
static void DispatchTiered(const MultifrontalPlan& plan, State& st,
                           cudaStream_t ks, const int* tb, const int* d_plc,
                           const int* h_plc, bool fwd) {
  constexpr int NT = MultifrontalPlan::kNumTiers;
  for (int t = 0; t < NT; ++t)
    if (tb[t + 1] > tb[t]) {
      if (fwd)
        FwdLevel(plan, st, ks, tb[t], tb[t + 1], d_plc, h_plc);
      else
        BwdLevel(plan, st, ks, tb[t], tb[t + 1], d_plc, h_plc);
    }
}

// Multi-stream sweep: replicate the factor subtree-fork pattern. Subtree
// streams sweep their own panel slices through all pre-spine levels
// independently (no global per-level barrier), join, run the spine on the main
// stream, then re-fork for the backward sweep.
static void SolveMultistreamSweep(const MultifrontalPlan& plan, State& st,
                                  cudaStream_t stream) {
  constexpr int NT = MultifrontalPlan::kNumTiers;
  const int* sub_tb = plan.h_subtree_level_tier_off.data();
  const bool have_sub = !plan.h_subtree_level_tier_off.empty();
  const int K = st.num_subtree_streams;
  const int spine_lo =
      (plan.spine_start_level >= 0) ? plan.spine_start_level : plan.num_plevels;

  // ---- Forward fork ----
  cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
  for (int k = 0; k < K; ++k)
    cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                        static_cast<cudaEvent_t>(st.fork_event), 0);

  // Per-subtree forward sweep over pre-spine levels.
  for (int k = 0; k < K; ++k) {
    cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
    const int* off =
        plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
    const int* cnt =
        plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
    for (int L = 0; L < spine_lo; ++L) {
      if (cnt[L] == 0) continue;
      if (have_sub)
        DispatchTiered(plan, st, ks,
                       sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
                       plan.d_plcols, plan.h_plcols.data(), /*fwd=*/true);
      else
        FwdLevel(plan, st, ks, off[L], off[L] + cnt[L], plan.d_plcols,
                 plan.h_plcols.data());
    }
    cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
  }

  // ---- Forward join, spine forward on main, then backward re-fork ----
  for (int k = 0; k < K; ++k)
    cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);

  if (spine_lo < plan.num_plevels) SolveSpineChain(plan, st, stream);

  cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
  for (int k = 0; k < K; ++k)
    cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                        static_cast<cudaEvent_t>(st.fork_event), 0);

  // Per-subtree backward sweep over pre-spine levels (top -> bottom).
  for (int k = 0; k < K; ++k) {
    cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
    const int* off =
        plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
    const int* cnt =
        plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
    for (int L = spine_lo - 1; L >= 0; --L) {
      if (cnt[L] == 0) continue;
      if (have_sub)
        DispatchTiered(plan, st, ks,
                       sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
                       plan.d_plcols, plan.h_plcols.data(), /*fwd=*/false);
      else
        BwdLevel(plan, st, ks, off[L], off[L] + cnt[L], plan.d_plcols,
                 plan.h_plcols.data());
    }
    cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
  }

  for (int k = 0; k < K; ++k)
    cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);
}

// Single-stream sweep: per-level occupancy-gated tier split, forward
// (leaves->root) then backward.
static void SolveSingleStreamSweep(const MultifrontalPlan& plan, State& st,
                                   cudaStream_t stream) {
  constexpr int NT = MultifrontalPlan::kNumTiers;
  const int* lvl_tb = plan.h_level_tier_off.data();
  const bool have_lvl = !plan.h_level_tier_off.empty() && plan.d_plcols_tier;
  const int spine_lo =
      (plan.spine_start_level >= 0 && !plan.h_spine_panels.empty())
          ? plan.spine_start_level
          : plan.num_plevels;
  for (int L = 0; L < spine_lo; ++L) {
    if (have_lvl)
      DispatchTiered(plan, st, stream, lvl_tb + (long)L * (NT + 1),
                     plan.d_plcols_tier, plan.h_plcols_tier.data(),
                     /*fwd=*/true);
    else
      FwdLevel(plan, st, stream, plan.panel_level_ptr[L],
               plan.panel_level_ptr[L + 1], plan.d_plcols,
               plan.h_plcols.data());
  }
  if (spine_lo < plan.num_plevels) SolveSpineChain(plan, st, stream);
  for (int L = spine_lo - 1; L >= 0; --L) {
    if (have_lvl)
      DispatchTiered(plan, st, stream, lvl_tb + (long)L * (NT + 1),
                     plan.d_plcols_tier, plan.h_plcols_tier.data(),
                     /*fwd=*/false);
    else
      BwdLevel(plan, st, stream, plan.panel_level_ptr[L],
               plan.panel_level_ptr[L + 1], plan.d_plcols,
               plan.h_plcols.data());
  }
}

// Forward (leaves->root) then backward (root->leaves) Solve. Multi-stream when
// the plan exposes independent subtrees with matching streams; otherwise a
// single-stream level walk.
static void IssueSolveLevels(const MultifrontalPlan& plan, State& st,
                             cudaStream_t stream) {
  // B == 1: single-system fwd/bwd schedule (inverted-pivot GEMVs, per-level
  // block sizing), matching the B=1 factor path's partitioned-inverse pivot
  // blocks (UseSingleSystem — the same decision point as schedule.cuh, so the
  // two paths cannot drift).
  if (UseSingleSystem(st)) {
    IssueSolveSingle(plan, st, stream);
    return;
  }
  // Batched: multistream subtree sweep when the plan exposes independent
  // subtrees, else single stream. (Each cell exposes B blocks, so the GPU stays
  // filled.)
  const bool use_multistream = st.num_subtree_streams > 1 &&
                               plan.num_subtrees == st.num_subtree_streams &&
                               !plan.h_subtree_level_off.empty();
  if (use_multistream)
    SolveMultistreamSweep(plan, st, stream);
  else
    SolveSingleStreamSweep(plan, st, stream);
}

}  // namespace
}  // namespace custom_linear_solver
