#pragma once

// Canonical solver constants and the front-size tier classifier.
//
// Single source of truth for the kernel-tier thresholds, shared-memory budgets,
// and fixed hardware/layout caps that the Analyze, Factorize, and Solve stages
// must all agree on. Keeping them here (instead of re-declaring per file)
// removes the "must match" coupling between Analyze/plan/lower.cu and the
// Factorize/Solve dispatchers.

#include <cstddef>
#include <cstdlib>

namespace custom_linear_solver {

// Warp width on all supported architectures.
inline constexpr int kWarpSize = 32;

// Front-size tiers (3). Each tier routes to ONE dedicated factor kernel as a
// deterministic function of front size; no occupancy/opt-in gating decides
// which kernel runs.
//
//   small (fsz <= 32) -> warp-packed sub-group kernel (FactorSmall)
//   mid   (fsz <= 64) -> whole-front shared-resident, 1 block/front (FactorMid)
//   big   (fsz >  64) -> global-resident; ONE front spread across MANY blocks
//                        via the multi-block triple FactorBigPivot ->
//                        FactorBigPanels -> FactorBigTrail (TF32: the trailing
//                        runs on tensor cores, FactorBigTrailTf32)
//
// Boundary rationale (both physical):
//   small|mid = warp width (32): sub-group packing (8/16/32 lanes per front)
//                lives inside one warp; below it, warp-per-front + __syncwarp
//                beats a block-per-front kernel that idles most threads and pays
//                a full block barrier on the many Leaf fronts.
//   mid|big   = whole-front shared occupancy crossover (64): whole-front
//                staging costs fsz²·elem of shared, so beyond ~64 the footprint
//                drops occupancy below ~2 blocks/SM. The big tier keeps the
//                front in global and stages only small tiles, spreading one
//                front's work across many blocks — fills the GPU when fronts are
//                few/large and absorbs fronts too large to be shared-resident at
//                all (FEM/circuit root separators).
//
// 65–111 fronts run the global big kernel (faster than a panel-resident tier);
// WholeFrontSharedMax() survives only as the big kernel's bounded-shared
// staging threshold.

// small|mid boundary = warp width.
inline constexpr int kSmallFrontMax = kWarpSize;  // 32

// mid|big boundary = whole-front shared occupancy crossover.
inline constexpr int kMidFrontMax = 64;

// Small-tier kernel packs this many warps per block.
inline constexpr int kSmallTierWarpsPerBlock = 8;

// Upper bound on pivot columns (nc) used to size per-warp shared slabs in the
// Solve kernels.
inline constexpr int kMaxPivotColumns = 64;

// Tensor-core trailing GEMM caps the staged pivot dimension at this many
// columns.
inline constexpr int kTensorCorePivotColumnCap = 32;

// Fronts with fsz <= this run the LuMidFront fused (Phase 1+2+3) path inside
// the mid kernel; larger fronts run the blocked panel-LU / U-Solve / trailing.
// Big fronts are always > this.
inline constexpr int kFusedMidFrontMax = 48;

// TC-eligibility cap for the TF32 trailing path (mid + big): the largest
// contribution dim (uc) whose padded L/U staging still fits the shared budget.
//
// HARDWARE-derived bound, not a tuning knob: whole-front staging is
// (2·uc_pad·nc_pad + 4·nc_pad)·4 B and must fit kDynamicSharedMemoryOptInBytes
// (99 KiB on sm_86), giving uc_pad·nc_pad <= ~12.6 K; with nc cap 32, 512 is
// safe for nc<=24. MUST be re-derived if kDynamicSharedMemoryOptInBytes changes.
// The max_uc clamp in front_range_caps.hpp tracks it so the dispatch sizes
// shared consistently (no OOB).
//
// Only gate is "shape fits the staging budget": nc <= kTensorCorePivotColumnCap,
// uc <= this. (uc=0 fronts run the TC path as a no-op trailing.)
inline constexpr int kTensorCoreUcCap = 512;

// Opt-in dynamic shared-memory size passed to
// cudaFuncAttributeMaxDynamicSharedMemorySize for the shared-resident kernels
// (sm_86 cap). This budget — together with the precision — sets the big-tier
// bounded-shared staging threshold (the tier boundary below (the largest front
// whose whole fsz×fsz staging fits here).
inline constexpr std::size_t kDynamicSharedMemoryOptInBytes = 99 * 1024;

// Bounded-shared threshold: the largest front (columns) whose dense fsz×fsz
// staging fits the budget above at the given precision — 159 (float, FP32/TF32)
// / 111 (double, FP64). DERIVED from the budget so the Analyze-time tier
// bucketing and the runtime dispatch agree, and so FP64 enters "large" sooner.
constexpr int WholeFrontSharedMax(bool fp64) {
  const long elem = fp64 ? (long)sizeof(double) : (long)sizeof(float);
  int f = 1;
  while ((long)(f + 1) * (f + 1) * elem <= (long)kDynamicSharedMemoryOptInBytes)
    ++f;
  return f;  // 159 float / 111 double
}
inline constexpr int kFloatSharedFrontMax = WholeFrontSharedMax(false);  // 159

// Maximum number of CUDA streams used for independent-subtree dispatch.
inline constexpr int kMaxSubtreeStreams = 8;

// Byte alignment of the per-array sub-allocations inside the single device
// arena.
inline constexpr int kArenaAlignmentBytes = 256;

// The three dedicated front-size tiers (see the boundary rationale at the top).
// One kernel per tier; classification is a pure function of front size and
// precision.
enum class FrontTier { kSmall, kMid, kBig };

inline FrontTier ClassifyFrontTier(int front_size, bool fp64) {
  (void)fp64;
  if (front_size <= kSmallFrontMax) return FrontTier::kSmall;
  if (front_size <= kMidFrontMax) return FrontTier::kMid;
  return FrontTier::kBig;  // fsz > 64: global-resident multi-block
}

// Number of dispatch tiers (== number of dedicated kernels). The Analyze-time
// tier-homogeneous dispatch order groups a level's panels into these tiers so
// each (level, tier) sub-launch hits exactly one kernel.
inline constexpr int kNumFrontBuckets = 3;

// Tier index (0=small, 1=mid, 2=big) for the Analyze-time bucketing. Must match
// ClassifyFrontTier exactly so a homogeneous bucket routes to a single kernel
// at dispatch.
inline int FrontBucket(int front_size, bool fp64) {
  return static_cast<int>(ClassifyFrontTier(front_size, fp64));
}

// Round `value` up to the next multiple of `alignment` (alignment > 0).
constexpr int RoundUpToMultiple(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace custom_linear_solver
