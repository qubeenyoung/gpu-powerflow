#pragma once

// Canonical solver constants and the front-size tier classifier.
//
// Single source of truth for the kernel-tier thresholds, shared-memory budgets, and fixed
// hardware/layout caps that the analyze, factorize, and solve stages must all agree on. Keeping
// them here (instead of re-declaring per file) removes the "must match" coupling between
// analyze/plan/lower.cu and the factorize/solve dispatchers.

#include <cstddef>

namespace custom_linear_solver {

// Warp width on all supported architectures.
inline constexpr int kWarpSize = 32;

// Small-tier boundary: fronts with front_size <= kSmallFrontMax use the warp-packed small kernel.
// Everything larger is a block-per-front "large" front whose kernel (shared-resident mid vs global
// big) is chosen at dispatch time by the shared-memory-fit test below — NOT by a fixed column cap.
inline constexpr int kSmallFrontMax = 32;

// How many leading tiers use the warp-packed small kernel (small only).
inline constexpr int kSmallTierCount = 1;

// Small-tier kernel packs this many warps per block.
inline constexpr int kSmallTierWarpsPerBlock = 8;

// Upper bound on pivot columns (nc) used to size per-warp shared slabs in the solve kernels.
inline constexpr int kMaxPivotColumns = 64;

// Tensor-core trailing GEMM caps the staged pivot dimension at this many columns.
inline constexpr int kTensorCorePivotColumnCap = 32;

// Fronts with fsz <= this run the lu_small_front fused (Phase 1+2+3) path inside the mid kernel;
// larger fronts run the blocked panel-LU / U-solve / trailing. Big fronts are always > this.
// (Was a scattered literal 48 in factorize/{mid,big}.cuh — consolidated here.)
inline constexpr int kFusedSmallFrontMax = 48;


// TC-eligibility caps for the TF32 trailing path (mid + big). Relaxed defaults (2026-06-11): nearly
// every mid/big front takes the tensor-core path — the uc>256 spine fronts that used to fall to
// scalar (the L25 factorize spike) now run TF32 too. Override via -D for A/Bs.
// Shared-budget bound: the whole-front staging is (2·uc_pad·nc_pad + 4·nc_pad)·4 B ≤ 99 KiB opt-in,
// i.e. uc_pad·nc_pad ≤ ~12.6 K. With nc cap 32 that allows uc up to 384; UC_CAP=512 is safe for
// nc≤24 (power-grid Jacobians have nc≤16, so 512 leaves ~35 % headroom). The max_uc clamp in
// front_range_caps.hpp tracks CLS_TC_UC_CAP so the dispatch sizes shared consistently (no OOB).
#ifndef CLS_TC_UC_CAP
#define CLS_TC_UC_CAP 512      // max contribution dim (uc) eligible for the TF32 mma trailing
#endif
#ifndef CLS_TC_NC_MIN
#define CLS_TC_NC_MIN 1        // mid: min pivot cols (K) for the blocked TF32 path
#endif
#ifndef CLS_TC_UC_MIN
#define CLS_TC_UC_MIN 1        // mid: min uc for the blocked TF32 path
#endif
#ifndef CLS_TC_FSZ_MIN
#define CLS_TC_FSZ_MIN 32      // mid: min front size for the blocked TF32 path (mid starts at 33)
#endif

// Opt-in dynamic shared-memory size passed to cudaFuncAttributeMaxDynamicSharedMemorySize for the
// shared-resident kernels (sm_86 cap). THIS budget — together with the runtime precision — is the
// mid/big boundary: dispatch_factor_mid keeps a front shared-resident (mid kernel) only while
// fsz*fsz*sizeof(elem) fits here; otherwise it falls through to the global big kernel.
inline constexpr std::size_t kDynamicSharedMemoryOptInBytes = 99 * 1024;

// Largest front (columns) whose dense fsz×fsz FP32 staging fits the budget above — i.e. the mid/big
// boundary for the float (FP32/TF32) path, DERIVED from the same budget so the analyze-time bucketing
// and the runtime dispatch agree (no hard-coded 128). FP64 fronts use sizeof(double) and hit the
// boundary sooner, so a mid-bucketed FP64 front simply falls through to big inside dispatch_factor_mid.
constexpr int float_shared_front_max()
{
    int f = 1;
    while ((long)(f + 1) * (f + 1) * (long)sizeof(float) <= (long)kDynamicSharedMemoryOptInBytes) ++f;
    return f;  // 159 for a 99 KiB budget
}
inline constexpr int kFloatSharedFrontMax = float_shared_front_max();

// Maximum number of CUDA streams used for independent-subtree dispatch.
inline constexpr int kMaxSubtreeStreams = 8;

// Byte alignment of the per-array sub-allocations inside the single device arena.
inline constexpr int kArenaAlignmentBytes = 256;

// Front-size tier for dispatch routing: small (warp-packed kernel) vs large (block-per-front). The
// mid-vs-big split among large fronts is decided at dispatch time by the shared-fit test, not here.
enum class FrontTier { kSmall, kLarge };

constexpr FrontTier classify_front_tier(int front_size)
{
    return front_size <= kSmallFrontMax ? FrontTier::kSmall : FrontTier::kLarge;
}

// Mid sub-tier boundary: the block-per-front range is split at kMidSplitFrontMax so a heterogeneous
// level dispatches its smaller-front majority with a tighter shared staging budget (Fs scales as
// fsz_cap², so a tighter cap raises occupancy).
inline constexpr int kMidSplitFrontMax = 64;

// Front-size buckets for the analyze-time tier ordering and the occupancy-gated dispatch split.
// Boundaries match the runtime dispatch: small | mid-low (<= kMidSplitFrontMax) | mid-high (fits the
// float shared budget) | large (overflows -> big). Each bucket's sub-launch carries its own fsz_cap,
// so anchoring the mid-high boundary at kFloatSharedFrontMax (not an arbitrary 128) keeps FP32/TF32
// fronts that fit shared on the mid kernel instead of being dragged to big by a larger neighbour.
inline constexpr int kMidBucketCount = 2;
inline constexpr int kBigBucketCount = 1;
inline constexpr int kNumFrontBuckets = kSmallTierCount + kMidBucketCount + kBigBucketCount;

constexpr int front_bucket(int front_size)
{
    if (front_size <= kSmallFrontMax) return 0;                  // small
    const int mid0 = kSmallTierCount;
    if (front_size <= kMidSplitFrontMax) return mid0;            // mid-low
    if (front_size <= kFloatSharedFrontMax) return mid0 + 1;     // mid-high (fits float shared)
    return kSmallTierCount + kMidBucketCount;                    // large (big kernel)
}

// Round `value` up to the next multiple of `alignment` (alignment > 0).
constexpr int round_up_to_multiple(int value, int alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace custom_linear_solver
