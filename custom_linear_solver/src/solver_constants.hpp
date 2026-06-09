#pragma once

// Canonical solver constants and the front-size tier classifier.
//
// Single source of truth for the kernel-tier thresholds, shared-memory budgets, and fixed
// hardware/layout caps that the analyze, factorize, and solve stages must all agree on. Keeping
// them here (instead of re-declaring per file) removes the "must match" coupling between
// plan/analyze.cu and the factorize/solve dispatchers.

#include <cstddef>

namespace custom_linear_solver {

// Warp width on all supported architectures.
inline constexpr int kWarpSize = 32;

// Front-size tier edges (inclusive upper bounds). A front of `front_size` columns is small when
// front_size <= kSmallFrontMax, mid when <= kMidFrontMax, big otherwise.
inline constexpr int kSmallFrontMax = 32;
inline constexpr int kMidFrontMax = 128;

// Number of tiers (small / mid / big) and how many leading tiers use the warp-packed small kernel.
inline constexpr int kNumFrontTiers = 3;
inline constexpr int kSmallTierCount = 1;

// Small-tier kernel packs this many warps per block.
inline constexpr int kSmallTierWarpsPerBlock = 8;

// Upper bound on pivot columns (nc) used to size per-warp shared slabs in the solve kernels.
inline constexpr int kMaxPivotColumns = 64;

// Tensor-core trailing GEMM caps the staged pivot dimension at this many columns.
inline constexpr int kTensorCorePivotColumnCap = 32;

// Per-block dynamic shared-memory budget for the mid tier (sm_86 cap), and the opt-in size passed
// to cudaFuncAttributeMaxDynamicSharedMemorySize for the shared-resident kernels.
inline constexpr std::size_t kMidSharedMemoryBudgetBytes = 96 * 1024;
inline constexpr std::size_t kDynamicSharedMemoryOptInBytes = 99 * 1024;

// Maximum number of CUDA streams used for independent-subtree dispatch.
inline constexpr int kMaxSubtreeStreams = 8;

// Byte alignment of the per-array sub-allocations inside the single device arena.
inline constexpr int kArenaAlignmentBytes = 256;

// Front-size tier. Enumerator order (kSmall=0, kMid=1, kBig=2) defines the tier index used by the
// analyze-time tier bucketing.
enum class FrontTier { kSmall, kMid, kBig };

// Classify a front by its column count against the tier edges.
constexpr FrontTier classify_front_tier(int front_size)
{
    if (front_size <= kSmallFrontMax) return FrontTier::kSmall;
    if (front_size <= kMidFrontMax) return FrontTier::kMid;
    return FrontTier::kBig;
}

// Tier index in [0, kNumFrontTiers): 0 = small, 1 = mid, 2 = big.
constexpr int front_tier_index(FrontTier tier) { return static_cast<int>(tier); }

// Mid sub-tier boundary: the mid tier (kSmallFrontMax < fsz <= kMidFrontMax) is split here for
// dispatch bucketing only — fronts at or below it form the "mid-low" bucket.
inline constexpr int kMidSplitFrontMax = 64;

// Front-size buckets for the analyze-time tier ordering and the occupancy-gated dispatch split.
// Finer than classify_front_tier: the mid tier is split at kMidSplitFrontMax so a heterogeneous mid
// level dispatches its small-mid majority with a tighter shared front-staging budget (Fs scales as
// fsz_cap², so a tighter cap raises occupancy and frees the shared headroom that front-per-block
// packing needs). Kernel SELECTION still uses classify_front_tier (small / mid / big) — both mid
// buckets run the mid kernel, just with their own fsz_cap. CLS_NO_MID_SPLIT reverts to 3 buckets.
#ifdef CLS_NO_MID_SPLIT
inline constexpr int kNumFrontBuckets = kNumFrontTiers;  // 3: small / mid / big
constexpr int front_bucket(int front_size)
{
    return front_tier_index(classify_front_tier(front_size));
}
#else
inline constexpr int kNumFrontBuckets = 4;  // small / mid-low / mid-high / big
constexpr int front_bucket(int front_size)
{
    if (front_size <= kSmallFrontMax) return 0;     // small
    if (front_size <= kMidSplitFrontMax) return 1;  // mid-low
    if (front_size <= kMidFrontMax) return 2;       // mid-high
    return 3;                                       // big
}
#endif

// Round `value` up to the next multiple of `alignment` (alignment > 0).
constexpr int round_up_to_multiple(int value, int alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace custom_linear_solver
