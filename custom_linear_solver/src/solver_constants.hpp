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
#ifdef CLS_SMALL_FRONT_MAX_16
inline constexpr int kSmallFrontMax = 16;
#else
inline constexpr int kSmallFrontMax = 32;
#endif
inline constexpr int kMidFrontMax = 128;

// Number of tiers (small / mid / big) and how many leading tiers use the warp-packed small kernel.
inline constexpr int kNumFrontTiers = 3;
#ifdef CLS_SMALL_BUCKET_SPLIT_16
inline constexpr int kSmallTierCount = 2;
#else
inline constexpr int kSmallTierCount = 1;
#endif

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
inline constexpr int kSmallSplitFrontMax = 16;
inline constexpr int kMidLowSplitFrontMax = 48;
inline constexpr int kMidSplitFrontMax = 64;
inline constexpr int kMidHighSplitFrontMax = 80;
inline constexpr int kBigLowSplitFrontMax = 159;

// Front-size buckets for the analyze-time tier ordering and the occupancy-gated dispatch split.
// Finer than classify_front_tier: the mid tier is split at kMidSplitFrontMax so a heterogeneous mid
// level dispatches its small-mid majority with a tighter shared front-staging budget (Fs scales as
// fsz_cap², so a tighter cap raises occupancy and frees the shared headroom that front-per-block
// packing needs). Kernel SELECTION still uses classify_front_tier (small / mid / big) — both mid
// buckets run the mid kernel, just with their own fsz_cap. CLS_NO_MID_SPLIT reverts to one
// mid bucket; CLS_SMALL_BUCKET_SPLIT_16 optionally splits only the small dispatch buckets.
#ifdef CLS_NO_MID_SPLIT
inline constexpr int kMidBucketCount = 1;
#elif defined(CLS_MID_LOW_SPLIT) && defined(CLS_MID_HIGH_SPLIT)
inline constexpr int kMidBucketCount = 4;
#elif defined(CLS_MID_LOW_SPLIT) || defined(CLS_MID_HIGH_SPLIT)
inline constexpr int kMidBucketCount = 3;
#else
inline constexpr int kMidBucketCount = 2;
#endif

#ifdef CLS_BIG_LOW_SPLIT
inline constexpr int kBigBucketCount = 2;
#else
inline constexpr int kBigBucketCount = 1;
#endif

inline constexpr int kNumFrontBuckets = kSmallTierCount + kMidBucketCount + kBigBucketCount;

constexpr int front_bucket(int front_size)
{
    if (front_size <= kSmallFrontMax) {
#ifdef CLS_SMALL_BUCKET_SPLIT_16
        return (front_size <= kSmallSplitFrontMax) ? 0 : 1;
#else
        return 0;
#endif
    }

    const int mid0 = kSmallTierCount;
#ifdef CLS_NO_MID_SPLIT
    if (front_size <= kMidFrontMax) return mid0;
#elif defined(CLS_MID_LOW_SPLIT) && defined(CLS_MID_HIGH_SPLIT)
    if (front_size <= kMidLowSplitFrontMax) return mid0;
    if (front_size <= kMidSplitFrontMax) return mid0 + 1;
    if (front_size <= kMidHighSplitFrontMax) return mid0 + 2;
    if (front_size <= kMidFrontMax) return mid0 + 3;
#elif defined(CLS_MID_LOW_SPLIT)
    if (front_size <= kMidLowSplitFrontMax) return mid0;
    if (front_size <= kMidSplitFrontMax) return mid0 + 1;
    if (front_size <= kMidFrontMax) return mid0 + 2;
#elif defined(CLS_MID_HIGH_SPLIT)
    if (front_size <= kMidSplitFrontMax) return mid0;
    if (front_size <= kMidHighSplitFrontMax) return mid0 + 1;
    if (front_size <= kMidFrontMax) return mid0 + 2;
#else
    if (front_size <= kMidSplitFrontMax) return mid0;
    if (front_size <= kMidFrontMax) return mid0 + 1;
#endif

    const int big0 = kSmallTierCount + kMidBucketCount;
#ifdef CLS_BIG_LOW_SPLIT
    if (front_size <= kBigLowSplitFrontMax) return big0;
    return big0 + 1;
#else
    return big0;
#endif
}

// Round `value` up to the next multiple of `alignment` (alignment > 0).
constexpr int round_up_to_multiple(int value, int alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace custom_linear_solver
