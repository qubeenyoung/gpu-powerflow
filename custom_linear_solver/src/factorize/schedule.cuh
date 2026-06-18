#pragma once

// FACTORIZE — elimination-tree schedule (entry point). One decision point,
// UseSingleSystem(st)
// (= B == 1, see internal/runtime/state.hpp), splits the two coherent paths:
//
//   B == 1  (single-system): the batched front-major schedule's per-level launch
//            / barrier latency is exposed, so factor each front with one block.
//            All precisions use the fused single-system kernels (Factorize/
//            single.cuh): the big-front triple is templated on the front type
//            (FP32 big fronts get the multi-block path too), and for TF32 the big
//            trailing runs on the Tensor Cores (FactorSingleBigTrailTf32). The
//            pivot blocks are then inverted (partitioned inverse) so the single-
//            system Solve runs parallel GEMVs.
//
//   B  > 1  (batched): walk the panel Etree level by level, forking independent
//   subtrees onto their
//            own streams, dispatching each tier-homogeneous range to its
//            dedicated kernel (small / mid / big,
//            Factorize/{small,mid,big}.cuh). Routing is a deterministic
//            function of front size (internal/types.hpp ClassifyFrontTier).
//
// IssueFactorLevels is wrapped by Factorize/Factorize.cu.

#include "factorize/big.cuh"
#include "factorize/mid.cuh"
#include "factorize/single.cuh"  // B=1 single-system factor schedule + partitioned-inverse
#include "factorize/small.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// Per-range dispatcher: scan the panels in plcols[b..e), pick the tier from the
// level's max front size, and dispatch one right-sized kernel. The big tier
// delegates to the mid-tier whole-front kernel at B==1 (see
// DispatchFactorBig).
static void IssueFactorLevelRange(const MultifrontalPlan& plan, State& st,
                                  cudaStream_t stream, int b, int e,
                                  const int* d_plc, const int* h_plc) {
  if (e <= b) return;
  const FrontRangeCaps caps = ScanFrontRange(plan, h_plc, b, e);
  const bool fp64 = (st.precision == Precision::FP64);
  switch (ClassifyFrontTier(caps.max_fsz, fp64)) {
    case FrontTier::kSmall: {
      // Occupancy gate: when the small tier under-fills the GPU (narrow levels)
      // give each front a whole block (FactorMid); when it saturates, pack
      // fronts into warps (FactorSmall). fp32 fronts are lighter, so count
      // their work ×2.
      const long work = (long)(e - b) * (long)st.batch_count;
      const long eff_work = fp64 ? work : work * 2;
      if (!FactorSaturates(eff_work))
        DispatchFactorMid(plan, st, stream, b, e, d_plc, h_plc, caps);
      else
        DispatchFactorSmall(plan, st, stream, b, e, d_plc, caps);
      return;
    }
    case FrontTier::kMid:
      DispatchFactorMid(plan, st, stream, b, e, d_plc, h_plc, caps);
      return;
    case FrontTier::kBig:
      DispatchFactorBig(plan, st, stream, b, e, d_plc, h_plc, caps);
      return;
  }
}

// Dispatch one (level- or cell-) range described by its (kNumTiers+1) tier
// boundaries `tb` into d_plc/h_plc: each non-empty tier sub-range -> its
// dedicated kernel. Same range = independent fronts, so the small->mid->big
// sub-launches are order-free and correct.
static void IssueFactorTiered(const MultifrontalPlan& plan, State& st,
                              cudaStream_t stream, const int* tb,
                              const int* d_plc, const int* h_plc) {
  constexpr int NT = MultifrontalPlan::kNumTiers;
  for (int t = 0; t < NT; ++t)
    if (tb[t + 1] > tb[t])
      IssueFactorLevelRange(plan, st, stream, tb[t], tb[t + 1], d_plc, h_plc);
}

// B == 1 single-system factor (all precisions), then partitioned-inverse the
// pivot blocks for the single-system Solve (invert applied once at the end).
//   FP64 -> scalar FactorSingleLevel + FP64 big-front triple.
//   FP32 -> scalar FactorSingleLevel + FT big-front triple (multi-block big).
//   TF32 -> same, but the multi-block big-front trailing runs on the Tensor Cores
//           (the dedicated FactorSingleBigTrailTf32: shared-staged Ozaki mma).
static void IssueFactorSingleSchedule(const MultifrontalPlan& plan, State& st,
                                      cudaStream_t stream) {
  IssueFactorSingleLevels(plan, st, stream);
  IssueFactorSingleInvert(plan, st, stream);
}

// B > 1 batched factor: multistream subtree sweep (when the plan exposes
// independent subtrees) or a single-stream tier-split level walk.
static void IssueFactorBatched(const MultifrontalPlan& plan, State& st,
                               cudaStream_t stream) {
  constexpr int NT = MultifrontalPlan::kNumTiers;
  const bool use_multistream = st.num_subtree_streams > 1 &&
                               plan.num_subtrees == st.num_subtree_streams &&
                               !plan.h_subtree_level_off.empty();
  const int* lvl_tb =
      plan.h_level_tier_off.data();  // single-stream: per-level ranges
  const int* sub_tb =
      plan.h_subtree_level_tier_off.data();  // multistream: per-cell ranges
  const bool have_lvl = !plan.h_level_tier_off.empty() && plan.d_plcols_tier;
  const bool have_sub = !plan.h_subtree_level_tier_off.empty();

  if (use_multistream) {
    const int spine_lo = (plan.spine_start_level >= 0) ? plan.spine_start_level
                                                       : plan.num_plevels;

    // Fork: record an event on the main stream and have each subtree stream
    // wait on it so the subtree work cannot start until any prior main-stream
    // work has retired.
    cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
    for (int k = 0; k < st.num_subtree_streams; ++k)
      cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                          static_cast<cudaEvent_t>(st.fork_event), 0);

    // Each subtree stream sweeps levels 0..spine_lo-1 over its own panel slice,
    // tier-splitting each (subtree, level) cell.
    for (int k = 0; k < st.num_subtree_streams; ++k) {
      cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
      const int* off =
          plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
      const int* cnt =
          plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
      for (int L = 0; L < plan.num_plevels; ++L) {
        if (L >= spine_lo) break;
        if (cnt[L] == 0) continue;
        if (have_sub)
          IssueFactorTiered(
              plan, st, ks,
              sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
              plan.d_plcols, plan.h_plcols.data());
        else
          IssueFactorLevelRange(plan, st, ks, off[L], off[L] + cnt[L],
                                plan.d_plcols, plan.h_plcols.data());
      }
      cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
    }

    // Join: main stream waits on every subtree's join event before issuing the
    // spine.
    for (int k = 0; k < st.num_subtree_streams; ++k)
      cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]),
                          0);

    // Spine levels (if any) on the main stream — cnt=1 chain, nothing to
    // tier-split.
    if (plan.spine_start_level >= 0)
      for (int L = plan.spine_start_level; L < plan.num_plevels; ++L)
        IssueFactorLevelRange(plan, st, stream, plan.panel_level_ptr[L],
                              plan.panel_level_ptr[L + 1], plan.d_plcols,
                              plan.h_plcols.data());
  } else {
    // Single stream: tier-split each level.
    for (int L = 0; L < plan.num_plevels; ++L) {
      if (have_lvl)
        IssueFactorTiered(plan, st, stream, lvl_tb + (long)L * (NT + 1),
                          plan.d_plcols_tier, plan.h_plcols_tier.data());
      else
        IssueFactorLevelRange(plan, st, stream, plan.panel_level_ptr[L],
                              plan.panel_level_ptr[L + 1], plan.d_plcols,
                              plan.h_plcols.data());
    }
  }
}

static void IssueFactorLevels(const MultifrontalPlan& plan, State& st,
                              cudaStream_t stream) {
  if (UseSingleSystem(st))
    IssueFactorSingleSchedule(plan, st, stream);
  else
    IssueFactorBatched(plan, st, stream);
}

}  // namespace
}  // namespace custom_linear_solver
