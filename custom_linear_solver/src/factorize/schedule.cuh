#pragma once

// FACTORIZE — elimination-tree schedule (entry point). Walks the panel-etree level by level,
// forking independent subtrees onto their own streams, and dispatches each range to the small /
// mid / big tier (factorize/{small,mid,big}.cuh). issue_factor_levels is wrapped by factorize/factorize.cu.

#include "factorize/small.cuh"
#include "factorize/mid.cuh"
#include "factorize/big.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// Per-range dispatcher: scan the panels in plcols[b..e), pick the tier from the level's max front
// size, and dispatch one right-sized kernel. A mid range whose shared layout overflows the
// per-block budget falls through to the big tier on the same range.
static void issue_factor_level_range(const MultifrontalPlan& plan, State& st,
                                     cudaStream_t stream, int b, int e,
                                     const int* d_plc, const int* h_plc)
{
    if (e <= b) return;
    const FrontRangeCaps caps = scan_front_range(plan, h_plc, b, e);
    switch (classify_front_tier(caps.max_fsz)) {
        case FrontTier::kSmall:
            dispatch_factor_small(plan, st, stream, b, e, d_plc, caps);
            return;
        case FrontTier::kLarge:
            // Large fronts try the shared-resident mid kernel first; the mid/big split is the
            // per-precision shared-budget fit decided inside dispatch_factor_mid (fsz*fsz*sizeof(elem)
            // <= the opt-in budget). Fronts that overflow fall through to the global big tier.
            if (dispatch_factor_mid(plan, st, stream, b, e, d_plc, h_plc, caps)) return;
            dispatch_factor_big(plan, st, stream, b, e, d_plc, h_plc, caps);
            return;
    }
}

// Dispatch one (level- or cell-) range described by its (kNumTiers+1) tier boundaries `tb` into
// d_plc/h_plc. Splits into tier-homogeneous sub-launches when the occupancy gate passes, else
// dispatches the whole range on the larger kernel (pre-split behaviour). Same range = independent
// fronts, so the small→mid→big sub-launches are order-free and correct.
static void issue_factor_tiered(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                const int* tb, const int* d_plc, const int* h_plc)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    constexpr int NS = MultifrontalPlan::kSmallTiers;
    const long small_cnt = tb[NS] - tb[0];
    const bool mixed = (tb[NT] - tb[0]) > small_cnt;  // a larger-tier front is present
    if (st.tier_split && small_cnt > 0 && mixed && small_cnt * (long)st.batch_count >= factor_warp_fill()) {
        for (int t = 0; t < NT; ++t)
            if (tb[t + 1] > tb[t])
                issue_factor_level_range(plan, st, stream, tb[t], tb[t + 1], d_plc, h_plc);
    } else {
        issue_factor_level_range(plan, st, stream, tb[0], tb[NT], d_plc, h_plc);
    }
}

static void issue_factor_levels(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    const bool use_multistream = st.num_subtree_streams > 1 &&
                                 plan.num_subtrees == st.num_subtree_streams &&
                                 !plan.h_subtree_level_off.empty();
    const int* lvl_tb = plan.h_level_tier_off.data();           // single-stream: per-level ranges
    const int* sub_tb = plan.h_subtree_level_tier_off.data();   // multistream: per-cell ranges
    const bool have_lvl = !plan.h_level_tier_off.empty() && plan.d_plcols_tier;
    const bool have_sub = !plan.h_subtree_level_tier_off.empty();

    if (use_multistream) {
        const int spine_lo = (plan.spine_start_level >= 0) ? plan.spine_start_level : plan.num_plevels;

        // Fork: record an event on the main stream and have each subtree stream wait on it
        // so the subtree work cannot start until any prior main-stream work has retired.
        cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                                static_cast<cudaEvent_t>(st.fork_event), 0);
        }

        // Each subtree stream sweeps levels 0..spine_lo-1 over its own panel slice, tier-splitting
        // each (subtree, level) cell under the same occupancy gate as the single-stream walk.
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
            const int* off = plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
            const int* cnt = plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
            for (int L = 0; L < plan.num_plevels; ++L) {
                if (L >= spine_lo) break;
                if (cnt[L] == 0) continue;
                if (have_sub)
                    issue_factor_tiered(plan, st, ks,
                                        sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
                                        plan.d_plcols, plan.h_plcols.data());
                else
                    issue_factor_level_range(plan, st, ks, off[L], off[L] + cnt[L],
                                             plan.d_plcols, plan.h_plcols.data());
            }
            cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
        }

        // Join: main stream waits on every subtree's join event before issuing the spine.
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);
        }

        // Spine levels (if any) on the main stream — cnt=1 chain, nothing to tier-split.
        if (plan.spine_start_level >= 0) {
            for (int L = plan.spine_start_level; L < plan.num_plevels; ++L) {
                const int b = plan.panel_level_ptr[L], e = plan.panel_level_ptr[L + 1];
                issue_factor_level_range(plan, st, stream, b, e,
                                         plan.d_plcols, plan.h_plcols.data());
            }
        }
    } else {
        // Single stream: tier-split each level under the occupancy gate (see issue_factor_tiered).
        for (int L = 0; L < plan.num_plevels; ++L) {
            if (have_lvl)
                issue_factor_tiered(plan, st, stream, lvl_tb + (long)L * (NT + 1),
                                    plan.d_plcols_tier, plan.h_plcols_tier.data());
            else
                issue_factor_level_range(plan, st, stream, plan.panel_level_ptr[L], plan.panel_level_ptr[L + 1],
                                         plan.d_plcols, plan.h_plcols.data());
        }
    }
}

}  // namespace
}  // namespace custom_linear_solver
