#pragma once

// Per-(level, range) front-size statistics shared by the factorize and solve dispatchers.
//
// A single host-side pass over the panels in a plcols sub-range yields the caps the kernels need
// for shared-layout sizing and tier selection. Keeping it here removes the near-duplicate scan
// loops the two dispatchers used to carry.

#include "plan/multifrontal_plan.hpp"

namespace custom_linear_solver {

struct LevelMetrics {
    int max_fsz = 0;        // largest front size in the range
    int max_uc = 1;        // largest contribution-block dim, clamped to <= 256 (shared-tile bound)
    int level_max_nc = 1;  // largest pivot-column count
    int level_max_uc = 1;  // largest contribution-block dim, unclamped
};

// Scan plcols[b..e) (indices into plan.h_front_ptr / plan.h_ncols) for the dispatch caps.
inline LevelMetrics scan_level_metrics(const plan::MultifrontalPlan& plan, const int* h_plc,
                                       int b, int e)
{
    LevelMetrics m;
    for (int q = b; q < e; ++q) {
        const int pp = h_plc[q];
        const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
        const int nc = plan.h_ncols[pp];
        const int uc = fsz - nc;
        if (fsz > m.max_fsz) m.max_fsz = fsz;
        if (uc > m.max_uc && uc <= 256) m.max_uc = uc;
        if (nc > m.level_max_nc) m.level_max_nc = nc;
        if (uc > m.level_max_uc) m.level_max_uc = uc;
    }
    return m;
}

}  // namespace custom_linear_solver
