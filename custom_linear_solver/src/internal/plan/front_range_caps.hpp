#pragma once

// Front-size caps over a contiguous range of panels, for host-side dispatch
// sizing.
//
// ScanFrontRange() makes one pass over the panels in a plcols sub-range and
// returns the maximum front size and contribution-block / pivot-column
// dimensions in that range — the caps the factor and Solve kernels need to size
// their shared layout and pick thread counts / tier. Shared by the Factorize
// and Solve dispatchers (replaces the near-duplicate scan loops they used to
// carry).

#include "internal/plan/multifrontal_plan.hpp"
#include "internal/types.hpp"  // CLS_TC_UC_CAP — keep the max_uc clamp in lock-step with the TF32 gate

namespace custom_linear_solver {

struct FrontRangeCaps {
  int max_fsz = 0;       // largest front size in the range
  int max_uc = 1;        // largest contribution-block dim, clamped to <=
                         // CLS_TC_UC_CAP (shared-tile bound)
  int level_max_nc = 1;  // largest pivot-column count
  int level_max_uc = 1;  // largest contribution-block dim, unclamped
};

// Scan plcols[b..e) (indices into plan.h_front_ptr / plan.h_ncols) for the
// dispatch caps.
inline FrontRangeCaps ScanFrontRange(const plan::MultifrontalPlan& plan,
                                     const int* h_plc, int b, int e) {
  FrontRangeCaps m;
  for (int q = b; q < e; ++q) {
    const int pp = h_plc[q];
    const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
    const int nc = plan.h_ncols[pp];
    const int uc = fsz - nc;
    if (fsz > m.max_fsz) m.max_fsz = fsz;
    if (uc > m.max_uc && uc <= CLS_TC_UC_CAP) m.max_uc = uc;
    if (nc > m.level_max_nc) m.level_max_nc = nc;
    if (uc > m.level_max_uc) m.level_max_uc = uc;
  }
  return m;
}

}  // namespace custom_linear_solver
