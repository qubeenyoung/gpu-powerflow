#include "symbolic/multifrontal.hpp"

#include <algorithm>
#include <thread>

namespace custom_linear_solver::symbolic {

namespace {
// cy260: parallelize independent per-panel work (each panel's front-rows / asm_idx
// depends only on read-only L + already-final parent rows). 12-core par_for, gated on
// panel count so small problems stay serial (thread overhead > gain).
template <typename Fn>
void par_panels(int lo, int hi, Fn&& fn)
{
    unsigned hw = std::thread::hardware_concurrency();
    const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
    if (hi - lo < 4096 || nth <= 1) { fn(lo, hi); return; }
    std::vector<std::thread> th;
    const int chunk = (hi - lo + nth - 1) / nth;
    for (int t = 0; t < nth; ++t) {
        const int a = lo + t * chunk, b = std::min(hi, a + chunk);
        if (a < b) th.emplace_back([&fn, a, b] { fn(a, b); });
    }
    for (auto& x : th) x.join();
}
}  // namespace

// Build the multifrontal symbolic structure from the fill pattern + panel partition.
//   In : Lp/Li (symbolic Cholesky), panels (amalgamated supernodes).
//   Out: per-panel front_ptr/front_rows (sorted member+CB rows), panel_parent, and the
//        extend-add map asm_ptr/asm_idx (where each CB row lands in the parent's front).
// Phases are parallelized across panels where independent (see the per-block notes).
MultifrontalSymbolic multifrontal_symbolic(int n, const std::vector<int>& Lp,
                                           const std::vector<int>& Li,
                                           const PanelPartition& panels)
{
    MultifrontalSymbolic mf;
    const int P = panels.num_panels;
    mf.num_panels = P;
    mf.panel_parent.assign(P, -1);
    mf.front_ptr.assign(static_cast<std::size_t>(P) + 1, 0);
    mf.asm_ptr.assign(static_cast<std::size_t>(P) + 1, 0);
    if (P == 0 || n <= 0) {
        return mf;
    }

    // Front rows = sorted union of the panel's member columns' L structures. The
    // marker `mark[i]==p` dedups within a panel; L is lower-triangular so every row
    // i >= the member column index >= panel's first column -> the panel's own
    // (contiguous) columns sort to the front, CB rows follow.
    // Phase 1 (parallel): each panel's front-rows depend only on read-only L. A
    // PER-THREAD marker array (mark[i]==p dedups within a panel; p is monotonic in a
    // chunk) makes panels fully independent -> byte-identical fr[p] regardless of
    // threading. The front_ptr prefix-sum stays serial after.
    std::vector<std::vector<int>> fr(P);
    par_panels(0, P, [&](int plo, int phi) {
        std::vector<int> mark(n, -1);  // per-thread (reused across this chunk's panels)
        for (int p = plo; p < phi; ++p) {
            const int first = panels.first[p];
            const int last = first + panels.ncols[p] - 1;
            std::vector<int>& rows = fr[p];
            rows.reserve(static_cast<std::size_t>(panels.width[p]));
            for (int j = first; j <= last; ++j)
                for (int q = Lp[j]; q < Lp[j + 1]; ++q) {
                    const int i = Li[q];
                    if (mark[i] != p) {
                        mark[i] = p;
                        rows.push_back(i);
                    }
                }
            std::sort(rows.begin(), rows.end());
        }
    });
    for (int p = 0; p < P; ++p)
        mf.front_ptr[p + 1] = mf.front_ptr[p] + static_cast<int>(fr[p].size());

    mf.front_rows.resize(static_cast<std::size_t>(mf.front_ptr[P]));
    par_panels(0, P, [&](int plo, int phi) {
        for (int p = plo; p < phi; ++p) {
            std::copy(fr[p].begin(), fr[p].end(), mf.front_rows.begin() + mf.front_ptr[p]);
        }
    });

    // Panel parent = the panel owning the first CB row (the etree parent of the
    // panel's last column is exactly that smallest sub-front row). Roots have no CB.
    for (int p = 0; p < P; ++p) {
        const int start = mf.front_ptr[p];
        const int nc = panels.ncols[p];
        if (mf.front_ptr[p + 1] - start > nc) {
            mf.panel_parent[p] = panels.panel_of[mf.front_rows[start + nc]];
        }
        mf.asm_ptr[p + 1] = mf.asm_ptr[p] + (mf.front_ptr[p + 1] - start - nc);
    }

    // Extend-add map: position of each CB row of P inside its parent's front_rows
    // (binary search; both are sorted). -1 flags an invariant violation.
    // Phase 4 (parallel): each panel writes a DISJOINT asm_idx[mf.asm_ptr[p]..) range
    // and only reads its parent's (already-final) front_rows -> race-free across panels.
    mf.asm_idx.assign(static_cast<std::size_t>(mf.asm_ptr[P]), -1);
    par_panels(0, P, [&](int plo, int phi) {
        for (int p = plo; p < phi; ++p) {
            const int par = mf.panel_parent[p];
            if (par < 0) {
                continue;
            }
            const int nc = panels.ncols[p];
            const int pstart = mf.front_ptr[par], pend = mf.front_ptr[par + 1];
            const int pfirst = panels.first[par];
            const int plast = pfirst + panels.ncols[par];
            int a = mf.asm_ptr[p];
            for (int k = mf.front_ptr[p] + nc; k < mf.front_ptr[p + 1]; ++k) {
                const int r = mf.front_rows[k];
                if (r >= pfirst && r < plast) {
                    mf.asm_idx[a] = pstart + (r - pfirst);
                } else {
                    const auto b = mf.front_rows.begin() + pstart;
                    const auto e = mf.front_rows.begin() + pend;
                    const auto it = std::lower_bound(b, e, r);
                    if (it != e && *it == r) {
                        mf.asm_idx[a] = static_cast<int>(it - mf.front_rows.begin());
                    }
                }
                ++a;
            }
        }
    });
    return mf;
}

}  // namespace custom_linear_solver::symbolic
