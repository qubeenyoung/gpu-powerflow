// Gating measurement for the cuDSS-grade GPU path (relaxed supernode amalgamation
// + dense batched BLAS). Fundamental supernodes here average ~2 columns (cycle 41)
// -> no BLAS-worthy blocks. RELAXED amalgamation merges etree chains into panels of
// up to S columns, padding each column to the panel's (top column's) row structure.
// If the padded-fill ratio stays modest (<~2x), dense GPU panels are viable; if it
// explodes, the dense path is closed (cuDSS must use a fine-grained scheme instead).

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <suitesparse/amd.h>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
#include "mysolver/symbolic/supernode.hpp"
#include "tools/matrix_io.hpp"

int main()
{
    namespace sym = mysolver::symbolic;
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    const std::vector<std::pair<std::string, std::filesystem::path>> cases = {
        {"case6468rte", nr / "case6468rte/J.mtx"},
        {"case_ACTIVSg25k", nr / "case_ACTIVSg25k/J.mtx"},
        {"case_SyntheticUSA", nr / "case_SyntheticUSA/J.mtx"},
    };
    std::printf("%-18s %-8s %-10s | padded-fill ratio (mean panel cols) at panel cap S\n",
                "matrix", "n", "true_fill");
    for (const auto& [name, path] : cases) {
        const auto csr = sparse_direct::io::read_matrix_market_csr(path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;
        std::vector<int> scp, sri;
        sym::symmetric_pattern(n, csc.col_ptr.data(), csc.row_idx.data(), scp, sri);
        std::vector<int> perm(n);
        double info[AMD_INFO];
        amd_order(n, scp.data(), sri.data(), perm.data(), nullptr, info);
        std::vector<int> psp, psi;
        sym::permute_pattern(n, scp.data(), sri.data(), perm, psp, psi);
        std::vector<int> parent = sym::etree(n, psp.data(), psi.data());
        std::vector<int> Lp, Li;
        sym::fill_pattern(n, psp.data(), psi.data(), parent, Lp, Li);
        std::vector<int> colcount(n);
        long true_fill = 0;
        for (int j = 0; j < n; ++j) {
            colcount[j] = Lp[j + 1] - Lp[j];
            true_fill += colcount[j];
        }
        std::printf("%-18s %-8d %-10ld |", name.c_str(), n, true_fill);
        // AMD postorders the etree, so etree chains are consecutive indices; the
        // relaxed-amalgamation panel builder merges chains (parent[j]==j+1) capped
        // at S columns. (Shared with the dense-panel multifrontal path.)
        for (int S : {8, 16, 32}) {
            const sym::PanelPartition pp = sym::relaxed_panels(n, parent, colcount, S);
            std::printf("  S=%d:%.2fx(%.1f)", S, double(pp.padded_fill) / true_fill,
                        double(n) / pp.num_panels);
        }
        // Right-looking GPU viability probe: how many updaters write each target
        // entry (i,j)? That is the atomicAdd contention a right-looking kernel would
        // pay. Build the symmetric fill from Lp/Li, then count ops per target.
        std::vector<std::vector<int>> cols(n);
        for (int j = 0; j < n; ++j)
            for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
                const int i = Li[p];
                cols[j].push_back(i);
                if (i != j) cols[i].push_back(j);
            }
        for (int j = 0; j < n; ++j) {
            std::sort(cols[j].begin(), cols[j].end());
            cols[j].erase(std::unique(cols[j].begin(), cols[j].end()), cols[j].end());
        }
        std::vector<int> mark(n, -1), tgtcnt;  // updaters per target (i,j)
        std::vector<int> pos(n, -1);
        long total_ops = 0;
        int maxc = 0;
        for (int j = 0; j < n; ++j) {
            for (int t = 0; t < (int)cols[j].size(); ++t) { mark[cols[j][t]] = j; pos[cols[j][t]] = t; }
            std::vector<int> cnt(cols[j].size(), 0);
            for (int k : cols[j]) {
                if (k >= j) break;
                for (int i : cols[k])
                    if (i > k && mark[i] == j) { ++cnt[pos[i]]; ++total_ops; }
            }
            for (int c : cnt) maxc = std::max(maxc, c);
        }
        const long ntargets = total_ops > 0 ? 0 : 0;
        (void)ntargets;
        std::printf("  | right-look: ops=%ld mean_updaters/target=%.1f max=%d\n", total_ops,
                    double(total_ops) / (Lp[n] * 2.0 - n), maxc);

        // Multifrontal go/no-go gate (computed from symbolic, no factorization).
        // The right-looking scatter does `total_ops` SCATTERED atomicAdds. A dense-
        // panel multifrontal does the SAME arithmetic but inside dense fronts
        // (coalesced/BLAS), leaving only the EXTEND-ADD (child contribution block ->
        // parent front) scattered. Per panel the CB is (fp-np)x(fp-np) dense, so
        // extend-add ops = Σ(fp-np)^2 with fp = front rows (~widest member colcount)
        // and np = panel cols. If Σ(fp-np)^2 << total_ops, multifrontal converts most
        // scattered atomics to dense work -> the GPU build is worth it. Also report
        // dense factor flops (Σ np*fp^2, the coalesced part) for completeness.
        for (int S : {8, 16, 32}) {
            const sym::PanelPartition pp = sym::relaxed_panels(n, parent, colcount, S);
            long extend_add = 0, dense_flops = 0;
            for (int p = 0; p < pp.num_panels; ++p) {
                const long fp = pp.width[p], np = pp.ncols[p];
                const long cb = fp > np ? fp - np : 0;
                extend_add += cb * cb;
                dense_flops += np * fp * fp;
            }
            std::printf("    [MF S=%d] extend_add=%-11ld (scatter/extend=%.1fx)  dense_flops=%ld\n",
                        S, extend_add,
                        extend_add > 0 ? double(total_ops) / extend_add : 0.0, dense_flops);
        }
    }
    return 0;
}
