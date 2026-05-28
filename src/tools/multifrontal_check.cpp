// Validate the multifrontal symbolic analysis (cycle 79, the dense-panel build's
// assembly backbone). For each power-grid Jacobian: AMD-order, build the fill
// pattern + relaxed panels, then multifrontal_symbolic, and check the multifrontal
// invariants that the dense-panel factorization relies on:
//   (1) each front's first `ncols` rows are exactly the panel's pivot columns;
//   (2) every CB row maps into the parent front (asm_idx >= 0) -- i.e.
//       cb_rows[child] subset of front_rows[parent];
//   (3) the panel etree is acyclic with parent panel id > child (postorder).
// Reports front-storage size + max front (the dense-kernel sizing) and any failures.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <suitesparse/amd.h>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
#include "mysolver/symbolic/multifrontal.hpp"
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
    int total_fail = 0;
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
        for (int j = 0; j < n; ++j) colcount[j] = Lp[j + 1] - Lp[j];

        const sym::PanelPartition pp = sym::relaxed_panels(n, parent, colcount, 16);
        const sym::MultifrontalSymbolic mf = sym::multifrontal_symbolic(n, Lp, Li, pp);

        int bad_pivot = 0, bad_asm = 0, bad_tree = 0, max_front = 0;
        long front_storage = 0;
        for (int p = 0; p < mf.num_panels; ++p) {
            const int start = mf.front_ptr[p], end = mf.front_ptr[p + 1];
            const int nc = pp.ncols[p], first = pp.first[p];
            const int fsz = end - start;
            front_storage += fsz;
            max_front = std::max(max_front, fsz);
            // (1) first nc rows == panel pivot columns first..first+nc-1
            for (int k = 0; k < nc; ++k)
                if (mf.front_rows[start + k] != first + k) ++bad_pivot;
            // (3) parent panel id strictly greater (postorder => acyclic, child first)
            if (mf.panel_parent[p] != -1 && mf.panel_parent[p] <= p) ++bad_tree;
        }
        // (2) every assembly index resolved (CB row found in parent front)
        for (int a : mf.asm_idx)
            if (a < 0) ++bad_asm;

        const int fail = bad_pivot + bad_asm + bad_tree;
        total_fail += fail;
        std::printf("%-18s n=%-7d panels=%-7d front_store=%-9ld max_front=%-4d "
                    "asm_ops=%-9zu | pivot_bad=%d asm_bad=%d tree_bad=%d %s\n",
                    name.c_str(), n, mf.num_panels, front_storage, max_front,
                    mf.asm_idx.size(), bad_pivot, bad_asm, bad_tree,
                    fail == 0 ? "OK" : "FAIL");
    }
    std::printf("%s (%d failures)\n", total_fail == 0 ? "ALL PASS" : "FAILURES", total_fail);
    return total_fail == 0 ? 0 : 1;
}
