// M2 exit tool (PLAN §254/§243/§264): run the symbolic stack on the real
// benchmark matrices, emit supernode statistics, and verify the elimination tree
// against CXSparse cs_etree on actual data.
//
// For each matrix the stats use the AMD ordering (KLU's default and mysolver's
// usual pick). Output columns:
//   matrix,rows,nnz,ordering,predicted_LU_fill,num_supernodes,avg_snode,max_snode,
//   num_levels,etree_matches_cxsparse

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <suitesparse/amd.h>
#include <suitesparse/cs.h>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
#include "mysolver/symbolic/schedule.hpp"
#include "mysolver/symbolic/supernode.hpp"
#include "tools/matrix_io.hpp"

namespace {

struct Case {
    std::string name;
    std::filesystem::path path;
};

std::vector<Case> m2_cases()
{
    const std::filesystem::path ss = "/datasets/benchmark_matrices/matrices";
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    return {
        {"memplus", ss / "memplus/memplus.mtx"},
        {"rajat27", ss / "rajat27/rajat27.mtx"},
        {"wang3", ss / "wang3/wang3.mtx"},
        {"onetone2", ss / "onetone2/onetone2.mtx"},
        {"rajat15", ss / "rajat15/rajat15.mtx"},
        {"case30", nr / "case30/J.mtx"},
        {"case118", nr / "case118/J.mtx"},
        {"case1197", nr / "case1197/J.mtx"},
        {"case_ACTIVSg2000", nr / "case_ACTIVSg2000/J.mtx"},
        {"case3012wp", nr / "case3012wp/J.mtx"},
        {"case6468rte", nr / "case6468rte/J.mtx"},
        {"case8387pegase", nr / "case8387pegase/J.mtx"},
    };
}

}  // namespace

int main()
{
    namespace sym = mysolver::symbolic;
    std::ofstream out("report/benchmark/symbolic_stats.csv");
    out << "matrix,rows,nnz,ordering,predicted_LU_fill,num_supernodes,"
           "avg_snode,max_snode,num_levels,etree_matches_cxsparse\n";

    int parity_failures = 0;
    for (const Case& c : m2_cases()) {
        const sparse_direct::matrix::CsrMatrix csr =
            sparse_direct::io::read_matrix_market_csr(c.path);
        const sparse_direct::matrix::CscMatrix csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;

        // AMD ordering on the symmetric pattern.
        std::vector<int> scp, sri;
        sym::symmetric_pattern(n, csc.col_ptr.data(), csc.row_idx.data(), scp, sri);
        std::vector<int> perm(n);
        double info[AMD_INFO];
        amd_order(n, scp.data(), sri.data(), perm.data(), nullptr, info);

        // Symbolic stack on the AMD-permuted symmetric pattern.
        std::vector<int> pcp, pri;
        sym::permute_pattern(n, csc.col_ptr.data(), csc.row_idx.data(), perm, pcp, pri);
        std::vector<int> spcp, spri;
        sym::symmetric_pattern(n, pcp.data(), pri.data(), spcp, spri);
        std::vector<int> parent = sym::etree(n, spcp.data(), spri.data());
        std::vector<int> post = sym::postorder(parent, n);
        std::vector<int> colcount = sym::column_counts(n, spcp.data(), spri.data(), parent, post);
        sym::SupernodePartition sp = sym::supernodes(n, parent, post, colcount);
        sym::SupernodeSchedule sched = sym::build_schedule(n, parent, sp);

        long lnz = 0;
        for (int v : colcount) {
            lnz += v;
        }
        const long predicted_fill = 2 * lnz - n;

        // Validate fill_pattern at scale: its nnz must equal lnz (colcount sum).
        std::vector<int> Lp, Li;
        sym::fill_pattern(n, spcp.data(), spri.data(), parent, Lp, Li);
        const bool fill_ok = (static_cast<long>(Li.size()) == lnz);
        int max_snode = 0;
        for (int s : sp.sizes) {
            if (s > max_snode) {
                max_snode = s;
            }
        }
        const double avg_snode =
            sp.num_supernodes > 0 ? static_cast<double>(n) / sp.num_supernodes : 0.0;

        // Verify etree against CXSparse on the real symmetric pattern.
        cs_di S{};
        S.nzmax = static_cast<int>(spri.size());
        S.m = n;
        S.n = n;
        S.p = spcp.data();
        S.i = spri.data();
        S.x = nullptr;
        S.nz = -1;
        int* cs_parent = cs_di_etree(&S, 0);
        bool matches = (cs_parent != nullptr);
        for (int k = 0; matches && k < n; ++k) {
            matches = (cs_parent[k] == parent[k]);
        }
        cs_di_free(cs_parent);
        if (!matches || !fill_ok) {
            ++parity_failures;
        }

        out << c.name << "," << n << "," << csc.nnz() << ",amd," << predicted_fill << ","
            << sp.num_supernodes << "," << avg_snode << "," << max_snode << ","
            << sched.num_levels << "," << (matches ? "true" : "false") << "\n";
        std::printf("%-18s n=%-7d snodes=%-7d avg=%6.2f max=%-6d levels=%-6d etree=%s fill=%s\n",
                    c.name.c_str(), n, sp.num_supernodes, avg_snode, max_snode,
                    sched.num_levels, matches ? "OK" : "MISMATCH", fill_ok ? "OK" : "BAD");
    }
    std::printf("etree-vs-cxsparse parity failures: %d\n", parity_failures);
    return parity_failures == 0 ? 0 : 1;
}
