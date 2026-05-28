// Measure the block-triangular-form (BTF) reducibility of the benchmark matrices
// (gate-safe diagnostic). KLU's speed on circuit matrices comes from BTF: it
// factors only the diagonal blocks. If a matrix is highly reducible (many small
// blocks, largest << n) BTF would dramatically cut my factor work; if it is one
// big block, BTF cannot help. Decides whether implementing BTF is worth it.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <suitesparse/btf.h>

#include "matrix/sparse_matrix.hpp"
#include "tools/matrix_io.hpp"

int main()
{
    const std::filesystem::path ss = "/datasets/benchmark_matrices/matrices";
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    const std::vector<std::pair<std::string, std::filesystem::path>> cases = {
        {"rajat27", ss / "rajat27/rajat27.mtx"},   {"onetone2", ss / "onetone2/onetone2.mtx"},
        {"rajat15", ss / "rajat15/rajat15.mtx"},   {"memplus", ss / "memplus/memplus.mtx"},
        {"case6468rte", nr / "case6468rte/J.mtx"}, {"case_ACTIVSg2000", nr / "case_ACTIVSg2000/J.mtx"},
    };
    std::printf("%-16s %-8s %-8s %-10s %-10s %s\n", "matrix", "n", "nblocks", "max_block",
                "max%", "BTF helps?");
    for (const auto& [name, path] : cases) {
        const auto csr = sparse_direct::io::read_matrix_market_csr(path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;
        std::vector<int> Ap(csc.col_ptr.begin(), csc.col_ptr.end());
        std::vector<int> Ai(csc.row_idx.begin(), csc.row_idx.end());
        std::vector<int> P(n), Q(n), R(n + 1), Work(5 * n);
        double work = 0.0;
        int nmatch = 0;
        const int nblocks = btf_order(n, Ap.data(), Ai.data(), 0.0, &work, P.data(), Q.data(),
                                      R.data(), &nmatch, Work.data());
        int maxblk = 0;
        for (int b = 0; b < nblocks; ++b) maxblk = std::max(maxblk, R[b + 1] - R[b]);
        const double maxpct = 100.0 * maxblk / n;
        std::printf("%-16s %-8d %-8d %-10d %-9.1f%% %s\n", name.c_str(), n, nblocks, maxblk, maxpct,
                    maxpct < 90.0 ? "YES" : "no (irreducible)");
    }
    return 0;
}
