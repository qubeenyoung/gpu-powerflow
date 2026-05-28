// Validate the MC64 max-product matching (gate-safe, standalone). (1) Small cases
// with a known optimal assignment. (2) The circuit matrices where the no-pivot
// own-numeric pipeline currently fails (rajat27/onetone2 zero-pivot, rajat15
// inaccurate): report the diagonal min/zero-count before vs. after the matching,
// to confirm MC64 moves large entries onto the diagonal.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/reordering/mc64.hpp"
#include "tools/matrix_io.hpp"

namespace {

bool is_permutation(const std::vector<int>& p)
{
    const int n = static_cast<int>(p.size());
    std::vector<char> seen(n, 0);
    for (int x : p) {
        if (x < 0 || x >= n || seen[x]) return false;
        seen[x] = 1;
    }
    return true;
}

void small_case(const char* name, int n, const std::vector<int>& Ap, const std::vector<int>& Ai,
                const std::vector<double>& Ax, const std::vector<int>& expect)
{
    std::vector<int> m;
    const bool ok = mysolver::reordering::mc64_match(n, Ap.data(), Ai.data(), Ax.data(), m);
    bool pass = ok && is_permutation(m) && m == expect;
    std::printf("  [small] %-10s match=", name);
    for (int x : m) std::printf("%d ", x);
    std::printf("expected=");
    for (int x : expect) std::printf("%d ", x);
    std::printf("-> %s\n", pass ? "PASS" : "FAIL");
}

}  // namespace

int main()
{
    std::printf("=== MC64 matching: known small cases ===\n");
    // 2x2: [[1,100],[100,1]] -> swap is optimal (100*100 >> 1*1).
    small_case("2x2", 2, {0, 2, 4}, {0, 1, 0, 1}, {1.0, 100.0, 100.0, 1.0}, {1, 0});
    // 3x3: tiny natural diagonal, large off-diagonals force a 3-cycle.
    small_case("3x3", 3, {0, 2, 4, 6}, {0, 1, 1, 2, 0, 2}, {0.01, 5.0, 0.01, 6.0, 7.0, 0.01},
               {1, 2, 0});

    std::printf("\n=== circuit matrices: diagonal before vs. after MC64 ===\n");
    const std::filesystem::path ss = "/datasets/benchmark_matrices/matrices";
    const std::vector<std::pair<std::string, std::filesystem::path>> cases = {
        {"rajat27", ss / "rajat27/rajat27.mtx"},
        {"onetone2", ss / "onetone2/onetone2.mtx"},
        {"rajat15", ss / "rajat15/rajat15.mtx"},
        {"memplus", ss / "memplus/memplus.mtx"},
    };
    std::printf("%-10s %-7s | %-12s %-8s | %-12s %-8s %s\n", "matrix", "n", "min|diag|0", "zeros0",
                "min|diag|MC", "zerosMC", "perm");
    for (const auto& [name, path] : cases) {
        const auto csr = sparse_direct::io::read_matrix_market_csr(path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;

        // helper: |A(i,j)| lookup in CSC column j
        auto aij = [&](int i, int j) -> double {
            for (int p = csc.col_ptr[j]; p < csc.col_ptr[j + 1]; ++p)
                if (csc.row_idx[p] == i) return std::abs(csc.values[p]);
            return 0.0;
        };

        double min0 = 1e300;
        int zeros0 = 0;
        for (int j = 0; j < n; ++j) {
            const double d = aij(j, j);
            min0 = std::min(min0, d);
            if (d < 1e-12) ++zeros0;
        }

        std::vector<int> m;
        const bool ok = mysolver::reordering::mc64_match(n, csc.col_ptr.data(),
                                                         csc.row_idx.data(), csc.values.data(), m);
        if (!ok) {
            std::printf("%-10s %-7d | %-12.3e %-8d | NO PERFECT MATCHING\n", name.c_str(), n, min0,
                        zeros0);
            continue;
        }
        double minM = 1e300;
        int zerosM = 0;
        for (int j = 0; j < n; ++j) {
            const double d = aij(m[j], j);  // matched diagonal entry
            minM = std::min(minM, d);
            if (d < 1e-12) ++zerosM;
        }
        std::printf("%-10s %-7d | %-12.3e %-8d | %-12.3e %-8d %s\n", name.c_str(), n, min0, zeros0,
                    minM, zerosM, is_permutation(m) ? "valid" : "INVALID");
    }
    return 0;
}
