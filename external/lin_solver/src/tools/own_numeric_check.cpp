// M3 empirical check (gate-safe, separate from the production solver): run the
// mysolver-own numeric pipeline (no KLU) on the real benchmark matrices and
// report the backward error, so we know which matrices the current no-pivot
// factorization already handles vs. which need matching/scaling/refinement.

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/own_pipeline.hpp"
#include "tools/compute_error.hpp"
#include "tools/matrix_io.hpp"

namespace {
struct Case {
    std::string name;
    std::filesystem::path path;
};
std::vector<Case> cases()
{
    const std::filesystem::path ss = "/datasets/benchmark_matrices/matrices";
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    return {
        {"memplus", ss / "memplus/memplus.mtx"}, {"rajat27", ss / "rajat27/rajat27.mtx"},
        {"wang3", ss / "wang3/wang3.mtx"}, {"onetone2", ss / "onetone2/onetone2.mtx"},
        {"rajat15", ss / "rajat15/rajat15.mtx"}, {"case30", nr / "case30/J.mtx"},
        {"case118", nr / "case118/J.mtx"}, {"case1197", nr / "case1197/J.mtx"},
        {"case_ACTIVSg2000", nr / "case_ACTIVSg2000/J.mtx"}, {"case3012wp", nr / "case3012wp/J.mtx"},
        {"case6468rte", nr / "case6468rte/J.mtx"}, {"case8387pegase", nr / "case8387pegase/J.mtx"},
        {"case_ACTIVSg25k", nr / "case_ACTIVSg25k/J.mtx"},
        {"case_SyntheticUSA", nr / "case_SyntheticUSA/J.mtx"},
    };
}
}  // namespace

int main()
{
    int ok_count = 0, total = 0;
    std::printf("%-18s %-8s %-11s %-11s %-7s %-7s %-7s %s\n", "matrix", "n", "berr", "abs_err",
                "anal_ms", "fac_ms", "sol_ms", "ok");
    for (const Case& c : cases()) {
        ++total;
        const auto csr = sparse_direct::io::read_matrix_market_csr(c.path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const auto rhs = sparse_direct::io::read_matrix_market_vector(c.path.parent_path() / "rhs.mtx");
        const auto xt = sparse_direct::io::read_matrix_market_vector(c.path.parent_path() / "x_true.mtx");

        std::vector<double> x;
        mysolver::OwnSolveStats st;
        const bool solved = mysolver::try_own_solve(csc, rhs.values, x, &st);
        if (!solved) {
            std::printf("%-18s %-8d %-11s %-11s %-7s %-7s %-7s %s\n", c.name.c_str(), csc.cols,
                        "FACT_FAIL", "-", "-", "-", "-", "no");
            continue;
        }
        const auto m = sparse_direct::error::compute_error(csr, rhs.values, x, xt.values);
        const bool good = m.berr <= 1e-10 && m.absolute_error <= 1e-8;
        if (good) {
            ++ok_count;
        }
        std::printf("%-18s %-8d %-11.3e %-11.3e %-7.1f %-7.1f %-7.1f %s\n", c.name.c_str(),
                    csc.cols, m.berr, m.absolute_error, st.analysis_ms, st.factor_ms, st.solve_ms,
                    good ? "YES" : "no");
    }
    std::printf("own-numeric (no-pivot) handles %d / %d matrices at berr<=1e-10\n", ok_count, total);
    return 0;
}
