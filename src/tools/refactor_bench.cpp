// Repeated-factorization (Newton-Raphson) benchmark: analyze once, then refactor
// the same sparsity with new values many times. This is the regime where cuDSS's
// amortized analyze wins; own_refactor reuses the matching + AMD order + symbolic
// fill so each NR iteration pays only equilibration + permute + factor + solve.
// Reports first-solve (full) vs per-refactor time, and validates the reuse path.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/own_pipeline.hpp"
#include "tools/compute_error.hpp"
#include "tools/matrix_io.hpp"

namespace {
using clk = std::chrono::steady_clock;
double ms(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}
double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

int main()
{
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    const std::vector<std::pair<std::string, std::filesystem::path>> cases = {
        {"case6468rte", nr / "case6468rte/J.mtx"},
        {"case_ACTIVSg25k", nr / "case_ACTIVSg25k/J.mtx"},
        {"case_SyntheticUSA", nr / "case_SyntheticUSA/J.mtx"},
    };
    std::printf("%-18s %-8s | %-11s %-11s %-7s | %s\n", "matrix", "n", "first_solve", "per_refactor",
                "speedup", "refactor berr");
    for (const auto& [name, path] : cases) {
        const auto csr = sparse_direct::io::read_matrix_market_csr(path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const auto rhs = sparse_direct::io::read_matrix_market_vector(path.parent_path() / "rhs.mtx");
        const auto& b = rhs.values;

        // First solve: full analyze + factor + solve, capturing the reusable analysis.
        std::vector<double> x;
        mysolver::OwnAnalysis an;
        auto t0 = clk::now();
        const bool ok = mysolver::try_own_solve(csc, b, x, nullptr, &an);
        const double first_ms = ms(t0, clk::now());
        if (!ok) {
            std::printf("%-18s %-8d | first solve FAILED\n", name.c_str(), csc.cols);
            continue;
        }

        // Repeated refactor (same pattern + values here; in NR the values change).
        std::vector<double> xr;
        mysolver::own_refactor(csc, an, b, xr, nullptr);  // warmup
        std::vector<double> rt;
        for (int r = 0; r < 5; ++r) {
            auto tr = clk::now();
            mysolver::own_refactor(csc, an, b, xr, nullptr);
            rt.push_back(ms(tr, clk::now()));
        }
        const double refac_ms = median(rt);
        const auto m = sparse_direct::error::compute_error(csr, b, xr, b);  // berr only (xt unused)
        std::printf("%-18s %-8d | %-11.1f %-11.1f %-6.2fx | %.2e\n", name.c_str(), csc.cols,
                    first_ms, refac_ms, first_ms / refac_ms, m.berr);
    }
    return 0;
}
