// Measure etree level structure (AMD order) of matrices across sizes, to decide
// whether per-level CPU parallelism can help. Cycle 48 found level-parallel gave
// no speedup on <=15k matrices (narrow levels: ~15-66 cols/level -> per-level
// barrier overhead dominates). Large power-grid matrices may have wider levels.
// Reports num_levels, mean/max cols/level, and the % of columns in levels with
// >= 256 columns (wide enough to amortize a barrier across 12 threads).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <suitesparse/amd.h>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/reordering/metis_nd.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
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
// P A Pᵀ with values (perm[new]=old).
void permute_vals(const sparse_direct::matrix::CscMatrix& A, const std::vector<int>& p,
                  sparse_direct::matrix::CscMatrix& o)
{
    const int n = A.cols;
    std::vector<int> ip(n);
    for (int k = 0; k < n; ++k) ip[p[k]] = k;
    o.rows = o.cols = n;
    o.col_ptr.assign(n + 1, 0);
    for (int c = 0; c < n; ++c) o.col_ptr[ip[c] + 1] += A.col_ptr[c + 1] - A.col_ptr[c];
    for (int c = 0; c < n; ++c) o.col_ptr[c + 1] += o.col_ptr[c];
    o.row_idx.assign(A.col_ptr[n], 0);
    o.values.assign(A.col_ptr[n], 0.0);
    std::vector<int> nx(o.col_ptr.begin(), o.col_ptr.end());
    for (int c = 0; c < n; ++c)
        for (int q = A.col_ptr[c]; q < A.col_ptr[c + 1]; ++q) {
            const int d = nx[ip[c]]++;
            o.row_idx[d] = ip[A.row_idx[q]];
            o.values[d] = A.values[q];
        }
}
}  // namespace

int main()
{
    namespace sym = mysolver::symbolic;
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    const std::vector<std::pair<std::string, std::filesystem::path>> cases = {
        {"case6468rte", nr / "case6468rte/J.mtx"},
        {"case8387pegase", nr / "case8387pegase/J.mtx"},
        {"case_ACTIVSg25k", nr / "case_ACTIVSg25k/J.mtx"},
        {"case_SyntheticUSA", nr / "case_SyntheticUSA/J.mtx"},
    };
    std::printf("%-18s %-8s | symP   amd    permP  etree  fillP  | factor (analysis sub-phase ms)\n",
                "matrix", "n");
    for (const auto& [name, path] : cases) {
        const auto csr = sparse_direct::io::read_matrix_market_csr(path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;
        {
            auto t = clk::now();
            std::vector<int> scp, sri;
            sym::symmetric_pattern(n, csc.col_ptr.data(), csc.row_idx.data(), scp, sri);
            const double symP = ms(t, clk::now());
            std::vector<int> perm(n);
            double info[AMD_INFO];
            t = clk::now();
            amd_order(n, scp.data(), sri.data(), perm.data(), nullptr, info);
            const double amd = ms(t, clk::now());
            t = clk::now();
            std::vector<int> psp, psi;
            sym::permute_pattern(n, scp.data(), sri.data(), perm, psp, psi);
            const double permP = ms(t, clk::now());
            t = clk::now();
            std::vector<int> parent = sym::etree(n, psp.data(), psi.data());
            const double etreeP = ms(t, clk::now());
            t = clk::now();
            std::vector<int> Lp, Li;
            sym::fill_pattern(n, psp.data(), psi.data(), parent, Lp, Li);
            const double fillP = ms(t, clk::now());
            sparse_direct::matrix::CscMatrix M;
            permute_vals(csc, perm, M);
            mysolver::numeric::SparseLU lu;
            mysolver::numeric::factor_nopiv(n, M.col_ptr.data(), M.row_idx.data(), M.values.data(),
                                            Lp, Li, lu);  // warmup
            std::vector<double> ft;
            for (int r = 0; r < 3; ++r) {
                auto a = clk::now();
                mysolver::numeric::factor_nopiv(n, M.col_ptr.data(), M.row_idx.data(),
                                                M.values.data(), Lp, Li, lu);
                ft.push_back(ms(a, clk::now()));
            }
            std::printf("%-18s %-8d | %-6.1f %-6.1f %-6.1f %-6.1f %-6.1f | %.1f\n", name.c_str(), n,
                        symP, amd, permP, etreeP, fillP, median(ft));
        }
    }
    return 0;
}
