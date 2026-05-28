// Compare the device-resident GPU triangular solve (cycle 86, graph-replayed
// fwd/bwd) against the CPU numeric::solve and the cuDSS solve reference, on
// METIS-ordered power-grid Jacobians. Factor with factor_nopiv (CPU), then time
// the solve phase only (the factor is already resident, as in cuDSS).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/gpu/gpu_solve.hpp"
#include "mysolver/reordering/metis_nd.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
#include "tools/matrix_io.hpp"

namespace {
using clk = std::chrono::steady_clock;
double ms(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}
void permute(const sparse_direct::matrix::CscMatrix& A, const std::vector<int>& p,
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
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
double berr(const sparse_direct::matrix::CscMatrix& M, const std::vector<double>& y,
            const std::vector<double>& b)
{
    const int n = M.cols;
    std::vector<double> r(b), den(n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int q = M.col_ptr[j]; q < M.col_ptr[j + 1]; ++q) {
            const int i = M.row_idx[q];
            r[i] -= M.values[q] * y[j];
            den[i] += std::abs(M.values[q]) * std::abs(y[j]);
        }
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = den[i] + std::abs(b[i]);
        if (d > 0.0) e = std::max(e, std::abs(r[i]) / d);
    }
    return e;
}
struct Case { std::string name; std::filesystem::path path; double cudss_solve; };
}  // namespace

int main()
{
    namespace sym = mysolver::symbolic;
    namespace num = mysolver::numeric;
    namespace gpu = mysolver::gpu;
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    std::vector<Case> cases = {
        {"case6468rte", nr / "case6468rte/J.mtx", 0.286},
        {"case8387pegase", nr / "case8387pegase/J.mtx", 0.363},
        {"case_ACTIVSg25k", nr / "case_ACTIVSg25k/J.mtx", 0.674},
        {"case_SyntheticUSA", nr / "case_SyntheticUSA/J.mtx", 1.443},
    };
    std::printf("GPU resident solve vs CPU solve vs cuDSS solve (METIS, solve phase only)\n");
    for (const Case& c : cases) {
        const auto csr = sparse_direct::io::read_matrix_market_csr(c.path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;
        std::vector<int> perm(n);
        mysolver::reordering::metis_nd(n, csc.col_ptr.data(), csc.row_idx.data(), perm);
        sparse_direct::matrix::CscMatrix M;
        permute(csc, perm, M);
        std::vector<int> mscp, msri;
        sym::symmetric_pattern(n, M.col_ptr.data(), M.row_idx.data(), mscp, msri);
        std::vector<int> parent = sym::etree(n, mscp.data(), msri.data());
        std::vector<int> Lp, Li;
        sym::fill_pattern(n, mscp.data(), msri.data(), parent, Lp, Li);

        num::SparseLU lu;
        num::factor_nopiv(n, M.col_ptr.data(), M.row_idx.data(), M.values.data(), Lp, Li, lu);
        std::vector<double> xt(n, 1.0), b(n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int q = M.col_ptr[j]; q < M.col_ptr[j + 1]; ++q)
                b[M.row_idx[q]] += M.values[q] * xt[j];

        // CPU solve.
        std::vector<double> yc;
        std::vector<double> tc;
        for (int r = 0; r < 7; ++r) { auto t0 = clk::now(); num::solve(lu, b, yc); tc.push_back(ms(t0, clk::now())); }
        const double berr_c = berr(M, yc, b);

        // GPU resident solve (graph-replayed), kernel ms.
        gpu::GpuSolvePlan sp = gpu::gpu_solve_analyze(lu, parent);
        std::vector<double> yg;
        gpu::gpu_solve_apply(sp, b, yg);  // warmup
        const double berr_g = berr(M, yg, b);
        std::vector<double> tk, tt;
        for (int r = 0; r < 7; ++r) {
            double k = 0.0;
            auto t0 = clk::now();
            gpu::gpu_solve_apply(sp, b, yg, &k);
            tt.push_back(ms(t0, clk::now()));
            tk.push_back(k);
        }
        std::printf("%-18s n=%-7d | CPU solve=%-8.3f berr=%-9.2e | GPU kern=%-8.3f tot=%-8.3f "
                    "berr=%-9.2e | cuDSS=%.3f\n",
                    c.name.c_str(), n, median(tc), berr_c, median(tk), median(tt), berr_g,
                    c.cudss_solve);
    }
    return 0;
}
