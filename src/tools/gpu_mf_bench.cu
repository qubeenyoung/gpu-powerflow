// Compare the multifrontal GPU factor (cycle 81) against the cy71 right-looking
// scatter and the cuDSS factor reference, on METIS-ordered power-grid Jacobians.
// Validates the MF path (berr on the real matrix) and times its kernel.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/gpu/gpu_factor.hpp"
#include "mysolver/gpu/gpu_mf.hpp"
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
    // cy262: race-free over source columns (ip bijection -> disjoint target columns). Parallelize.
    auto fill = [&](int clo, int chi) {
        for (int c = clo; c < chi; ++c)
            for (int q = A.col_ptr[c]; q < A.col_ptr[c + 1]; ++q) {
                const int d = nx[ip[c]]++;
                o.row_idx[d] = ip[A.row_idx[q]];
                o.values[d] = A.values[q];
            }
    };
    unsigned hw = std::thread::hardware_concurrency();
    const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
    if (n < 32768 || nth <= 1) {
        fill(0, n);
    } else {
        std::vector<std::thread> th;
        const int chunk = (n + nth - 1) / nth;
        for (int t = 0; t < nth; ++t) {
            const int a = t * chunk, b = std::min(n, a + chunk);
            if (a < b) th.emplace_back([&fill, a, b] { fill(a, b); });
        }
        for (auto& x : th) x.join();
    }
}
double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
double componentwise_berr(const sparse_direct::matrix::CscMatrix& M,
                          const std::vector<double>& y, const std::vector<double>& b)
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
struct Case { std::string name; std::filesystem::path path; double cudss_fac; double cudss_solve; };
}  // namespace

int main()
{
    namespace sym = mysolver::symbolic;
    namespace gpu = mysolver::gpu;
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    // cuDSS factor/solve refs = WARM median-7 (CUDSS_REPEAT=7, handle reused), measured cy340 by
    // `benchmark --solver cudss-gpu` on THIS machine -- the fair warm-vs-warm match to our gpu_mf
    // warm median-7. (Superseded the older 0.612/0.286-style refs, which were optimistic for cuDSS:
    // the on-machine warm cuDSS is slightly slower, e.g. case6468 F 0.667 not 0.612.)
    std::vector<Case> cases = {
        {"case6468rte", nr / "case6468rte/J.mtx", 0.667, 0.331},
        {"case8387pegase", nr / "case8387pegase/J.mtx", 1.145, 0.423},
        {"case_ACTIVSg25k", nr / "case_ACTIVSg25k/J.mtx", 2.113, 0.786},
        {"case_SyntheticUSA", nr / "case_SyntheticUSA/J.mtx", 5.754, 1.700},
    };
    std::printf("MF GPU factor + solve vs cuDSS (METIS ordering)\n");
    for (const Case& c : cases) {
        const auto csr = sparse_direct::io::read_matrix_market_csr(c.path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;
        const auto a0 = clk::now();
        std::vector<int> perm(n);
        const bool ptime = std::getenv("PREP_TIME") != nullptr;  // cy178: prep sub-phase profiling
        auto pt = clk::now();
        auto plap = [&](const char* nm) { if (ptime) { std::fprintf(stderr, "    [prep] %-18s %.1f ms\n", nm, ms(pt, clk::now())); pt = clk::now(); } };
        mysolver::reordering::metis_nd(n, csc.col_ptr.data(), csc.row_idx.data(), perm);
        plap("metis_nd(order)");
        sparse_direct::matrix::CscMatrix M;
        permute(csc, perm, M);
        plap("permute");
        std::vector<int> mscp, msri;
        sym::symmetric_pattern(n, M.col_ptr.data(), M.row_idx.data(), mscp, msri);
        plap("symmetric_pattern");
        std::vector<int> parent = sym::etree(n, mscp.data(), msri.data());
        plap("etree");
        std::vector<int> Lp, Li;
        sym::fill_pattern(n, mscp.data(), msri.data(), parent, Lp, Li);
        plap("fill_pattern");
        if (std::getenv("MF_FILL"))
            std::fprintf(stderr, "  [fill] %-18s n=%d fill_nnz=%lld (%.1fx input)\n", c.name.c_str(),
                         n, (long long)Lp[n], Lp[n] / (double)csr.values.size());
        const double prep_ms = ms(a0, clk::now());  // ordering + symbolic (analysis prep)

        // True solution probe b = M * ones.
        std::vector<double> xt(n, 1.0), b(n, 0.0), y;
        for (int j = 0; j < n; ++j)
            for (int q = M.col_ptr[j]; q < M.col_ptr[j + 1]; ++q)
                b[M.row_idx[q]] += M.values[q] * xt[j];

        // cy71 right-looking.
        gpu::GpuFactorPlan rl = gpu::gpu_analyze(n, M.col_ptr.data(), M.row_idx.data(), Lp, Li,
                                                 parent);
        mysolver::numeric::SparseLU lu_rl;
        gpu::gpu_factorize(rl, M.values.data(), lu_rl);
        mysolver::numeric::solve(lu_rl, b, y);
        const double berr_rl = componentwise_berr(M, y, b);
        std::vector<double> krl;
        for (int r = 0; r < 7; ++r) {
            double k = 0.0;
            gpu::gpu_factorize(rl, M.values.data(), lu_rl, &k);
            krl.push_back(k);
        }

        // cycle 81 multifrontal. Time gpu_mf_analyze (symbolic + maps + upload); the
        // full GPU analysis A = prep (ordering+symbolic) + this (one-time, NR-amortized).
        const auto an0 = clk::now();
        const bool fp32 = std::getenv("MF_FP32") != nullptr;  // mixed-precision factor+solve
        gpu::GpuMfPlan mf = gpu::gpu_mf_analyze(n, M.col_ptr.data(), M.row_idx.data(), Lp, Li,
                                               parent, 8, fp32);
        const double analyze_ms = prep_ms + ms(an0, clk::now());
        double berr_mf = -1.0, kmf_med = -1.0;
        double gsolve_med = -1.0, csolve_med = -1.0, berr_gs = -1.0;
        int nplev = mf.num_plevels;
        if (std::getenv("MF_LEVELW")) {  // diagnostic: max fronts per level (cooperative-grid fit)
            int mw = 0;
            for (int L = 0; L < mf.num_plevels; ++L)
                mw = std::max(mw, mf.plptr[L + 1] - mf.plptr[L]);
            std::fprintf(stderr, "  %s plev=%d max_level_width=%d\n", c.name.c_str(),
                         mf.num_plevels, mw);
        }
        if (mf.num_panels > 0) {
            mysolver::numeric::SparseLU lu_mf;
            const bool ok = gpu::gpu_mf_factorize(mf, M.values.data(), lu_mf);
            std::vector<double> ym;
            mysolver::numeric::solve(lu_mf, b, ym);
            berr_mf = ok ? componentwise_berr(M, ym, b) : 1e9;
            std::vector<double> kmf;
            for (int r = 0; r < 7; ++r) {
                double k = 0.0;
                gpu::gpu_mf_factorize(mf, M.values.data(), lu_mf, &k);
                kmf.push_back(k);
            }
            kmf_med = median(kmf);

            // MF GPU solve (reuses the just-factored fronts) vs CPU solve.
            std::vector<double> xs;
            gpu::gpu_mf_solve(mf, b, xs);  // warmup
            berr_gs = componentwise_berr(M, xs, b);
            std::vector<double> kg, tc;
            for (int r = 0; r < 7; ++r) {
                double k = 0.0;
                gpu::gpu_mf_solve(mf, b, xs, &k);
                kg.push_back(k);
            }
            gsolve_med = median(kg);
            std::vector<double> xc;
            for (int r = 0; r < 7; ++r) {
                auto t0 = clk::now();
                mysolver::numeric::solve(lu_mf, b, xc);
                tc.push_back(ms(t0, clk::now()));
            }
            csolve_med = median(tc);
        }
        std::printf("%-18s n=%-7d | A=%-7.2f F=%-7.3f S=%-7.3f E2E=%-8.2f | cuDSS F=%-6.3f "
                    "S=%-6.3f berr=%-9.2e\n",
                    c.name.c_str(), n, analyze_ms, kmf_med, gsolve_med,
                    analyze_ms + kmf_med + gsolve_med, c.cudss_fac, c.cudss_solve, berr_gs);
    }
    return 0;
}
