// Probe whether the multifrontal GPU factor (power-grid-tuned) extends to high-fill
// CIRCUIT matrices (rajat/onetone), where the cy71 right-looking kernel OOM'd. The
// MF analyze bails (returns num_panels==0) if the front arena would exceed ~8GB, so
// this reports: does MF run, what's the front memory, and is it correct (berr)?

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>


#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/gpu/gpu_mf.hpp"
#include "mysolver/reordering/mc64.hpp"
#include "mysolver/reordering/metis_nd.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
#include "tools/matrix_io.hpp"

namespace {
using Csc = sparse_direct::matrix::CscMatrix;
using clk = std::chrono::steady_clock;
double ms(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}
// Row/col equilibration As = Dr A Dc (mirrors own_pipeline) -- conditions the
// no-pivot factor.
void equilibrate(const Csc& A, Csc& As)
{
    const int n = A.cols;
    std::vector<double> rs(n, 1.0), cs(n, 1.0), rmax(n, 0.0), cmax(n, 0.0);
    for (int c = 0; c < n; ++c)
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p)
            rmax[A.row_idx[p]] = std::max(rmax[A.row_idx[p]], std::fabs(A.values[p]));
    for (int i = 0; i < n; ++i)
        if (rmax[i] > 0.0) rs[i] = 1.0 / rmax[i];
    for (int c = 0; c < n; ++c) {
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p)
            cmax[c] = std::max(cmax[c], std::fabs(A.values[p]) * rs[A.row_idx[p]]);
        if (cmax[c] > 0.0) cs[c] = 1.0 / cmax[c];
    }
    As = A;
    for (int c = 0; c < n; ++c)
        for (int p = As.col_ptr[c]; p < As.col_ptr[c + 1]; ++p)
            As.values[p] = A.values[p] * rs[As.row_idx[p]] * cs[c];
}
// Column permutation B(:,i) = A(:,q[i]).
void colperm(const Csc& A, const std::vector<int>& q, Csc& B)
{
    const int n = A.cols;
    B.rows = A.rows;
    B.cols = n;
    B.col_ptr.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) B.col_ptr[i + 1] = A.col_ptr[q[i] + 1] - A.col_ptr[q[i]];
    for (int i = 0; i < n; ++i) B.col_ptr[i + 1] += B.col_ptr[i];
    B.row_idx.resize(B.col_ptr[n]);
    B.values.resize(B.col_ptr[n]);
    for (int i = 0; i < n; ++i) {
        int d = B.col_ptr[i];
        for (int p = A.col_ptr[q[i]]; p < A.col_ptr[q[i] + 1]; ++p, ++d) {
            B.row_idx[d] = A.row_idx[p];
            B.values[d] = A.values[p];
        }
    }
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
}  // namespace

int main()
{
    namespace sym = mysolver::symbolic;
    namespace gpu = mysolver::gpu;
    const std::string root = "/datasets/benchmark_matrices/matrices/";
    // cuDSS factor/solve refs = WARM median-7 (CUDSS_REPEAT=7, handle reused), measured cy340 on
    // this machine by benchmark --solver cudss-gpu (fair warm-vs-warm). Superseded the older refs.
    struct C { std::string name; double cf, cs; };
    const std::vector<C> names = {{"rajat27", 0.954, 0.446}, {"memplus", 1.189, 0.487},
                                  {"onetone2", 9.677, 1.809}, {"rajat15", 3.949, 1.018}};
    std::printf("MC64 + GPU multifrontal factor+solve on circuits vs cuDSS\n");
    for (const C& cc : names) {
        const std::string nm = cc.name;
        const auto csr = sparse_direct::io::read_matrix_market_csr(root + nm + "/" + nm + ".mtx");
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;

        // MC64 + scaling preprocessing (the production circuit path): equilibrate,
        // then max-product match large entries onto the diagonal so the no-pivot MF
        // stays stable. Power-grid skips this (strong natural diagonal); circuits
        // need it.
        Csc As;
        equilibrate(csc, As);
        std::vector<int> match_col, match(n);
        bool matched = mysolver::reordering::mc64_match(n, As.col_ptr.data(), As.row_idx.data(),
                                                        As.values.data(), match_col);
        Csc Bm;
        if (matched) {
            std::fill(match.begin(), match.end(), -1);
            for (int j = 0; j < n; ++j) match[match_col[j]] = j;
            colperm(As, match, Bm);
        } else {
            Bm = As;
        }
        // METIS nested dissection (vs AMD): minimizes etree depth -> wider, shallower
        // panel tree -> more parallelism per level on the deep narrow circuit etrees.
        std::vector<int> perm(n);
        mysolver::reordering::metis_nd(n, Bm.col_ptr.data(), Bm.row_idx.data(), perm);
        sparse_direct::matrix::CscMatrix M;
        permute(Bm, perm, M);
        std::vector<int> mscp, msri;
        sym::symmetric_pattern(n, M.col_ptr.data(), M.row_idx.data(), mscp, msri);
        std::vector<int> parent = sym::etree(n, mscp.data(), msri.data());
        std::vector<int> Lp, Li;
        sym::fill_pattern(n, mscp.data(), msri.data(), parent, Lp, Li);
        const long fill = Lp[n];

        // cy169: cap=8 to MATCH production (mysolver_gpu_solver passes 8). This tool had
        // been benchmarking at cap=16, making the reported circuit S pessimistic vs what
        // production actually runs (cap=8 solve is -14..-32% on the circuits because the
        // solve is fill-work-bound and wider amalgamated panels do more padded work).
        gpu::GpuMfPlan mf = gpu::gpu_mf_analyze(n, M.col_ptr.data(), M.row_idx.data(), Lp, Li,
                                               parent, 8);
        if (mf.num_panels == 0) {
            std::printf("%-10s n=%-7d fill=%-9ld | MF BAILED (front arena > 8GB) -> CPU fallback\n",
                        nm.c_str(), n, fill);
            continue;
        }
        std::vector<double> xt(n, 1.0), b(n, 0.0), x;
        for (int j = 0; j < n; ++j)
            for (int q = M.col_ptr[j]; q < M.col_ptr[j + 1]; ++q)
                b[M.row_idx[q]] += M.values[q] * xt[j];
        mysolver::numeric::SparseLU lu;
        double kms = 0.0, sms = 0.0;
        const bool ok = gpu::gpu_mf_factorize(mf, M.values.data(), lu, &kms);
        gpu::gpu_mf_solve(mf, b, x);  // warmup
        const double e = berr(M, x, b);
        std::vector<double> kf, ks;
        for (int r = 0; r < 5; ++r) {
            double kk = 0.0, ss = 0.0;
            gpu::gpu_mf_factorize(mf, M.values.data(), lu, &kk);
            gpu::gpu_mf_solve(mf, b, x, &ss);
            kf.push_back(kk);
            ks.push_back(ss);
        }
        std::sort(kf.begin(), kf.end());
        std::sort(ks.begin(), ks.end());
        kms = kf[kf.size() / 2];
        sms = ks[ks.size() / 2];
        std::printf("%-10s n=%-7d fill=%-8ld plev=%-4d | MF ok=%d FACTOR=%-8.3f cuDSS=%-6.3f | "
                    "SOLVE=%-7.3f cuDSS=%-6.3f berr=%.2e front_MB=%.0f\n",
                    nm.c_str(), n, fill, mf.num_plevels, ok, kms, cc.cf, sms, cc.cs, e,
                    mf.front_total * 8.0 / 1e6);
    }
    return 0;
}
