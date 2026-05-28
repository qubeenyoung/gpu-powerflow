// Measure the GPU level-set factorization vs the CPU factor on real matrices
// (the cuDSS-chase metric). AMD-orders each matrix, then times gpu_factor
// (warm + median) and factor_nopiv. Reports factor_ms; cuDSS factor for
// reference (from earlier benchmark runs).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <suitesparse/amd.h>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/gpu/gpu_factor.hpp"
#include "mysolver/reordering/metis_nd.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
#include "mysolver/symbolic/supernode.hpp"
#include "tools/matrix_io.hpp"

namespace {
using clk = std::chrono::steady_clock;
double ms(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}
// P A Pᵀ with values.
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
double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Componentwise backward error of y as a solution of M y = b (Arioli-Demmel-Duff):
// max_i |b - M y|_i / (|M| |y| + |b|)_i. Validates the GPU factor on the real
// matrices (gpu_test only covers n=5 toys), and is independent of GPU contention.
double componentwise_berr(const sparse_direct::matrix::CscMatrix& M,
                          const std::vector<double>& y, const std::vector<double>& b)
{
    const int n = M.cols;
    std::vector<double> r(b), den(n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int q = M.col_ptr[j]; q < M.col_ptr[j + 1]; ++q) {
            const int i = M.row_idx[q];
            r[i] -= M.values[q] * y[j];               // r = b - M y
            den[i] += std::abs(M.values[q]) * std::abs(y[j]);  // |M| |y|
        }
    double berr = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = den[i] + std::abs(b[i]);
        if (d > 0.0) berr = std::max(berr, std::abs(r[i]) / d);
    }
    return berr;
}

// GPU-aware ordering cost proxy. Builds the symmetric filled pattern from the
// lower fill Lp/Li, counts Schur ops per column, and returns the critical-path
// cost = Σ over etree levels of the heaviest column's op count. Columns in a
// level run in parallel, so the level's latency tracks its heaviest column;
// summing over levels captures depth × per-level load — the quantity that the
// occupancy-bound kernel actually pays (vs. raw fill, which only counts work).
long predict_gpu_cost(int n, const std::vector<int>& Lp, const std::vector<int>& Li,
                      const std::vector<int>& parent)
{
    std::vector<std::vector<int>> cols(n);  // symmetric filled pattern
    for (int j = 0; j < n; ++j)
        for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
            const int i = Li[p];
            cols[j].push_back(i);
            if (i != j) cols[i].push_back(j);
        }
    for (int j = 0; j < n; ++j) {
        std::sort(cols[j].begin(), cols[j].end());
        cols[j].erase(std::unique(cols[j].begin(), cols[j].end()), cols[j].end());
    }
    std::vector<int> mark(n, -1), opc(n, 0);
    for (int j = 0; j < n; ++j) {
        for (int i : cols[j]) mark[i] = j;
        long cnt = 0;
        for (int k : cols[j]) {
            if (k >= j) break;
            for (int i : cols[k])
                if (i > k && mark[i] == j) ++cnt;
        }
        opc[j] = static_cast<int>(cnt);
    }
    std::vector<int> lev(n, 0);
    int nlev = 0;
    for (int j = 0; j < n; ++j)
        if (parent[j] != -1) lev[parent[j]] = std::max(lev[parent[j]], lev[j] + 1);
    for (int j = 0; j < n; ++j) nlev = std::max(nlev, lev[j] + 1);
    std::vector<int> lmax(nlev, 0);
    for (int j = 0; j < n; ++j) lmax[lev[j]] = std::max(lmax[lev[j]], opc[j]);
    long crit = 0;
    for (int L = 0; L < nlev; ++L) crit += lmax[L];
    return crit;
}
struct Case { std::string name; std::filesystem::path path; double cudss_fac; };
}  // namespace

int main()
{
    namespace sym = mysolver::symbolic;
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    std::vector<Case> cases = {
        {"case3012wp", nr / "case3012wp/J.mtx", 0.471},
        {"case_ACTIVSg2000", nr / "case_ACTIVSg2000/J.mtx", 0.647},
        {"case6468rte", nr / "case6468rte/J.mtx", 0.618},
        {"case8387pegase", nr / "case8387pegase/J.mtx", 1.112},
        {"case_ACTIVSg25k", nr / "case_ACTIVSg25k/J.mtx", 0.0},
        {"case_SyntheticUSA", nr / "case_SyntheticUSA/J.mtx", 9.8},  // cuDSS GPU factor ref
    };
    std::printf("AMD vs METIS-ND ordering: GPU factor kernel ms + etree occupancy\n");
    for (const Case& c : cases) {
        const auto csr = sparse_direct::io::read_matrix_market_csr(c.path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;
        std::vector<int> scp, sri;
        sym::symmetric_pattern(n, csc.col_ptr.data(), csc.row_idx.data(), scp, sri);

        // Compare orderings: AMD (current own-pipeline default) vs METIS nested
        // dissection (cuDSS-style). ND aims for a wider, shallower etree -> more
        // columns per level -> better GPU occupancy on the narrow-etree cases.
        // The AUTO row picks, per matrix, the ordering with the lower predicted
        // GPU cost (predict_gpu_cost) to get best-of-both without overfitting.
        double kms_ord[2];
        long cost_ord[2];
        for (int oi = 0; oi < 2; ++oi) {
            const char* ord = oi == 0 ? "AMD" : "METIS";
            std::vector<int> perm(n);
            if (oi == 0) {
                double info[AMD_INFO];
                amd_order(n, scp.data(), sri.data(), perm.data(), nullptr, info);
            } else {
                mysolver::reordering::metis_nd(n, csc.col_ptr.data(), csc.row_idx.data(), perm);
            }
            sparse_direct::matrix::CscMatrix M;
            permute(csc, perm, M);
            std::vector<int> mscp, msri;
            sym::symmetric_pattern(n, M.col_ptr.data(), M.row_idx.data(), mscp, msri);
            std::vector<int> parent = sym::etree(n, mscp.data(), msri.data());
            std::vector<int> Lp, Li;
            sym::fill_pattern(n, mscp.data(), msri.data(), parent, Lp, Li);

            const long cost = predict_gpu_cost(n, Lp, Li, parent);

            // Supernode structure: do these matrices have BLAS-worthy dense blocks?
            // (cycle 23 found pure supernodal slower on tiny supernodes.) Report
            // the fundamental supernode size distribution to decide if a supernodal
            // / amalgamated GPU path is viable here.
            {
                std::vector<int> post = sym::postorder(parent, n);
                std::vector<int> cc = sym::column_counts(n, mscp.data(), msri.data(), parent, post);
                sym::SupernodePartition sn = sym::supernodes(n, parent, post, cc);
                int maxsz = 0, ge4 = 0, ge8 = 0;
                for (int s = 0; s < sn.num_supernodes; ++s) {
                    maxsz = std::max(maxsz, sn.sizes[s]);
                    if (sn.sizes[s] >= 4) ge4 += sn.sizes[s];
                    if (sn.sizes[s] >= 8) ge8 += sn.sizes[s];
                }
                std::printf("  [snode] %-16s %-6s nsuper=%-6d mean=%.2f max=%-4d "
                            ">=4:%.0f%% >=8:%.0f%%\n",
                            c.name.c_str(), ord, sn.num_supernodes,
                            double(n) / sn.num_supernodes, maxsz, 100.0 * ge4 / n,
                            100.0 * ge8 / n);
            }

            // Analyze once (amortized symbolic + alloc + structure upload), then
            // time the repeated factorize path (value scatter + kernel + download).
            mysolver::gpu::GpuFactorPlan plan =
                mysolver::gpu::gpu_analyze(n, M.col_ptr.data(), M.row_idx.data(), Lp, Li, parent);
            mysolver::numeric::SparseLU lu;
            mysolver::gpu::gpu_factorize(plan, M.values.data(), lu);  // warmup

            // Correctness on the real matrix: solve M y = b for b = M*ones, berr.
            std::vector<double> xt(n, 1.0), b(n, 0.0), y;
            for (int j = 0; j < n; ++j)
                for (int q = M.col_ptr[j]; q < M.col_ptr[j + 1]; ++q)
                    b[M.row_idx[q]] += M.values[q] * xt[j];
            mysolver::numeric::solve(lu, b, y);
            const double berr = componentwise_berr(M, y, b);

            std::vector<double> kt, gt;
            for (int r = 0; r < 7; ++r) {
                double kms = 0.0;
                auto t0 = clk::now();
                mysolver::gpu::gpu_factorize(plan, M.values.data(), lu, &kms);
                gt.push_back(ms(t0, clk::now()));
                kt.push_back(kms);
            }
            kms_ord[oi] = median(kt);
            cost_ord[oi] = cost;
            std::printf("%-16s %-6s n=%-6d | refac_tot=%-9.3f gpu_kern=%-9.3f berr=%-9.2e "
                        "fill=%-7ld pred_cost=%-8ld cuDSS=%.3f\n",
                        c.name.c_str(), ord, n, median(gt), kms_ord[oi], berr, (long)Lp[n],
                        cost, c.cudss_fac);
        }
        const int pick = cost_ord[1] < cost_ord[0] ? 1 : 0;       // lower predicted cost
        const int best = kms_ord[1] < kms_ord[0] ? 1 : 0;         // empirically faster
        std::printf("%-16s AUTO   -> picked %-6s gpu_kern=%-9.3f  (best=%s %s)\n",
                    c.name.c_str(), pick ? "METIS" : "AMD", kms_ord[pick],
                    best ? "METIS" : "AMD", pick == best ? "OK" : "MISS");
    }
    return 0;
}
