// Validate the CPU multifrontal numeric factor (cycle 80) against factor_nopiv
// (the established left-looking reference). Both produce no-pivot LU on the same
// fill pattern, so the SparseLU values must agree to rounding, and the solve must
// have the same backward error. AMD-orders each matrix, builds the etree/fill,
// factors both ways, and reports max|Sx_mf - Sx_ref| + componentwise berr.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <suitesparse/amd.h>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
#include "tools/matrix_io.hpp"

namespace {
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
    namespace num = mysolver::numeric;
    const std::filesystem::path nr = "/datasets/power_system/nr_linear_systems";
    const std::vector<std::pair<std::string, std::filesystem::path>> cases = {
        {"case1197", nr / "case1197/J.mtx"},
        {"case_ACTIVSg2000", nr / "case_ACTIVSg2000/J.mtx"},
        {"case3012wp", nr / "case3012wp/J.mtx"},
        {"case6468rte", nr / "case6468rte/J.mtx"},
        {"case8387pegase", nr / "case8387pegase/J.mtx"},
        {"case_ACTIVSg25k", nr / "case_ACTIVSg25k/J.mtx"},
    };
    int fails = 0;
    for (const auto& [name, path] : cases) {
        const auto csr = sparse_direct::io::read_matrix_market_csr(path);
        const auto csc = sparse_direct::matrix::to_csc(csr);
        const int n = csc.cols;
        std::vector<int> scp, sri;
        sym::symmetric_pattern(n, csc.col_ptr.data(), csc.row_idx.data(), scp, sri);
        std::vector<int> perm(n);
        double info[AMD_INFO];
        amd_order(n, scp.data(), sri.data(), perm.data(), nullptr, info);
        sparse_direct::matrix::CscMatrix M;
        permute(csc, perm, M);
        std::vector<int> mscp, msri;
        sym::symmetric_pattern(n, M.col_ptr.data(), M.row_idx.data(), mscp, msri);
        std::vector<int> parent = sym::etree(n, mscp.data(), msri.data());
        std::vector<int> Lp, Li;
        sym::fill_pattern(n, mscp.data(), msri.data(), parent, Lp, Li);

        num::SparseLU ref, mfac;
        const bool ok_ref = num::factor_nopiv(n, M.col_ptr.data(), M.row_idx.data(),
                                              M.values.data(), Lp, Li, ref);
        const bool ok_mf = num::multifrontal_factor(n, M.col_ptr.data(), M.row_idx.data(),
                                                    M.values.data(), Lp, Li, parent, mfac, 16);

        double maxdiff = 0.0;
        for (std::size_t q = 0; q < ref.x.size(); ++q)
            maxdiff = std::max(maxdiff, std::abs(ref.x[q] - mfac.x[q]));

        // Solve M y = b for b = M*ones; compare both factors' backward error.
        std::vector<double> xt(n, 1.0), b(n, 0.0), yr, ym;
        for (int j = 0; j < n; ++j)
            for (int q = M.col_ptr[j]; q < M.col_ptr[j + 1]; ++q)
                b[M.row_idx[q]] += M.values[q] * xt[j];
        num::solve(ref, b, yr);
        num::solve(mfac, b, ym);
        const double br = berr(M, yr, b), bm = berr(M, ym, b);

        const bool pass = ok_ref && ok_mf && maxdiff < 1e-9 && bm < 1e-10;
        if (!pass) ++fails;
        std::printf("%-18s n=%-7d ok=%d/%d maxSxdiff=%-10.2e berr_ref=%-9.2e berr_mf=%-9.2e %s\n",
                    name.c_str(), n, ok_ref, ok_mf, maxdiff, br, bm, pass ? "OK" : "FAIL");
    }
    std::printf("%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
