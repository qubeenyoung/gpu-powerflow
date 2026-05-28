// GPU research-track validation: CSR SpMV pipeline, and the level-set GPU
// factorization (plan M3 factorize_driver small-path) vs CPU reference.

#include <cmath>
#include <cstdio>
#include <vector>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/gpu/gpu_factor.hpp"
#include "mysolver/gpu/gpu_solve.hpp"
#include "mysolver/gpu/gpu_spmv.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"

namespace {
int g_fail = 0, g_chk = 0;
void check(bool ok, const char* what)
{
    ++g_chk;
    if (!ok) { std::printf("  FAIL: %s\n", what); ++g_fail; }
}
double maxdiff(const std::vector<double>& a, const std::vector<double>& b)
{
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Factor `csr` on the GPU and solve A x = b; compare to x_true.
double gpu_solve_resid(const sparse_direct::matrix::CsrMatrix& csr,
                       const std::vector<double>& b, const std::vector<double>& x_true,
                       bool* ok)
{
    namespace sym = mysolver::symbolic;
    const auto csc = sparse_direct::matrix::to_csc(csr);
    const int n = csc.cols;
    std::vector<int> scp, sri;
    sym::symmetric_pattern(n, csc.col_ptr.data(), csc.row_idx.data(), scp, sri);
    std::vector<int> parent = sym::etree(n, scp.data(), sri.data());
    std::vector<int> Lp, Li;
    sym::fill_pattern(n, scp.data(), sri.data(), parent, Lp, Li);
    mysolver::numeric::SparseLU lu;
    *ok = mysolver::gpu::gpu_factor(n, csc.col_ptr.data(), csc.row_idx.data(),
                                    csc.values.data(), Lp, Li, parent, lu);
    std::vector<double> x;
    mysolver::numeric::solve(lu, b, x);
    return maxdiff(x, x_true);
}
}  // namespace

int main()
{
    // 1) GPU SpMV vs CPU on tridiag(-1,4,-1).
    {
        const int n = 5;
        std::vector<int> rp = {0, 2, 5, 8, 11, 13};
        std::vector<int> ci = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
        std::vector<double> v = {4, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1, -1, 4};
        std::vector<double> x = {1, 2, 3, 4, 5};
        std::vector<double> y_cpu(n, 0.0);
        for (int r = 0; r < n; ++r)
            for (int p = rp[r]; p < rp[r + 1]; ++p) y_cpu[r] += v[p] * x[ci[p]];
        const auto y_gpu = mysolver::gpu::csr_spmv(n, rp, ci, v, x);
        check(maxdiff(y_gpu, y_cpu) < 1e-12, "gpu_spmv == cpu");
    }

    // 2) GPU level-set factorization: tridiagonal (chain etree).
    {
        sparse_direct::matrix::CsrMatrix t;
        t.rows = t.cols = 5;
        t.row_ptr = {0, 2, 5, 8, 11, 13};
        t.col_idx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
        t.values = {4, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1, -1, 4};
        bool ok = false;
        const double d = gpu_solve_resid(t, {2, 4, 6, 8, 16}, {1, 2, 3, 4, 5}, &ok);
        check(ok && d < 1e-10, "gpu_factor solves tridiagonal to x_true");
    }

    // 3) GPU level-set factorization: arrow (real fill, parallel level).
    {
        sparse_direct::matrix::CsrMatrix a;
        a.rows = a.cols = 5;
        a.row_ptr = {0, 5, 7, 9, 11, 13};
        a.col_idx = {0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4};
        a.values = {10, 1, 1, 1, 1, 1, 10, 1, 10, 1, 10, 1, 10};
        const std::vector<double> xt = {1, 2, 3, 4, 5};
        const auto acsc = sparse_direct::matrix::to_csc(a);
        std::vector<double> b(5, 0.0);
        for (int j = 0; j < 5; ++j)
            for (int p = acsc.col_ptr[j]; p < acsc.col_ptr[j + 1]; ++p)
                b[acsc.row_idx[p]] += acsc.values[p] * xt[j];
        bool ok = false;
        const double d = gpu_solve_resid(a, b, xt, &ok);
        check(ok && d < 1e-10, "gpu_factor solves arrow (with fill) to x_true");
    }

    // 4) End-to-end GPU factor + GPU triangular solve (M4) on the arrow matrix.
    {
        namespace sym = mysolver::symbolic;
        sparse_direct::matrix::CsrMatrix a;
        a.rows = a.cols = 5;
        a.row_ptr = {0, 5, 7, 9, 11, 13};
        a.col_idx = {0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4};
        a.values = {10, 1, 1, 1, 1, 1, 10, 1, 10, 1, 10, 1, 10};
        const std::vector<double> xt = {1, 2, 3, 4, 5};
        const auto csc = sparse_direct::matrix::to_csc(a);
        const int n = csc.cols;
        std::vector<double> b(n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int p = csc.col_ptr[j]; p < csc.col_ptr[j + 1]; ++p)
                b[csc.row_idx[p]] += csc.values[p] * xt[j];
        std::vector<int> scp, sri;
        sym::symmetric_pattern(n, csc.col_ptr.data(), csc.row_idx.data(), scp, sri);
        std::vector<int> parent = sym::etree(n, scp.data(), sri.data());
        std::vector<int> Lp, Li;
        sym::fill_pattern(n, scp.data(), sri.data(), parent, Lp, Li);
        mysolver::numeric::SparseLU lu;
        mysolver::gpu::gpu_factor(n, csc.col_ptr.data(), csc.row_idx.data(), csc.values.data(),
                                  Lp, Li, parent, lu);
        std::vector<double> x;
        mysolver::gpu::gpu_solve(lu, parent, b, x);
        check(maxdiff(x, xt) < 1e-10, "gpu_factor + gpu_solve (M4) arrow to x_true");
    }

    std::printf("%d/%d passed (gpu_test)\n", g_chk - g_fail, g_chk);
    return g_fail == 0 ? 0 : 1;
}
