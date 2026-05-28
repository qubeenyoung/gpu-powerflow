// Unit test for the mysolver M0 phase API (PLAN §7, end-to-end row).
//
// Solves a known 5x5 diagonally-dominant system and checks that the cuDSS-style
// analyze -> factorize -> solve contract reproduces the exact solution, and
// that an AnalyzeResult can be reused across multiple factorizations.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <suitesparse/cs.h>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/dense_lu.hpp"
#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/factorize/supernodal.hpp"
#include "mysolver/reordering/metis_nd.hpp"
#include "mysolver/solver.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
#include "mysolver/symbolic/schedule.hpp"
#include "mysolver/symbolic/supernode.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void expect_lt(double value, double limit, const char* what)
{
    ++g_checks;
    if (!(value < limit)) {
        std::printf("  FAIL: %s (%.3e >= %.3e)\n", what, value, limit);
        ++g_failures;
    }
}

void expect_true(bool ok, const char* what)
{
    ++g_checks;
    if (!ok) {
        std::printf("  FAIL: %s\n", what);
        ++g_failures;
    }
}

// A permutation of 0..n-1 must contain each index exactly once.
bool is_permutation(const std::vector<int>& perm, int n)
{
    if (static_cast<int>(perm.size()) != n) {
        return false;
    }
    std::vector<char> seen(n, 0);
    for (int v : perm) {
        if (v < 0 || v >= n || seen[v]) {
            return false;
        }
        seen[v] = 1;
    }
    return true;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
{
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

}  // namespace

int main()
{
    using sparse_direct::matrix::CsrMatrix;
    using sparse_direct::matrix::to_csc;

    // A = tridiag(-1, 4, -1), n = 5  (symmetric, diagonally dominant).
    CsrMatrix csr;
    csr.rows = 5;
    csr.cols = 5;
    csr.row_ptr = {0, 2, 5, 8, 11, 13};
    csr.col_idx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    csr.values = {4, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1, -1, 4};

    const std::vector<double> x_true = {1, 2, 3, 4, 5};
    const std::vector<double> b = {2, 4, 6, 8, 16};  // A * x_true

    const auto csc = to_csc(csr);

    mysolver::AnalyzeResult a = mysolver::analyze(csc);

    mysolver::FactorState f1;
    mysolver::factorize(csc, a, &f1);
    std::vector<double> x1;
    mysolver::solve(f1, a, b, x1);
    expect_lt(max_abs_diff(x1, x_true), 1e-10, "solve reproduces x_true");

    // Reuse the same analysis for a second factorization (NR-iteration pattern).
    mysolver::FactorState f2;
    mysolver::factorize(csc, a, &f2);
    std::vector<double> x2;
    mysolver::solve(f2, a, b, x2);
    expect_lt(max_abs_diff(x2, x_true), 1e-10, "analysis reuse reproduces x_true");

    // METIS ND ordering on the same 5x5 pattern must be a valid permutation.
    std::vector<int> nd_perm;
    const bool nd_ok = mysolver::reordering::metis_nd(csc.cols, csc.col_ptr.data(),
                                                      csc.row_idx.data(), nd_perm);
    expect_true(nd_ok, "metis_nd succeeds on 5x5 pattern");
    expect_true(is_permutation(nd_perm, csc.cols), "metis_nd returns a valid permutation");

    // Elimination tree of the symmetric pattern. For the tridiagonal pattern the
    // etree is the path 0->1->2->3->4 (parent = [1,2,3,4,-1]).
    std::vector<int> sym_cp, sym_ri;
    mysolver::symbolic::symmetric_pattern(csc.cols, csc.col_ptr.data(),
                                          csc.row_idx.data(), sym_cp, sym_ri);
    std::vector<int> parent = mysolver::symbolic::etree(csc.cols, sym_cp.data(), sym_ri.data());
    const std::vector<int> expected_parent = {1, 2, 3, 4, -1};
    expect_true(parent == expected_parent, "etree matches known tridiagonal path");

    // Parity against CXSparse cs_di_etree on the same symmetric pattern (PLAN §243).
    cs_di sym{};
    sym.nzmax = static_cast<int>(sym_ri.size());
    sym.m = csc.cols;
    sym.n = csc.cols;
    sym.p = sym_cp.data();
    sym.i = sym_ri.data();
    sym.x = nullptr;
    sym.nz = -1;  // compressed-column form
    int* cs_parent = cs_di_etree(&sym, 0);
    bool etree_parity = (cs_parent != nullptr);
    for (int k = 0; etree_parity && k < csc.cols; ++k) {
        etree_parity = (cs_parent[k] == parent[k]);
    }
    expect_true(etree_parity, "etree matches CXSparse cs_di_etree");

    // Column counts of L: known tridiagonal result + CXSparse cs_di_counts parity.
    std::vector<int> post = mysolver::symbolic::postorder(parent, csc.cols);
    std::vector<int> colcount = mysolver::symbolic::column_counts(
        csc.cols, sym_cp.data(), sym_ri.data(), parent, post);
    const std::vector<int> expected_colcount = {2, 2, 2, 2, 1};
    expect_true(colcount == expected_colcount, "column_counts matches tridiagonal L");

    // cs_di_counts accepts any valid postorder of the etree; reuse our own
    // (our parent matches cs_parent, verified above).
    int* cs_cc = cs_di_counts(&sym, cs_parent, post.data(), 0);
    bool cc_parity = (cs_cc != nullptr);
    for (int k = 0; cc_parity && k < csc.cols; ++k) {
        cc_parity = (cs_cc[k] == colcount[k]);
    }
    expect_true(cc_parity, "column_counts matches CXSparse cs_di_counts");
    cs_di_free(cs_cc);
    cs_di_free(cs_parent);

    // Supernodes: for the tridiagonal, only the trailing 2x2 nests, so the
    // partition is {0},{1},{2},{3,4} -> 4 supernodes, sizes summing to n.
    mysolver::symbolic::SupernodePartition sp =
        mysolver::symbolic::supernodes(csc.cols, parent, post, colcount);
    int snode_total = 0;
    for (int s : sp.sizes) {
        snode_total += s;
    }
    expect_true(sp.num_supernodes == 4 && snode_total == csc.cols && sp.sizes.back() == 2,
                "tridiagonal -> {0},{1},{2},{3,4} supernodes summing to n");

    // A fully dense 4x4 pattern -> a single supernode of size 4.
    sparse_direct::matrix::CsrMatrix dense;
    dense.rows = 4;
    dense.cols = 4;
    dense.row_ptr = {0, 4, 8, 12, 16};
    dense.col_idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    dense.values.assign(16, 1.0);
    std::vector<int> dcp, dri;
    mysolver::symbolic::symmetric_pattern(4, dense.row_ptr.data(), dense.col_idx.data(), dcp, dri);
    std::vector<int> dparent = mysolver::symbolic::etree(4, dcp.data(), dri.data());
    std::vector<int> dpost = mysolver::symbolic::postorder(dparent, 4);
    std::vector<int> dcc = mysolver::symbolic::column_counts(4, dcp.data(), dri.data(), dparent, dpost);
    mysolver::symbolic::SupernodePartition dsp =
        mysolver::symbolic::supernodes(4, dparent, dpost, dcc);
    expect_true(dsp.num_supernodes == 1 && dsp.sizes.size() == 1 && dsp.sizes[0] == 4,
                "dense 4x4 -> single supernode of size 4");

    // Schedule: tridiagonal supernodes form a chain {0}->{1}->{2}->{3,4}, so 4
    // levels each with one supernode (no parallelism in a path graph).
    mysolver::symbolic::SupernodeSchedule sched =
        mysolver::symbolic::build_schedule(csc.cols, parent, sp);
    const std::vector<int> expected_super_parent = {1, 2, 3, -1};
    bool chain_ok = (sched.num_levels == 4) && (sched.super_parent == expected_super_parent);
    for (int L = 0; chain_ok && L < sched.num_levels; ++L) {
        chain_ok = (sched.levels[L].size() == 1);
    }
    expect_true(chain_ok, "tridiagonal schedule is a 4-level chain");

    // Dense 4x4: one supernode, one level.
    mysolver::symbolic::SupernodeSchedule dsched =
        mysolver::symbolic::build_schedule(4, dparent, dsp);
    expect_true(dsched.num_levels == 1 && dsched.super_parent == std::vector<int>{-1},
                "dense 4x4 schedule has a single level");

    // Fill pattern: total nnz(L) must equal sum(column_counts) (independent
    // cross-check), and the dense 4x4 must be fully filled (10 = 4+3+2+1).
    std::vector<int> Lp, Li;
    mysolver::symbolic::fill_pattern(csc.cols, sym_cp.data(), sym_ri.data(), parent, Lp, Li);
    long colcount_sum = 0;
    for (int c : colcount) {
        colcount_sum += c;
    }
    expect_true(static_cast<long>(Li.size()) == colcount_sum && Lp[csc.cols] == (int)Li.size(),
                "fill_pattern nnz(L) == sum(column_counts) [tridiagonal]");

    std::vector<int> dLp, dLi;
    mysolver::symbolic::fill_pattern(4, dcp.data(), dri.data(), dparent, dLp, dLi);
    expect_true(dLi.size() == 10 && dLp[4] == 10, "fill_pattern dense 4x4 -> nnz(L)=10");

    // Supernodal panel structure (postorder = identity for these toys).
    // Tridiagonal: supernode {3,4} panel = rows {3,4}; supernode {0} panel = {0,1}.
    mysolver::numeric::SupernodalStruct ssn =
        mysolver::numeric::build_supernodal(csc.cols, Lp, Li, sp);
    {
        const int last = ssn.num_supernodes - 1;  // supernode {3,4}
        const int beg = ssn.panel_ptr[last], end = ssn.panel_ptr[last + 1];
        bool ok = (ssn.num_supernodes == 4) && (ssn.sn_first[last] == 3) &&
                  (ssn.sn_size[last] == 2) && (end - beg == 2) &&
                  (ssn.panel_rows[beg] == 3) && (ssn.panel_rows[beg + 1] == 4);
        expect_true(ok, "supernodal panel: tridiagonal {3,4} -> rows {3,4}");
    }
    // Dense 4x4: one supernode, panel = all 4 rows.
    mysolver::numeric::SupernodalStruct dssn =
        mysolver::numeric::build_supernodal(4, dLp, dLi, dsp);
    expect_true(dssn.num_supernodes == 1 && dssn.sn_size[0] == 4 &&
                (dssn.panel_ptr[1] - dssn.panel_ptr[0]) == 4,
                "supernodal panel: dense 4x4 -> single 4-row panel");

    // Supernodal numeric factorization must solve to x_true (tridiagonal: size-1
    // supernodes + the {3,4} block + Schur updates).
    {
        mysolver::numeric::SparseLU slu2;
        const bool ok = mysolver::numeric::factor_supernodal(
            csc.cols, csc.col_ptr.data(), csc.row_idx.data(), csc.values.data(), Lp, Li, ssn, slu2);
        std::vector<double> xs;
        mysolver::numeric::solve(slu2, b, xs);
        expect_true(ok, "factor_supernodal factors tridiagonal");
        expect_lt(max_abs_diff(xs, x_true), 1e-10, "factor_supernodal solves tridiagonal to x_true");
    }

    // Dense block LU (M3 kernel): 100 random 8x8 diagonally-dominant systems
    // must solve to a tiny residual (PLAN §666).
    {
        const int m = 8;
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        double worst = 0.0;
        int singular = 0;
        for (int trial = 0; trial < 100; ++trial) {
            std::vector<double> A(m * m);
            for (int j = 0; j < m; ++j) {
                for (int i = 0; i < m; ++i) {
                    A[i + j * m] = dist(rng);
                }
                A[j + j * m] += m;  // diagonal dominance -> well conditioned
            }
            std::vector<double> x_true(m);
            for (int i = 0; i < m; ++i) {
                x_true[i] = dist(rng);
            }
            std::vector<double> b(m, 0.0);
            for (int j = 0; j < m; ++j) {
                for (int i = 0; i < m; ++i) {
                    b[i] += A[i + j * m] * x_true[j];
                }
            }
            std::vector<double> lu = A;
            std::vector<int> piv;
            if (!mysolver::numeric::dense_lu(m, lu, piv)) {
                ++singular;
                continue;
            }
            std::vector<double> x = b;
            mysolver::numeric::dense_lu_solve(m, lu, piv, x);
            worst = std::max(worst, max_abs_diff(x, x_true));
        }
        expect_true(singular == 0 && worst < 1e-10,
                    "dense_lu: 100 random 8x8 solves residual < 1e-10");
    }

    // Sparse LU on the static fill pattern: tridiagonal (no fill) end-to-end.
    {
        mysolver::numeric::SparseLU slu;
        const bool ok = mysolver::numeric::factor_nopiv(
            csc.cols, csc.col_ptr.data(), csc.row_idx.data(), csc.values.data(), Lp, Li, slu);
        std::vector<double> xs;
        mysolver::numeric::solve(slu, b, xs);
        expect_true(ok, "sparse_lu factors tridiagonal");
        expect_lt(max_abs_diff(xs, x_true), 1e-10, "sparse_lu solves tridiagonal to x_true");
    }

    // Sparse LU with real fill: an up-left "arrow" (dense first row/col) whose
    // natural-order elimination fills the trailing block; diagonally dominant.
    {
        const int m = 5;
        CsrMatrix arr;
        arr.rows = m;
        arr.cols = m;
        arr.row_ptr = {0, 5, 7, 9, 11, 13};
        arr.col_idx = {0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4};
        arr.values = {10, 1, 1, 1, 1, 1, 10, 1, 10, 1, 10, 1, 10};
        const auto acsc = to_csc(arr);
        std::vector<int> acp, ari;
        mysolver::symbolic::symmetric_pattern(m, arr.row_ptr.data(), arr.col_idx.data(), acp, ari);
        std::vector<int> aparent = mysolver::symbolic::etree(m, acp.data(), ari.data());
        std::vector<int> aLp, aLi;
        mysolver::symbolic::fill_pattern(m, acp.data(), ari.data(), aparent, aLp, aLi);

        const std::vector<double> arr_xtrue = {1, 2, 3, 4, 5};
        std::vector<double> arr_b(m, 0.0);  // b = A * x_true
        for (int j = 0; j < m; ++j) {
            for (int p = acsc.col_ptr[j]; p < acsc.col_ptr[j + 1]; ++p) {
                arr_b[acsc.row_idx[p]] += acsc.values[p] * arr_xtrue[j];
            }
        }
        mysolver::numeric::SparseLU aslu;
        const bool ok = mysolver::numeric::factor_nopiv(
            m, acsc.col_ptr.data(), acsc.row_idx.data(), acsc.values.data(), aLp, aLi, aslu);
        std::vector<double> ax;
        mysolver::numeric::solve(aslu, arr_b, ax);
        expect_true(ok, "sparse_lu factors arrow (with fill)");
        expect_lt(max_abs_diff(ax, arr_xtrue), 1e-10, "sparse_lu solves arrow to x_true");

        // Supernodal numeric on the arrow: the up-left arrow fills the trailing
        // 4x4 -> a size-4 supernode, exercising the dense-block + multi-col Schur.
        std::vector<int> apost = mysolver::symbolic::postorder(aparent, m);
        std::vector<int> acc = mysolver::symbolic::column_counts(m, acp.data(), ari.data(), aparent, apost);
        mysolver::symbolic::SupernodePartition asp =
            mysolver::symbolic::supernodes(m, aparent, apost, acc);
        mysolver::numeric::SupernodalStruct assn =
            mysolver::numeric::build_supernodal(m, aLp, aLi, asp);
        mysolver::numeric::SparseLU aslu2;
        const bool ok2 = mysolver::numeric::factor_supernodal(
            m, acsc.col_ptr.data(), acsc.row_idx.data(), acsc.values.data(), aLp, aLi, assn, aslu2);
        std::vector<double> ax2;
        mysolver::numeric::solve(aslu2, arr_b, ax2);
        expect_true(ok2, "factor_supernodal factors arrow");
        expect_lt(max_abs_diff(ax2, arr_xtrue), 1e-10, "factor_supernodal solves arrow to x_true");
    }

    const int passed = g_checks - g_failures;
    std::printf("%d/%d passed (tests/mysolver)\n", passed, g_checks);
    return g_failures == 0 ? 0 : 1;
}
