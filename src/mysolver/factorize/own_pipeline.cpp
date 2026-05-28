#include "mysolver/factorize/own_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <suitesparse/amd.h>
#include <suitesparse/btf.h>

#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/reordering/mc64.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"

namespace mysolver {

namespace {

// Build P·A·Pᵀ with values (perm[new] = old), CSC. Columns need not be sorted
// (factor_nopiv finds positions in the symmetric fill pattern, which is sorted).
void permute_csc(const sparse_direct::matrix::CscMatrix& A, const std::vector<int>& perm,
                 sparse_direct::matrix::CscMatrix& out)
{
    const int n = A.cols;
    std::vector<int> iperm(n);
    for (int k = 0; k < n; ++k) {
        iperm[perm[k]] = k;
    }
    out.rows = n;
    out.cols = n;
    out.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int c = 0; c < n; ++c) {
        out.col_ptr[iperm[c] + 1] += A.col_ptr[c + 1] - A.col_ptr[c];
    }
    for (int c = 0; c < n; ++c) {
        out.col_ptr[c + 1] += out.col_ptr[c];
    }
    const int nnz = A.col_ptr[n];
    out.row_idx.assign(nnz, 0);
    out.values.assign(nnz, 0.0);
    std::vector<int> next(out.col_ptr.begin(), out.col_ptr.end());
    for (int c = 0; c < n; ++c) {
        const int nc = iperm[c];
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p) {
            const int dst = next[nc]++;
            out.row_idx[dst] = iperm[A.row_idx[p]];
            out.values[dst] = A.values[p];
        }
    }
}

// Column-permute A so that B(:,i) = A(:,qcol[i]).
void colperm_csc(const sparse_direct::matrix::CscMatrix& A, const std::vector<int>& qcol,
                 sparse_direct::matrix::CscMatrix& out)
{
    const int n = A.cols;
    out.rows = A.rows;
    out.cols = n;
    out.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) {
        const int c = qcol[i];
        out.col_ptr[i + 1] = out.col_ptr[i] + (A.col_ptr[c + 1] - A.col_ptr[c]);
    }
    out.row_idx.resize(out.col_ptr[n]);
    out.values.resize(out.col_ptr[n]);
    for (int i = 0; i < n; ++i) {
        const int c = qcol[i];
        int dst = out.col_ptr[i];
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p, ++dst) {
            out.row_idx[dst] = A.row_idx[p];
            out.values[dst] = A.values[p];
        }
    }
}

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Row/col equilibration As = Dr A Dc (entries bounded near 1) — conditions the
// no-pivot factorization. Value-dependent, so recomputed per factorization.
void equilibrate(const sparse_direct::matrix::CscMatrix& A, sparse_direct::matrix::CscMatrix& As,
                 std::vector<double>& rscale, std::vector<double>& cscale)
{
    const int n = A.cols;
    rscale.assign(n, 1.0);
    cscale.assign(n, 1.0);
    std::vector<double> rmax(n, 0.0), cmax(n, 0.0);
    for (int c = 0; c < n; ++c)
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p)
            rmax[A.row_idx[p]] = std::max(rmax[A.row_idx[p]], std::fabs(A.values[p]));
    for (int i = 0; i < n; ++i)
        if (rmax[i] > 0.0) rscale[i] = 1.0 / rmax[i];
    for (int c = 0; c < n; ++c) {
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p)
            cmax[c] = std::max(cmax[c], std::fabs(A.values[p]) * rscale[A.row_idx[p]]);
        if (cmax[c] > 0.0) cscale[c] = 1.0 / cmax[c];
    }
    As = A;
    for (int c = 0; c < n; ++c)
        for (int p = As.col_ptr[c]; p < As.col_ptr[c + 1]; ++p)
            As.values[p] = A.values[p] * rscale[As.row_idx[p]] * cscale[c];
}

// Apply column permutation `match` (B(:,i)=As(:,match[i])). Identity needs no
// permutation -> move As (no O(nnz) copy). As may be moved-from on return.
void build_matched(sparse_direct::matrix::CscMatrix& As, const std::vector<int>& match,
                   sparse_direct::matrix::CscMatrix& B)
{
    bool identity = true;
    for (int i = 0; i < static_cast<int>(match.size()); ++i)
        if (match[i] != i) { identity = false; break; }
    if (identity)
        B = std::move(As);
    else
        colperm_csc(As, match, B);
}

// Shared per-factorization tail: M = P B Pᵀ, no-pivot factor, scaled solve +
// adaptive iterative refinement. Reuses match/perm/Lp/Li (pattern + matching).
bool factor_solve_refine(const sparse_direct::matrix::CscMatrix& B,
                         const std::vector<double>& rscale, const std::vector<double>& cscale,
                         const std::vector<int>& match, const std::vector<int>& perm,
                         const std::vector<int>& Lp, const std::vector<int>& Li,
                         const sparse_direct::matrix::CscMatrix& A, const std::vector<double>& b,
                         std::vector<double>& x_out, OwnSolveStats* stats)
{
    const int n = A.cols;
    sparse_direct::matrix::CscMatrix M;
    permute_csc(B, perm, M);
    const auto t_factor = clk::now();
    numeric::SparseLU lu;
    if (!numeric::factor_nopiv(n, M.col_ptr.data(), M.row_idx.data(), M.values.data(), Lp, Li, lu))
        return false;
    if (stats) stats->factor_ms = ms_since(t_factor);
    const auto t_solve = clk::now();
    auto solve_As = [&](const std::vector<double>& rhs_s) {
        std::vector<double> c(n), w, y(n), xp(n);
        for (int a = 0; a < n; ++a) c[a] = rhs_s[perm[a]];
        numeric::solve(lu, c, w);
        for (int bc = 0; bc < n; ++bc) y[perm[bc]] = w[bc];
        for (int i = 0; i < n; ++i) xp[match[i]] = y[i];
        return xp;
    };
    auto solve_orig = [&](const std::vector<double>& rhs) {
        std::vector<double> rs(n);
        for (int i = 0; i < n; ++i) rs[i] = rscale[i] * rhs[i];
        std::vector<double> xp = solve_As(rs);
        std::vector<double> x(n);
        for (int j = 0; j < n; ++j) x[j] = cscale[j] * xp[j];
        return x;
    };
    double bnorm = 0.0;
    for (double v : b) bnorm = std::max(bnorm, std::fabs(v));
    x_out = solve_orig(b);
    for (int iter = 0; iter < 3; ++iter) {
        std::vector<double> r = b;  // r = b - A x_out
        double rnorm = 0.0;
        for (int c = 0; c < n; ++c)
            for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p)
                r[A.row_idx[p]] -= A.values[p] * x_out[c];
        for (double v : r) rnorm = std::max(rnorm, std::fabs(v));
        if (rnorm <= 1e-13 * bnorm) break;
        std::vector<double> dx = solve_orig(r);
        for (int j = 0; j < n; ++j) x_out[j] += dx[j];
    }
    if (stats) stats->solve_ms = ms_since(t_solve);
    for (double v : x_out)
        if (!std::isfinite(v)) return false;
    return true;
}

}  // namespace

bool try_own_solve(const sparse_direct::matrix::CscMatrix& A,
                   const std::vector<double>& b, std::vector<double>& x_out,
                   OwnSolveStats* stats, OwnAnalysis* save)
{
    namespace sym = symbolic;
    const auto t_analysis = clk::now();
    const int n = A.cols;
    if (n <= 0 || A.rows != n || static_cast<int>(b.size()) != n) {
        return false;
    }

    // 1) Equilibration.
    sparse_direct::matrix::CscMatrix As;
    std::vector<double> rscale, cscale;
    equilibrate(A, As, rscale, cscale);

    // 2) Matching. If the NATURAL diagonal is already strong after equilibration
    // (power-grid: large self-terms) no matching is needed -> identity. Only
    // circuit matrices (zero/tiny natural diagonal) pay for MC64 max-product
    // matching, avoiding the O(n·m·log) MC64 SSP on large power-grid (cycle 52/53).
    double min_nat_diag = 1e300;
    for (int j = 0; j < n; ++j) {
        double d = 0.0;
        for (int p = As.col_ptr[j]; p < As.col_ptr[j + 1]; ++p)
            if (As.row_idx[p] == j) { d = std::fabs(As.values[p]); break; }
        min_nat_diag = std::min(min_nat_diag, d);
    }
    std::vector<int> match(n);
    if (min_nat_diag >= 0.10) {
        for (int i = 0; i < n; ++i) match[i] = i;
    } else {
        std::vector<int> match_col;
        if (!reordering::mc64_match(n, As.col_ptr.data(), As.row_idx.data(), As.values.data(),
                                    match_col)) {
            return false;  // structurally singular
        }
        std::fill(match.begin(), match.end(), -1);
        for (int j = 0; j < n; ++j) match[match_col[j]] = j;
    }
    {
        std::vector<char> seen(n, 0);
        for (int i = 0; i < n; ++i) {
            if (match[i] < 0 || match[i] >= n || seen[match[i]]) return false;
            seen[match[i]] = 1;
        }
    }
    sparse_direct::matrix::CscMatrix B;
    build_matched(As, match, B);

    // 3) Fill-reducing AMD order + symbolic fill pattern.
    std::vector<int> scp, sri;
    sym::symmetric_pattern(n, B.col_ptr.data(), B.row_idx.data(), scp, sri);
    std::vector<int> perm(n);
    double info[AMD_INFO];
    if (amd_order(n, scp.data(), sri.data(), perm.data(), nullptr, info) != AMD_OK) {
        return false;
    }
    std::vector<int> pscp, psri;
    sym::permute_pattern(n, scp.data(), sri.data(), perm, pscp, psri);
    std::vector<int> parent = sym::etree(n, pscp.data(), psri.data());
    std::vector<int> Lp, Li;
    sym::fill_pattern(n, pscp.data(), psri.data(), parent, Lp, Li);
    if (stats) stats->analysis_ms = ms_since(t_analysis);

    if (save) {
        save->n = n;
        save->match = match;
        save->perm = perm;
        save->Lp = Lp;
        save->Li = Li;
    }

    // 4) Numeric factor + scaled solve + refinement (shared tail).
    return factor_solve_refine(B, rscale, cscale, match, perm, Lp, Li, A, b, x_out, stats);
}

bool own_refactor(const sparse_direct::matrix::CscMatrix& A, const OwnAnalysis& an,
                  const std::vector<double>& b, std::vector<double>& x_out, OwnSolveStats* stats)
{
    const auto t0 = clk::now();
    const int n = A.cols;
    if (n != an.n || A.rows != n || static_cast<int>(b.size()) != n) return false;

    // Reuse the matching + ordering + symbolic fill (skips MC64/AMD/symbolic, the
    // expensive analysis). Only re-equilibrate (values changed) and re-apply the
    // stored column permutation, then factor + solve.
    sparse_direct::matrix::CscMatrix As;
    std::vector<double> rscale, cscale;
    equilibrate(A, As, rscale, cscale);
    sparse_direct::matrix::CscMatrix B;
    build_matched(As, an.match, B);
    if (stats) stats->analysis_ms = ms_since(t0);  // reuse: just equilibration + permute
    return factor_solve_refine(B, rscale, cscale, an.match, an.perm, an.Lp, an.Li, A, b, x_out,
                               stats);
}

}  // namespace mysolver
