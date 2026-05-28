#include "mysolver/factorize/sparse_lu.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#include "mysolver/symbolic/multifrontal.hpp"
#include "mysolver/symbolic/supernode.hpp"

namespace mysolver::numeric {

namespace {

// Build the symmetric filled pattern S = fill(L) + fill(L)ᵀ (CSC, sorted) from
// the lower fill pattern Lp/Li (which already includes the diagonal).
void build_symmetric_filled(int n, const std::vector<int>& Lp, const std::vector<int>& Li,
                            std::vector<int>& Sp, std::vector<int>& Si)
{
    // S = fill(L) + fill(L)ᵀ. L's rows are unique per column (symbolic Cholesky),
    // and the mirrored entries (row < col) are disjoint from L's (row >= col), so
    // S has no duplicates -> a counting two-pass CSR build (no per-column vectors,
    // no unique) replaces the old vector<vector<int>>; only a per-column sort
    // remains. Much faster on large matrices (was n heap allocations).
    Sp.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
            const int i = Li[p];  // i >= j
            ++Sp[j + 1];
            if (i != j) ++Sp[i + 1];  // mirror
        }
    }
    for (int j = 0; j < n; ++j) Sp[j + 1] += Sp[j];
    Si.resize(Sp[n]);
    std::vector<int> next(Sp.begin(), Sp.end());
    for (int j = 0; j < n; ++j) {
        for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
            const int i = Li[p];
            Si[next[j]++] = i;
            if (i != j) Si[next[i]++] = j;
        }
    }
    for (int j = 0; j < n; ++j) std::sort(Si.begin() + Sp[j], Si.begin() + Sp[j + 1]);
}

// Position of row i in column j of S (sorted), or -1.
int find_pos(const std::vector<int>& Sp, const std::vector<int>& Si, int j, int i)
{
    const auto first = Si.begin() + Sp[j];
    const auto last = Si.begin() + Sp[j + 1];
    const auto it = std::lower_bound(first, last, i);
    if (it != last && *it == i) {
        return static_cast<int>(it - Si.begin());
    }
    return -1;
}

}  // namespace

bool factor_nopiv(int n, const int* Ap, const int* Ai, const double* Ax,
                  const std::vector<int>& Lp, const std::vector<int>& Li,
                  SparseLU& out)
{
    out.n = n;
    build_symmetric_filled(n, Lp, Li, out.Sp, out.Si);
    out.x.assign(out.Si.size(), 0.0);

    // Scatter A into the filled pattern.
    for (int j = 0; j < n; ++j) {
        for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
            const int pos = find_pos(out.Sp, out.Si, j, Ai[p]);
            if (pos >= 0) {
                out.x[pos] = Ax[p];
            }
        }
    }

    // Left-looking LU without pivoting on the known pattern.
    std::vector<double> work(n, 0.0);
    for (int j = 0; j < n; ++j) {
        const int cj0 = out.Sp[j], cj1 = out.Sp[j + 1];
        for (int p = cj0; p < cj1; ++p) {
            work[out.Si[p]] = out.x[p];
        }
        // Apply updates from columns k < j (sorted, so visited in increasing k).
        for (int p = cj0; p < cj1; ++p) {
            const int k = out.Si[p];
            if (k >= j) {
                break;
            }
            const double ukj = work[k];  // U(k,j), fully updated by k'<k
            if (ukj == 0.0) {
                continue;
            }
            for (int q = out.Sp[k]; q < out.Sp[k + 1]; ++q) {
                const int i = out.Si[q];
                if (i > k) {
                    work[i] -= out.x[q] * ukj;  // x[q] = L(i,k)
                }
            }
        }
        double pivot = 0.0;
        const int dpos = find_pos(out.Sp, out.Si, j, j);
        if (dpos >= 0) {
            pivot = work[j];
        }
        if (pivot == 0.0) {
            return false;
        }
        for (int p = cj0; p < cj1; ++p) {
            const int i = out.Si[p];
            if (i > j) {
                work[i] /= pivot;  // L(i,j)
            }
            out.x[p] = work[i];
            work[i] = 0.0;  // reset for the next column
        }
    }
    return true;
}

bool multifrontal_factor(int n, const int* Ap, const int* Ai, const double* Ax,
                         const std::vector<int>& Lp, const std::vector<int>& Li,
                         const std::vector<int>& parent, SparseLU& out, int panel_cap)
{
    namespace sym = mysolver::symbolic;
    out.n = n;
    build_symmetric_filled(n, Lp, Li, out.Sp, out.Si);
    out.x.assign(out.Si.size(), 0.0);
    if (n <= 0) {
        return true;
    }
    const std::vector<int>& Sp = out.Sp;
    const std::vector<int>& Si = out.Si;

    std::vector<int> colcount(n);
    for (int j = 0; j < n; ++j) colcount[j] = Lp[j + 1] - Lp[j];
    const sym::PanelPartition panels = sym::relaxed_panels(n, parent, colcount, panel_cap);
    const sym::MultifrontalSymbolic mf = sym::multifrontal_symbolic(n, Lp, Li, panels);
    const int P = panels.num_panels;

    // Local index of global row g within panel p's (sorted) front_rows slice.
    auto fidx = [&](int p, int g) {
        const auto b = mf.front_rows.begin() + mf.front_ptr[p];
        const auto e = mf.front_rows.begin() + mf.front_ptr[p + 1];
        return static_cast<int>(std::lower_bound(b, e, g) - b);
    };

    // Bucket each A entry (i,j) into its owner front = panel_of[min(i,j)]; both i
    // and j lie in that front (symmetric pattern). Precomputed once.
    std::vector<std::vector<std::array<int, 3>>> abuck(P);  // (row_local, col_local, A pos)
    for (int j = 0; j < n; ++j)
        for (int q = Ap[j]; q < Ap[j + 1]; ++q) {
            const int i = Ai[q];
            const int owner = panels.panel_of[i < j ? i : j];
            abuck[owner].push_back({fidx(owner, i), fidx(owner, j), q});
        }

    std::vector<std::vector<int>> children(P);
    for (int c = 0; c < P; ++c)
        if (mf.panel_parent[c] != -1) children[mf.panel_parent[c]].push_back(c);

    // Contribution blocks, kept until the parent consumes them (postorder => a
    // panel's children all have smaller ids and are processed first).
    std::vector<std::vector<double>> cb(P);

    for (int p = 0; p < P; ++p) {
        const int s = mf.front_ptr[p];
        const int fsz = mf.front_ptr[p + 1] - s;
        const int nc = panels.ncols[p];
        std::vector<double> F(static_cast<std::size_t>(fsz) * fsz, 0.0);

        // Assemble: original A entries + children contribution blocks (extend-add).
        for (const std::array<int, 3>& a : abuck[p])
            F[static_cast<std::size_t>(a[0]) * fsz + a[1]] += Ax[a[2]];
        for (int c : children[p]) {
            const int ccb = (mf.front_ptr[c + 1] - mf.front_ptr[c]) - panels.ncols[c];
            const int abase = mf.asm_ptr[c];
            const std::vector<double>& C = cb[c];
            for (int a = 0; a < ccb; ++a) {
                const int row = mf.asm_idx[abase + a] - s;  // parent-local row
                for (int b = 0; b < ccb; ++b) {
                    const int col = mf.asm_idx[abase + b] - s;
                    F[static_cast<std::size_t>(row) * fsz + col] +=
                        C[static_cast<std::size_t>(a) * ccb + b];
                }
            }
            cb[c] = std::vector<double>();  // free
        }

        // Dense no-pivot LU on the nc pivots; the trailing block becomes the CB.
        for (int k = 0; k < nc; ++k) {
            const double piv = F[static_cast<std::size_t>(k) * fsz + k];
            if (piv == 0.0) {
                return false;
            }
            for (int i = k + 1; i < fsz; ++i) F[static_cast<std::size_t>(i) * fsz + k] /= piv;
            for (int i = k + 1; i < fsz; ++i) {
                const double lik = F[static_cast<std::size_t>(i) * fsz + k];
                if (lik == 0.0) continue;
                for (int j = k + 1; j < fsz; ++j)
                    F[static_cast<std::size_t>(i) * fsz + j] -=
                        lik * F[static_cast<std::size_t>(k) * fsz + j];
            }
        }

        // Save the contribution block (trailing (fsz-nc) x (fsz-nc) Schur block).
        const int cbsz = fsz - nc;
        if (cbsz > 0 && mf.panel_parent[p] != -1) {
            cb[p].assign(static_cast<std::size_t>(cbsz) * cbsz, 0.0);
            for (int i = 0; i < cbsz; ++i)
                for (int j = 0; j < cbsz; ++j)
                    cb[p][static_cast<std::size_t>(i) * cbsz + j] =
                        F[static_cast<std::size_t>(nc + i) * fsz + (nc + j)];
        }

        // Emit every non-CB front entry (the panel's final L/U) into Sx. The CB
        // block (both indices >= nc) belongs to ancestors. Padded zeros fall
        // outside S (find_pos < 0) and are skipped.
        for (int ri = 0; ri < fsz; ++ri) {
            const int R = mf.front_rows[s + ri];
            for (int ci = 0; ci < fsz; ++ci) {
                if (ri >= nc && ci >= nc) continue;
                const int pos = find_pos(Sp, Si, mf.front_rows[s + ci], R);
                if (pos >= 0) out.x[pos] = F[static_cast<std::size_t>(ri) * fsz + ci];
            }
        }
    }
    return true;
}

void solve(const SparseLU& lu, const std::vector<double>& b, std::vector<double>& x_out)
{
    const int n = lu.n;
    x_out = b;
    // Forward: unit-lower L (column-oriented).
    for (int j = 0; j < n; ++j) {
        for (int p = lu.Sp[j]; p < lu.Sp[j + 1]; ++p) {
            const int i = lu.Si[p];
            if (i > j) {
                x_out[i] -= lu.x[p] * x_out[j];
            }
        }
    }
    // Backward: U (column-oriented), diagonal is the pivot U(j,j).
    for (int j = n - 1; j >= 0; --j) {
        double diag = 1.0;
        for (int p = lu.Sp[j]; p < lu.Sp[j + 1]; ++p) {
            if (lu.Si[p] == j) {
                diag = lu.x[p];
                break;
            }
        }
        const double xj = x_out[j] / diag;
        x_out[j] = xj;
        for (int p = lu.Sp[j]; p < lu.Sp[j + 1]; ++p) {
            const int i = lu.Si[p];
            if (i < j) {
                x_out[i] -= lu.x[p] * xj;  // U(i,j)
            }
        }
    }
}

}  // namespace mysolver::numeric
