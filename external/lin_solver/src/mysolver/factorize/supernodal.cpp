#include "mysolver/factorize/supernodal.hpp"

#include <algorithm>
#include <cstddef>

#include <mkl_cblas.h>

namespace mysolver::numeric {

namespace {

void build_symmetric_filled(int n, const std::vector<int>& Lp, const std::vector<int>& Li,
                            std::vector<int>& Sp, std::vector<int>& Si)
{
    std::vector<std::vector<int>> cols(n);
    for (int j = 0; j < n; ++j) {
        for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
            const int i = Li[p];
            cols[j].push_back(i);
            if (i != j) {
                cols[i].push_back(j);
            }
        }
    }
    Sp.assign(static_cast<std::size_t>(n) + 1, 0);
    Si.clear();
    for (int j = 0; j < n; ++j) {
        std::sort(cols[j].begin(), cols[j].end());
        cols[j].erase(std::unique(cols[j].begin(), cols[j].end()), cols[j].end());
        for (int i : cols[j]) {
            Si.push_back(i);
        }
        Sp[j + 1] = static_cast<int>(Si.size());
    }
}

int find_pos(const std::vector<int>& Sp, const std::vector<int>& Si, int j, int i)
{
    const auto first = Si.begin() + Sp[j];
    const auto last = Si.begin() + Sp[j + 1];
    const auto it = std::lower_bound(first, last, i);
    return (it != last && *it == i) ? static_cast<int>(it - Si.begin()) : -1;
}

}  // namespace

SupernodalStruct build_supernodal(int n, const std::vector<int>& Lp, const std::vector<int>& Li,
                                  const symbolic::SupernodePartition& sp)
{
    // Assumes the matrix is in postorder, so each supernode occupies a
    // contiguous column range [sn_first, sn_first + sn_size).
    SupernodalStruct ss;
    ss.n = n;
    ss.num_supernodes = sp.num_supernodes;
    const int S = sp.num_supernodes;
    ss.sn_first.assign(S, -1);
    ss.sn_size.assign(S, 0);
    for (int j = 0; j < n; ++j) {
        const int s = sp.snode_of[j];
        if (ss.sn_first[s] == -1 || j < ss.sn_first[s]) {
            ss.sn_first[s] = j;
        }
        ++ss.sn_size[s];
    }

    // Panel rows of a supernode = the fill structure of its first column (the
    // largest, since within a fundamental supernode the structures nest).
    ss.panel_ptr.assign(static_cast<std::size_t>(S) + 1, 0);
    for (int s = 0; s < S; ++s) {
        const int c0 = ss.sn_first[s];
        ss.panel_ptr[s + 1] = ss.panel_ptr[s] + (Lp[c0 + 1] - Lp[c0]);
    }
    ss.panel_rows.resize(ss.panel_ptr[S]);
    for (int s = 0; s < S; ++s) {
        const int c0 = ss.sn_first[s];
        int dst = ss.panel_ptr[s];
        for (int p = Lp[c0]; p < Lp[c0 + 1]; ++p) {
            ss.panel_rows[dst++] = Li[p];
        }
    }
    return ss;
}

bool factor_supernodal(int n, const int* Ap, const int* Ai, const double* Ax,
                       const std::vector<int>& Lp, const std::vector<int>& Li,
                       const SupernodalStruct& ss, SparseLU& out)
{
    out.n = n;
    build_symmetric_filled(n, Lp, Li, out.Sp, out.Si);
    out.x.assign(out.Si.size(), 0.0);
    for (int j = 0; j < n; ++j) {
        for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
            const int pos = find_pos(out.Sp, out.Si, j, Ai[p]);
            if (pos >= 0) {
                out.x[pos] = Ax[p];
            }
        }
    }
    auto Sref = [&](int i, int j) -> double* {
        const int p = find_pos(out.Sp, out.Si, j, i);
        return p >= 0 ? &out.x[p] : nullptr;
    };

    for (int s = 0; s < ss.num_supernodes; ++s) {
        const int c0 = ss.sn_first[s];
        const int sz = ss.sn_size[s];
        const int pb = ss.panel_ptr[s];
        const int nr = ss.panel_ptr[s + 1] - pb;
        const int nb = nr - sz;
        const int* rows = &ss.panel_rows[pb];  // sorted; rows[0..sz-1] = c0..c0+sz-1

        // Gather the dense column panel (L side) and right-row panel (U side).
        std::vector<double> Pcol(static_cast<std::size_t>(nr) * sz, 0.0);
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < sz; ++c) {
                double* p = Sref(rows[r], c0 + c);
                Pcol[static_cast<std::size_t>(r) * sz + c] = p ? *p : 0.0;
            }
        }
        std::vector<double> Prow(static_cast<std::size_t>(sz) * nb, 0.0);
        for (int c = 0; c < sz; ++c) {
            for (int a = 0; a < nb; ++a) {
                double* p = Sref(c0 + c, rows[sz + a]);
                Prow[static_cast<std::size_t>(c) * nb + a] = p ? *p : 0.0;
            }
        }

        // 1) No-pivot LU of the column panel (diagonal block + below rows).
        for (int kk = 0; kk < sz; ++kk) {
            const double piv = Pcol[static_cast<std::size_t>(kk) * sz + kk];
            if (piv == 0.0) {
                return false;
            }
            for (int r = kk + 1; r < nr; ++r) {
                Pcol[static_cast<std::size_t>(r) * sz + kk] /= piv;
            }
            for (int jj = kk + 1; jj < sz; ++jj) {
                const double pkj = Pcol[static_cast<std::size_t>(kk) * sz + jj];
                if (pkj == 0.0) {
                    continue;
                }
                for (int r = kk + 1; r < nr; ++r) {
                    Pcol[static_cast<std::size_t>(r) * sz + jj] -=
                        Pcol[static_cast<std::size_t>(r) * sz + kk] * pkj;
                }
            }
        }
        // 2) Right-U panel: forward solve L_d * Uright = origRight (unit-lower L_d).
        for (int a = 0; a < nb; ++a) {
            for (int kk = 0; kk < sz; ++kk) {
                double val = Prow[static_cast<std::size_t>(kk) * nb + a];
                for (int c = 0; c < kk; ++c) {
                    val -= Pcol[static_cast<std::size_t>(kk) * sz + c] *
                           Prow[static_cast<std::size_t>(c) * nb + a];
                }
                Prow[static_cast<std::size_t>(kk) * nb + a] = val;
            }
        }
        // 3) Schur update S(i,k) -= L_below · U_right. Dense GEMM (cblas) into a
        // buffer, then scatter into S (the extend-add). This is the dominant cost
        // and the dense-BLAS win on large supernodes.
        if (nb > 0) {
            std::vector<double> C(static_cast<std::size_t>(nb) * nb, 0.0);
            const double* Lbelow = &Pcol[static_cast<std::size_t>(sz) * sz];  // nb x sz, lda=sz
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nb, nb, sz,
                        1.0, Lbelow, sz, Prow.data(), nb, 0.0, C.data(), nb);
            for (int a = 0; a < nb; ++a) {
                const int i = rows[sz + a];
                for (int b = 0; b < nb; ++b) {
                    const double upd = C[static_cast<std::size_t>(a) * nb + b];
                    if (upd != 0.0) {
                        double* pp = Sref(i, rows[sz + b]);
                        if (pp) {
                            *pp -= upd;
                        }
                    }
                }
            }
        }
        // Store the factored panels back into S.
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < sz; ++c) {
                double* pp = Sref(rows[r], c0 + c);
                if (pp) {
                    *pp = Pcol[static_cast<std::size_t>(r) * sz + c];
                }
            }
        }
        for (int c = 0; c < sz; ++c) {
            for (int a = 0; a < nb; ++a) {
                double* pp = Sref(c0 + c, rows[sz + a]);
                if (pp) {
                    *pp = Prow[static_cast<std::size_t>(c) * nb + a];
                }
            }
        }
    }
    return true;
}

}  // namespace mysolver::numeric
