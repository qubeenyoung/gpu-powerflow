#include "mysolver/factorize/dense_lu.hpp"

#include <cmath>
#include <utility>

namespace mysolver::numeric {

namespace {
inline double& at(std::vector<double>& a, int n, int i, int j) { return a[i + static_cast<std::size_t>(j) * n]; }
inline double at(const std::vector<double>& a, int n, int i, int j) { return a[i + static_cast<std::size_t>(j) * n]; }
}  // namespace

bool dense_lu(int n, std::vector<double>& a, std::vector<int>& piv)
{
    piv.assign(n, 0);
    for (int k = 0; k < n; ++k) {
        // Partial pivot: largest magnitude in column k at or below the diagonal.
        int p = k;
        double best = std::fabs(at(a, n, k, k));
        for (int i = k + 1; i < n; ++i) {
            const double v = std::fabs(at(a, n, i, k));
            if (v > best) {
                best = v;
                p = i;
            }
        }
        piv[k] = p;
        if (best == 0.0) {
            return false;  // singular
        }
        if (p != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(at(a, n, k, j), at(a, n, p, j));
            }
        }
        const double akk = at(a, n, k, k);
        for (int i = k + 1; i < n; ++i) {
            at(a, n, i, k) /= akk;  // multipliers -> L
        }
        // Right-looking trailing-submatrix update.
        for (int j = k + 1; j < n; ++j) {
            const double akj = at(a, n, k, j);
            if (akj == 0.0) {
                continue;
            }
            for (int i = k + 1; i < n; ++i) {
                at(a, n, i, j) -= at(a, n, i, k) * akj;
            }
        }
    }
    return true;
}

void dense_lu_solve(int n, const std::vector<double>& lu, const std::vector<int>& piv,
                    std::vector<double>& x)
{
    // Apply row pivots in factorization order.
    for (int k = 0; k < n; ++k) {
        if (piv[k] != k) {
            std::swap(x[k], x[piv[k]]);
        }
    }
    // Forward substitution with unit-lower L.
    for (int i = 0; i < n; ++i) {
        double s = x[i];
        for (int k = 0; k < i; ++k) {
            s -= at(lu, n, i, k) * x[k];
        }
        x[i] = s;
    }
    // Back substitution with U.
    for (int i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (int k = i + 1; k < n; ++k) {
            s -= at(lu, n, i, k) * x[k];
        }
        x[i] = s / at(lu, n, i, i);
    }
}

}  // namespace mysolver::numeric
