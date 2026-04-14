#include "iterative_solvers.hpp"
#include "manual_bicgstab.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace exp_20260413::iterative::probe {
namespace {

// ILU(0): keep exactly the original sparsity pattern of J.
// No fill entries are created; this is the lightest ILU point in this experiment.
class Ilu0Preconditioner {
public:
    bool compute(const SparseMatrix& matrix, double pivot_tol)
    {
        pivot_tol_ = pivot_tol;
        n_ = static_cast<int32_t>(matrix.rows());
        if (matrix.rows() != matrix.cols()) {
            return false;
        }

        const int32_t nnz = static_cast<int32_t>(matrix.nonZeros());
        row_ptr_.assign(matrix.outerIndexPtr(), matrix.outerIndexPtr() + n_ + 1);
        col_idx_.assign(matrix.innerIndexPtr(), matrix.innerIndexPtr() + nnz);
        lu_.assign(matrix.valuePtr(), matrix.valuePtr() + nnz);
        diag_pos_.assign(static_cast<std::size_t>(n_), -1);
        row_pos_.clear();
        row_pos_.resize(static_cast<std::size_t>(n_));

        for (int32_t row = 0; row < n_; ++row) {
            auto& positions = row_pos_[static_cast<std::size_t>(row)];
            positions.reserve(static_cast<std::size_t>(row_ptr_[row + 1] - row_ptr_[row]));
            for (int32_t p = row_ptr_[row]; p < row_ptr_[row + 1]; ++p) {
                const int32_t col = col_idx_[static_cast<std::size_t>(p)];
                positions[col] = p;
                if (col == row) {
                    diag_pos_[static_cast<std::size_t>(row)] = p;
                }
            }
            if (diag_pos_[static_cast<std::size_t>(row)] < 0) {
                return false;
            }
        }

        for (int32_t row = 0; row < n_; ++row) {
            for (int32_t p = row_ptr_[row]; p < row_ptr_[row + 1]; ++p) {
                const int32_t col = col_idx_[static_cast<std::size_t>(p)];
                double value = lu_[static_cast<std::size_t>(p)];
                const int32_t max_elim_col = std::min(row, col);

                for (int32_t q = row_ptr_[row]; q < row_ptr_[row + 1]; ++q) {
                    const int32_t elim_col = col_idx_[static_cast<std::size_t>(q)];
                    if (elim_col >= max_elim_col) {
                        break;
                    }

                    const auto& elim_row = row_pos_[static_cast<std::size_t>(elim_col)];
                    const auto found = elim_row.find(col);
                    if (found != elim_row.end()) {
                        value -= lu_[static_cast<std::size_t>(q)] *
                                 lu_[static_cast<std::size_t>(found->second)];
                    }
                }

                if (col < row) {
                    const int32_t diag_pos = diag_pos_[static_cast<std::size_t>(col)];
                    double diag = lu_[static_cast<std::size_t>(diag_pos)];
                    if (std::abs(diag) < pivot_tol_) {
                        diag = safe_pivot(diag, pivot_tol_);
                        lu_[static_cast<std::size_t>(diag_pos)] = diag;
                    }
                    lu_[static_cast<std::size_t>(p)] = value / diag;
                } else {
                    if (col == row && std::abs(value) < pivot_tol_) {
                        value = safe_pivot(value, pivot_tol_);
                    }
                    lu_[static_cast<std::size_t>(p)] = value;
                }
            }
        }

        return true;
    }

    Vector solve(const Vector& rhs) const
    {
        Vector y(rhs.size());
        Vector x(rhs.size());
        y.setZero();
        x.setZero();

        for (int32_t row = 0; row < n_; ++row) {
            double sum = rhs[row];
            for (int32_t p = row_ptr_[row]; p < row_ptr_[row + 1]; ++p) {
                const int32_t col = col_idx_[static_cast<std::size_t>(p)];
                if (col >= row) {
                    break;
                }
                sum -= lu_[static_cast<std::size_t>(p)] * y[col];
            }
            y[row] = sum;
        }

        for (int32_t row = n_ - 1; row >= 0; --row) {
            double sum = y[row];
            double diag = 0.0;
            for (int32_t p = row_ptr_[row]; p < row_ptr_[row + 1]; ++p) {
                const int32_t col = col_idx_[static_cast<std::size_t>(p)];
                const double value = lu_[static_cast<std::size_t>(p)];
                if (col == row) {
                    diag = value;
                } else if (col > row) {
                    sum -= value * x[col];
                }
            }
            x[row] = sum / safe_pivot(diag, pivot_tol_);
        }

        return x;
    }

private:
    int32_t n_ = 0;
    double pivot_tol_ = 1e-12;
    std::vector<int32_t> row_ptr_;
    std::vector<int32_t> col_idx_;
    std::vector<double> lu_;
    std::vector<int32_t> diag_pos_;
    std::vector<std::unordered_map<int32_t, int32_t>> row_pos_;
};

}  // namespace

ProbeResult solve_bicgstab_ilu0(const SparseMatrix& matrix,
                                const Vector& rhs,
                                const Snapshot& snapshot,
                                const SolverOptions& options,
                                const std::filesystem::path& snapshot_dir)
{
    return solve_with_manual_preconditioner<Ilu0Preconditioner>(
        "bicgstab_ilu0",
        matrix,
        rhs,
        snapshot,
        options,
        snapshot_dir,
        [&](Ilu0Preconditioner& preconditioner) {
            return preconditioner.compute(matrix, options.ilu_pivot_tol);
        });
}

}  // namespace exp_20260413::iterative::probe
