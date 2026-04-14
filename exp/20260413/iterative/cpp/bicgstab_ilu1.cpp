#include "iterative_solvers.hpp"
#include "manual_bicgstab.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace exp_20260413::iterative::probe {
namespace {

// ILU(1): allow one level of fill during incomplete factorization.
// This is still level-based ILU, not ILUT's dual-threshold dropping rule.
class LevelIluPreconditioner {
public:
    bool compute(const SparseMatrix& matrix, int32_t max_level, double pivot_tol)
    {
        pivot_tol_ = pivot_tol;
        max_level_ = max_level;
        n_ = static_cast<int32_t>(matrix.rows());
        if (matrix.rows() != matrix.cols()) {
            return false;
        }

        rows_.clear();
        rows_.resize(static_cast<std::size_t>(n_));
        diag_pos_.assign(static_cast<std::size_t>(n_), -1);

        for (int32_t row = 0; row < n_; ++row) {
            std::map<int32_t, Entry> current;
            for (int32_t p = matrix.outerIndexPtr()[row]; p < matrix.outerIndexPtr()[row + 1]; ++p) {
                current[matrix.innerIndexPtr()[p]] =
                    Entry{matrix.innerIndexPtr()[p], matrix.valuePtr()[p], 0};
            }
            if (current.find(row) == current.end()) {
                current[row] = Entry{row, 0.0, 0};
            }

            for (auto it = current.begin(); it != current.end(); ++it) {
                const int32_t elim_col = it->first;
                if (elim_col >= row) {
                    break;
                }
                if (it->second.level > max_level_) {
                    continue;
                }

                const int32_t elim_diag_pos = diag_pos_[static_cast<std::size_t>(elim_col)];
                if (elim_diag_pos < 0) {
                    return false;
                }

                const double elim_diag = rows_[static_cast<std::size_t>(elim_col)]
                                             [static_cast<std::size_t>(elim_diag_pos)].value;
                const double multiplier = it->second.value / safe_pivot(elim_diag, pivot_tol_);
                it->second.value = multiplier;

                const auto& elim_row = rows_[static_cast<std::size_t>(elim_col)];
                for (const Entry& elim_entry : elim_row) {
                    if (elim_entry.col <= elim_col) {
                        continue;
                    }
                    const int32_t fill_level = it->second.level + elim_entry.level + 1;
                    if (fill_level > max_level_) {
                        continue;
                    }

                    auto found = current.find(elim_entry.col);
                    if (found == current.end()) {
                        current[elim_entry.col] =
                            Entry{elim_entry.col, -multiplier * elim_entry.value, fill_level};
                    } else {
                        found->second.value -= multiplier * elim_entry.value;
                        found->second.level = std::min(found->second.level, fill_level);
                    }
                }
            }

            auto diag = current.find(row);
            if (diag == current.end()) {
                current[row] = Entry{row, safe_pivot(0.0, pivot_tol_), 0};
            } else if (std::abs(diag->second.value) < pivot_tol_) {
                diag->second.value = safe_pivot(diag->second.value, pivot_tol_);
            }

            auto& out_row = rows_[static_cast<std::size_t>(row)];
            out_row.reserve(current.size());
            int32_t diag_index = -1;
            for (const auto& [col, entry] : current) {
                if (entry.level <= max_level_ || col == row) {
                    if (col == row) {
                        diag_index = static_cast<int32_t>(out_row.size());
                    }
                    out_row.push_back(entry);
                }
            }
            diag_pos_[static_cast<std::size_t>(row)] = diag_index;
            if (diag_index < 0) {
                return false;
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
            for (const Entry& entry : rows_[static_cast<std::size_t>(row)]) {
                if (entry.col >= row) {
                    break;
                }
                sum -= entry.value * y[entry.col];
            }
            y[row] = sum;
        }

        for (int32_t row = n_ - 1; row >= 0; --row) {
            double sum = y[row];
            double diag = 0.0;
            for (const Entry& entry : rows_[static_cast<std::size_t>(row)]) {
                if (entry.col == row) {
                    diag = entry.value;
                } else if (entry.col > row) {
                    sum -= entry.value * x[entry.col];
                }
            }
            x[row] = sum / safe_pivot(diag, pivot_tol_);
        }

        return x;
    }

private:
    struct Entry {
        int32_t col = 0;
        double value = 0.0;
        int32_t level = 0;
    };

    int32_t n_ = 0;
    int32_t max_level_ = 1;
    double pivot_tol_ = 1e-12;
    std::vector<std::vector<Entry>> rows_;
    std::vector<int32_t> diag_pos_;
};

}  // namespace

ProbeResult solve_bicgstab_ilu1(const SparseMatrix& matrix,
                                const Vector& rhs,
                                const Snapshot& snapshot,
                                const SolverOptions& options,
                                const std::filesystem::path& snapshot_dir)
{
    constexpr int32_t kLevel = 1;
    return solve_with_manual_preconditioner<LevelIluPreconditioner>(
        "bicgstab_ilu1",
        matrix,
        rhs,
        snapshot,
        options,
        snapshot_dir,
        [&](LevelIluPreconditioner& preconditioner) {
            return preconditioner.compute(matrix, kLevel, options.ilu_pivot_tol);
        });
}

}  // namespace exp_20260413::iterative::probe
