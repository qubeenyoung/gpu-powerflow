#include "iterative_solvers.hpp"
#include "manual_bicgstab.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace exp_20260413::iterative::probe {
namespace {

// Block-Jacobi: invert only small contiguous diagonal blocks.
// It avoids triangular solves, which is why it is a useful GPU-oriented foil to ILU.
class BlockJacobiPreconditioner {
public:
    bool compute(const SparseMatrix& matrix, int32_t block_size, double pivot_tol)
    {
        if (matrix.rows() != matrix.cols()) {
            return false;
        }

        n_ = static_cast<int32_t>(matrix.rows());
        block_size_ = block_size;
        pivot_tol_ = pivot_tol;
        blocks_.clear();
        blocks_.reserve(static_cast<std::size_t>((n_ + block_size_ - 1) / block_size_));

        for (int32_t start = 0; start < n_; start += block_size_) {
            const int32_t size = std::min(block_size_, n_ - start);
            Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(size, size);

            for (int32_t row = start; row < start + size; ++row) {
                for (int32_t p = matrix.outerIndexPtr()[row]; p < matrix.outerIndexPtr()[row + 1]; ++p) {
                    const int32_t col = matrix.innerIndexPtr()[p];
                    if (col >= start && col < start + size) {
                        dense(row - start, col - start) = matrix.valuePtr()[p];
                    }
                }
            }

            Block block;
            block.start = start;
            block.size = size;

            Eigen::FullPivLU<Eigen::MatrixXd> lu(dense);
            lu.setThreshold(pivot_tol_);
            if (!lu.isInvertible()) {
                for (int32_t i = 0; i < size; ++i) {
                    dense(i, i) += pivot_tol_;
                }
                lu.compute(dense);
                lu.setThreshold(pivot_tol_);
            }

            if (lu.isInvertible()) {
                block.inverse = lu.inverse();
            } else {
                block.diagonal_fallback = true;
                block.diag_inv.resize(size);
                for (int32_t i = 0; i < size; ++i) {
                    block.diag_inv[i] = 1.0 / safe_pivot(dense(i, i), pivot_tol_);
                }
            }

            blocks_.push_back(std::move(block));
        }

        return true;
    }

    Vector solve(const Vector& rhs) const
    {
        Vector out(rhs.size());
        out.setZero();
        for (const Block& block : blocks_) {
            if (block.diagonal_fallback) {
                for (int32_t i = 0; i < block.size; ++i) {
                    out[block.start + i] = block.diag_inv[i] * rhs[block.start + i];
                }
            } else {
                out.segment(block.start, block.size) =
                    block.inverse * rhs.segment(block.start, block.size);
            }
        }
        return out;
    }

private:
    struct Block {
        int32_t start = 0;
        int32_t size = 0;
        bool diagonal_fallback = false;
        Eigen::MatrixXd inverse;
        Vector diag_inv;
    };

    int32_t n_ = 0;
    int32_t block_size_ = 32;
    double pivot_tol_ = 1e-12;
    std::vector<Block> blocks_;
};

}  // namespace

ProbeResult solve_bicgstab_block_jacobi(const SparseMatrix& matrix,
                                        const Vector& rhs,
                                        const Snapshot& snapshot,
                                        const SolverOptions& options,
                                        const std::filesystem::path& snapshot_dir)
{
    return solve_with_manual_preconditioner<BlockJacobiPreconditioner>(
        "bicgstab_block_jacobi",
        matrix,
        rhs,
        snapshot,
        options,
        snapshot_dir,
        [&](BlockJacobiPreconditioner& preconditioner) {
            return preconditioner.compute(matrix, options.block_size, options.ilu_pivot_tol);
        });
}

}  // namespace exp_20260413::iterative::probe
