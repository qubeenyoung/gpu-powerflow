#include "iterative_probe_common.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace exp_20260413::iterative::probe {

SparseMatrix make_sparse_matrix(const Snapshot& snapshot)
{
    std::vector<Eigen::Triplet<double, int32_t>> triplets;
    triplets.reserve(snapshot.values.size());

    for (int32_t row = 0; row < snapshot.rows; ++row) {
        for (int32_t k = snapshot.row_ptr[static_cast<std::size_t>(row)];
             k < snapshot.row_ptr[static_cast<std::size_t>(row + 1)];
             ++k) {
            triplets.emplace_back(row,
                                  snapshot.col_idx[static_cast<std::size_t>(k)],
                                  snapshot.values[static_cast<std::size_t>(k)]);
        }
    }

    SparseMatrix matrix(snapshot.rows, snapshot.cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

Vector make_rhs(const Snapshot& snapshot)
{
    Vector rhs(static_cast<Eigen::Index>(snapshot.rhs.size()));
    for (std::size_t i = 0; i < snapshot.rhs.size(); ++i) {
        rhs[static_cast<Eigen::Index>(i)] = snapshot.rhs[i];
    }
    return rhs;
}

std::vector<double> to_std_vector(const Vector& vector)
{
    std::vector<double> out(static_cast<std::size_t>(vector.size()));
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
        out[static_cast<std::size_t>(i)] = vector[i];
    }
    return out;
}

double vector_inf_norm(const Vector& vector)
{
    double norm = 0.0;
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
        norm = std::max(norm, std::abs(vector[i]));
    }
    return norm;
}

double safe_pivot(double value, double pivot_tol)
{
    if (std::abs(value) >= pivot_tol) {
        return value;
    }
    return value < 0.0 ? -pivot_tol : pivot_tol;
}

ProbeResult make_probe_result(const std::filesystem::path& snapshot_dir,
                              const std::string& solver_name,
                              const Snapshot& snapshot)
{
    ProbeResult result;
    result.snapshot_dir = snapshot_dir;
    result.solver = solver_name;
    result.rhs_inf = exp_20260413::iterative::inf_norm(snapshot.rhs);
    return result;
}

void finalize_probe_result(ProbeResult& result,
                           const Snapshot& snapshot,
                           const Vector& x)
{
    const std::vector<double> x_host = to_std_vector(x);
    result.residual_inf = exp_20260413::iterative::residual_inf_norm(snapshot, x_host);
    result.relative_residual_inf =
        result.rhs_inf > 0.0 ? result.residual_inf / result.rhs_inf : result.residual_inf;
    result.x_inf = exp_20260413::iterative::inf_norm(x_host);

    if (!snapshot.x_direct.empty()) {
        result.direct_residual_inf =
            exp_20260413::iterative::residual_inf_norm(snapshot, snapshot.x_direct);
        result.x_delta_direct_inf =
            exp_20260413::iterative::diff_inf_norm(x_host, snapshot.x_direct);
    }
}

}  // namespace exp_20260413::iterative::probe
