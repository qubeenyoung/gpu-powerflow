#pragma once

#include "linear_system_io.hpp"

#include <Eigen/Sparse>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace exp_20260413::iterative::probe {

using Snapshot = exp_20260413::iterative::LinearSystemSnapshot;
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;
using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Clock = std::chrono::steady_clock;

struct SolverOptions {
    std::string solver = "bicgstab_ilut";
    double tolerance = 1e-8;
    int32_t max_iter = 1000;
    double ilut_drop_tol = 1e-4;
    int32_t ilut_fill_factor = 10;
    int32_t block_size = 32;
    double ilu_pivot_tol = 1e-12;
};

struct ProbeResult {
    std::filesystem::path snapshot_dir;
    std::string solver;
    bool success = false;
    int iterations = 0;
    double estimated_error = std::numeric_limits<double>::quiet_NaN();
    double setup_sec = 0.0;
    double solve_sec = 0.0;
    double residual_inf = std::numeric_limits<double>::quiet_NaN();
    double relative_residual_inf = std::numeric_limits<double>::quiet_NaN();
    double rhs_inf = 0.0;
    double x_inf = std::numeric_limits<double>::quiet_NaN();
    double direct_residual_inf = std::numeric_limits<double>::quiet_NaN();
    double x_delta_direct_inf = std::numeric_limits<double>::quiet_NaN();
};

SparseMatrix make_sparse_matrix(const Snapshot& snapshot);
Vector make_rhs(const Snapshot& snapshot);
std::vector<double> to_std_vector(const Vector& vector);
double vector_inf_norm(const Vector& vector);

// The experiment is about solver behavior, so we keep a tiny pivot guard here
// instead of failing every time a local incomplete factorization sees a zero-ish diagonal.
double safe_pivot(double value, double pivot_tol);

ProbeResult make_probe_result(const std::filesystem::path& snapshot_dir,
                              const std::string& solver_name,
                              const Snapshot& snapshot);

void finalize_probe_result(ProbeResult& result,
                           const Snapshot& snapshot,
                           const Vector& x);

}  // namespace exp_20260413::iterative::probe
