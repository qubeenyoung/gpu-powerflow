#pragma once

#include "linear/block_ilu_preconditioner.hpp"
#include "linear/csr_spmv.hpp"
#include "utils/cuda_utils.hpp"

#include <cublas_v2.h>

#include <cstdint>
#include <string>

namespace exp_20260415::block_ilu {

struct BicgstabOptions {
    double relative_tolerance = 1e-6;
    int32_t max_iterations = 500;
    bool collect_timing_breakdown = true;
};

struct BicgstabTiming {
    double preconditioner_sec = 0.0;
    double spmv_sec = 0.0;
    double reduction_sec = 0.0;
    double vector_update_sec = 0.0;
    double small_solve_sec = 0.0;
    double residual_refresh_sec = 0.0;
};

struct BicgstabStats {
    bool converged = false;
    int32_t iterations = 0;
    int32_t spmv_calls = 0;
    int32_t preconditioner_applies = 0;
    double initial_residual_norm = 0.0;
    double final_residual_norm = 0.0;
    double relative_residual_norm = 0.0;
    double solve_sec = 0.0;
    double avg_iteration_sec = 0.0;
    int32_t reduction_calls = 0;
    BicgstabTiming timing;
    std::string failure_reason;
};

class BicgstabSolver {
public:
    BicgstabSolver();
    ~BicgstabSolver();

    BicgstabSolver(const BicgstabSolver&) = delete;
    BicgstabSolver& operator=(const BicgstabSolver&) = delete;

    BicgstabStats solve(const CsrSpmv& matrix,
                        BlockIluPreconditioner& preconditioner,
                        int32_t n,
                        const double* rhs_device,
                        double* x_device,
                        const BicgstabOptions& options);

private:
    void ensure_workspace(int32_t n);
    double dot(int32_t n, const double* x, const double* y);
    double norm2(int32_t n, const double* x);

    cublasHandle_t cublas_ = nullptr;
    DeviceBuffer<double> r_;
    DeviceBuffer<double> r_hat_;
    DeviceBuffer<double> p_;
    DeviceBuffer<double> v_;
    DeviceBuffer<double> s_;
    DeviceBuffer<double> t_;
    DeviceBuffer<double> p_hat_;
    DeviceBuffer<double> s_hat_;
    DeviceBuffer<double> ax_;
};

}  // namespace exp_20260415::block_ilu
