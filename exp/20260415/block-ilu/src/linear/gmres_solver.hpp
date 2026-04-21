#pragma once

#include "linear/block_ilu_preconditioner.hpp"
#include "linear/csr_spmv.hpp"
#include "utils/cuda_utils.hpp"

#include <cublas_v2.h>

#include <cstdint>
#include <string>
#include <vector>

namespace exp_20260415::block_ilu {

struct GmresOptions {
    double relative_tolerance = 1e-6;
    int32_t max_iterations = 500;
    int32_t restart = 30;
    int32_t residual_check_interval = 5;
    bool collect_timing_breakdown = true;
};

struct GmresTiming {
    double preconditioner_sec = 0.0;
    double spmv_sec = 0.0;
    double reduction_sec = 0.0;
    double vector_update_sec = 0.0;
    double small_solve_sec = 0.0;
    double residual_refresh_sec = 0.0;
};

struct GmresStats {
    bool converged = false;
    int32_t iterations = 0;
    int32_t restart_cycles = 0;
    int32_t spmv_calls = 0;
    int32_t preconditioner_applies = 0;
    int32_t reduction_calls = 0;
    double initial_residual_norm = 0.0;
    double final_residual_norm = 0.0;
    double relative_residual_norm = 0.0;
    double solve_sec = 0.0;
    double iteration_sec = 0.0;
    double avg_iteration_sec = 0.0;
    GmresTiming timing;
    std::string failure_reason;
};

class GmresSolver {
public:
    GmresSolver();
    ~GmresSolver();

    GmresSolver(const GmresSolver&) = delete;
    GmresSolver& operator=(const GmresSolver&) = delete;

    GmresStats solve(const CsrSpmv& matrix,
                     BlockIluPreconditioner& preconditioner,
                     int32_t n,
                     const double* rhs_device,
                     double* x_device,
                     const GmresOptions& options);

private:
    void ensure_workspace(int32_t n, int32_t restart);
    void set_zero(int32_t n, double* x);
    void copy_vector(int32_t n, const double* src, double* dst);
    void norm_to_device(int32_t n, const double* x, double* out_device);
    void dot_to_device(int32_t n, const double* x, const double* y, double* out_device);
    double device_scalar_to_host(const double* scalar_device);
    double recompute_residual(const CsrSpmv& matrix,
                              int32_t n,
                              const double* rhs_device,
                              const double* x_device,
                              double rhs_norm,
                              GmresStats& stats,
                              const GmresOptions& options);
    void update_solution(int32_t n, int32_t basis_count, double* x_device);
    bool solve_small_upper(int32_t basis_count);
    double apply_givens_and_residual(int32_t column, double rhs_norm);

    cublasHandle_t cublas_ = nullptr;
    int32_t workspace_n_ = 0;
    int32_t workspace_restart_ = 0;
    DeviceBuffer<double> r_;
    DeviceBuffer<double> w_;
    DeviceBuffer<double> ax_;
    DeviceBuffer<double> v_basis_;
    DeviceBuffer<double> z_basis_;
    DeviceBuffer<double> d_h_col_;
    DeviceBuffer<double> d_y_;
    std::vector<double> h_col_host_;
    std::vector<double> hessenberg_;
    std::vector<double> givens_c_;
    std::vector<double> givens_s_;
    std::vector<double> g_;
    std::vector<double> y_host_;
};

}  // namespace exp_20260415::block_ilu
