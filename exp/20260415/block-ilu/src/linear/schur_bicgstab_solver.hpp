#pragma once

#include "linear/implicit_schur_operator.hpp"
#include "utils/cuda_utils.hpp"

#include <cublas_v2.h>

#include <cstdint>
#include <string>
#include <vector>

namespace exp_20260415::block_ilu {

struct SchurBicgstabOptions {
    double relative_tolerance = 1e-6;
    int32_t max_iterations = 500;
    int32_t gmres_restart = 30;
    int32_t gmres_residual_check_interval = 5;
    bool collect_timing_breakdown = true;
};

struct SchurBicgstabTiming {
    double schur_rhs_sec = 0.0;
    double schur_matvec_sec = 0.0;
    double schur_recover_sec = 0.0;
    double schur_spmv_sec = 0.0;
    double schur_j11_solve_sec = 0.0;
    double schur_preconditioner_sec = 0.0;
    double reduction_sec = 0.0;
    double vector_update_sec = 0.0;
    double small_solve_sec = 0.0;
    double residual_refresh_sec = 0.0;
};

struct SchurBicgstabStats {
    bool converged = false;
    int32_t iterations = 0;
    int32_t restart_cycles = 0;
    int32_t schur_matvec_calls = 0;
    int32_t schur_preconditioner_calls = 0;
    int32_t spmv_calls = 0;
    int32_t j11_solve_calls = 0;
    int32_t reduction_calls = 0;
    double initial_residual_norm = 0.0;
    double final_residual_norm = 0.0;
    double relative_residual_norm = 0.0;
    double solve_sec = 0.0;
    double avg_iteration_sec = 0.0;
    SchurBicgstabTiming timing;
    std::string failure_reason;
};

class SchurBicgstabSolver {
public:
    SchurBicgstabSolver();
    ~SchurBicgstabSolver();

    SchurBicgstabSolver(const SchurBicgstabSolver&) = delete;
    SchurBicgstabSolver& operator=(const SchurBicgstabSolver&) = delete;

    SchurBicgstabStats solve(ImplicitSchurOperator& op,
                             const double* rhs_full_device,
                             double* dx_full_device,
                             const SchurBicgstabOptions& options);

private:
    void ensure_workspace(int32_t n);
    void dot_to_device(int32_t n, const double* x, const double* y, double* out_device);

    cublasHandle_t cublas_ = nullptr;
    DeviceBuffer<double> rhs_s_;
    DeviceBuffer<double> dvm_;
    DeviceBuffer<double> r_;
    DeviceBuffer<double> r_hat_;
    DeviceBuffer<double> p_;
    DeviceBuffer<double> v_;
    DeviceBuffer<double> s_;
    DeviceBuffer<double> t_;
    DeviceBuffer<double> ax_;
    DeviceBuffer<double> d_scalars_;
    DeviceBuffer<double> d_reduce1_;
    DeviceBuffer<double> d_reduce2_;
    DeviceBuffer<int32_t> d_status_;
    std::vector<double> h_scalars_;
    std::vector<int32_t> h_status_;
};

}  // namespace exp_20260415::block_ilu
