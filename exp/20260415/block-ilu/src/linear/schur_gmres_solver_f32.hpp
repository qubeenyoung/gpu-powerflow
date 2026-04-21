#pragma once

#include "linear/implicit_schur_operator_f32.hpp"
#include "linear/schur_bicgstab_solver.hpp"
#include "utils/cuda_utils.hpp"

#include <cublas_v2.h>

#include <cstdint>
#include <vector>

namespace exp_20260415::block_ilu {

class SchurGmresSolverF32 {
public:
    SchurGmresSolverF32();
    ~SchurGmresSolverF32();

    SchurGmresSolverF32(const SchurGmresSolverF32&) = delete;
    SchurGmresSolverF32& operator=(const SchurGmresSolverF32&) = delete;

    SchurBicgstabStats solve(ImplicitSchurOperatorF32& op,
                             const double* rhs_full_device,
                             double* dx_full_device,
                             const SchurBicgstabOptions& options);

private:
    void ensure_workspace(int32_t n, int32_t restart);
    void set_zero(int32_t n, float* x);
    void copy_vector(int32_t n, const float* src, float* dst);
    void norm_to_device(int32_t n, const float* x, float* out_device);
    void dot_to_device(int32_t n, const float* x, const float* y, float* out_device);
    float device_scalar_to_host(const float* scalar_device);
    float recompute_residual(ImplicitSchurOperatorF32& op,
                             int32_t n,
                             const float* rhs_device,
                             const float* x_device,
                             float rhs_norm,
                             SchurBicgstabStats& stats,
                             SchurOperatorStats& op_stats,
                             const SchurBicgstabOptions& options);
    void update_solution(int32_t n, int32_t basis_count, float* x_device);
    bool solve_small_upper(int32_t basis_count);
    float apply_givens_and_residual(int32_t column, float rhs_norm);

    cublasHandle_t cublas_ = nullptr;
    int32_t workspace_n_ = 0;
    int32_t workspace_restart_ = 0;
    DeviceBuffer<float> rhs_s_;
    DeviceBuffer<float> dvm_;
    DeviceBuffer<float> r_;
    DeviceBuffer<float> w_;
    DeviceBuffer<float> ax_;
    DeviceBuffer<float> v_basis_;
    DeviceBuffer<float> z_basis_;
    DeviceBuffer<float> d_h_col_;
    DeviceBuffer<float> d_y_;
    std::vector<float> h_col_host_;
    std::vector<float> hessenberg_;
    std::vector<float> givens_c_;
    std::vector<float> givens_s_;
    std::vector<float> g_;
    std::vector<float> y_host_;
};

}  // namespace exp_20260415::block_ilu
