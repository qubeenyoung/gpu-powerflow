#pragma once

#include "linear/implicit_schur_operator_f32.hpp"
#include "linear/schur_bicgstab_solver.hpp"
#include "utils/cuda_utils.hpp"

#include <cublas_v2.h>

#include <cstdint>
#include <vector>

namespace exp_20260415::block_ilu {

class SchurBicgstabSolverF32 {
public:
    SchurBicgstabSolverF32();
    ~SchurBicgstabSolverF32();

    SchurBicgstabSolverF32(const SchurBicgstabSolverF32&) = delete;
    SchurBicgstabSolverF32& operator=(const SchurBicgstabSolverF32&) = delete;

    SchurBicgstabStats solve(ImplicitSchurOperatorF32& op,
                             const double* rhs_full_device,
                             double* dx_full_device,
                             const SchurBicgstabOptions& options);

private:
    void ensure_workspace(int32_t n);
    void dot_to_device(int32_t n, const float* x, const float* y, float* out_device);

    cublasHandle_t cublas_ = nullptr;
    DeviceBuffer<float> rhs_s_;
    DeviceBuffer<float> dvm_;
    DeviceBuffer<float> r_;
    DeviceBuffer<float> r_hat_;
    DeviceBuffer<float> p_;
    DeviceBuffer<float> v_;
    DeviceBuffer<float> s_;
    DeviceBuffer<float> t_;
    DeviceBuffer<float> p_hat_;
    DeviceBuffer<float> s_hat_;
    DeviceBuffer<float> ax_;
    DeviceBuffer<float> d_scalars_;
    DeviceBuffer<float> d_reduce1_;
    DeviceBuffer<float> d_reduce2_;
    DeviceBuffer<int32_t> d_status_;
    std::vector<float> h_scalars_;
    std::vector<int32_t> h_status_;
};

}  // namespace exp_20260415::block_ilu
