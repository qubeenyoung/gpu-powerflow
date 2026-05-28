#pragma once

#include "newton_solver/core/newton_solver.hpp"

// Internal CUDA/PyTorch interop bridge.
//
// These entry points intentionally live outside NewtonSolver's public API.
// Torch bindings validate tensor shape/dtype/device before passing data_ptr()
// values here. C++ callers should prefer solve/solve_batch/solve_adjoint.
namespace cupf::torch_api {

void solve_backward(
    NewtonSolver& solver,
    const void* grad_va_device_ptr,
    const void* grad_vm_device_ptr,
    void* grad_load_p_device_ptr,
    void* grad_load_q_device_ptr,
    int32_t batch_size,
    int32_t n_bus,
    const char* dtype,
    const AdjointOptions& options,
    AdjointResult& result);

void solve_forward(
    NewtonSolver& solver,
    const void* sbus_base_re_device_ptr,
    const void* sbus_base_im_device_ptr,
    const void* load_p_device_ptr,
    const void* load_q_device_ptr,
    const void* v0_va_device_ptr,
    const void* v0_vm_device_ptr,
    void* va_out_device_ptr,
    void* vm_out_device_ptr,
    int32_t batch_size,
    int32_t n_bus,
    const char* dtype,
    const NRConfig& config,
    const SolveOptions& solve_options,
    AdjointResult& result);

}  // namespace cupf::torch_api
