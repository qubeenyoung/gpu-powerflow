#ifdef CUPF_WITH_TORCH

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/core/newton_solver_cuda_bridge.hpp"
#include "newton_solver/core/newton_solver_types.hpp"
#include "utils/cuda_utils.hpp"

#include <string>

namespace {

const char* tensor_dtype_name(const at::Tensor& t)
{
    if (t.scalar_type() == at::kDouble) return "float64";
    if (t.scalar_type() == at::kFloat) return "float32";
    TORCH_CHECK(false, "cuPF torch extension supports only torch.float64 and torch.float32 tensors");
}

void check_cuda_contiguous(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kDouble || t.scalar_type() == at::kFloat,
                name, " must be torch.float64 or torch.float32");
}

void check_same_device_dtype(const at::Tensor& ref, const at::Tensor& t, const char* name)
{
    check_cuda_contiguous(t, name);
    TORCH_CHECK(t.scalar_type() == ref.scalar_type(),
                name, " dtype must match load/input dtype");
    TORCH_CHECK(t.device().index() == ref.device().index(),
                name, " device index must match load/input tensor");
}

int32_t checked_batch_bus_shape(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.dim() == 2, name, " must have shape [batch_size, n_bus]");
    TORCH_CHECK(t.size(0) > 0 && t.size(1) > 0, name, " shape must be positive");
    return static_cast<int32_t>(t.size(1));
}

void check_batch_bus_like(const at::Tensor& ref, const at::Tensor& t, const char* name)
{
    check_same_device_dtype(ref, t, name);
    TORCH_CHECK(t.dim() == 2, name, " must have shape [batch_size, n_bus]");
    TORCH_CHECK(t.sizes() == ref.sizes(), name, " shape must match reference [batch_size, n_bus]");
}

void check_bus_vector_like(const at::Tensor& ref_batch, const at::Tensor& t, const char* name)
{
    check_same_device_dtype(ref_batch, t, name);
    TORCH_CHECK(t.dim() == 1, name, " must have shape [n_bus]");
    TORCH_CHECK(t.size(0) == ref_batch.size(1), name, " length must equal n_bus");
}

}  // namespace


AdjointResult solve_with_adjoint_cache_torch_binding(NewtonSolver& self,
                                                     at::Tensor sbus_base_re,
                                                     at::Tensor sbus_base_im,
                                                     at::Tensor load_p,
                                                     at::Tensor load_q,
                                                     at::Tensor v0_va,
                                                     at::Tensor v0_vm,
                                                     at::Tensor va_out,
                                                     at::Tensor vm_out,
                                                     const NRConfig& config,
                                                     const SolveOptions& solve_options)
{
    check_cuda_contiguous(load_p, "load_p");
    const int32_t n_bus = checked_batch_bus_shape(load_p, "load_p");
    const int32_t batch_size = static_cast<int32_t>(load_p.size(0));
    check_batch_bus_like(load_p, load_q, "load_q");
    check_batch_bus_like(load_p, va_out, "va_out");
    check_batch_bus_like(load_p, vm_out, "vm_out");
    check_bus_vector_like(load_p, sbus_base_re, "sbus_base_re");
    check_bus_vector_like(load_p, sbus_base_im, "sbus_base_im");
    check_bus_vector_like(load_p, v0_va, "v0_va");
    check_bus_vector_like(load_p, v0_vm, "v0_vm");

    const int device_index = load_p.device().index();
    at::cuda::CUDAGuard guard(load_p.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();
    ScopedCudaStream scoped_stream(stream);

    AdjointResult result;
    cupf::torch_api::solve_forward(self,
                                   sbus_base_re.data_ptr(),
                                   sbus_base_im.data_ptr(),
                                   load_p.data_ptr(),
                                   load_q.data_ptr(),
                                   v0_va.data_ptr(),
                                   v0_vm.data_ptr(),
                                   va_out.data_ptr(),
                                   vm_out.data_ptr(),
                                   batch_size,
                                   n_bus,
                                   tensor_dtype_name(load_p),
                                   config,
                                   solve_options,
                                   result);
    result.torch_extension_zero_copy = true;
    result.raw_pointer_api_used = false;
    result.current_stream_integrated = true;
    result.includes_host_device_transfer = false;
    result.zero_copy = true;
    return result;
}


AdjointResult solve_adjoint_torch_binding(NewtonSolver& self,
                                          at::Tensor grad_va,
                                          at::Tensor grad_vm,
                                          at::Tensor grad_load_p_out,
                                          at::Tensor grad_load_q_out,
                                          const AdjointOptions& options)
{
    check_cuda_contiguous(grad_va, "grad_va");
    const int32_t n_bus = checked_batch_bus_shape(grad_va, "grad_va");
    const int32_t batch_size = static_cast<int32_t>(grad_va.size(0));
    check_batch_bus_like(grad_va, grad_vm, "grad_vm");
    check_batch_bus_like(grad_va, grad_load_p_out, "grad_load_p_out");
    check_batch_bus_like(grad_va, grad_load_q_out, "grad_load_q_out");

    const int device_index = grad_va.device().index();
    at::cuda::CUDAGuard guard(grad_va.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();
    ScopedCudaStream scoped_stream(stream);

    AdjointResult result;
    cupf::torch_api::solve_backward(self,
                                    grad_va.data_ptr(),
                                    grad_vm.data_ptr(),
                                    grad_load_p_out.data_ptr(),
                                    grad_load_q_out.data_ptr(),
                                    batch_size,
                                    n_bus,
                                    tensor_dtype_name(grad_va),
                                    options,
                                    result);
    result.torch_extension_zero_copy = true;
    result.raw_pointer_api_used = false;
    result.current_stream_integrated = true;
    result.includes_host_device_transfer = false;
    result.zero_copy = true;
    return result;
}

#endif  // CUPF_WITH_TORCH
