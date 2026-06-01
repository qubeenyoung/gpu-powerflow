// ---------------------------------------------------------------------------
// torch_bridge.cpp
//
// Zero-copy bridge between PyTorch tensors and the NewtonSolver. The forward
// and backward entry points take raw CUDA device pointers (from torch tensors)
// and run the solve / adjoint solve directly on them, so no host copies or
// extra device allocations are needed. The cupf::torch_api free functions are
// the C++ surface the Python extension binds to.
// ---------------------------------------------------------------------------

#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/core/newton_solver_cuda_bridge.hpp"
#include "newton_solver/core/pipeline.hpp"
#include "newton_solver/core/solver_contexts.hpp"

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

#ifdef CUPF_WITH_CUDA
#include "newton_solver/ops/linear_solve/cuda_linear_solve_kernels.hpp"
#endif


// ===========================================================================
// Translation-unit-local helpers
// ===========================================================================
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#ifdef CUPF_WITH_CUDA
// Resize every device buffer to match the current case/batch dimensions.
// FP64 is limited to batch_size==1; FP32/Mixed scale each buffer by batch_size.
// (All size_t casts widen positive int32 dimensions before multiplying, which
//  both matches the buffer API and guards the products against int overflow.)
void ensure_cuda_tensor_batch(CudaFp64Storage& buf, int32_t batch_size)
{
    if (batch_size != 1) {
        throw std::runtime_error(
            "CUDA FP64 torch extension path currently supports batch_size=1; use fp32 or mixed for batched runs");
    }
    const std::size_t bus_count = static_cast<std::size_t>(buf.n_bus);
    const std::size_t residual_count = static_cast<std::size_t>(buf.dimF);
    const auto ensure_size = [](auto& b, std::size_t count) {
        if (b.size() != count) b.resize(count);
    };
    ensure_size(buf.d_F, residual_count);
    ensure_size(buf.d_normF, 1);
    ensure_size(buf.d_dx, residual_count);
    ensure_size(buf.d_Va, bus_count);
    ensure_size(buf.d_Vm, bus_count);
    ensure_size(buf.d_V_re, bus_count);
    ensure_size(buf.d_V_im, bus_count);
    ensure_size(buf.d_Sbus_re, bus_count);
    ensure_size(buf.d_Sbus_im, bus_count);
    ensure_size(buf.d_Ibus_re, bus_count);
    ensure_size(buf.d_Ibus_im, bus_count);
}

void ensure_cuda_tensor_batch(CudaFp32Storage& buf, int32_t batch_size)
{
    if (batch_size <= 0) {
        throw std::invalid_argument("torch extension path requires a positive batch size");
    }
    buf.batch_size = batch_size;
    buf.ybus_values_batched = false;
    const std::size_t bus_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.n_bus);
    const std::size_t residual_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.dimF);
    const std::size_t jacobian_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.nnz_J);
    const auto ensure_size = [](auto& b, std::size_t count) {
        if (b.size() != count) b.resize(count);
    };
    ensure_size(buf.d_J_values, jacobian_count);
    ensure_size(buf.d_F, residual_count);
    ensure_size(buf.d_normF, static_cast<std::size_t>(batch_size));
    ensure_size(buf.d_dx, residual_count);
    ensure_size(buf.d_Va, bus_count);
    ensure_size(buf.d_Vm, bus_count);
    ensure_size(buf.d_V_re, bus_count);
    ensure_size(buf.d_V_im, bus_count);
    ensure_size(buf.d_Sbus_re, bus_count);
    ensure_size(buf.d_Sbus_im, bus_count);
    ensure_size(buf.d_Ibus_re, bus_count);
    ensure_size(buf.d_Ibus_im, bus_count);
}

void ensure_cuda_tensor_batch(CudaMixedStorage& buf, int32_t batch_size)
{
    if (batch_size <= 0) {
        throw std::invalid_argument("torch extension path requires a positive batch size");
    }
    buf.batch_size = batch_size;
    buf.ybus_values_batched = false;
    const std::size_t bus_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.n_bus);
    const std::size_t residual_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.dimF);
    const std::size_t jacobian_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.nnz_J);
    const auto ensure_size = [](auto& b, std::size_t count) {
        if (b.size() != count) b.resize(count);
    };
    ensure_size(buf.d_J_values, jacobian_count);
    ensure_size(buf.d_F, residual_count);
    ensure_size(buf.d_normF, static_cast<std::size_t>(batch_size));
    ensure_size(buf.d_dx, residual_count);
    ensure_size(buf.d_Va, bus_count);
    ensure_size(buf.d_Vm, bus_count);
    ensure_size(buf.d_V_re, bus_count);
    ensure_size(buf.d_V_im, bus_count);
    ensure_size(buf.d_Sbus_re, bus_count);
    ensure_size(buf.d_Sbus_im, bus_count);
    ensure_size(buf.d_Ibus_re, bus_count);
    ensure_size(buf.d_Ibus_im, bus_count);
}

// --- Compile-time pipeline classification (drives dtype/pointer selection) ---

// True for any FP64 CUDA pipeline (cuDSS or, when built, the custom solver).
template <typename PipelineT>
struct IsCudaFp64Pipeline : std::bool_constant<
    std::is_same_v<PipelineT, CudaFp64Pipeline>
#ifdef CUPF_ENABLE_CUSTOM_SOLVER
    || std::is_same_v<PipelineT, CudaFp64CustomPipeline>
#endif
> {};

template <typename PipelineT>
constexpr bool is_cuda_fp64_pipeline_v = IsCudaFp64Pipeline<PipelineT>::value;

// Expected torch dtype string for a pipeline (used to validate caller tensors).
template <typename PipelineT>
const char* cuda_pipeline_dtype_name()
{
    if constexpr (is_cuda_fp64_pipeline_v<PipelineT>) {
        return "float64";
    } else {
        return "float32";
    }
}

// Human-readable backend tag for a pipeline (diagnostics / result metadata).
template <typename PipelineT>
const char* cuda_pipeline_backend_name()
{
    if constexpr (std::is_same_v<PipelineT, CudaFp64Pipeline>) {
        return "cuda_cudss_fp64";
#ifdef CUPF_ENABLE_CUSTOM_SOLVER
    } else if constexpr (std::is_same_v<PipelineT, CudaFp64CustomPipeline>) {
        return "cuda_custom_fp64";
#endif
    } else if constexpr (std::is_same_v<PipelineT, CudaFp32Pipeline>) {
        return "cuda_cudss_fp32";
    } else {
        return "cuda_cudss_mixed";
    }
}

// Populate the result flags shared by both torch entry points (zero-copy,
// cache reuse, J^T provenance, shape/backend tags) from the pipeline state.
template <typename PipelineT>
void fill_common_cuda_adjoint_metadata(PipelineT& p, AdjointResult& result)
{
    result.success = true;
    result.used_adjoint_cache = p.adjoint_cache.has_adjoint_cache;
    result.adjoint_cache_matches_final_state = p.adjoint_cache.adjoint_cache_matches_final_state;
    result.reused_forward_factorization = p.adjoint_cache.reused_forward_factorization;
    result.reused_final_state_factorization = p.adjoint_cache.has_adjoint_cache;
    result.refactorized_for_backward = false;
    result.used_explicit_transpose = p.adjoint_cache.used_explicit_transpose;
    result.used_python_scipy = false;
    result.includes_host_device_transfer = false;
    result.zero_copy = true;
    result.torch_extension_zero_copy = true;
    result.raw_pointer_api_used = false;
    result.current_stream_integrated = true;
    result.jt_symbolic_analyzed_at_initialize = p.adjoint_cache.jt_symbolic_analyzed_at_initialize;
    result.jt_values_transposed_on_device = p.adjoint_cache.jt_values_transposed_on_device;
    result.jt_factorized_during_forward_cache = p.adjoint_cache.jt_factorized_during_forward_cache;
    result.jt_refactorized_during_backward = false;
    result.host_roundtrip_for_jt_transpose = p.adjoint_cache.host_roundtrip_for_jt_transpose;
    result.n_bus = p.buf.n_bus;
    result.batch_size = cuda_storage_batch_size(p.buf);
    result.dimF = p.buf.dimF;
    result.backend = p.adjoint_cache.backend_name.empty()
        ? cuda_pipeline_backend_name<PipelineT>()
        : p.adjoint_cache.backend_name;
    result.transpose_solve_backend = p.adjoint_cache.transpose_solve_backend_name;
}
#endif  // CUPF_WITH_CUDA

}  // namespace


// ===========================================================================
// NewtonSolver torch entry points (operate directly on torch device pointers)
// ===========================================================================

// Backward pass: given upstream gradients dL/dVa, dL/dVm on the device, reuse
// the cached J^T factorization to produce load gradients in place. Requires an
// exact cached adjoint factorization from a prior forward solve.
void NewtonSolver::solve_torch_backward(
    const void* grad_va_device_ptr,
    const void* grad_vm_device_ptr,
    void* grad_load_p_device_ptr,
    void* grad_load_q_device_ptr,
    int32_t batch_size,
    int32_t n_bus,
    const char* dtype,
    const AdjointOptions& options,
    AdjointResult& result)
{
#ifndef CUPF_WITH_CUDA
    (void)grad_va_device_ptr;
    (void)grad_vm_device_ptr;
    (void)grad_load_p_device_ptr;
    (void)grad_load_q_device_ptr;
    (void)batch_size;
    (void)n_bus;
    (void)dtype;
    (void)options;
    (void)result;
    throw std::runtime_error("cupf::torch_api::solve_backward requires a CUDA build");
#else
    if (!initialized_) {
        throw std::runtime_error("cupf::torch_api::solve_backward(): initialize() and forward solve must be called first");
    }
    if (grad_va_device_ptr == nullptr || grad_vm_device_ptr == nullptr ||
        grad_load_p_device_ptr == nullptr || grad_load_q_device_ptr == nullptr) {
        throw std::invalid_argument("cupf::torch_api::solve_backward(): null device pointer");
    }
    const std::string dtype_name = dtype ? std::string(dtype) : std::string();

    result = {};
    std::visit([&](auto& p) {
        using PipelineT = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<PipelineT, CpuFp64Pipeline>) {
            throw std::runtime_error("cupf::torch_api::solve_backward(): CPU pipeline is not supported");
        } else {
            using ValueT = std::conditional_t<
                is_cuda_fp64_pipeline_v<PipelineT>, double, float>;
            const char* expected_dtype = cuda_pipeline_dtype_name<PipelineT>();
            if (dtype_name != expected_dtype) {
                throw std::invalid_argument(
                    std::string("cupf::torch_api::solve_backward(): dtype must be ") +
                    expected_dtype + " for this cuPF compute policy");
            }
            if (batch_size != cuda_storage_batch_size(p.buf) || n_bus != p.buf.n_bus) {
                throw std::invalid_argument("cupf::torch_api::solve_backward(): shape does not match cached forward solve");
            }
            const bool cache_ok =
                p.adjoint_cache.has_adjoint_cache &&
                p.adjoint_cache.adjoint_cache_matches_final_state &&
                p.linear_solve.has_adjoint_cache();
            if (!cache_ok) {
                throw std::runtime_error(
                    "cupf::torch_api::solve_backward(): missing exact cached adjoint factorization");
            }

            const auto total_start = Clock::now();
            // The torch tensors arrive as untyped device pointers; reinterpret
            // them as the pipeline's scalar type (ValueT = float or double).
            const ValueT* grad_va = reinterpret_cast<const ValueT*>(grad_va_device_ptr);
            const ValueT* grad_vm = reinterpret_cast<const ValueT*>(grad_vm_device_ptr);
            ValueT* grad_p = reinterpret_cast<ValueT*>(grad_load_p_device_ptr);
            ValueT* grad_q = reinterpret_cast<ValueT*>(grad_load_q_device_ptr);
            const int32_t n_pv = p.buf.n_pvpq - p.buf.n_pq;

            // Gather dL/dx into the adjoint RHS -> solve J^T -> project the
            // solution back onto the P/Q load gradients.
            launch_gather_adjoint_rhs<ValueT>(
                grad_va, grad_vm, p.linear_solve.adjoint_rhs_data(),
                p.buf.d_pv.data(), n_pv,
                p.buf.d_pq.data(), p.buf.n_pq,
                p.buf.n_bus, batch_size);
            p.linear_solve.solve_adjoint_explicit_transpose_cached(result.solve_time_ms);
            result.transpose_solve_time_ms = result.solve_time_ms;
            launch_project_load_gradients<ValueT>(
                p.linear_solve.adjoint_solution_data(),
                grad_p,
                grad_q,
                p.buf.d_pv.data(), n_pv,
                p.buf.d_pq.data(), p.buf.n_pq,
                p.buf.n_bus, batch_size);

            fill_common_cuda_adjoint_metadata(p, result);
            result.used_adjoint_cache = true;
            result.adjoint_cache_matches_final_state = true;
            result.reused_final_state_factorization = true;
            result.factorization_time_ms = 0.0;
            result.total_time_ms = elapsed_ms(total_start, Clock::now());
            result.transpose_solve_backend += "_torch_extension";
        }
    }, pipeline_->v);
#endif
}


// Forward pass: assemble PF inputs from base power + load tensors, run the NR
// loop on the device, and write converged Va/Vm back into the output tensors.
// Optionally caches an adjoint factorization for a following backward pass.
void NewtonSolver::solve_torch_forward(
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
    AdjointResult& result)
{
#ifndef CUPF_WITH_CUDA
    (void)sbus_base_re_device_ptr;
    (void)sbus_base_im_device_ptr;
    (void)load_p_device_ptr;
    (void)load_q_device_ptr;
    (void)v0_va_device_ptr;
    (void)v0_vm_device_ptr;
    (void)va_out_device_ptr;
    (void)vm_out_device_ptr;
    (void)batch_size;
    (void)n_bus;
    (void)dtype;
    (void)config;
    (void)solve_options;
    (void)result;
    throw std::runtime_error("cupf::torch_api::solve_forward requires a CUDA build");
#else
    if (!initialized_) {
        throw std::runtime_error("cupf::torch_api::solve_forward(): initialize() must be called first");
    }
    if (!sbus_base_re_device_ptr || !sbus_base_im_device_ptr || !load_p_device_ptr ||
        !load_q_device_ptr || !v0_va_device_ptr || !v0_vm_device_ptr ||
        !va_out_device_ptr || !vm_out_device_ptr) {
        throw std::invalid_argument("cupf::torch_api::solve_forward(): null device pointer");
    }
    const std::string dtype_name = dtype ? std::string(dtype) : std::string();
    const auto total_start = Clock::now();

    result = {};
    std::visit([&](auto& p) {
        using PipelineT = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<PipelineT, CpuFp64Pipeline>) {
            throw std::runtime_error("cupf::torch_api::solve_forward(): CPU pipeline is not supported");
        } else {
            const char* expected_dtype = cuda_pipeline_dtype_name<PipelineT>();
            if (dtype_name != expected_dtype) {
                throw std::invalid_argument(
                    std::string("cupf::torch_api::solve_forward(): dtype must be ") +
                    expected_dtype + " for this cuPF compute policy");
            }
            if (n_bus != p.buf.n_bus) {
                throw std::invalid_argument("cupf::torch_api::solve_forward(): n_bus does not match initialized solver");
            }
            if (batch_size != 1 && !PipelineT::batch_supported) {
                throw std::runtime_error("cupf::torch_api::solve_forward(): requested batch size is not supported by this CUDA pipeline");
            }

            // Size buffers for this batch and clear stale state.
            p.adjoint_cache = AdjointCache{};
            ensure_cuda_tensor_batch(p.buf, batch_size);
            p.buf.d_F.memsetZero();
            p.buf.d_normF.memsetZero();
            p.buf.d_dx.memsetZero();
            p.buf.d_J_values.memsetZero();
            p.buf.d_Ibus_re.memsetZero();
            p.buf.d_Ibus_im.memsetZero();

            // Build Sbus / V from the input tensors. Each branch reinterprets
            // the untyped device pointers as the input precision the pipeline
            // expects (FP64, FP32, or FP64-in/FP32-compute for Mixed).
            if constexpr (is_cuda_fp64_pipeline_v<PipelineT>) {
                const double* base_re = static_cast<const double*>(sbus_base_re_device_ptr);
                const double* base_im = static_cast<const double*>(sbus_base_im_device_ptr);
                const double* load_p = static_cast<const double*>(load_p_device_ptr);
                const double* load_q = static_cast<const double*>(load_q_device_ptr);
                const double* v0_va = static_cast<const double*>(v0_va_device_ptr);
                const double* v0_vm = static_cast<const double*>(v0_vm_device_ptr);
                launch_set_pf_inputs_from_load<double, double>(
                    base_re, base_im, load_p, load_q, v0_va, v0_vm,
                    p.buf.d_Sbus_re.data(), p.buf.d_Sbus_im.data(),
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    p.buf.d_V_re.data(), p.buf.d_V_im.data(),
                    p.buf.n_bus, batch_size);
            } else if constexpr (std::is_same_v<PipelineT, CudaFp32Pipeline>) {
                const float* base_re = static_cast<const float*>(sbus_base_re_device_ptr);
                const float* base_im = static_cast<const float*>(sbus_base_im_device_ptr);
                const float* load_p = static_cast<const float*>(load_p_device_ptr);
                const float* load_q = static_cast<const float*>(load_q_device_ptr);
                const float* v0_va = static_cast<const float*>(v0_va_device_ptr);
                const float* v0_vm = static_cast<const float*>(v0_vm_device_ptr);
                launch_set_pf_inputs_from_load<float, float>(
                    base_re, base_im, load_p, load_q, v0_va, v0_vm,
                    p.buf.d_Sbus_re.data(), p.buf.d_Sbus_im.data(),
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    p.buf.d_V_re.data(), p.buf.d_V_im.data(),
                    p.buf.n_bus, batch_size);
            } else {
                const float* base_re = static_cast<const float*>(sbus_base_re_device_ptr);
                const float* base_im = static_cast<const float*>(sbus_base_im_device_ptr);
                const float* load_p = static_cast<const float*>(load_p_device_ptr);
                const float* load_q = static_cast<const float*>(load_q_device_ptr);
                const float* v0_va = static_cast<const float*>(v0_va_device_ptr);
                const float* v0_vm = static_cast<const float*>(v0_vm_device_ptr);
                launch_set_pf_inputs_from_load<float, double>(
                    base_re, base_im, load_p, load_q, v0_va, v0_vm,
                    p.buf.d_Sbus_re.data(), p.buf.d_Sbus_im.data(),
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    p.buf.d_V_re.data(), p.buf.d_V_im.data(),
                    p.buf.n_bus, batch_size);
            }

            IterationContext iter_ctx{
                .config = config,
                .pv = nullptr, .n_pv = p.buf.n_pvpq - p.buf.n_pq,
                .pq = nullptr, .n_pq = p.buf.n_pq,
            };
            const int32_t iterations = run_iteration_stages(iter_ctx);
            (void)iterations;

            if (solve_options.prepare_adjoint_cache) {
                prepare_adjoint_cache(iter_ctx, solve_options, iter_ctx.normF);
            }

            // Write converged Va/Vm into the caller's output tensors, casting
            // the output pointers to the tensor precision per pipeline.
            const int32_t total_bus = batch_size * p.buf.n_bus;
            if constexpr (is_cuda_fp64_pipeline_v<PipelineT>) {
                launch_copy_voltage_outputs<double, double>(
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    static_cast<double*>(va_out_device_ptr),
                    static_cast<double*>(vm_out_device_ptr),
                    total_bus);
            } else if constexpr (std::is_same_v<PipelineT, CudaFp32Pipeline>) {
                launch_copy_voltage_outputs<float, float>(
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    static_cast<float*>(va_out_device_ptr),
                    static_cast<float*>(vm_out_device_ptr),
                    total_bus);
            } else {
                launch_copy_voltage_outputs<double, float>(
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    static_cast<float*>(va_out_device_ptr),
                    static_cast<float*>(vm_out_device_ptr),
                    total_bus);
            }

            fill_common_cuda_adjoint_metadata(p, result);
            result.success = true;
            result.torch_extension_zero_copy = true;
            result.raw_pointer_api_used = false;
            result.current_stream_integrated = true;
            result.n_bus = p.buf.n_bus;
            result.batch_size = batch_size;
            result.dimF = p.buf.dimF;
            result.factorization_time_ms = p.adjoint_cache.factorization_time_ms;
            result.total_time_ms = elapsed_ms(total_start, Clock::now());
        }
    }, pipeline_->v);
#endif
}


// ===========================================================================
// Public C++ API (bound by the Python torch extension)
//
// Thin forwarders that adapt free-function calls to the NewtonSolver methods.
// ===========================================================================
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
    AdjointResult& result)
{
    solver.solve_torch_backward(grad_va_device_ptr,
                                grad_vm_device_ptr,
                                grad_load_p_device_ptr,
                                grad_load_q_device_ptr,
                                batch_size,
                                n_bus,
                                dtype,
                                options,
                                result);
}

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
    AdjointResult& result)
{
    solver.solve_torch_forward(sbus_base_re_device_ptr,
                               sbus_base_im_device_ptr,
                               load_p_device_ptr,
                               load_q_device_ptr,
                               v0_va_device_ptr,
                               v0_vm_device_ptr,
                               va_out_device_ptr,
                               vm_out_device_ptr,
                               batch_size,
                               n_bus,
                               dtype,
                               config,
                               solve_options,
                               result);
}

}  // namespace cupf::torch_api
