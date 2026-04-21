#include "solver/powerflow_block_ilu_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace exp_20260415::block_ilu {
namespace {

constexpr int32_t kBlockSize = 256;

double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start)
        .count();
}

__global__ void mismatch_kernel(int32_t dim,
                                int32_t n_pv,
                                int32_t n_pq,
                                const double* __restrict__ y_re,
                                const double* __restrict__ y_im,
                                const int32_t* __restrict__ y_row_ptr,
                                const int32_t* __restrict__ y_col,
                                const double* __restrict__ v_re,
                                const double* __restrict__ v_im,
                                const double* __restrict__ sbus_re,
                                const double* __restrict__ sbus_im,
                                const int32_t* __restrict__ pv,
                                const int32_t* __restrict__ pq,
                                double* __restrict__ mismatch)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) {
        return;
    }

    int32_t bus = 0;
    bool take_q = false;
    if (tid < n_pv) {
        bus = pv[tid];
    } else if (tid < n_pv + n_pq) {
        bus = pq[tid - n_pv];
    } else {
        bus = pq[tid - n_pv - n_pq];
        take_q = true;
    }

    double i_re = 0.0;
    double i_im = 0.0;
    for (int32_t pos = y_row_ptr[bus]; pos < y_row_ptr[bus + 1]; ++pos) {
        const int32_t col = y_col[pos];
        const double yr = y_re[pos];
        const double yi = y_im[pos];
        const double vr = v_re[col];
        const double vi = v_im[col];
        i_re += yr * vr - yi * vi;
        i_im += yr * vi + yi * vr;
    }

    const double vr = v_re[bus];
    const double vi = v_im[bus];
    const double mis_p = vr * i_re + vi * i_im - sbus_re[bus];
    const double mis_q = vi * i_re - vr * i_im - sbus_im[bus];
    mismatch[tid] = take_q ? mis_q : mis_p;
}

__global__ void negate_kernel(int32_t n,
                              const double* __restrict__ input,
                              double* __restrict__ output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = -input[i];
    }
}

__global__ void reduce_absmax_kernel(int32_t n,
                                     const double* __restrict__ input,
                                     double* __restrict__ partial)
{
    __shared__ double values[kBlockSize];
    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;
    values[tid] = (i < n) ? fabs(input[i]) : 0.0;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] = fmax(values[tid], values[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = values[0];
    }
}

__global__ void decompose_voltage_kernel(int32_t n_bus,
                                         const double* __restrict__ v_re,
                                         const double* __restrict__ v_im,
                                         double* __restrict__ va,
                                         double* __restrict__ vm)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }
    va[bus] = atan2(v_im[bus], v_re[bus]);
    vm[bus] = hypot(v_re[bus], v_im[bus]);
}

__global__ void update_voltage_kernel(int32_t dim,
                                      int32_t n_pv,
                                      int32_t n_pq,
                                      double alpha,
                                      const double* __restrict__ dx,
                                      const int32_t* __restrict__ pv,
                                      const int32_t* __restrict__ pq,
                                      double* __restrict__ va,
                                      double* __restrict__ vm)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) {
        return;
    }

    if (tid < n_pv) {
        va[pv[tid]] += alpha * dx[tid];
    } else if (tid < n_pv + n_pq) {
        va[pq[tid - n_pv]] += alpha * dx[tid];
    } else {
        vm[pq[tid - n_pv - n_pq]] += alpha * dx[tid];
    }
}

__global__ void reconstruct_voltage_kernel(int32_t n_bus,
                                           const double* __restrict__ va,
                                           const double* __restrict__ vm,
                                           double* __restrict__ v_re,
                                           double* __restrict__ v_im)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }
    v_re[bus] = vm[bus] * cos(va[bus]);
    v_im[bus] = vm[bus] * sin(va[bus]);
}

void ensure_size(DeviceBuffer<double>& buffer, std::size_t count)
{
    if (buffer.size() != count) {
        buffer.resize(count);
    }
}

double norm_inf(DeviceBuffer<double>& partial,
                DeviceBuffer<double>& scratch,
                int32_t n,
                const double* values)
{
    int32_t current = (n + kBlockSize - 1) / kBlockSize;
    ensure_size(partial, static_cast<std::size_t>(current));
    reduce_absmax_kernel<<<current, kBlockSize>>>(n, values, partial.data());
    CUDA_CHECK(cudaGetLastError());

    bool in_partial = true;
    while (current > 1) {
        const int32_t next = (current + kBlockSize - 1) / kBlockSize;
        if (in_partial) {
            ensure_size(scratch, static_cast<std::size_t>(next));
            reduce_absmax_kernel<<<next, kBlockSize>>>(current, partial.data(), scratch.data());
        } else {
            ensure_size(partial, static_cast<std::size_t>(next));
            reduce_absmax_kernel<<<next, kBlockSize>>>(current, scratch.data(), partial.data());
        }
        CUDA_CHECK(cudaGetLastError());
        current = next;
        in_partial = !in_partial;
    }

    double result = 0.0;
    if (in_partial) {
        partial.copyTo(&result, 1);
    } else {
        scratch.copyTo(&result, 1);
    }
    return result;
}

}  // namespace

InnerPrecision parse_inner_precision(const std::string& name)
{
    std::string lowered = name;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered == "fp64" || lowered == "f64" || lowered == "double") {
        return InnerPrecision::Fp64;
    }
    if (lowered == "fp32" || lowered == "f32" || lowered == "float") {
        return InnerPrecision::Fp32;
    }
    throw std::runtime_error("unknown inner precision: " + name);
}

const char* inner_precision_name(InnerPrecision precision)
{
    switch (precision) {
    case InnerPrecision::Fp64:
        return "fp64";
    case InnerPrecision::Fp32:
        return "fp32";
    default:
        return "unknown";
    }
}

SchurLinearSolverKind parse_schur_linear_solver_kind(const std::string& name)
{
    std::string lowered = name;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered == "bicgstab" || lowered == "implicit_schur_bicgstab") {
        return SchurLinearSolverKind::Bicgstab;
    }
    if (lowered == "gmres" || lowered == "implicit_schur_gmres") {
        return SchurLinearSolverKind::Gmres;
    }
    throw std::runtime_error("unknown Schur linear solver: " + name);
}

const char* schur_linear_solver_kind_name(SchurLinearSolverKind kind)
{
    switch (kind) {
    case SchurLinearSolverKind::Bicgstab:
        return "implicit_schur_bicgstab";
    case SchurLinearSolverKind::Gmres:
        return "implicit_schur_gmres";
    default:
        return "unknown";
    }
}

PowerFlowBlockIluSolver::PowerFlowBlockIluSolver(BlockIluSolverOptions options)
    : options_(options)
{
    if (options_.nonlinear_tolerance <= 0.0 ||
        options_.linear_tolerance <= 0.0 ||
        options_.max_outer_iterations <= 0 ||
        options_.max_inner_iterations <= 0 ||
        options_.gmres_restart <= 0 ||
        options_.gmres_residual_check_interval <= 0 ||
        options_.j11_dense_block_size <= 0 ||
        options_.line_search_max_trials <= 0 ||
        options_.line_search_reduction <= 0.0 ||
        options_.line_search_reduction >= 1.0) {
        throw std::runtime_error("PowerFlowBlockIluSolver received invalid options");
    }
    if (options_.inner_precision == InnerPrecision::Fp32 &&
        options_.j11_solver_kind != J11SolverKind::PartitionDenseLu &&
        options_.j11_solver_kind != J11SolverKind::ExactKlu) {
        throw std::runtime_error("FP32 inner mode currently requires partition-dense-lu or exact-klu J11");
    }
    if (options_.j11_solver_kind == J11SolverKind::ExactKlu &&
        options_.inner_precision != InnerPrecision::Fp32) {
        throw std::runtime_error("exact-klu J11 oracle currently requires FP32 inner mode");
    }
    if (options_.schur_preconditioner_kind != SchurPreconditionerKind::None &&
        options_.inner_precision != InnerPrecision::Fp32) {
        throw std::runtime_error("Schur preconditioner currently requires FP32 inner mode");
    }
    if (options_.linear_solver_kind == SchurLinearSolverKind::Gmres &&
        options_.inner_precision != InnerPrecision::Fp32) {
        throw std::runtime_error("implicit Schur GMRES currently requires FP32 inner mode");
    }
    if (options_.j11_dense_backend == J11DenseBackend::TcNoPivot &&
        (options_.inner_precision != InnerPrecision::Fp32 ||
         options_.j11_solver_kind != J11SolverKind::PartitionDenseLu)) {
        throw std::runtime_error("TC dense backend requires FP32 inner and partition-dense-lu J11");
    }
    if (options_.j11_dense_backend == J11DenseBackend::CusolverGetrf &&
        (options_.inner_precision != InnerPrecision::Fp32 ||
         options_.j11_solver_kind != J11SolverKind::PartitionDenseLu)) {
        throw std::runtime_error("cuSolver dense backend currently requires FP32 inner and partition-dense-lu J11");
    }
}

void PowerFlowBlockIluSolver::analyze(const HostPowerFlowStructure& structure)
{
    if (structure.n_bus <= 0) {
        throw std::runtime_error("PowerFlowBlockIluSolver::analyze requires a nonempty case");
    }

    patterns_ = build_reduced_jacobian_patterns(structure.n_bus,
                                                structure.pv,
                                                structure.pq,
                                                structure.ybus_row_ptr,
                                                structure.ybus_col_idx);
    jacobian_.analyze(patterns_);

    d_pv_.assign(patterns_.index.pv.data(), patterns_.index.pv.size());
    d_pq_.assign(patterns_.index.pq.data(), patterns_.index.pq.size());
    d_mismatch_.resize(static_cast<std::size_t>(patterns_.index.dim));
    d_rhs_.resize(static_cast<std::size_t>(patterns_.index.dim));
    d_dx_.resize(static_cast<std::size_t>(patterns_.index.dim));
    d_base_voltage_re_.resize(static_cast<std::size_t>(patterns_.index.n_bus));
    d_base_voltage_im_.resize(static_cast<std::size_t>(patterns_.index.n_bus));
    d_va_.resize(static_cast<std::size_t>(patterns_.index.n_bus));
    d_vm_.resize(static_cast<std::size_t>(patterns_.index.n_bus));

    analyzed_ = true;
    preconditioner_analyzed_ = false;
}

void PowerFlowBlockIluSolver::ensure_analyzed() const
{
    if (!analyzed_) {
        throw std::runtime_error("PowerFlowBlockIluSolver used before analyze");
    }
}

void PowerFlowBlockIluSolver::download_last_dx(std::vector<double>& dx) const
{
    ensure_analyzed();
    dx.resize(static_cast<std::size_t>(patterns_.index.dim));
    d_dx_.copyTo(dx.data(), dx.size());
}

double PowerFlowBlockIluSolver::assemble_mismatch_and_norm(const DevicePowerFlowState& state)
{
    ensure_analyzed();
    const int32_t dim = patterns_.index.dim;
    const int32_t grid = (dim + kBlockSize - 1) / kBlockSize;
    mismatch_kernel<<<grid, kBlockSize>>>(dim,
                                          patterns_.index.n_pv,
                                          patterns_.index.n_pq,
                                          state.ybus_re,
                                          state.ybus_im,
                                          state.ybus_row_ptr,
                                          state.ybus_col_idx,
                                          state.voltage_re,
                                          state.voltage_im,
                                          state.sbus_re,
                                          state.sbus_im,
                                          d_pv_.data(),
                                          d_pq_.data(),
                                          d_mismatch_.data());
    CUDA_CHECK(cudaGetLastError());
    return norm_inf(d_partial_, d_scratch_, dim, d_mismatch_.data());
}

void PowerFlowBlockIluSolver::apply_voltage_update(const DevicePowerFlowState& state,
                                                   double alpha)
{
    apply_voltage_update_from_base(state, state.voltage_re, state.voltage_im, alpha);
}

void PowerFlowBlockIluSolver::apply_voltage_update_from_base(
    const DevicePowerFlowState& state,
    const double* base_voltage_re,
    const double* base_voltage_im,
    double alpha)
{
    ensure_analyzed();
    const int32_t n_bus = patterns_.index.n_bus;
    const int32_t dim = patterns_.index.dim;
    const int32_t grid_bus = (n_bus + kBlockSize - 1) / kBlockSize;
    const int32_t grid_dim = (dim + kBlockSize - 1) / kBlockSize;

    decompose_voltage_kernel<<<grid_bus, kBlockSize>>>(
        n_bus, base_voltage_re, base_voltage_im, d_va_.data(), d_vm_.data());
    CUDA_CHECK(cudaGetLastError());
    update_voltage_kernel<<<grid_dim, kBlockSize>>>(dim,
                                                    patterns_.index.n_pv,
                                                    patterns_.index.n_pq,
                                                    alpha,
                                                    d_dx_.data(),
                                                    d_pv_.data(),
                                                    d_pq_.data(),
                                                    d_va_.data(),
                                                    d_vm_.data());
    CUDA_CHECK(cudaGetLastError());
    reconstruct_voltage_kernel<<<grid_bus, kBlockSize>>>(
        n_bus, d_va_.data(), d_vm_.data(), state.voltage_re, state.voltage_im);
    CUDA_CHECK(cudaGetLastError());
}

BlockIluSolveStats PowerFlowBlockIluSolver::solve(const DevicePowerFlowState& state)
{
    ensure_analyzed();
    if (state.ybus_row_ptr == nullptr || state.ybus_col_idx == nullptr ||
        state.ybus_re == nullptr || state.ybus_im == nullptr ||
        state.sbus_re == nullptr || state.sbus_im == nullptr ||
        state.voltage_re == nullptr || state.voltage_im == nullptr) {
        throw std::runtime_error("PowerFlowBlockIluSolver::solve received incomplete state");
    }

    BlockIluSolveStats total;
    const int32_t dim = patterns_.index.dim;

    for (int32_t outer = 0; outer < options_.max_outer_iterations; ++outer) {
        const auto outer_start = std::chrono::steady_clock::now();
        BlockIluStepTrace step;
        step.outer_iteration = outer + 1;
        total.outer_iterations = outer + 1;
        const auto append_dx_trace = [&](const BlockIluStepTrace& recorded_step) {
            if (!options_.collect_dx_trace) {
                return;
            }
            BlockIluDxTrace trace;
            trace.outer_iteration = recorded_step.outer_iteration;
            trace.dx_was_applied = recorded_step.dx_was_applied;
            trace.alpha = recorded_step.line_search_alpha;
            trace.dx.resize(static_cast<std::size_t>(dim));
            d_dx_.copyTo(trace.dx.data(), trace.dx.size());
            total.dx_trace.push_back(std::move(trace));
        };

        const auto mismatch_start = std::chrono::steady_clock::now();
        total.final_mismatch = assemble_mismatch_and_norm(state);
        step.mismatch_sec = seconds_since(mismatch_start);
        step.before_mismatch = total.final_mismatch;
        if (total.final_mismatch <= options_.nonlinear_tolerance) {
            total.converged = true;
            step.after_mismatch = total.final_mismatch;
            step.outer_iteration_sec = seconds_since(outer_start);
            total.step_trace.push_back(step);
            return total;
        }

        const auto assembly_start = std::chrono::steady_clock::now();
        jacobian_.assemble(state.ybus_re, state.ybus_im, state.voltage_re, state.voltage_im);
        CUDA_CHECK(cudaDeviceSynchronize());
        step.jacobian_assembly_sec = seconds_since(assembly_start);
        if (!preconditioner_analyzed_) {
            if (options_.inner_precision == InnerPrecision::Fp32) {
                schur_operator_f32_.analyze(jacobian_.j11_view(),
                                            patterns_.j11,
                                            patterns_.j22,
                                            jacobian_.j12_view(),
                                            jacobian_.j21_view(),
                                            jacobian_.j22_view(),
                                            patterns_.index.n_pvpq,
                                            patterns_.index.n_pq,
                                            options_.j11_solver_kind,
                                            options_.j11_reorder_mode,
                                            options_.j11_dense_block_size,
                                            options_.j11_dense_backend,
                                            options_.j11_partition_mode,
                                            options_.schur_preconditioner_kind);
            } else {
                schur_operator_.analyze(jacobian_.j11_view(),
                                        patterns_.j11,
                                        jacobian_.j12_view(),
                                        jacobian_.j21_view(),
                                        jacobian_.j22_view(),
                                        patterns_.index.n_pvpq,
                                        patterns_.index.n_pq,
                                        options_.j11_reorder_mode,
                                        options_.j11_solver_kind,
                                        options_.j11_dense_block_size,
                                        options_.j11_partition_mode);
            }
            preconditioner_analyzed_ = true;
        }
        const auto factorize_start = std::chrono::steady_clock::now();
        try {
            if (options_.inner_precision == InnerPrecision::Fp32) {
                schur_operator_f32_.factorize_j11();
            } else {
                schur_operator_.factorize_j11();
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            step.preconditioner_factorized = true;
        } catch (const std::exception&) {
            step.preconditioner_factorize_sec = seconds_since(factorize_start);
            step.after_mismatch = total.final_mismatch;
            step.outer_iteration_sec = seconds_since(outer_start);
            total.j11_zero_pivot =
                (options_.inner_precision == InnerPrecision::Fp32)
                    ? schur_operator_f32_.j11_zero_pivot()
                    : schur_operator_.j11_zero_pivot();
            total.j22_zero_pivot =
                (options_.inner_precision == InnerPrecision::Fp32)
                    ? schur_operator_f32_.j22_zero_pivot()
                    : -1;
            total.step_trace.push_back(step);
            total.failure_reason = "preconditioner_factorization_failed";
            return total;
        }
        step.preconditioner_factorize_sec = seconds_since(factorize_start);
        ++total.preconditioner_rebuilds;
        total.j11_zero_pivot =
            (options_.inner_precision == InnerPrecision::Fp32)
                ? schur_operator_f32_.j11_zero_pivot()
                : schur_operator_.j11_zero_pivot();
        total.j22_zero_pivot =
            (options_.inner_precision == InnerPrecision::Fp32)
                ? schur_operator_f32_.j22_zero_pivot()
                : -1;

        const int32_t grid = (dim + kBlockSize - 1) / kBlockSize;
        negate_kernel<<<grid, kBlockSize>>>(dim, d_mismatch_.data(), d_rhs_.data());
        CUDA_CHECK(cudaGetLastError());

        SchurBicgstabOptions linear_options;
        linear_options.relative_tolerance = options_.linear_tolerance;
        linear_options.max_iterations = options_.max_inner_iterations;
        linear_options.gmres_restart = options_.gmres_restart;
        linear_options.gmres_residual_check_interval =
            options_.gmres_residual_check_interval;
        linear_options.collect_timing_breakdown = options_.collect_timing_breakdown;
        SchurBicgstabStats linear;
        if (options_.linear_solver_kind == SchurLinearSolverKind::Gmres) {
            linear = gmres_solver_f32_.solve(
                schur_operator_f32_, d_rhs_.data(), d_dx_.data(), linear_options);
        } else if (options_.inner_precision == InnerPrecision::Fp32) {
            linear = linear_solver_f32_.solve(
                schur_operator_f32_, d_rhs_.data(), d_dx_.data(), linear_options);
        } else {
            linear = linear_solver_.solve(
                schur_operator_, d_rhs_.data(), d_dx_.data(), linear_options);
        }

        total.total_inner_iterations += linear.iterations;
        total.total_spmv_calls += linear.spmv_calls;
        total.total_preconditioner_applies += linear.j11_solve_calls;
        total.total_schur_preconditioner_applies += linear.schur_preconditioner_calls;
        total.total_reduction_calls += linear.reduction_calls;
        total.total_restart_cycles += linear.restart_cycles;
        total.total_schur_matvec_calls += linear.schur_matvec_calls;
        total.total_j11_solve_calls += linear.j11_solve_calls;
        total.total_linear_solve_sec += linear.solve_sec;
        total.total_linear_iteration_sec +=
            linear.avg_iteration_sec * static_cast<double>(linear.iterations);
        total.total_linear_timing.schur_rhs_sec += linear.timing.schur_rhs_sec;
        total.total_linear_timing.schur_matvec_sec += linear.timing.schur_matvec_sec;
        total.total_linear_timing.schur_recover_sec += linear.timing.schur_recover_sec;
        total.total_linear_timing.schur_spmv_sec += linear.timing.schur_spmv_sec;
        total.total_linear_timing.schur_j11_solve_sec += linear.timing.schur_j11_solve_sec;
        total.total_linear_timing.schur_preconditioner_sec +=
            linear.timing.schur_preconditioner_sec;
        total.total_linear_timing.reduction_sec += linear.timing.reduction_sec;
        total.total_linear_timing.vector_update_sec += linear.timing.vector_update_sec;
        total.total_linear_timing.small_solve_sec += linear.timing.small_solve_sec;
        total.total_linear_timing.residual_refresh_sec += linear.timing.residual_refresh_sec;

        if (!linear.converged && !options_.continue_on_linear_failure) {
            step.linear_converged = false;
            step.dx_was_applied = false;
            step.after_mismatch = total.final_mismatch;
            step.linear_relative_residual = linear.relative_residual_norm;
            step.linear_solve_sec = linear.solve_sec;
            step.linear_avg_iteration_sec = linear.avg_iteration_sec;
            step.linear_preconditioner_sec = linear.timing.schur_j11_solve_sec;
            step.linear_schur_preconditioner_sec =
                linear.timing.schur_preconditioner_sec;
            step.linear_spmv_sec = linear.timing.schur_spmv_sec;
            step.linear_reduction_sec = linear.timing.reduction_sec;
            step.linear_vector_update_sec = linear.timing.vector_update_sec;
            step.linear_small_solve_sec = linear.timing.small_solve_sec;
            step.linear_residual_refresh_sec = linear.timing.residual_refresh_sec;
            step.schur_rhs_sec = linear.timing.schur_rhs_sec;
            step.schur_matvec_sec = linear.timing.schur_matvec_sec;
            step.schur_recover_sec = linear.timing.schur_recover_sec;
            step.inner_iterations = linear.iterations;
            step.restart_cycles = linear.restart_cycles;
            step.spmv_calls = linear.spmv_calls;
            step.preconditioner_applies = linear.j11_solve_calls;
            step.schur_preconditioner_applies = linear.schur_preconditioner_calls;
            step.schur_matvec_calls = linear.schur_matvec_calls;
            step.j11_solve_calls = linear.j11_solve_calls;
            step.reduction_calls = linear.reduction_calls;
            step.outer_iteration_sec = seconds_since(outer_start);
            append_dx_trace(step);
            total.step_trace.push_back(step);
            total.failure_reason = "linear_" + linear.failure_reason;
            return total;
        }

        const double before = total.final_mismatch;
        double after = before;
        double accepted_alpha = 1.0;
        int32_t line_search_trials = 1;
        bool line_search_accepted = true;
        const auto update_start = std::chrono::steady_clock::now();
        if (options_.enable_line_search) {
            const std::size_t voltage_bytes =
                static_cast<std::size_t>(patterns_.index.n_bus) * sizeof(double);
            CUDA_CHECK(cudaMemcpy(d_base_voltage_re_.data(),
                                  state.voltage_re,
                                  voltage_bytes,
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_base_voltage_im_.data(),
                                  state.voltage_im,
                                  voltage_bytes,
                                  cudaMemcpyDeviceToDevice));
            line_search_accepted = false;
            accepted_alpha = 1.0;
            double last_trial_alpha = accepted_alpha;
            for (line_search_trials = 1;
                 line_search_trials <= options_.line_search_max_trials;
                 ++line_search_trials) {
                last_trial_alpha = accepted_alpha;
                apply_voltage_update_from_base(state,
                                               d_base_voltage_re_.data(),
                                               d_base_voltage_im_.data(),
                                               accepted_alpha);
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto after_mismatch_start = std::chrono::steady_clock::now();
                after = assemble_mismatch_and_norm(state);
                step.after_mismatch_sec += seconds_since(after_mismatch_start);
                if (after < before || after <= options_.nonlinear_tolerance) {
                    line_search_accepted = true;
                    break;
                }
                accepted_alpha *= options_.line_search_reduction;
            }
            if (!line_search_accepted) {
                line_search_trials = options_.line_search_max_trials;
                accepted_alpha = last_trial_alpha;
                CUDA_CHECK(cudaMemcpy(state.voltage_re,
                                      d_base_voltage_re_.data(),
                                      voltage_bytes,
                                      cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(state.voltage_im,
                                      d_base_voltage_im_.data(),
                                      voltage_bytes,
                                      cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaDeviceSynchronize());
                after = before;
            }
        } else {
            apply_voltage_update(state, 1.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto after_mismatch_start = std::chrono::steady_clock::now();
            after = assemble_mismatch_and_norm(state);
            step.after_mismatch_sec = seconds_since(after_mismatch_start);
        }
        step.voltage_update_sec = seconds_since(update_start);
        step.line_search_sec = options_.enable_line_search ? step.voltage_update_sec : 0.0;
        step.line_search_alpha = accepted_alpha;
        step.line_search_trials = line_search_trials;
        step.line_search_accepted = line_search_accepted;
        total.final_mismatch = after;

        step.linear_converged = linear.converged;
        step.dx_was_applied = line_search_accepted;
        step.before_mismatch = before;
        step.after_mismatch = after;
        step.linear_relative_residual = linear.relative_residual_norm;
        step.linear_solve_sec = linear.solve_sec;
        step.linear_avg_iteration_sec = linear.avg_iteration_sec;
        step.linear_preconditioner_sec = linear.timing.schur_j11_solve_sec;
        step.linear_schur_preconditioner_sec = linear.timing.schur_preconditioner_sec;
        step.linear_spmv_sec = linear.timing.schur_spmv_sec;
        step.linear_reduction_sec = linear.timing.reduction_sec;
        step.linear_vector_update_sec = linear.timing.vector_update_sec;
        step.linear_small_solve_sec = linear.timing.small_solve_sec;
        step.linear_residual_refresh_sec = linear.timing.residual_refresh_sec;
        step.schur_rhs_sec = linear.timing.schur_rhs_sec;
        step.schur_matvec_sec = linear.timing.schur_matvec_sec;
        step.schur_recover_sec = linear.timing.schur_recover_sec;
        step.inner_iterations = linear.iterations;
        step.restart_cycles = linear.restart_cycles;
        step.spmv_calls = linear.spmv_calls;
        step.preconditioner_applies = linear.j11_solve_calls;
        step.schur_preconditioner_applies = linear.schur_preconditioner_calls;
        step.schur_matvec_calls = linear.schur_matvec_calls;
        step.j11_solve_calls = linear.j11_solve_calls;
        step.reduction_calls = linear.reduction_calls;
        step.outer_iteration_sec = seconds_since(outer_start);
        append_dx_trace(step);
        total.step_trace.push_back(step);

        if (!line_search_accepted) {
            total.failure_reason = "line_search_failed";
            return total;
        }

        if (after <= options_.nonlinear_tolerance) {
            total.converged = true;
            return total;
        }
    }

    if (!total.converged && total.failure_reason.empty()) {
        total.failure_reason = "max_outer_iterations";
    }
    return total;
}

}  // namespace exp_20260415::block_ilu
