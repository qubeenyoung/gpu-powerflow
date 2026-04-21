#include "jfnk_amgx_solver.hpp"

#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

struct LinearSolveCallbacks {
    CsrSpmv* jacobian = nullptr;
    PreconditionerKind preconditioner_kind = PreconditionerKind::ScalarAmgx;
    AmgxPreconditioner* scalar_amgx = nullptr;
    BlockAmgxPreconditioner* block_amgx = nullptr;
    BusBlockJacobiPreconditioner* bus_block_jacobi = nullptr;
};

void apply_jacobian_callback(const double* input_device, double* output_device, void* user)
{
    auto* callbacks = static_cast<LinearSolveCallbacks*>(user);
    callbacks->jacobian->apply(input_device, output_device);
}

void apply_preconditioner_callback(const double* input_device, double* output_device, void* user)
{
    auto* callbacks = static_cast<LinearSolveCallbacks*>(user);
    switch (callbacks->preconditioner_kind) {
    case PreconditionerKind::ScalarAmgx:
        callbacks->scalar_amgx->apply(input_device, output_device, callbacks->jacobian->rows());
        return;
    case PreconditionerKind::Block2x2Amgx:
    case PreconditionerKind::Block2x2AmgxJacobi:
        callbacks->block_amgx->apply(input_device, output_device, callbacks->jacobian->rows());
        return;
    case PreconditionerKind::BusBlockJacobi:
        callbacks->bus_block_jacobi->apply(input_device, output_device, callbacks->jacobian->rows());
        return;
    }
    throw std::runtime_error("unknown preconditioner kind in FGMRES callback");
}

void validate_device_state(const DevicePowerFlowState& state)
{
    if (state.ybus_row_ptr == nullptr || state.ybus_col_idx == nullptr ||
        state.ybus_re == nullptr || state.ybus_im == nullptr ||
        state.sbus_re == nullptr || state.sbus_im == nullptr ||
        state.voltage_re == nullptr || state.voltage_im == nullptr) {
        throw std::runtime_error("JfnkAmgxSolver::solve received an incomplete device state");
    }
}

bool should_rebuild_preconditioner(int32_t outer,
                                   int32_t last_rebuild_outer,
                                   int32_t rebuild_interval)
{
    if (last_rebuild_outer < 0) {
        return true;
    }
    if (rebuild_interval <= 0) {
        return false;
    }
    return outer - last_rebuild_outer >= rebuild_interval;
}

}  // namespace

JfnkAmgxSolver::JfnkAmgxSolver(SolverOptions options)
    : options_(options)
{}

void JfnkAmgxSolver::analyze(const HostPowerFlowStructure& structure)
{
    if (structure.n_bus <= 0) {
        throw std::runtime_error("JfnkAmgxSolver::analyze requires a nonempty bus graph");
    }

    ordering_ = build_bus_ordering(structure.n_bus,
                                   structure.ybus_row_ptr,
                                   structure.ybus_col_idx,
                                   structure.ordering);
    index_ = build_bus_local_index(structure.n_bus,
                                   structure.pv,
                                   structure.pq,
                                   ordering_);
    pattern_ = build_bus_local_jacobian_pattern(index_,
                                                structure.ybus_row_ptr,
                                                structure.ybus_col_idx);

    jacobian_assembler_.analyze(index_,
                                pattern_,
                                structure.ybus_row_ptr,
                                structure.ybus_col_idx);
    direct_block_jacobian_.analyze(index_, structure.ybus_row_ptr, structure.ybus_col_idx);
    residual_assembler_.analyze(index_);
    voltage_update_.analyze(index_);

    d_rhs_.resize(static_cast<std::size_t>(index_.dim));
    d_dx_.resize(static_cast<std::size_t>(index_.dim));
    analyzed_ = true;
}

SolveStats JfnkAmgxSolver::solve(const DevicePowerFlowState& state)
{
    if (!analyzed_) {
        throw std::runtime_error("JfnkAmgxSolver::solve called before analyze");
    }
    validate_device_state(state);

    SolveStats total;
    const int32_t dim = residual_assembler_.dim();
    int32_t last_preconditioner_rebuild_outer = -1;

    for (int32_t outer = 0; outer < options_.max_outer_iterations; ++outer) {
        total.outer_iterations = outer + 1;

        residual_assembler_.assemble(state.ybus_row_ptr,
                                     state.ybus_col_idx,
                                     state.ybus_re,
                                     state.ybus_im,
                                     state.voltage_re,
                                     state.voltage_im,
                                     state.sbus_re,
                                     state.sbus_im);
        total.final_mismatch = vector_ops_.norm_inf(dim, residual_assembler_.values_device());
        if (total.final_mismatch <= options_.nonlinear_tolerance) {
            total.converged = true;
            return total;
        }

        jacobian_assembler_.assemble(state.ybus_re,
                                     state.ybus_im,
                                     state.voltage_re,
                                     state.voltage_im);
        const CsrMatrixView jacobian = jacobian_assembler_.device_matrix_view();
        jacobian_spmv_.bind(jacobian);

        const bool rebuild_preconditioner =
            should_rebuild_preconditioner(outer,
                                          last_preconditioner_rebuild_outer,
                                          options_.preconditioner_rebuild_interval);
        if (rebuild_preconditioner) {
            if (options_.preconditioner == PreconditionerKind::Block2x2Amgx ||
                options_.preconditioner == PreconditionerKind::Block2x2AmgxJacobi ||
                options_.preconditioner == PreconditionerKind::BusBlockJacobi) {
                direct_block_jacobian_.assemble(state.ybus_re,
                                                state.ybus_im,
                                                state.voltage_re,
                                                state.voltage_im);
            }
            if (options_.preconditioner == PreconditionerKind::Block2x2Amgx) {
                block_amgx_.setup(direct_block_jacobian_.device_matrix_view(),
                                  AmgxBlockSmoother::MulticolorDilu);
            } else if (options_.preconditioner == PreconditionerKind::Block2x2AmgxJacobi) {
                block_amgx_.setup(direct_block_jacobian_.device_matrix_view(),
                                  AmgxBlockSmoother::BlockJacobi);
            } else if (options_.preconditioner == PreconditionerKind::BusBlockJacobi) {
                bus_block_jacobi_.setup(direct_block_jacobian_.jacobi_view());
            } else {
                amgx_.setup(jacobian);
            }
            last_preconditioner_rebuild_outer = outer;
            ++total.preconditioner_rebuilds;
        }
        const int32_t preconditioner_age = outer - last_preconditioner_rebuild_outer;

        vector_ops_.negate(dim, residual_assembler_.values_device(), d_rhs_.data());

        LinearSolveCallbacks callbacks{
            .jacobian = &jacobian_spmv_,
            .preconditioner_kind = options_.preconditioner,
            .scalar_amgx = &amgx_,
            .block_amgx = &block_amgx_,
            .bus_block_jacobi = &bus_block_jacobi_,
        };
        SolveStats linear = fgmres_.solve(dim,
                                          d_rhs_.data(),
                                          d_dx_.data(),
                                          options_,
                                          apply_jacobian_callback,
                                          apply_preconditioner_callback,
                                          &callbacks);

        total.total_inner_iterations += linear.total_inner_iterations;
        total.total_jv_calls += linear.total_jv_calls;
        total.linear_failures += linear.linear_failures;

        if (!linear.converged && !options_.continue_on_linear_failure) {
            if (options_.track_dx_residual) {
                total.step_trace.push_back(NonlinearStepTrace{
                    .outer_iteration = outer + 1,
                    .linear_converged = linear.converged,
                    .inner_iterations = linear.total_inner_iterations,
                    .jv_calls = linear.total_jv_calls,
                    .linear_failures = linear.linear_failures,
                    .preconditioner_age = preconditioner_age,
                    .linear_residual = linear.final_mismatch,
                    .before_mismatch = total.final_mismatch,
                    .after_mismatch = total.final_mismatch,
                    .after_before_ratio = 1.0,
                    .preconditioner_rebuilt = rebuild_preconditioner,
                    .dx_was_applied = false,
                });
            }
            total.failure_reason = "linear_" + linear.failure_reason;
            return total;
        }

        const double before_update_mismatch = total.final_mismatch;
        voltage_update_.apply(d_dx_.data(), state.voltage_re, state.voltage_im);
        if (options_.track_dx_residual) {
            residual_assembler_.assemble(state.ybus_row_ptr,
                                         state.ybus_col_idx,
                                         state.ybus_re,
                                         state.ybus_im,
                                         state.voltage_re,
                                         state.voltage_im,
                                         state.sbus_re,
                                         state.sbus_im);
            const double after_update_mismatch =
                vector_ops_.norm_inf(dim, residual_assembler_.values_device());
            total.final_mismatch = after_update_mismatch;
            total.step_trace.push_back(NonlinearStepTrace{
                .outer_iteration = outer + 1,
                .linear_converged = linear.converged,
                .inner_iterations = linear.total_inner_iterations,
                .jv_calls = linear.total_jv_calls,
                .linear_failures = linear.linear_failures,
                .preconditioner_age = preconditioner_age,
                .linear_residual = linear.final_mismatch,
                .before_mismatch = before_update_mismatch,
                .after_mismatch = after_update_mismatch,
                .after_before_ratio = after_update_mismatch / before_update_mismatch,
                .preconditioner_rebuilt = rebuild_preconditioner,
                .dx_was_applied = true,
            });
            if (after_update_mismatch <= options_.nonlinear_tolerance) {
                total.converged = true;
                return total;
            }
        }
    }

    if (!total.converged && total.failure_reason.empty()) {
        total.failure_reason = "max_outer_iterations";
    }
    return total;
}

}  // namespace exp_20260414::amgx_v2
