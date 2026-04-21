#pragma once

#include "assembly/reduced_jacobian_assembler.hpp"
#include "linear/implicit_schur_operator.hpp"
#include "linear/implicit_schur_operator_f32.hpp"
#include "linear/schur_bicgstab_solver.hpp"
#include "linear/schur_bicgstab_solver_f32.hpp"
#include "linear/schur_gmres_solver_f32.hpp"
#include "model/reduced_jacobian.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace exp_20260415::block_ilu {

enum class InnerPrecision {
    Fp64,
    Fp32,
};

InnerPrecision parse_inner_precision(const std::string& name);
const char* inner_precision_name(InnerPrecision precision);

enum class SchurLinearSolverKind {
    Bicgstab,
    Gmres,
};

SchurLinearSolverKind parse_schur_linear_solver_kind(const std::string& name);
const char* schur_linear_solver_kind_name(SchurLinearSolverKind kind);

struct HostPowerFlowStructure {
    int32_t n_bus = 0;
    std::vector<int32_t> ybus_row_ptr;
    std::vector<int32_t> ybus_col_idx;
    std::vector<int32_t> pv;
    std::vector<int32_t> pq;
};

struct DevicePowerFlowState {
    const int32_t* ybus_row_ptr = nullptr;
    const int32_t* ybus_col_idx = nullptr;
    const double* ybus_re = nullptr;
    const double* ybus_im = nullptr;
    const double* sbus_re = nullptr;
    const double* sbus_im = nullptr;
    double* voltage_re = nullptr;
    double* voltage_im = nullptr;
};

struct BlockIluSolverOptions {
    double nonlinear_tolerance = 1e-8;
    double linear_tolerance = 1e-6;
    int32_t max_outer_iterations = 10;
    int32_t max_inner_iterations = 500;
    SchurLinearSolverKind linear_solver_kind = SchurLinearSolverKind::Bicgstab;
    int32_t gmres_restart = 30;
    int32_t gmres_residual_check_interval = 5;
    J11ReorderMode j11_reorder_mode = J11ReorderMode::None;
    J11SolverKind j11_solver_kind = J11SolverKind::Ilu0;
    int32_t j11_dense_block_size = 32;
    J11DenseBackend j11_dense_backend = J11DenseBackend::CublasGetrf;
    J11PartitionMode j11_partition_mode = J11PartitionMode::Bfs;
    InnerPrecision inner_precision = InnerPrecision::Fp64;
    bool collect_timing_breakdown = true;
    bool continue_on_linear_failure = false;
    bool collect_dx_trace = false;
    bool enable_line_search = false;
    SchurPreconditionerKind schur_preconditioner_kind =
        SchurPreconditionerKind::None;
    int32_t line_search_max_trials = 8;
    double line_search_reduction = 0.5;
};

struct BlockIluStepTrace {
    int32_t outer_iteration = 0;
    bool linear_converged = false;
    bool dx_was_applied = false;
    bool preconditioner_factorized = false;
    double before_mismatch = 0.0;
    double after_mismatch = 0.0;
    double outer_iteration_sec = 0.0;
    double mismatch_sec = 0.0;
    double jacobian_assembly_sec = 0.0;
    double preconditioner_factorize_sec = 0.0;
    double linear_relative_residual = 0.0;
    double linear_solve_sec = 0.0;
    double linear_avg_iteration_sec = 0.0;
    double linear_preconditioner_sec = 0.0;
    double linear_schur_preconditioner_sec = 0.0;
    double linear_spmv_sec = 0.0;
    double linear_reduction_sec = 0.0;
    double linear_vector_update_sec = 0.0;
    double linear_small_solve_sec = 0.0;
    double linear_residual_refresh_sec = 0.0;
    double schur_rhs_sec = 0.0;
    double schur_matvec_sec = 0.0;
    double schur_recover_sec = 0.0;
    int32_t schur_matvec_calls = 0;
    int32_t schur_preconditioner_applies = 0;
    int32_t j11_solve_calls = 0;
    double voltage_update_sec = 0.0;
    double after_mismatch_sec = 0.0;
    double line_search_sec = 0.0;
    double line_search_alpha = 1.0;
    int32_t line_search_trials = 0;
    bool line_search_accepted = true;
    int32_t inner_iterations = 0;
    int32_t restart_cycles = 0;
    int32_t spmv_calls = 0;
    int32_t preconditioner_applies = 0;
    int32_t reduction_calls = 0;
};

struct BlockIluDxTrace {
    int32_t outer_iteration = 0;
    bool dx_was_applied = false;
    double alpha = 1.0;
    std::vector<double> dx;
};

struct BlockIluSolveStats {
    bool converged = false;
    int32_t outer_iterations = 0;
    int32_t total_inner_iterations = 0;
    int32_t total_spmv_calls = 0;
    int32_t total_preconditioner_applies = 0;
    int32_t total_reduction_calls = 0;
    int32_t total_restart_cycles = 0;
    int32_t total_schur_matvec_calls = 0;
    int32_t total_schur_preconditioner_applies = 0;
    int32_t total_j11_solve_calls = 0;
    double total_linear_solve_sec = 0.0;
    double total_linear_iteration_sec = 0.0;
    SchurBicgstabTiming total_linear_timing;
    int32_t preconditioner_rebuilds = 0;
    int32_t j11_zero_pivot = -1;
    int32_t j22_zero_pivot = -1;
    double final_mismatch = 0.0;
    std::string failure_reason;
    std::vector<BlockIluStepTrace> step_trace;
    std::vector<BlockIluDxTrace> dx_trace;
};

class PowerFlowBlockIluSolver {
public:
    explicit PowerFlowBlockIluSolver(BlockIluSolverOptions options);

    void analyze(const HostPowerFlowStructure& structure);
    BlockIluSolveStats solve(const DevicePowerFlowState& state);
    void download_last_dx(std::vector<double>& dx) const;

private:
    double assemble_mismatch_and_norm(const DevicePowerFlowState& state);
    void apply_voltage_update(const DevicePowerFlowState& state, double alpha);
    void apply_voltage_update_from_base(const DevicePowerFlowState& state,
                                        const double* base_voltage_re,
                                        const double* base_voltage_im,
                                        double alpha);
    void ensure_analyzed() const;

    BlockIluSolverOptions options_;
    ReducedJacobianPatterns patterns_;
    ReducedJacobianAssembler jacobian_;
    ImplicitSchurOperator schur_operator_;
    ImplicitSchurOperatorF32 schur_operator_f32_;
    SchurBicgstabSolver linear_solver_;
    SchurBicgstabSolverF32 linear_solver_f32_;
    SchurGmresSolverF32 gmres_solver_f32_;
    bool analyzed_ = false;
    bool preconditioner_analyzed_ = false;

    DeviceBuffer<int32_t> d_pv_;
    DeviceBuffer<int32_t> d_pq_;
    DeviceBuffer<double> d_mismatch_;
    DeviceBuffer<double> d_rhs_;
    DeviceBuffer<double> d_dx_;
    DeviceBuffer<double> d_base_voltage_re_;
    DeviceBuffer<double> d_base_voltage_im_;
    DeviceBuffer<double> d_va_;
    DeviceBuffer<double> d_vm_;
    DeviceBuffer<double> d_partial_;
    DeviceBuffer<double> d_scratch_;
};

}  // namespace exp_20260415::block_ilu
