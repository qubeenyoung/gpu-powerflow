#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace exp_20260414::amgx_v2 {

enum class PreconditionerKind {
    ScalarAmgx,
    Block2x2Amgx,
    Block2x2AmgxJacobi,
    BusBlockJacobi,
};

inline const char* preconditioner_kind_name(PreconditionerKind kind)
{
    switch (kind) {
    case PreconditionerKind::ScalarAmgx:
        return "scalar_amgx";
    case PreconditionerKind::Block2x2Amgx:
        return "block2x2_amgx";
    case PreconditionerKind::Block2x2AmgxJacobi:
        return "block2x2_amgx_jacobi";
    case PreconditionerKind::BusBlockJacobi:
        return "bus_block_jacobi";
    }
    return "unknown";
}

inline PreconditionerKind parse_preconditioner_kind(const std::string& name)
{
    if (name == "scalar_amgx" || name == "scalar") {
        return PreconditionerKind::ScalarAmgx;
    }
    if (name == "block2x2_amgx" || name == "block2x2" || name == "block") {
        return PreconditionerKind::Block2x2Amgx;
    }
    if (name == "block2x2_amgx_jacobi" || name == "amgx_block_jacobi") {
        return PreconditionerKind::Block2x2AmgxJacobi;
    }
    if (name == "bus_block_jacobi" || name == "block_jacobi") {
        return PreconditionerKind::BusBlockJacobi;
    }
    throw std::runtime_error("unknown preconditioner kind: " + name);
}

struct SolverOptions {
    double nonlinear_tolerance = 1e-8;
    double linear_tolerance = 1e-2;
    int32_t max_outer_iterations = 20;
    int32_t max_inner_iterations = 500;
    int32_t gmres_restart = 200;
    int32_t preconditioner_rebuild_interval = 2;
    PreconditionerKind preconditioner = PreconditionerKind::ScalarAmgx;
    bool continue_on_linear_failure = false;
    bool track_dx_residual = false;
};

struct NonlinearStepTrace {
    int32_t outer_iteration = 0;
    bool linear_converged = false;
    int64_t inner_iterations = 0;
    int64_t jv_calls = 0;
    int32_t linear_failures = 0;
    int32_t preconditioner_age = 0;
    double linear_residual = 0.0;
    double before_mismatch = 0.0;
    double after_mismatch = 0.0;
    double after_before_ratio = 0.0;
    bool preconditioner_rebuilt = false;
    bool dx_was_applied = false;
};

struct SolveStats {
    bool converged = false;
    int32_t outer_iterations = 0;
    int32_t linear_failures = 0;
    int32_t preconditioner_rebuilds = 0;
    int64_t total_inner_iterations = 0;
    int64_t total_jv_calls = 0;
    double final_mismatch = 0.0;
    std::string failure_reason;
    std::vector<NonlinearStepTrace> step_trace;
};

}  // namespace exp_20260414::amgx_v2
