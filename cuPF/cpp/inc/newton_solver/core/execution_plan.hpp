#pragma once

#include "newton_solver/ops/op_interfaces.hpp"

#include <memory>


// ---------------------------------------------------------------------------
// ExecutionPlan: the fully assembled solver for one NewtonOptions profile.
//
// Owns Storage (buffers + library handles) and the four Op instances that
// correspond to one concrete execution profile such as CPU FP64 or CUDA Mixed.
//
// Built by PlanBuilder::build(options) — never constructed directly.
//
// NR hot path:
//   plan.mismatch->run(ctx);
//   if (ctx.converged) break;
//   plan.jacobian->run(ctx);
//   plan.linear_solve->run(ctx);
//   plan.voltage_update->run(ctx);
// ---------------------------------------------------------------------------
struct ExecutionPlan {
    std::unique_ptr<IStorage>         storage;
    std::unique_ptr<IMismatchOp>      mismatch;
    std::unique_ptr<IJacobianOp>      jacobian;
    std::unique_ptr<ILinearSolveOp>   linear_solve;
    std::unique_ptr<IVoltageUpdateOp> voltage_update;

    // True once PlanBuilder has successfully assembled all components.
    bool ready = false;
};
