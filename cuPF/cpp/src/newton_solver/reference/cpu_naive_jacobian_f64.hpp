#pragma once

#include "newton_solver/ops/op_interfaces.hpp"


class CpuFp64Storage;


// ---------------------------------------------------------------------------
// CpuNaiveJacobianOpF64: PyPower-like Jacobian assembly on CPU FP64.
//
// Mirrors v2 naive CPU backend / PYPOWER's dSbus_dV workflow:
//   1. build full dS_dVa and dS_dVm sparse matrices
//   2. slice J11/J12/J21/J22
//   3. rebuild J from triplets every iteration
// ---------------------------------------------------------------------------
class CpuNaiveJacobianOpF64 final : public IJacobianOp {
public:
    explicit CpuNaiveJacobianOpF64(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    CpuFp64Storage& storage_;
};
