#pragma once

#include "newton_solver/ops/op_interfaces.hpp"


class CpuFp64Storage;


// ---------------------------------------------------------------------------
// CpuJacobianOpF64: edge-based FP64 Jacobian fill on CPU.
//
// Uses pre-computed JacobianMaps to scatter Ybus entries into J.values.
// Supports both EdgeBased and VertexBased fill algorithms (selected by
// the JacobianBuilderType stored in the maps).
// ---------------------------------------------------------------------------
class CpuJacobianOpF64 final : public IJacobianOp {
public:
    explicit CpuJacobianOpF64(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    CpuFp64Storage& storage_;
};
