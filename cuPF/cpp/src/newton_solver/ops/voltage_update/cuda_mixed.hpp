#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaVoltageUpdateMixed: voltage update for the Mixed precision path.
//
// dx (FP32) → apply to Va/Vm (FP64) → reconstruct V (FP64 complex)
// ---------------------------------------------------------------------------
class CudaVoltageUpdateMixed final : public IVoltageUpdateOp {
public:
    explicit CudaVoltageUpdateMixed(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    IStorage& storage_;
};

#endif  // CUPF_WITH_CUDA
