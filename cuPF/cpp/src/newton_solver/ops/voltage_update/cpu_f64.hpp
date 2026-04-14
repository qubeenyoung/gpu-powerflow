#pragma once

#include "newton_solver/ops/op_interfaces.hpp"


class CpuFp64Storage;


// ---------------------------------------------------------------------------
// CpuVoltageUpdateF64: apply dx correction and reconstruct complex voltage.
//
// dx layout: [Va[pv], Va[pq], Vm[pq]]
// Updates Va[pvpq], Vm[pq], then recomputes V = Vm * exp(j*Va).
// ---------------------------------------------------------------------------
class CpuVoltageUpdateF64 final : public IVoltageUpdateOp {
public:
    explicit CpuVoltageUpdateF64(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    CpuFp64Storage& storage_;
};
