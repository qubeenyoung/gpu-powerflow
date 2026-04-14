#pragma once

#include "newton_solver/ops/op_interfaces.hpp"


class CpuFp64Storage;


// ---------------------------------------------------------------------------
// CpuMismatchOpF64: Eigen-based FP64 mismatch on CPU.
//
// Computes:
//   Ibus     = Ybus * V          (Eigen SpMV)
//   mismatch = conj(V) * (Ibus - Sbus/V*)   (power mismatch)
//   F        = pack [Re(mis[pvpq]), Im(mis[pq])]
//   normF    = max|F_i|
// ---------------------------------------------------------------------------
class CpuMismatchOpF64 final : public IMismatchOp {
public:
    explicit CpuMismatchOpF64(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    CpuFp64Storage& storage_;
};
