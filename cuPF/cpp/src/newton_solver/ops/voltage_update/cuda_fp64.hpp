#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaVoltageUpdateFp64: FP64 전압 갱신 (3단계 커널 파이프라인).
//
//   1. decompose_voltage_kernel  : V(복소) → Va(각도), Vm(크기)
//   2. update_voltage_fp64_kernel: dx(FP64) 보정 → Va[pv/pq], Vm[pq]
//   3. reconstruct_voltage_kernel: Va, Vm → V(복소)
//
// 모든 버퍼가 FP64이므로 정밀도 손실 없음. CudaFp64Storage 전용.
// ---------------------------------------------------------------------------
class CudaVoltageUpdateFp64 final : public IVoltageUpdateOp {
public:
    explicit CudaVoltageUpdateFp64(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    IStorage& storage_;
};

#endif  // CUPF_WITH_CUDA
