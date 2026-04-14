#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"


// ---------------------------------------------------------------------------
// CudaJacobianOpVertexFp64: vertex-based FP64 Jacobian fill kernel.
//
// warp 하나(32 레인)가 버스(정점) 하나를 처리한다.
// 레인이 버스 행의 원소를 warp_size 스트라이드로 분담하고,
// 대각 기여는 레지스터 누산 → warp_sum(butterfly) → lane 0 단일 write.
// 오프 대각은 직접 write (같은 열을 여러 warp가 처리하지 않으므로 atomic 불필요).
// 고차수(degree) 버스가 많은 계통에서 edge-based보다 효율적.
// ---------------------------------------------------------------------------
class CudaJacobianOpVertexFp64 final : public IJacobianOp {
public:
    explicit CudaJacobianOpVertexFp64(IStorage& storage);
    void run(IterationContext& ctx) override;

private:
    IStorage& storage_;
};

#endif  // CUPF_WITH_CUDA
