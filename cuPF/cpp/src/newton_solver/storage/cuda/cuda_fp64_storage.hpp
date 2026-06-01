#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/storage/cuda/cuda_batched_storage.hpp"


// ---------------------------------------------------------------------------
// CudaFp64Storage: full-FP64 CUDA 경로의 device 버퍼.
//
// 상태와 Jacobian/solve 모두 FP64 (정밀도 최우선, 정밀도 변환 없음). 모든 동작은
// CudaBatchedStorage<double, double>가 제공한다(레이아웃/생애주기는 거기 문서 참조).
// batch_size > 1도 FP32/Mixed와 동일한 cuDSS uniform-batch 경로로 지원한다.
// ---------------------------------------------------------------------------
struct CudaFp64Storage : CudaBatchedStorage<double, double> {};

#endif  // CUPF_WITH_CUDA
