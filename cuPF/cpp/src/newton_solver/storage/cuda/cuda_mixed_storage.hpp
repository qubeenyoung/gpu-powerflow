#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/storage/cuda/cuda_batched_storage.hpp"


// ---------------------------------------------------------------------------
// CudaMixedStorage: mixed precision CUDA 경로의 device 버퍼.
//
// 상태(Ybus/V/Va/Vm/Sbus/Ibus/F/normF)는 FP64로 유지하고, Jacobian 값과
// solve 결과(d_J_values/d_dx)만 FP32로 둔다 → cuDSS factorize/solve를 더 싸게.
// 모든 동작은 CudaBatchedStorage<double, float>가 제공한다(레이아웃/생애주기는
// 거기 문서 참조). 최종 전압은 FP64 상태에서 재구성한다.
// ---------------------------------------------------------------------------
struct CudaMixedStorage : CudaBatchedStorage<double, float> {};

#endif  // CUPF_WITH_CUDA
