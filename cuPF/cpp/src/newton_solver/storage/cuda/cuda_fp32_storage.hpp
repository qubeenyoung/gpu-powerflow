#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/storage/cuda/cuda_batched_storage.hpp"


// ---------------------------------------------------------------------------
// CudaFp32Storage: full-FP32 CUDA 경로의 device 버퍼.
//
// 상태(state)와 Jacobian/solve 모두 FP32. public I/O는 FP64지만 upload에서
// float로 down-cast하고, download에서 다시 FP64로 up-cast한다. 모든 동작은
// CudaBatchedStorage<float, float>가 제공한다(레이아웃/생애주기는 거기 문서 참조).
// ---------------------------------------------------------------------------
struct CudaFp32Storage : CudaBatchedStorage<float, float> {};

#endif  // CUPF_WITH_CUDA
