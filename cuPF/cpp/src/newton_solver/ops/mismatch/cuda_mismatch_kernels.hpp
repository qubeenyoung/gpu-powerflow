#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"

void launch_compute_ibus_batch_fp32(CudaMixedStorage& storage);
void launch_compute_mismatch_batch_f64(CudaMixedStorage& storage);
void launch_reduce_norm_batch_f64(CudaMixedStorage& storage);

#endif  // CUPF_WITH_CUDA
