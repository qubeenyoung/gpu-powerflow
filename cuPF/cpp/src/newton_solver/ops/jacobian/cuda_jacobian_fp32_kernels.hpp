#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"

void launch_fill_jacobian_edge_offdiag_fp32(CudaMixedStorage& storage);
void launch_fill_jacobian_vertex_offdiag_fp32(CudaMixedStorage& storage);
void launch_fill_jacobian_diag_from_ibus_fp32(CudaMixedStorage& storage);

#endif  // CUPF_WITH_CUDA
