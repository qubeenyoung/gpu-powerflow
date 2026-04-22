#pragma once

#ifdef CUPF_WITH_CUDA

#include <cstdint>

void launch_prepare_rhs(const double* src, float* dst, int32_t count);

#endif  // CUPF_WITH_CUDA
