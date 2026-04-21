#pragma once

#ifdef CUPF_WITH_CUDA

#include <cstdint>

void launch_cast_rhs_f64_to_f32(const double* src, float* dst, int32_t count);

#endif  // CUPF_WITH_CUDA
