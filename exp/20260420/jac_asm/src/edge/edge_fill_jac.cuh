#pragma once

#include "data_types.hpp"

#include <cstdint>

namespace exp20260420::newton_solver {

__global__ void fill_jacobian_edge(YbusGraph ybus,
                                   const float* v_re,
                                   const float* v_im,
                                   const float* v_norm_re,
                                   const float* v_norm_im,
                                   const int32_t* mapJ11,
                                   const int32_t* mapJ21,
                                   const int32_t* mapJ12,
                                   const int32_t* mapJ22,
                                   const int32_t* diagJ11,
                                   const int32_t* diagJ21,
                                   const int32_t* diagJ12,
                                   const int32_t* diagJ22,
                                   float* J_values);

}  // namespace exp20260420::newton_solver

__global__ void fill_jacobian_edge(YbusGraph ybus,
                                   const float* v_re,
                                   const float* v_im,
                                   const float* v_norm_re,
                                   const float* v_norm_im,
                                   const int32_t* mapJ11,
                                   const int32_t* mapJ21,
                                   const int32_t* mapJ12,
                                   const int32_t* mapJ22,
                                   const int32_t* diagJ11,
                                   const int32_t* diagJ21,
                                   const int32_t* diagJ12,
                                   const int32_t* diagJ22,
                                   float* J_values);
