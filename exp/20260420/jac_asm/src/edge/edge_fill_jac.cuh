#pragma once

#include "data_types.hpp"

#include <cstdint>

__global__ void fill_jacobian_edge(
    const YbusGraph ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,

    const int32_t* __restrict__ offdiagJ11,
    const int32_t* __restrict__ offdiagJ21,
    const int32_t* __restrict__ offdiagJ12,
    const int32_t* __restrict__ offdiagJ22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    float* __restrict__ J_values);
