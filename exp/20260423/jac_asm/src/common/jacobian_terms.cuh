#pragma once

#include "data_types.hpp"

struct JacobianTerms {
    float term_va_re = 0.0f;
    float term_va_im = 0.0f;
    float term_vm_re = 0.0f;
    float term_vm_im = 0.0f;
    float diag_vm_re = 0.0f;
    float diag_vm_im = 0.0f;
};


__device__ __forceinline__ JacobianTerms computeJacobianTerms(
    const YbusGraph ybus,
    int32_t y_index,
    int32_t row_bus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im
) {
    const int32_t col_bus = ybus.col[y_index];

    const float yr = ybus.real[y_index];
    const float yi = ybus.imag[y_index];
    const float vi_re = v_re[row_bus];
    const float vi_im = v_im[row_bus];
    const float vj_re = v_re[col_bus];
    const float vj_im = v_im[col_bus];
    const float vi_norm_re = v_norm_re[row_bus];
    const float vi_norm_im = v_norm_im[row_bus];
    const float vj_norm_re = v_norm_re[col_bus];
    const float vj_norm_im = v_norm_im[col_bus];

    const float curr_re = yr * vj_re - yi * vj_im;
    const float curr_im = yr * vj_im + yi * vj_re;

    const float neg_j_vi_re = vi_im;
    const float neg_j_vi_im = -vi_re;

    JacobianTerms terms;
    terms.term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
    terms.term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

    const float norm_curr_re = yr * vj_norm_re - yi * vj_norm_im;
    const float norm_curr_im = yr * vj_norm_im + yi * vj_norm_re;
    terms.term_vm_re = vi_re * norm_curr_re + vi_im * norm_curr_im;
    terms.term_vm_im = vi_im * norm_curr_re - vi_re * norm_curr_im;

    terms.diag_vm_re = vi_norm_re * curr_re + vi_norm_im * curr_im;
    terms.diag_vm_im = vi_norm_im * curr_re - vi_norm_re * curr_im;

    return terms;
}
