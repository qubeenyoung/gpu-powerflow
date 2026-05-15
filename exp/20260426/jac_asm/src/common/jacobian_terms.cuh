#pragma once

#include <cstdint>

// Shared per-edge Jacobian contributions. The real/imaginary components map to
// the P/Q rows used by the four Newton Jacobian blocks.
struct JacobianTerms {
    // Angle derivative contribution used by J11 and J21.
    float term_va_re = 0.0f;
    float term_va_im = 0.0f;
    // Voltage-magnitude derivative contribution used by J12 and J22.
    float term_vm_re = 0.0f;
    float term_vm_im = 0.0f;
    // Diagonal voltage-magnitude contribution accumulated by the owning bus.
    float diag_vm_re = 0.0f;
    float diag_vm_im = 0.0f;
};

// Computes all local Jacobian terms for one Ybus entry (row_bus, col_bus).
__device__ __forceinline__ JacobianTerms computeJacobianTerms(
    int32_t row_bus,
    int32_t col_bus,
    float yr,
    float yi,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im
) {
    // Load the sending-end voltage once; every local derivative uses it.
    const float vi_re = v_re[row_bus];
    const float vi_im = v_im[row_bus];

    // Current contribution Y_ij * V_j from the selected Ybus entry.
    const float curr_re = yr * v_re[col_bus] - yi * v_im[col_bus];
    const float curr_im = yr * v_im[col_bus] + yi * v_re[col_bus];

    // Angle derivative multiplies the sending voltage by -j.
    const float neg_j_vi_re = vi_im;
    const float neg_j_vi_im = -vi_re;

    JacobianTerms terms;
    // Off-diagonal angle derivatives: real part feeds P, imaginary feeds Q.
    terms.term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
    terms.term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

    // Off-diagonal voltage-magnitude derivatives use the normalized V_j phasor.
    const float norm_curr_re = yr * v_norm_re[col_bus] - yi * v_norm_im[col_bus];
    const float norm_curr_im = yr * v_norm_im[col_bus] + yi * v_norm_re[col_bus];
    terms.term_vm_re = vi_re * norm_curr_re + vi_im * norm_curr_im;
    terms.term_vm_im = vi_im * norm_curr_re - vi_re * norm_curr_im;

    // Diagonal voltage-magnitude derivatives use the normalized sending voltage.
    terms.diag_vm_re = v_norm_re[row_bus] * curr_re + v_norm_im[row_bus] * curr_im;
    terms.diag_vm_im = v_norm_im[row_bus] * curr_re - v_norm_re[row_bus] * curr_im;

    return terms;
}
