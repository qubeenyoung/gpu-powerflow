#include "edge_fill_jac.cuh"
#include "jacobian_terms.cuh"

__global__ void
fill_jacobian_edge(
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
    float* __restrict__ J_values
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ybus.n_edges) {
        return;
    }

    const int32_t i = ybus.row[tid];
    const int32_t j = ybus.col[tid];

    const JacobianTerms terms = computeJacobianTerms(
        ybus, tid, i, v_re, v_im, v_norm_re, v_norm_im);

    // Off-diagonal entries are unique, so they can be written directly.
    // Self edges share the diagonal slot, so they are folded into the diagonal path below.
    if (i != j) {
        if (offdiagJ11[tid] >= 0) J_values[offdiagJ11[tid]] = terms.term_va_re;
        if (offdiagJ21[tid] >= 0) J_values[offdiagJ21[tid]] = terms.term_va_im;
        if (offdiagJ12[tid] >= 0) J_values[offdiagJ12[tid]] = terms.term_vm_re;
        if (offdiagJ22[tid] >= 0) J_values[offdiagJ22[tid]] = terms.term_vm_im;

        if (diag11[i] >= 0) atomicAdd(&J_values[diag11[i]], -terms.term_va_re);
        if (diag21[i] >= 0) atomicAdd(&J_values[diag21[i]], -terms.term_va_im);
    }

    float diag_vm_re = terms.diag_vm_re;
    float diag_vm_im = terms.diag_vm_im;

    if (i == j) {
        diag_vm_re += terms.term_vm_re;
        diag_vm_im += terms.term_vm_im;
    }

    if (diag12[i] >= 0) atomicAdd(&J_values[diag12[i]], diag_vm_re);
    if (diag22[i] >= 0) atomicAdd(&J_values[diag22[i]], diag_vm_im);
}
