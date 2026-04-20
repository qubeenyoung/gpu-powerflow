#include "edge_fill_jac.cuh"

namespace {

__device__ __forceinline__ void fill_jacobian_edge_value(
    const YbusGraph ybus,
    int32_t k,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,
    const int32_t* __restrict__ mapJ11,
    const int32_t* __restrict__ mapJ21,
    const int32_t* __restrict__ mapJ12,
    const int32_t* __restrict__ mapJ22,
    const int32_t* __restrict__ diagJ11,
    const int32_t* __restrict__ diagJ21,
    const int32_t* __restrict__ diagJ12,
    const int32_t* __restrict__ diagJ22,
    float* __restrict__ J_values)
{
    const int32_t i = ybus.row[k];
    const int32_t j = ybus.col[k];

    const float yr = ybus.real[k];
    const float yi = ybus.imag[k];
    const float vi_re = v_re[i];
    const float vi_im = v_im[i];
    const float vj_re = v_re[j];
    const float vj_im = v_im[j];

    const float curr_re = yr * vj_re - yi * vj_im;
    const float curr_im = yr * vj_im + yi * vj_re;

    const float neg_j_vi_re = vi_im;
    const float neg_j_vi_im = -vi_re;
    const float term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
    const float term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

    const float vnj_re = v_norm_re[j];
    const float vnj_im = v_norm_im[j];
    const float norm_curr_re = yr * vnj_re - yi * vnj_im;
    const float norm_curr_im = yr * vnj_im + yi * vnj_re;
    const float term_vm_re = vi_re * norm_curr_re + vi_im * norm_curr_im;
    const float term_vm_im = vi_im * norm_curr_re - vi_re * norm_curr_im;

    const int32_t j11 = mapJ11[k];
    const int32_t j21 = mapJ21[k];
    const int32_t j12 = mapJ12[k];
    const int32_t j22 = mapJ22[k];
    if (j11 >= 0) atomicAdd(&J_values[j11], term_va_re);
    if (j21 >= 0) atomicAdd(&J_values[j21], term_va_im);
    if (j12 >= 0) atomicAdd(&J_values[j12], term_vm_re);
    if (j22 >= 0) atomicAdd(&J_values[j22], term_vm_im);

    const int32_t d11 = diagJ11[i];
    const int32_t d21 = diagJ21[i];
    if (d11 >= 0) atomicAdd(&J_values[d11], -term_va_re);
    if (d21 >= 0) atomicAdd(&J_values[d21], -term_va_im);

    const float vni_re = v_norm_re[i];
    const float vni_im = v_norm_im[i];
    const float diag_vm_re = vni_re * curr_re + vni_im * curr_im;
    const float diag_vm_im = vni_im * curr_re - vni_re * curr_im;

    const int32_t d12 = diagJ12[i];
    const int32_t d22 = diagJ22[i];
    if (d12 >= 0) atomicAdd(&J_values[d12], diag_vm_re);
    if (d22 >= 0) atomicAdd(&J_values[d22], diag_vm_im);
}

}  // namespace

namespace exp20260420::newton_solver {

__global__ void fill_jacobian_edge(YbusGraph ybus,
                                   const float* __restrict__ v_re,
                                   const float* __restrict__ v_im,
                                   const float* __restrict__ v_norm_re,
                                   const float* __restrict__ v_norm_im,
                                   const int32_t* __restrict__ mapJ11,
                                   const int32_t* __restrict__ mapJ21,
                                   const int32_t* __restrict__ mapJ12,
                                   const int32_t* __restrict__ mapJ22,
                                   const int32_t* __restrict__ diagJ11,
                                   const int32_t* __restrict__ diagJ21,
                                   const int32_t* __restrict__ diagJ12,
                                   const int32_t* __restrict__ diagJ22,
                                   float* __restrict__ J_values)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < ybus.n_edges) {
        fill_jacobian_edge_value(
            ybus, k, v_re, v_im, v_norm_re, v_norm_im,
            mapJ11, mapJ21, mapJ12, mapJ22,
            diagJ11, diagJ21, diagJ12, diagJ22, J_values);
    }
}

}  // namespace exp20260420::newton_solver

__global__ void fill_jacobian_edge(YbusGraph ybus,
                                   const float* __restrict__ v_re,
                                   const float* __restrict__ v_im,
                                   const float* __restrict__ v_norm_re,
                                   const float* __restrict__ v_norm_im,
                                   const int32_t* __restrict__ mapJ11,
                                   const int32_t* __restrict__ mapJ21,
                                   const int32_t* __restrict__ mapJ12,
                                   const int32_t* __restrict__ mapJ22,
                                   const int32_t* __restrict__ diagJ11,
                                   const int32_t* __restrict__ diagJ21,
                                   const int32_t* __restrict__ diagJ12,
                                   const int32_t* __restrict__ diagJ22,
                                   float* __restrict__ J_values)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < ybus.n_edges) {
        fill_jacobian_edge_value(
            ybus, k, v_re, v_im, v_norm_re, v_norm_im,
            mapJ11, mapJ21, mapJ12, mapJ22,
            diagJ11, diagJ21, diagJ12, diagJ22, J_values);
    }
}
