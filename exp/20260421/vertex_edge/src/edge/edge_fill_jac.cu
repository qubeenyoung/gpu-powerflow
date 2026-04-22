#include "edge_fill_jac.cuh"

#include <cmath>

namespace {

__device__ __forceinline__ void fill_jacobian_edge_value(
    const YbusGraph ybus,
    int32_t k,
    int32_t batch,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const int32_t* __restrict__ mapJ11,
    const int32_t* __restrict__ mapJ21,
    const int32_t* __restrict__ mapJ12,
    const int32_t* __restrict__ mapJ22,
    const int32_t* __restrict__ diagJ11,
    const int32_t* __restrict__ diagJ21,
    const int32_t* __restrict__ diagJ12,
    const int32_t* __restrict__ diagJ22,
    int32_t jac_nnz,
    float* __restrict__ J_values)
{
    const int32_t i = ybus.row[k];
    const int32_t j = ybus.col[k];
    const int32_t bus_base = batch * ybus.n_bus;
    const int32_t jac_base = batch * jac_nnz;

    const float yr = ybus.real[k];
    const float yi = ybus.imag[k];
    const float vi_re = static_cast<float>(v_re[bus_base + i]);
    const float vi_im = static_cast<float>(v_im[bus_base + i]);
    const float vj_re = static_cast<float>(v_re[bus_base + j]);
    const float vj_im = static_cast<float>(v_im[bus_base + j]);

    const float curr_re = yr * vj_re - yi * vj_im;
    const float curr_im = yr * vj_im + yi * vj_re;
    const bool self_edge = (i == j);

    if (!self_edge) {
        const float neg_j_vi_re = vi_im;
        const float neg_j_vi_im = -vi_re;
        const float term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
        const float term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

        const float vj_abs = hypotf(vj_re, vj_im);
        float term_vm_re = 0.0f;
        float term_vm_im = 0.0f;
        if (vj_abs > 1e-6f) {
            const float scaled_re = curr_re / vj_abs;
            const float scaled_im = curr_im / vj_abs;
            term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
            term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
        }

        if (mapJ11[k] >= 0) J_values[jac_base + mapJ11[k]] = term_va_re;
        if (mapJ21[k] >= 0) J_values[jac_base + mapJ21[k]] = term_va_im;
        if (mapJ12[k] >= 0) J_values[jac_base + mapJ12[k]] = term_vm_re;
        if (mapJ22[k] >= 0) J_values[jac_base + mapJ22[k]] = term_vm_im;

        if (diagJ11[i] >= 0) atomicAdd(&J_values[jac_base + diagJ11[i]], -term_va_re);
        if (diagJ21[i] >= 0) atomicAdd(&J_values[jac_base + diagJ21[i]], -term_va_im);
    }

    const float vi_abs = hypotf(vi_re, vi_im);
    if (vi_abs > 1e-6f) {
        const float vi_norm_re = vi_re / vi_abs;
        const float vi_norm_im = vi_im / vi_abs;
        const float diag_vm_re = vi_norm_re * curr_re + vi_norm_im * curr_im;
        const float diag_vm_im = vi_norm_im * curr_re - vi_norm_re * curr_im;
        const float diag_scale = self_edge ? 2.0f : 1.0f;

        if (diagJ12[i] >= 0) atomicAdd(&J_values[jac_base + diagJ12[i]], diag_scale * diag_vm_re);
        if (diagJ22[i] >= 0) atomicAdd(&J_values[jac_base + diagJ22[i]], diag_scale * diag_vm_im);
    }
}

__device__ __forceinline__ void fill_jacobian_edge_value_single(
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

    if (mapJ11[k] >= 0) atomicAdd(&J_values[mapJ11[k]], term_va_re);
    if (mapJ21[k] >= 0) atomicAdd(&J_values[mapJ21[k]], term_va_im);
    if (mapJ12[k] >= 0) atomicAdd(&J_values[mapJ12[k]], term_vm_re);
    if (mapJ22[k] >= 0) atomicAdd(&J_values[mapJ22[k]], term_vm_im);

    if (diagJ11[i] >= 0) atomicAdd(&J_values[diagJ11[i]], -term_va_re);
    if (diagJ21[i] >= 0) atomicAdd(&J_values[diagJ21[i]], -term_va_im);

    const float vni_re = v_norm_re[i];
    const float vni_im = v_norm_im[i];
    const float diag_vm_re = vni_re * curr_re + vni_im * curr_im;
    const float diag_vm_im = vni_im * curr_re - vni_re * curr_im;

    if (diagJ12[i] >= 0) atomicAdd(&J_values[diagJ12[i]], diag_vm_re);
    if (diagJ22[i] >= 0) atomicAdd(&J_values[diagJ22[i]], diag_vm_im);
}

}  // namespace

namespace exp20260421::vertex_edge::newton_solver {

__global__ void fill_jacobian_edge_batch(YbusGraph ybus,
                                         const double* __restrict__ v_re,
                                         const double* __restrict__ v_im,
                                         int32_t batch_size,
                                         const int32_t* __restrict__ mapJ11,
                                         const int32_t* __restrict__ mapJ21,
                                         const int32_t* __restrict__ mapJ12,
                                         const int32_t* __restrict__ mapJ22,
                                         const int32_t* __restrict__ diagJ11,
                                         const int32_t* __restrict__ diagJ21,
                                         const int32_t* __restrict__ diagJ12,
                                         const int32_t* __restrict__ diagJ22,
                                         int32_t jac_nnz,
                                         float* __restrict__ J_values)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t batch = blockIdx.y;
    if (k < ybus.n_edges && batch < batch_size) {
        fill_jacobian_edge_value(
            ybus,
            k,
            batch,
            v_re,
            v_im,
            mapJ11,
            mapJ21,
            mapJ12,
            mapJ22,
            diagJ11,
            diagJ21,
            diagJ12,
            diagJ22,
            jac_nnz,
            J_values);
    }
}

}  // namespace exp20260421::vertex_edge::newton_solver

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
        fill_jacobian_edge_value_single(
            ybus, k, v_re, v_im, v_norm_re, v_norm_im,
            mapJ11, mapJ21, mapJ12, mapJ22,
            diagJ11, diagJ21, diagJ12, diagJ22, J_values);
    }
}
