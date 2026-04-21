#include "data_types.hpp"
#include "jacobian_terms.cuh"

namespace {

__device__ __forceinline__ float warpReduceSum(float value)
{
    constexpr unsigned kFullMask = 0xffffffffu;
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    return value;
}

}  // namespace

__global__ void
fill_jacobian_vertex(
    const YbusGraph ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,

    int32_t n_rows,
    const int32_t* __restrict__ pvpq,
    const int32_t* __restrict__ offdiagJ11,
    const int32_t* __restrict__ offdiagJ21,
    const int32_t* __restrict__ offdiagJ12,
    const int32_t* __restrict__ offdiagJ22,
    const int32_t* __restrict__ diagJ11,
    const int32_t* __restrict__ diagJ21,
    const int32_t* __restrict__ diagJ12,
    const int32_t* __restrict__ diagJ22,

    float* __restrict__ J_values
) {
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int32_t bus = pvpq[row];
    const bool has_q_row = diagJ21[bus] >= 0;

    float diag11_acc = 0.0f;
    float diag21_acc = 0.0f;
    float diag12_acc = 0.0f;
    float diag22_acc = 0.0f;

    for (int32_t k = ybus.row_ptr[bus]; k < ybus.row_ptr[bus + 1]; ++k) {
        const int32_t col = ybus.col[k];

        if (col == bus) {
            if (has_q_row) {
                const JacobianTerms terms = computeJacobianTerms(
                    ybus, k, bus, v_re, v_im, v_norm_re, v_norm_im);
                diag12_acc += terms.diag_vm_re + terms.term_vm_re;
                diag22_acc += terms.diag_vm_im + terms.term_vm_im;
            }
            continue;
        }

        const JacobianTerms terms = computeJacobianTerms(
            ybus, k, bus, v_re, v_im, v_norm_re, v_norm_im);

        diag11_acc += -terms.term_va_re;
        if (has_q_row) {
            diag21_acc += -terms.term_va_im;
            diag12_acc += terms.diag_vm_re;
            diag22_acc += terms.diag_vm_im;
        }

        if (offdiagJ11[k] >= 0) J_values[offdiagJ11[k]] = terms.term_va_re;
        if (offdiagJ12[k] >= 0) J_values[offdiagJ12[k]] = terms.term_vm_re;

        if (has_q_row) {
            if (offdiagJ21[k] >= 0) J_values[offdiagJ21[k]] = terms.term_va_im;
            if (offdiagJ22[k] >= 0) J_values[offdiagJ22[k]] = terms.term_vm_im;
        }
    }

    J_values[diagJ11[bus]] = diag11_acc;

    if (has_q_row) {
        J_values[diagJ21[bus]] = diag21_acc;
        J_values[diagJ12[bus]] = diag12_acc;
        J_values[diagJ22[bus]] = diag22_acc;
    }
}

__global__ void
fill_jacobian_vertex_warp(
    const YbusGraph ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,

    int32_t n_rows,
    const int32_t* __restrict__ pvpq,
    const int32_t* __restrict__ offdiagJ11,
    const int32_t* __restrict__ offdiagJ21,
    const int32_t* __restrict__ offdiagJ12,
    const int32_t* __restrict__ offdiagJ22,
    const int32_t* __restrict__ diagJ11,
    const int32_t* __restrict__ diagJ21,
    const int32_t* __restrict__ diagJ12,
    const int32_t* __restrict__ diagJ22,

    float* __restrict__ J_values
) {
    constexpr int32_t kWarpSize = 32;
    const int32_t lane = threadIdx.x & (kWarpSize - 1);
    const int32_t warp_in_block = threadIdx.x / kWarpSize;
    const int32_t warps_per_block = blockDim.x / kWarpSize;
    const int32_t row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= n_rows) {
        return;
    }

    const int32_t bus = pvpq[row];
    const bool has_q_row = diagJ21[bus] >= 0;

    float diag11_acc = 0.0f;
    float diag21_acc = 0.0f;
    float diag12_acc = 0.0f;
    float diag22_acc = 0.0f;

    const int32_t row_begin = ybus.row_ptr[bus];
    const int32_t row_end = ybus.row_ptr[bus + 1];
    for (int32_t k = row_begin + lane; k < row_end; k += kWarpSize) {
        const int32_t col = ybus.col[k];

        if (col == bus) {
            if (has_q_row) {
                const JacobianTerms terms = computeJacobianTerms(
                    ybus, k, bus, v_re, v_im, v_norm_re, v_norm_im);
                diag12_acc += terms.diag_vm_re + terms.term_vm_re;
                diag22_acc += terms.diag_vm_im + terms.term_vm_im;
            }
            continue;
        }

        const JacobianTerms terms = computeJacobianTerms(
            ybus, k, bus, v_re, v_im, v_norm_re, v_norm_im);

        diag11_acc += -terms.term_va_re;
        if (has_q_row) {
            diag21_acc += -terms.term_va_im;
            diag12_acc += terms.diag_vm_re;
            diag22_acc += terms.diag_vm_im;
        }

        if (offdiagJ11[k] >= 0) J_values[offdiagJ11[k]] = terms.term_va_re;
        if (offdiagJ12[k] >= 0) J_values[offdiagJ12[k]] = terms.term_vm_re;

        if (has_q_row) {
            if (offdiagJ21[k] >= 0) J_values[offdiagJ21[k]] = terms.term_va_im;
            if (offdiagJ22[k] >= 0) J_values[offdiagJ22[k]] = terms.term_vm_im;
        }
    }

    diag11_acc = warpReduceSum(diag11_acc);
    diag21_acc = warpReduceSum(diag21_acc);
    diag12_acc = warpReduceSum(diag12_acc);
    diag22_acc = warpReduceSum(diag22_acc);

    if (lane == 0) {
        J_values[diagJ11[bus]] = diag11_acc;

        if (has_q_row) {
            J_values[diagJ21[bus]] = diag21_acc;
            J_values[diagJ12[bus]] = diag12_acc;
            J_values[diagJ22[bus]] = diag22_acc;
        }
    }
}
