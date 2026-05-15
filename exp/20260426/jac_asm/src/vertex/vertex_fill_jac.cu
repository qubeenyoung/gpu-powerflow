#include "data_types.hpp"
#include "jacobian_terms.cuh"

namespace {

// Fills one reduced Jacobian row by traversing the corresponding Ybus CSR row.
// PQ-only work is selected by a runtime branch instead of template specialization.
__device__ __forceinline__ void fill_jacobian_vertex_row(
    const YbusCsr ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,
    int32_t row,
    bool is_pq_bus,
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
    // Reduced row index maps back to the physical bus stored in pvpq order.
    const int32_t bus = pvpq[row];

    // Diagonal entries are accumulated locally across the bus adjacency list and
    // written once at the end, avoiding atomics in the vertex formulation.
    float diag11_acc = 0.0f;
    float diag21_acc = 0.0f;
    float diag12_acc = 0.0f;
    float diag22_acc = 0.0f;

    // Walk every Ybus entry in this bus row and emit the matching Jacobian terms.
    for (int32_t k = ybus.row_ptr[bus]; k < ybus.row_ptr[bus + 1]; ++k) {
        const int32_t col = ybus.col[k];

        // The explicit self entry contributes only to voltage-magnitude
        // diagonals here; angle diagonals are built from neighboring terms.
        if (col == bus) {
            if (is_pq_bus) {
                const JacobianTerms terms = computeJacobianTerms(
                    bus, col, ybus.real[k], ybus.imag[k],
                    v_re, v_im, v_norm_re, v_norm_im);
                diag12_acc += terms.diag_vm_re + terms.term_vm_re;
                diag22_acc += terms.diag_vm_im + terms.term_vm_im;
            }
            continue;
        }

        const JacobianTerms terms = computeJacobianTerms(
            bus, col, ybus.real[k], ybus.imag[k],
            v_re, v_im, v_norm_re, v_norm_im);

        // Non-self entries contribute to angle diagonals as a negative sum.
        diag11_acc += -terms.term_va_re;
        if (is_pq_bus) {
            diag21_acc += -terms.term_va_im;
            diag12_acc += terms.diag_vm_re;
            diag22_acc += terms.diag_vm_im;
        }

        // Off-diagonal P-row blocks exist for both PV and PQ buses.
        if (offdiagJ11[k] >= 0) J_values[offdiagJ11[k]] = terms.term_va_re;
        if (offdiagJ12[k] >= 0) J_values[offdiagJ12[k]] = terms.term_vm_re;

        // Q-row blocks exist only for PQ buses.
        if (is_pq_bus) {
            if (offdiagJ21[k] >= 0) J_values[offdiagJ21[k]] = terms.term_va_im;
            if (offdiagJ22[k] >= 0) J_values[offdiagJ22[k]] = terms.term_vm_im;
        }
    }

    // Store the accumulated diagonal block entries owned by this row.
    J_values[diagJ11[bus]] = diag11_acc;

    if (is_pq_bus) {
        J_values[diagJ21[bus]] = diag21_acc;
        J_values[diagJ12[bus]] = diag12_acc;
        J_values[diagJ22[bus]] = diag22_acc;
    }
}

}  // namespace

// Vertex-parallel fill: one thread owns one reduced P row, and PQ rows also
// produce the matching Q-row and voltage-magnitude block values.
__global__ void
fill_jacobian_vertex(
    const YbusCsr ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,
    int32_t n_pv,
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
    // Kernel rows are reduced Jacobian P rows, not raw bus ids.
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    // pvpq is stored as [PV..., PQ...], so this row-level flag controls whether
    // the inlined row fill also emits Q-row and voltage-magnitude diagonal work.
    const bool is_pq_bus = row >= n_pv;
    fill_jacobian_vertex_row(
        ybus, v_re, v_im, v_norm_re, v_norm_im, row, is_pq_bus, pvpq,
        offdiagJ11, offdiagJ21, offdiagJ12, offdiagJ22,
        diagJ11, diagJ21, diagJ12, diagJ22,
        J_values);
}
