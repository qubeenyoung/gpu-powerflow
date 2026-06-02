// ---------------------------------------------------------------------------
// jacobian_gpu_common.hpp
//
// Shared building blocks for the three CUDA Jacobian-assembly variants
// (fill_jacobian_edge_kernel.hpp, fill_jacobian_edge_atomic_kernel.hpp,
// fill_jacobian_vertex_warp_kernel.hpp). Holds the device atomics, the warp
// reduction, the per-edge complex sensitivity math the variants all share, and
// the optional host-side dump helper.
//
// Background — what the Jacobian is. For polar-form Newton power flow the
// residual is the bus power mismatch and the unknowns are voltage angle (Va)
// and magnitude (Vm). The 2x2 block Jacobian is
//     [ dP/dVa  dP/dVm ] = [ J11  J12 ]
//     [ dQ/dVa  dQ/dVm ]   [ J21  J22 ]
// Each Ybus nonzero (i,j) contributes to the off-diagonal (i!=j) entries of all
// four blocks; the diagonal (i==i) entries also gather a self term built from
// the bus current Ibus_i = sum_j Y_ij V_j. The off-diagonal contribution is the
// same expression in every variant and lives in compute_edge_sensitivity();
// only the diagonal accumulation strategy differs between variants.
//
// Header-only (anonymous namespace) so the single dispatch TU
// (fill_jacobian_gpu.cu) instantiates the templates it needs per storage type.
// ---------------------------------------------------------------------------

#pragma once

#ifdef CUPF_WITH_CUDA

#include "utils/cuda_utils.hpp"
#include "utils/dump.hpp"

#include <cstdint>
#include <vector>


namespace {

// Float atomic add maps straight to the hardware intrinsic.
__device__ __forceinline__ float cupf_atomic_add(float* address, float value)
{
    return atomicAdd(address, value);
}

// Double atomic add: native on SM 6.0+, otherwise emulated with a CAS loop on
// the 64-bit bit pattern (so the FP32/Mixed-on-old-arch builds still link).
__device__ __forceinline__ double cupf_atomic_add(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}


// Sum `value` across a full 32-lane warp via butterfly shuffles; lane 0 holds
// the total on return (other lanes hold partial sums). Used by the vertex-warp
// variant to reduce a bus's per-edge diagonal contributions.
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T value)
{
    constexpr unsigned kFullMask = 0xffffffffu;
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    return value;
}


// Off-diagonal sensitivity of one Ybus edge (i -> j), shared by all variants.
//
//   curr = Y_ij * V_j                          (the current edge contributes)
//   angle block (J11/J21):  -i * V_i * conj(curr)
//   magnitude block (J12/J22): V_i * conj(curr / |V_j|)   (skip if |V_j| ~ 0)
//
// Returned as real/imag pairs: *_re feeds the P rows (J11/J12), *_im the Q rows
// (J21/J22). AccumScalar is the precision the kernel accumulates in. The exact
// expression order here matches the original inlined kernels, so results are
// bit-identical across the refactor.
template <typename AccumScalar>
struct EdgeSensitivity {
    AccumScalar curr_re,    curr_im;     // Y_ij * V_j
    AccumScalar term_va_re, term_va_im;  // angle-block contribution
    AccumScalar term_vm_re, term_vm_im;  // magnitude-block contribution
};

template <typename AccumScalar>
__device__ __forceinline__ EdgeSensitivity<AccumScalar> compute_edge_sensitivity(
    AccumScalar yr,    AccumScalar yi,      // Y_ij (real, imag)
    AccumScalar vi_re, AccumScalar vi_im,   // V_i  (source bus voltage)
    AccumScalar vj_re, AccumScalar vj_im,   // V_j  (target bus voltage)
    AccumScalar vj_abs)                     // |V_j|
{
    EdgeSensitivity<AccumScalar> s;

    // curr = Y_ij * V_j  (complex multiply written in real arithmetic).
    s.curr_re = yr * vj_re - yi * vj_im;
    s.curr_im = yr * vj_im + yi * vj_re;

    // Angle term = (-i * V_i) * conj(curr). Pre-rotate V_i by -i: (-i)(a+bi) =
    // b - a i, hence neg_j_vi = (vi_im, -vi_re).
    const AccumScalar neg_j_vi_re = vi_im;
    const AccumScalar neg_j_vi_im = -vi_re;
    s.term_va_re = neg_j_vi_re * s.curr_re + neg_j_vi_im * s.curr_im;
    s.term_va_im = neg_j_vi_im * s.curr_re - neg_j_vi_re * s.curr_im;

    // Magnitude term = V_i * conj(curr / |V_j|). Undefined at |V_j| == 0, so a
    // tiny-magnitude guard leaves it at zero (that bus has no Vm unknown anyway).
    s.term_vm_re = AccumScalar(0);
    s.term_vm_im = AccumScalar(0);
    if (vj_abs > AccumScalar(1.0e-12)) {
        const AccumScalar scaled_re = s.curr_re / vj_abs;
        const AccumScalar scaled_im = s.curr_im / vj_abs;
        s.term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
        s.term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
    }
    return s;
}


// Debug-only: copy the assembled CSR Jacobian back to host and hand it to the
// dump machinery. No-op unless dumping is enabled (CUPF dump env/flag).
template <typename ValueType>
void dump_cuda_jacobian_if_enabled(const char* name,
                                   int32_t iteration,
                                   int32_t dim,
                                   const DeviceBuffer<int32_t>& d_row_ptr,
                                   const DeviceBuffer<int32_t>& d_col_idx,
                                   const DeviceBuffer<ValueType>& d_values,
                                   int32_t nnz)
{
    if (!newton_solver::utils::isDumpEnabled()) {
        return;
    }
    if (dim <= 0 || nnz <= 0) {
        return;
    }

    std::vector<int32_t> row_ptr(dim + 1);
    std::vector<int32_t> col_idx(nnz);
    std::vector<ValueType> values(nnz);
    d_row_ptr.copyTo(row_ptr.data(), row_ptr.size());
    d_col_idx.copyTo(col_idx.data(), col_idx.size());
    d_values.copyTo(values.data(), values.size());
    newton_solver::utils::dumpCSR(name,
                                  iteration,
                                  row_ptr.data(),
                                  col_idx.data(),
                                  values.data(),
                                  dim,
                                  dim);
}

}  // namespace

#endif  // CUPF_WITH_CUDA
