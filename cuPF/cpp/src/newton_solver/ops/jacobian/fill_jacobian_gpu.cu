// ---------------------------------------------------------------------------
// fill_jacobian_gpu.cu
//
// CUDA assembly of the power-flow Jacobian — the GPU counterpart of
// fill_jacobian.cpp. This translation unit is just the per-pipeline dispatch:
// it selects one of three assembly variants (chosen by CudaJacobianOp::kind)
// and forwards the storage buffers with the right scalar types. The variants
// and the shared math live in:
//   jacobian_gpu_common.hpp              - atomics, warp reduce, per-edge
//                                          sensitivity, host dump helper
//   fill_jacobian_edge_kernel.hpp        - Edge (default, cached-Ibus diagonal)
//   fill_jacobian_edge_atomic_kernel.hpp - EdgeAtomic (atomic scatter)
//   fill_jacobian_vertex_warp_kernel.hpp - VertexWarp (warp-per-row reduce)
//
// Scalar types per profile (JScalar, YbusScalar, VoltageScalar[, IbusScalar]):
//   FP64  : double, double, double         (state and J all double)
//   Mixed : float , double, double         (double state -> float J)
//   FP32  : float , float , float          (everything float)
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "fill_jacobian_gpu.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"

#include "newton_solver/ops/jacobian/jacobian_gpu_common.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian_edge_kernel.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian_edge_atomic_kernel.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian_vertex_warp_kernel.hpp"


// ===========================================================================
// Per-pipeline dispatch. Each overload picks the kernel scalar types for its
// storage layout and forwards the batch / cached-Ibus / batched-Ybus flags.
// The default (Edge) branch reuses the cached d_Ibus for the diagonal; the
// EdgeAtomic / VertexWarp branches are alternative assembly layouts selected
// via CudaJacobianOp::kind.
// ===========================================================================

// FP64: all-double inputs and Jacobian, batched.
void CudaJacobianOp<double>::run(CudaFp64Storage& buf, IterationContext& ctx)
{
    if (kind == CudaJacobianKind::EdgeAtomic) {
        launch_fill_jacobian_edge_atomic<double, double, double>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else if (kind == CudaJacobianKind::VertexWarp) {
        launch_fill_jacobian_vertex_warp<double, double, double>(
            buf.n_pvpq, buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_pvpq,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else {
        launch_fill_jacobian_gpu<double, double, double, double>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            // use_cached_ibus=true: the ibus stage ran just before jacobian in the
            // NR loop (and in prepare_adjoint_cache), so d_Ibus is current. Reusing
            // it avoids recomputing the per-diagonal current injection in-kernel.
            true,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            &buf.d_Ibus_re, &buf.d_Ibus_im,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    }
    dump_cuda_jacobian_if_enabled("jacobian",
                                  ctx.iter,
                                  buf.dimF,
                                  buf.d_J_row_ptr,
                                  buf.d_J_col_idx,
                                  buf.d_J_values,
                                  buf.nnz_J);
}


// Mixed: FP64 Ybus/voltage/Ibus inputs assembled into an FP32 Jacobian, batched.
void CudaJacobianOp<float>::run(CudaMixedStorage& buf, IterationContext& ctx)
{
    if (kind == CudaJacobianKind::EdgeAtomic) {
        launch_fill_jacobian_edge_atomic<float, double, double>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else if (kind == CudaJacobianKind::VertexWarp) {
        launch_fill_jacobian_vertex_warp<float, double, double>(
            buf.n_pvpq, buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_pvpq,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else {
        launch_fill_jacobian_gpu<float, double, double, double>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            true,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            &buf.d_Ibus_re, &buf.d_Ibus_im,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    }
    dump_cuda_jacobian_if_enabled("jacobian",
                                  ctx.iter,
                                  buf.dimF,
                                  buf.d_J_row_ptr,
                                  buf.d_J_col_idx,
                                  buf.d_J_values,
                                  buf.nnz_J);
}


// FP32: all-float inputs and Jacobian, batched.
void CudaJacobianOp<float>::run(CudaFp32Storage& buf, IterationContext& ctx)
{
    if (kind == CudaJacobianKind::EdgeAtomic) {
        launch_fill_jacobian_edge_atomic<float, float, float>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else if (kind == CudaJacobianKind::VertexWarp) {
        launch_fill_jacobian_vertex_warp<float, float, float>(
            buf.n_pvpq, buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            buf.d_pvpq,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    } else {
        launch_fill_jacobian_gpu<float, float, float, float>(
            buf.nnz_ybus, buf.nnz_J, buf.n_bus, buf.batch_size,
            buf.ybus_values_batched,
            true,
            buf.d_Ybus_re, buf.d_Ybus_im,
            buf.d_Ybus_row, buf.d_Ybus_indices, buf.d_Ybus_indptr,
            buf.d_V_re, buf.d_V_im, buf.d_Vm,
            &buf.d_Ibus_re, &buf.d_Ibus_im,
            buf.d_mapJ11, buf.d_mapJ21, buf.d_mapJ12, buf.d_mapJ22,
            buf.d_diagJ11, buf.d_diagJ21, buf.d_diagJ12, buf.d_diagJ22,
            buf.d_J_values);
    }
    dump_cuda_jacobian_if_enabled("jacobian",
                                  ctx.iter,
                                  buf.dimF,
                                  buf.d_J_row_ptr,
                                  buf.d_J_col_idx,
                                  buf.d_J_values,
                                  buf.nnz_J);
}

#endif  // CUPF_WITH_CUDA
