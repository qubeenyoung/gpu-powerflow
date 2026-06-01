#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "utils/cuda_utils.hpp"

#include <cstddef>


// ===========================================================================
// CudaBatchedStorage<StateScalar, JacScalar>
//
// Single templated implementation behind all three CUDA storage profiles. The
// FP32 / FP64 / Mixed device storages used to be three ~330-line files that
// differed only in their buffer element types; this template captures that
// difference in two scalar parameters and the concrete profiles are thin
// derived structs (see cuda_fp32_storage.hpp / cuda_fp64_storage.hpp /
// cuda_mixed_storage.hpp):
//
//     using semantics             StateScalar   JacScalar
//     CudaFp32Storage  : <float , float >          float        float
//     CudaFp64Storage  : <double, double>          double       double
//     CudaMixedStorage : <double, float >          double       float
//
//   - StateScalar — precision of the *physical state* and everything derived
//     from it: Ybus values, voltage (rectangular d_V_re/d_V_im and polar
//     d_Va/d_Vm), Sbus, Ibus, the mismatch residual d_F and its per-case norm
//     d_normF.
//   - JacScalar — precision of the *linear-solve* objects handed to cuDSS: the
//     Jacobian values d_J_values and the solution/step d_dx. Mixed keeps FP64
//     state but assembles an FP32 Jacobian for a cheaper factorize/solve.
//
// Index buffers (CSR pointers/indices, scatter maps, bus-type lists) are always
// int32 and precision-independent.
//
// Layout: batch-major and contiguous. For batch case `b`:
//   per-bus arrays      live at  [b * n_bus + bus]        (d_V_*, d_Va, d_Vm,
//                                                          d_Sbus_*, d_Ibus_*)
//   per-residual arrays live at  [b * dimF  + row]        (d_F, d_dx)
//   per-J-value arrays  live at  [b * nnz_J + pos]        (d_J_values)
//   d_normF             lives at [b]                      (one L-inf per case)
// The Ybus *pattern* (indptr/indices/row, scatter maps) is shared across the
// batch; Ybus *values* are shared unless ybus_values_batched is set. This is
// exactly the contiguous uniform-batch layout cuDSS expects, so the same
// descriptors drive B == 1 and B > 1 (see cuda_cudss.cpp).
//
// Lifecycle (called by the pipeline in pipeline.hpp; not a virtual interface —
// dispatch is static on the concrete type):
//   prepare()        once per initialize(): allocate buffers, upload the
//                    time-invariant topology (Ybus pattern+values, J pattern,
//                    scatter maps, bus-type indices).
//   upload()         once per solve(): (re)size for the batch and push the
//                    per-solve inputs (Ybus/Sbus values, V0 seed).
//   download()       single-case (B == 1) voltage readback.
//   download_batch() batched voltage readback + per-case final mismatch norms.
// ===========================================================================
template <typename StateScalar, typename JacScalar>
struct CudaBatchedStorage {
    void prepare(const InitializeContext& ctx);
    void upload(const SolveContext& ctx);
    void download(NRResult& result) const;
    void download_batch(NRBatchResult& result) const;

    // Flat index of bus `bus` in case `batch` (per-bus arrays are [B * n_bus]).
    std::size_t bus_offset(int32_t batch, int32_t bus) const
    {
        return static_cast<std::size_t>(batch) * static_cast<std::size_t>(n_bus) +
               static_cast<std::size_t>(bus);
    }

    // Flat index of residual row `row` in case `batch` (d_F/d_dx are [B * dimF]).
    std::size_t residual_offset(int32_t batch, int32_t row) const
    {
        return static_cast<std::size_t>(batch) * static_cast<std::size_t>(dimF) +
               static_cast<std::size_t>(row);
    }

    // Flat index of J value `pos` in case `batch` (d_J_values is [B * nnz_J]).
    std::size_t jacobian_offset(int32_t batch, int32_t pos) const
    {
        return static_cast<std::size_t>(batch) * static_cast<std::size_t>(nnz_J) +
               static_cast<std::size_t>(pos);
    }

    // Flat index of Ybus value `pos` in case `batch`. When the Ybus values are
    // shared across the batch (the common case) every case reads slot `pos`.
    std::size_t ybus_value_offset(int32_t batch, int32_t pos) const
    {
        return ybus_values_batched
            ? static_cast<std::size_t>(batch) * static_cast<std::size_t>(nnz_ybus) +
                  static_cast<std::size_t>(pos)
            : static_cast<std::size_t>(pos);
    }

    // --- Ybus (pattern shared across batch; values shared unless batched) ----
    DeviceBuffer<StateScalar> d_Ybus_re;
    DeviceBuffer<StateScalar> d_Ybus_im;
    DeviceBuffer<int32_t>     d_Ybus_indptr;   // CSR row pointers   [n_bus + 1]
    DeviceBuffer<int32_t>     d_Ybus_indices;  // CSR column indices [nnz_ybus]
    DeviceBuffer<int32_t>     d_Ybus_row;      // CSR row of each nz [nnz_ybus]

    // --- Jacobian (FP precision = JacScalar; pattern shared, values batched) -
    DeviceBuffer<JacScalar>   d_J_values;      // [B * nnz_J]
    DeviceBuffer<int32_t>     d_J_row_ptr;
    DeviceBuffer<int32_t>     d_J_col_idx;

    // --- Newton residual / step ----------------------------------------------
    DeviceBuffer<StateScalar> d_F;             // mismatch residual [B * dimF]
    DeviceBuffer<StateScalar> d_normF;         // per-case L-inf norm [B]
    DeviceBuffer<JacScalar>   d_dx;            // linear-solve step  [B * dimF]

    // --- Voltage state (polar Va/Vm authoritative; rectangular V cached) -----
    DeviceBuffer<StateScalar> d_Va;            // [B * n_bus]
    DeviceBuffer<StateScalar> d_Vm;            // [B * n_bus]
    DeviceBuffer<StateScalar> d_V_re;          // [B * n_bus]
    DeviceBuffer<StateScalar> d_V_im;          // [B * n_bus]

    // --- Power injection / bus current ---------------------------------------
    DeviceBuffer<StateScalar> d_Sbus_re;       // [B * n_bus]
    DeviceBuffer<StateScalar> d_Sbus_im;       // [B * n_bus]
    DeviceBuffer<StateScalar> d_Ibus_re;       // [B * n_bus] (cached Ybus*V)
    DeviceBuffer<StateScalar> d_Ibus_im;       // [B * n_bus]

    // --- Scatter maps / bus-type indices (precision-independent) --------------
    DeviceBuffer<int32_t> d_mapJ11, d_mapJ12, d_mapJ21, d_mapJ22;
    DeviceBuffer<int32_t> d_diagJ11, d_diagJ12, d_diagJ21, d_diagJ22;
    DeviceBuffer<int32_t> d_pvpq, d_pv, d_pq;

    int32_t n_bus  = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq   = 0;
    int32_t dimF   = 0;
    int32_t nnz_ybus = 0;
    int32_t nnz_J    = 0;
    int32_t batch_size = 1;
    bool    ybus_values_batched = false;
};


// Uniform accessors for the batch size and per-case Jacobian nnz of any CUDA
// storage profile (FP32/FP64/Mixed all derive from CudaBatchedStorage and
// expose these members). Defined once here so the value cannot drift per TU —
// these used to be copy-pasted into each .cpp as overload sets, and the FP64
// batch flag once went out of sync across those copies.
template <typename Storage>
int32_t cuda_storage_batch_size(const Storage& b) { return b.batch_size; }

template <typename Storage>
int32_t cuda_storage_nnz_j(const Storage& b) { return b.nnz_J; }  // per case, not batch-multiplied

#endif  // CUPF_WITH_CUDA
