// ---------------------------------------------------------------------------
// cuda_batched_storage.cpp
//
// One templated implementation of the CUDA batched device storage, shared by
// all three precision profiles via explicit instantiation at the bottom:
//
//     CudaFp32Storage  = CudaBatchedStorage<float , float >
//     CudaFp64Storage  = CudaBatchedStorage<double, double>
//     CudaMixedStorage = CudaBatchedStorage<double, float >
//
// StateScalar drives the physical-state buffers (Ybus/V/Va/Vm/Sbus/Ibus/F/
// normF); JacScalar drives the linear-solve buffers (d_J_values/d_dx). The host
// inputs are always FP64 (std::complex<double>); upload casts down to
// StateScalar and download casts back up. See cuda_batched_storage.hpp for the
// batch-major layout contract.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_batched_storage.hpp"
#include "newton_solver/storage/cuda/storage_convert.hpp"
#include "utils/cuda_utils.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>


namespace {

// Validate that a caller-supplied array pointer is non-null whenever it is
// expected to point at `count` elements. A null pointer with count == 0 is
// allowed (empty array), so we only throw when count > 0. `name` is echoed in
// the message so the caller can tell which input was missing.
template <typename T>
void require_pointer(const T* ptr, const char* name, int32_t count)
{
    if (count > 0 && ptr == nullptr) {
        throw std::invalid_argument(std::string(name) + " must not be null");
    }
}

// Ybus is stored CSR: indptr[row] gives the row's first nonzero and indices[k]
// gives each nonzero's *column*, but the row is implicit (you only know it from
// which indptr bucket k falls in). The edge-parallel kernels (Ibus, Jacobian)
// launch one thread per nonzero `k` and need that nonzero's *row* in O(1), so
// here we materialize the inverse: rows[k] = source row of nonzero k. Built once
// in prepare() and uploaded to d_Ybus_row.
std::vector<int32_t> build_ybus_row_index(const YbusView& ybus)
{
    std::vector<int32_t> rows(ybus.nnz, 0);
    for (int32_t row = 0; row < ybus.rows; ++row) {
        // Every nonzero in [indptr[row], indptr[row+1]) belongs to this row.
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
            rows[k] = row;
        }
    }
    return rows;
}

// Host-side split of a batched complex array (e.g. Sbus) into separate real/imag
// device buffers at the storage precision (cast double -> StorageScalar).
//
// Only the strided fallback in upload() uses this; the common contiguous path
// converts on-device via launch_split_complex (no host copy). `logical_count`
// is the element count per case (n_bus); `stride` is the caller's per-case step
// in `src` (>= logical_count, may exceed it if the source is a sub-view of a
// wider buffer). Output is packed batch-major with stride == logical_count
// (no gaps), which is what the kernels assume.
template <typename StorageScalar>
void upload_complex_components(DeviceBuffer<StorageScalar>& dst_re,
                               DeviceBuffer<StorageScalar>& dst_im,
                               const std::complex<double>* src,
                               int32_t logical_count,
                               int32_t batch_size,
                               int64_t stride)
{
    const std::size_t total = batch_size * logical_count;
    std::vector<StorageScalar> h_re(total);
    std::vector<StorageScalar> h_im(total);

    for (int32_t b = 0; b < batch_size; ++b) {
        // Destination is gap-free (logical_count); source may be strided.
        const std::size_t dst_base = b * logical_count;
        const std::size_t src_base = b * stride;
        for (int32_t i = 0; i < logical_count; ++i) {
            const auto& val = src[src_base + i];
            h_re[dst_base + i] = static_cast<StorageScalar>(val.real());
            h_im[dst_base + i] = static_cast<StorageScalar>(val.imag());
        }
    }
    // Single bulk H2D per component (DeviceBuffer::assign = cudaMemcpy).
    dst_re.assign(h_re.data(), h_re.size());
    dst_im.assign(h_im.data(), h_im.size());
}

// Same re/im split for the Ybus *values*. Unlike Sbus/V0, Ybus is usually
// identical across the batch (same network, different injections), so by
// default we store a single shared copy (`stored == 1`) and every case's kernel
// reads slot k. Only when values_batched is set (per-case admittances, e.g. a
// contingency sweep that perturbs lines) do we store one full copy per case.
// The Ybus *sparsity pattern* is always shared and lives in the index buffers.
template <typename StorageScalar>
void upload_ybus_components(DeviceBuffer<StorageScalar>& dst_re,
                            DeviceBuffer<StorageScalar>& dst_im,
                            const std::complex<double>* src,
                            int32_t nnz_ybus,
                            int32_t batch_size,
                            int64_t stride,
                            bool values_batched)
{
    // How many copies of the nnz values we actually store.
    const int32_t stored = values_batched ? batch_size : 1;
    const std::size_t total = stored * nnz_ybus;
    std::vector<StorageScalar> h_re(total);
    std::vector<StorageScalar> h_im(total);

    for (int32_t b = 0; b < stored; ++b) {
        const std::size_t dst_base = b * nnz_ybus;
        const std::size_t src_base = values_batched ? b * stride : 0;
        for (int32_t k = 0; k < nnz_ybus; ++k) {
            const auto& val = src[src_base + k];
            h_re[dst_base + k] = static_cast<StorageScalar>(val.real());
            h_im[dst_base + k] = static_cast<StorageScalar>(val.imag());
        }
    }
    dst_re.assign(h_re.data(), h_re.size());
    dst_im.assign(h_im.data(), h_im.size());
}

}  // namespace


// Allocate device buffers and upload the time-invariant topology once: Ybus
// pattern + values, Jacobian pattern, and the scatter maps. Buffers are sized
// for a single case here; upload() resizes them for the actual batch. Per-solve
// values arrive later via upload().
template <typename StateScalar, typename JacScalar>
void CudaBatchedStorage<StateScalar, JacScalar>::prepare(const InitializeContext& ctx)
{
    // Inputs are caller-owned raw pointers; fail fast (with the field name) if a
    // required one is missing before we start allocating device memory.
    require_pointer(ctx.ybus.indptr,  "InitializeContext.ybus.indptr",  ctx.ybus.rows + 1);
    require_pointer(ctx.ybus.indices, "InitializeContext.ybus.indices", ctx.ybus.nnz);
    require_pointer(ctx.pv, "InitializeContext.pv", ctx.n_pv);
    require_pointer(ctx.pq, "InitializeContext.pq", ctx.n_pq);

    // Problem dimensions. dimF = n_pvpq + n_pq is the Newton system size:
    // n_pvpq angle equations (PV+PQ buses) plus n_pq magnitude equations.
    n_bus  = ctx.n_bus;
    n_pvpq = ctx.maps.n_pvpq;
    n_pq   = ctx.maps.n_pq;
    dimF   = n_pvpq + n_pq;

    // nnz counts are fixed by the sparsity pattern (constant across the solve).
    // Default to a single case; upload() promotes batch_size when B > 1.
    nnz_ybus = ctx.ybus.nnz;
    nnz_J    = ctx.J.nnz;
    batch_size = 1;
    ybus_values_batched = false;

    // --- Ybus: values (re/im) + CSR structure + per-nz row index -------------
    // Pattern and (by default) values are shared across the batch, so they are
    // uploaded once here and never touched again in upload() unless batched.
    d_Ybus_re.resize(nnz_ybus);
    d_Ybus_im.resize(nnz_ybus);
    upload_ybus_components(d_Ybus_re, d_Ybus_im, ctx.ybus.data,
                           nnz_ybus, 1, nnz_ybus, false);
    d_Ybus_indptr.assign(ctx.ybus.indptr,  ctx.ybus.rows + 1);
    d_Ybus_indices.assign(ctx.ybus.indices, nnz_ybus);

    // Inverse of the CSR pattern (nz -> row) for the edge-parallel kernels.
    const std::vector<int32_t> h_y_row = build_ybus_row_index(ctx.ybus);
    d_Ybus_row.assign(h_y_row.data(), h_y_row.size());

    // --- Jacobian: values buffer (filled each NR iter) + CSR structure -------
    // Only the structure is known now; d_J_values is sized to one case here and
    // resized to B*nnz_J in upload(). row_ptr/col_idx are shared across batch.
    d_J_values.resize(nnz_J);
    d_J_row_ptr.assign(ctx.J.row_ptr.data(), ctx.J.dim + 1);
    d_J_col_idx.assign(ctx.J.col_idx.data(), nnz_J);

    // --- Newton residual / norm / step (single-case sizing for now) ----------
    d_F.resize(dimF);
    d_normF.resize(1);
    d_dx.resize(dimF);

    // --- Voltage state: polar (Va, Vm) is authoritative, (V_re, V_im) cached --
    d_Va.resize(n_bus);
    d_Vm.resize(n_bus);
    d_V_re.resize(n_bus);
    d_V_im.resize(n_bus);

    // --- Power injection (Sbus) and bus current (Ibus = Ybus*V) cache --------
    d_Sbus_re.resize(n_bus);
    d_Sbus_im.resize(n_bus);
    d_Ibus_re.resize(n_bus);
    d_Ibus_im.resize(n_bus);

    // Scatter maps and bus-type index lists are pure topology (precision- and
    // batch-independent), so upload them once here and never again.
    const auto upload_map = [](DeviceBuffer<int32_t>& dst, const std::vector<int32_t>& src) {
        dst.assign(src.data(), src.size());
    };

    upload_map(d_mapJ11,  ctx.maps.mapJ11);   // off-diagonal J value positions
    upload_map(d_mapJ12,  ctx.maps.mapJ12);
    upload_map(d_mapJ21,  ctx.maps.mapJ21);
    upload_map(d_mapJ22,  ctx.maps.mapJ22);
    upload_map(d_diagJ11, ctx.maps.diagJ11);  // diagonal J value positions
    upload_map(d_diagJ12, ctx.maps.diagJ12);
    upload_map(d_diagJ21, ctx.maps.diagJ21);
    upload_map(d_diagJ22, ctx.maps.diagJ22);
    upload_map(d_pvpq,    ctx.maps.pvpq);      // PV+PQ bus order (Jacobian rows)

    d_pv.assign(ctx.pv, ctx.n_pv);
    d_pq.assign(ctx.pq, ctx.n_pq);

    // Zero the accumulators/outputs so a solve that reads before the first write
    // (or a dump on iteration 0) sees defined memory rather than malloc garbage.
    d_F.memsetZero();
    d_normF.memsetZero();
    d_dx.memsetZero();
    d_J_values.memsetZero();
    d_Ibus_re.memsetZero();
    d_Ibus_im.memsetZero();
}


// Push per-solve inputs: resize for the batch, upload Ybus/Sbus (cast to
// StateScalar), and seed V/Va/Vm from the V0 guess.
template <typename StateScalar, typename JacScalar>
void CudaBatchedStorage<StateScalar, JacScalar>::upload(const SolveContext& ctx)
{
    if (ctx.ybus == nullptr || ctx.sbus == nullptr || ctx.V0 == nullptr) {
        throw std::invalid_argument("CudaBatchedStorage::upload: solve context is incomplete");
    }

    // The sparsity pattern was baked into the device buffers (and the cuDSS
    // symbolic analysis) in prepare(); a different shape now would corrupt every
    // index buffer, so reject it rather than silently misbehave.
    const YbusView& ybus = *ctx.ybus;
    if (ybus.rows != n_bus || ybus.cols != n_bus || ybus.nnz != nnz_ybus) {
        throw std::runtime_error("CudaBatchedStorage::upload: Ybus dimensions do not match initialize()");
    }

    require_pointer(ybus.data, "SolveContext.ybus->data", ybus.nnz);
    require_pointer(ctx.sbus,  "SolveContext.sbus",       n_bus);
    require_pointer(ctx.V0,    "SolveContext.V0",         n_bus);

    if (ctx.batch_size <= 0) {
        throw std::invalid_argument("CudaBatchedStorage::upload: batch_size must be positive");
    }
    // Each case must span at least n_bus elements in the source; a smaller
    // stride would make case b overlap case b-1.
    if (ctx.sbus_stride < n_bus || ctx.V0_stride < n_bus) {
        throw std::invalid_argument("CudaBatchedStorage::upload: batch strides must be at least n_bus");
    }

    batch_size = ctx.batch_size;
    ybus_values_batched = ctx.ybus_values_batched;

    // Per-solve buffer sizes (batch-major totals). nnz_J/dimF/n_bus are fixed;
    // only batch_size varies between solves on the same initialized solver.
    const std::size_t bus_count      = batch_size * n_bus;
    const std::size_t residual_count = batch_size * dimF;
    const std::size_t jacobian_count = batch_size * nnz_J;

    // Grow (or shrink) only when the batch size actually changed — repeated
    // solves at the same B reuse the existing device allocations (no churn).
    const auto ensure_size = [](auto& buf, std::size_t count) {
        if (buf.size() != count) buf.resize(count);
    };

    ensure_size(d_J_values,  jacobian_count);   // [B * nnz_J]
    ensure_size(d_F,         residual_count);   // [B * dimF]
    ensure_size(d_normF,     batch_size);  // [B]
    ensure_size(d_dx,        residual_count);   // [B * dimF]
    ensure_size(d_Va,        bus_count);        // [B * n_bus] (and the rest)
    ensure_size(d_Vm,        bus_count);
    ensure_size(d_V_re,      bus_count);
    ensure_size(d_V_im,      bus_count);
    ensure_size(d_Sbus_re,   bus_count);
    ensure_size(d_Sbus_im,   bus_count);
    ensure_size(d_Ibus_re,   bus_count);
    ensure_size(d_Ibus_im,   bus_count);

    // Refresh Ybus values (pattern is unchanged from prepare()). Shared across
    // the batch unless ybus_values_batched.
    upload_ybus_components(d_Ybus_re, d_Ybus_im, ybus.data,
                           ybus.nnz, batch_size, ctx.ybus_value_stride, ybus_values_batched);

    const int32_t total_bus = batch_size * n_bus;

    // Fast path (contiguous batch, stride == n_bus): bulk H2D the raw FP64
    // Sbus/V0 once and convert/cast on device, instead of per-element host
    // cast + arg/abs loops that scale with batch_size * n_bus. The device
    // kernels emit StateScalar (float for FP32, double for FP64/Mixed).
    if (ctx.sbus_stride == n_bus && ctx.V0_stride == n_bus) {
        // std::complex<double> is layout-compatible with two contiguous doubles
        // (re, im), so a B*n_bus complex array reinterprets as 2*B*n_bus doubles
        // with no copy. We H2D that raw block once into a scratch buffer and let
        // the device kernels do the re/im split and the polar conversion.
        DeviceBuffer<double> d_raw;
        d_raw.resize(total_bus * 2);

        // Sbus: just split interleaved (re,im) into d_Sbus_re / d_Sbus_im.
        d_raw.assign(reinterpret_cast<const double*>(ctx.sbus),
                     total_bus * 2);
        launch_split_complex<StateScalar>(d_raw.data(), d_Sbus_re.data(), d_Sbus_im.data(), total_bus);

        // V0: split into rectangular (V_re,V_im) AND seed polar state
        // (Va = atan2(im,re), Vm = hypot(re,im)) in one kernel pass.
        d_raw.assign(reinterpret_cast<const double*>(ctx.V0),
                     total_bus * 2);
        launch_seed_state_from_v0<StateScalar>(d_raw.data(), d_V_re.data(), d_V_im.data(),
                                               d_Va.data(), d_Vm.data(), total_bus);
    } else {
        // General (strided) fallback: the source isn't a clean contiguous block
        // (each case is sub-strided), so we can't reinterpret-and-bulk-copy.
        // Convert on the host instead, packing into gap-free batch-major order.
        upload_complex_components(d_Sbus_re, d_Sbus_im, ctx.sbus, n_bus, batch_size, ctx.sbus_stride);

        std::vector<StateScalar> h_V_re(bus_count);
        std::vector<StateScalar> h_V_im(bus_count);
        std::vector<StateScalar> h_Va(bus_count);
        std::vector<StateScalar> h_Vm(bus_count);
        for (int32_t b = 0; b < batch_size; ++b) {
            const std::size_t dst_base = b * n_bus;
            const std::size_t src_base = b * ctx.V0_stride;
            for (int32_t bus = 0; bus < n_bus; ++bus) {
                const auto& v0  = ctx.V0[src_base + bus];
                const std::size_t dst = dst_base + bus;
                // Keep both representations: rectangular for the mismatch/Ibus
                // kernels, polar for the Newton angle/magnitude state.
                h_V_re[dst] = static_cast<StateScalar>(v0.real());
                h_V_im[dst] = static_cast<StateScalar>(v0.imag());
                h_Va[dst]   = static_cast<StateScalar>(std::arg(v0));   // angle
                h_Vm[dst]   = static_cast<StateScalar>(std::abs(v0));   // magnitude
            }
        }
        d_V_re.assign(h_V_re.data(), h_V_re.size());
        d_V_im.assign(h_V_im.data(), h_V_im.size());
        d_Va.assign(h_Va.data(), h_Va.size());
        d_Vm.assign(h_Vm.data(), h_Vm.size());
    }

    // Reset per-iteration accumulators for this fresh solve: residual, its norm,
    // the step, the Jacobian values, and the cached Ibus (recomputed each iter).
    d_F.memsetZero();
    d_normF.memsetZero();
    d_dx.memsetZero();
    d_J_values.memsetZero();
    d_Ibus_re.memsetZero();
    d_Ibus_im.memsetZero();
}


// Single-case voltage readback: pack Re/Im into interleaved complex<double> on
// device (up-cast from StateScalar), then one bulk D2H.
template <typename StateScalar, typename JacScalar>
void CudaBatchedStorage<StateScalar, JacScalar>::download(NRResult& result) const
{
    if (batch_size != 1) {
        throw std::runtime_error("CudaBatchedStorage::download: use download_batch for batch_size > 1");
    }

    result.V.resize(n_bus);
    DeviceBuffer<double> d_out;
    d_out.resize(n_bus * 2);
    launch_pack_complex_to_double<StateScalar>(d_V_re.data(), d_V_im.data(), d_out.data(), n_bus);
    d_out.copyTo(reinterpret_cast<double*>(result.V.data()), n_bus * 2);
}


// Batched voltage readback (StateScalar -> FP64) plus per-case final mismatch
// norms. Layout is batch-major: result.V[b * n_bus + bus].
template <typename StateScalar, typename JacScalar>
void CudaBatchedStorage<StateScalar, JacScalar>::download_batch(NRBatchResult& result) const
{
    const std::size_t total = batch_size * n_bus;

    result.n_bus      = n_bus;
    result.batch_size = batch_size;
    result.V.resize(total);

    // Pack reconstructed d_V_re/d_V_im to interleaved complex<double> on device,
    // then one bulk D2H (replaces an O(batch*n_bus) host interleave loop).
    DeviceBuffer<double> d_out;
    d_out.resize(total * 2);
    launch_pack_complex_to_double<StateScalar>(d_V_re.data(), d_V_im.data(), d_out.data(),
                                               static_cast<int32_t>(total));
    d_out.copyTo(reinterpret_cast<double*>(result.V.data()), total * 2);

    if (!d_normF.empty()) {
        // d_normF holds one StateScalar per case; widen each to the public FP64
        // result (the cast is the identity when StateScalar == double).
        std::vector<StateScalar> h_norm(batch_size);
        d_normF.copyTo(h_norm.data(), h_norm.size());
        result.final_mismatch.resize(batch_size);
        for (int32_t b = 0; b < batch_size; ++b) {
            result.final_mismatch[b] = static_cast<double>(h_norm[b]);
        }
    }
}


// --- Explicit instantiations for the three CUDA precision profiles ----------
template struct CudaBatchedStorage<float,  float>;   // CudaFp32Storage
template struct CudaBatchedStorage<double, double>;  // CudaFp64Storage
template struct CudaBatchedStorage<double, float>;   // CudaMixedStorage

#endif  // CUPF_WITH_CUDA
