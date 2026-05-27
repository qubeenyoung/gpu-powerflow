#ifdef CUPF_WITH_CUDA

#include "cuda_fp32_storage.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>


namespace {

template <typename T>
void require_pointer(const T* ptr, const char* name, int32_t count)
{
    if (count > 0 && ptr == nullptr) {
        throw std::invalid_argument(std::string(name) + " must not be null");
    }
}

std::vector<int32_t> build_ybus_row_index(const YbusView& ybus)
{
    std::vector<int32_t> rows(static_cast<std::size_t>(ybus.nnz), 0);
    for (int32_t row = 0; row < ybus.rows; ++row) {
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
            rows[static_cast<std::size_t>(k)] = row;
        }
    }
    return rows;
}

void upload_complex_components(DeviceBuffer<float>& dst_re,
                               DeviceBuffer<float>& dst_im,
                               const std::complex<double>* src,
                               int32_t logical_count,
                               int32_t batch_size,
                               int64_t stride)
{
    const std::size_t total =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(logical_count);
    std::vector<float> h_re(total);
    std::vector<float> h_im(total);

    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t dst_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(logical_count);
        const std::size_t src_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(stride);
        for (int32_t i = 0; i < logical_count; ++i) {
            const auto& val = src[src_base + static_cast<std::size_t>(i)];
            h_re[dst_base + static_cast<std::size_t>(i)] = static_cast<float>(val.real());
            h_im[dst_base + static_cast<std::size_t>(i)] = static_cast<float>(val.imag());
        }
    }
    dst_re.assign(h_re.data(), h_re.size());
    dst_im.assign(h_im.data(), h_im.size());
}

void upload_ybus_components(DeviceBuffer<float>& dst_re,
                            DeviceBuffer<float>& dst_im,
                            const std::complex<double>* src,
                            int32_t nnz_ybus,
                            int32_t batch_size,
                            int64_t stride,
                            bool values_batched)
{
    const int32_t stored = values_batched ? batch_size : 1;
    const std::size_t total =
        static_cast<std::size_t>(stored) * static_cast<std::size_t>(nnz_ybus);
    std::vector<float> h_re(total);
    std::vector<float> h_im(total);

    for (int32_t b = 0; b < stored; ++b) {
        const std::size_t dst_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(nnz_ybus);
        const std::size_t src_base =
            values_batched ? static_cast<std::size_t>(b) * static_cast<std::size_t>(stride) : 0;
        for (int32_t k = 0; k < nnz_ybus; ++k) {
            const auto& val = src[src_base + static_cast<std::size_t>(k)];
            h_re[dst_base + static_cast<std::size_t>(k)] = static_cast<float>(val.real());
            h_im[dst_base + static_cast<std::size_t>(k)] = static_cast<float>(val.imag());
        }
    }
    dst_re.assign(h_re.data(), h_re.size());
    dst_im.assign(h_im.data(), h_im.size());
}

}  // namespace


void CudaFp32Buffers::prepare(const InitializeContext& ctx)
{
    require_pointer(ctx.ybus.indptr,  "InitializeContext.ybus.indptr",  ctx.ybus.rows + 1);
    require_pointer(ctx.ybus.indices, "InitializeContext.ybus.indices", ctx.ybus.nnz);
    require_pointer(ctx.pv, "InitializeContext.pv", ctx.n_pv);
    require_pointer(ctx.pq, "InitializeContext.pq", ctx.n_pq);

    n_bus  = ctx.n_bus;
    n_pvpq = ctx.maps.n_pvpq;
    n_pq   = ctx.maps.n_pq;
    dimF   = n_pvpq + n_pq;

    nnz_ybus = ctx.ybus.nnz;
    nnz_J    = ctx.J.nnz;
    batch_size = 1;
    ybus_values_batched = false;

    d_Ybus_re.resize(nnz_ybus);
    d_Ybus_im.resize(nnz_ybus);
    upload_ybus_components(d_Ybus_re, d_Ybus_im, ctx.ybus.data,
                           nnz_ybus, 1, nnz_ybus, false);
    d_Ybus_indptr.assign(ctx.ybus.indptr,  static_cast<std::size_t>(ctx.ybus.rows + 1));
    d_Ybus_indices.assign(ctx.ybus.indices, static_cast<std::size_t>(nnz_ybus));

    const std::vector<int32_t> h_y_row = build_ybus_row_index(ctx.ybus);
    d_Y_row.assign(h_y_row.data(), h_y_row.size());

    d_J_values.resize(nnz_J);
    d_J_row_ptr.assign(ctx.J.row_ptr.data(), static_cast<std::size_t>(ctx.J.dim + 1));
    d_J_col_idx.assign(ctx.J.col_idx.data(), static_cast<std::size_t>(nnz_J));

    d_F.resize(dimF);
    d_normF.resize(1);
    d_dx.resize(dimF);

    d_Va.resize(n_bus);
    d_Vm.resize(n_bus);
    d_V_re.resize(n_bus);
    d_V_im.resize(n_bus);

    d_Sbus_re.resize(n_bus);
    d_Sbus_im.resize(n_bus);
    d_Ibus_re.resize(n_bus);
    d_Ibus_im.resize(n_bus);

    const auto upload_map = [](DeviceBuffer<int32_t>& dst, const std::vector<int32_t>& src) {
        dst.assign(src.data(), src.size());
    };

    upload_map(d_mapJ11,  ctx.maps.mapJ11);
    upload_map(d_mapJ12,  ctx.maps.mapJ12);
    upload_map(d_mapJ21,  ctx.maps.mapJ21);
    upload_map(d_mapJ22,  ctx.maps.mapJ22);
    upload_map(d_diagJ11, ctx.maps.diagJ11);
    upload_map(d_diagJ12, ctx.maps.diagJ12);
    upload_map(d_diagJ21, ctx.maps.diagJ21);
    upload_map(d_diagJ22, ctx.maps.diagJ22);
    upload_map(d_pvpq,    ctx.maps.pvpq);

    d_pv.assign(ctx.pv, static_cast<std::size_t>(ctx.n_pv));
    d_pq.assign(ctx.pq, static_cast<std::size_t>(ctx.n_pq));

    d_F.memsetZero();
    d_normF.memsetZero();
    d_dx.memsetZero();
    d_J_values.memsetZero();
    d_Ibus_re.memsetZero();
    d_Ibus_im.memsetZero();
}


void CudaFp32Buffers::upload(const SolveContext& ctx)
{
    if (ctx.ybus == nullptr || ctx.sbus == nullptr || ctx.V0 == nullptr) {
        throw std::invalid_argument("CudaFp32Buffers::upload: solve context is incomplete");
    }

    const YbusView& ybus = *ctx.ybus;
    if (ybus.rows != n_bus || ybus.cols != n_bus || ybus.nnz != nnz_ybus) {
        throw std::runtime_error("CudaFp32Buffers::upload: Ybus dimensions do not match initialize()");
    }

    require_pointer(ybus.data, "SolveContext.ybus->data", ybus.nnz);
    require_pointer(ctx.sbus,  "SolveContext.sbus",       n_bus);
    require_pointer(ctx.V0,    "SolveContext.V0",         n_bus);

    if (ctx.batch_size <= 0) {
        throw std::invalid_argument("CudaFp32Buffers::upload: batch_size must be positive");
    }
    if (ctx.sbus_stride < n_bus || ctx.V0_stride < n_bus) {
        throw std::invalid_argument("CudaFp32Buffers::upload: batch strides must be at least n_bus");
    }

    batch_size = ctx.batch_size;
    ybus_values_batched = ctx.ybus_values_batched;

    const std::size_t bus_count      = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(n_bus);
    const std::size_t residual_count = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(dimF);
    const std::size_t jacobian_count = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(nnz_J);

    const auto ensure_size = [](auto& buf, std::size_t count) {
        if (buf.size() != count) buf.resize(count);
    };

    ensure_size(d_J_values,  jacobian_count);
    ensure_size(d_F,         residual_count);
    ensure_size(d_normF,     static_cast<std::size_t>(batch_size));
    ensure_size(d_dx,        residual_count);
    ensure_size(d_Va,        bus_count);
    ensure_size(d_Vm,        bus_count);
    ensure_size(d_V_re,      bus_count);
    ensure_size(d_V_im,      bus_count);
    ensure_size(d_Sbus_re,   bus_count);
    ensure_size(d_Sbus_im,   bus_count);
    ensure_size(d_Ibus_re,   bus_count);
    ensure_size(d_Ibus_im,   bus_count);

    upload_ybus_components(d_Ybus_re, d_Ybus_im, ybus.data,
                           ybus.nnz, batch_size, ctx.ybus_value_stride, ybus_values_batched);
    upload_complex_components(d_Sbus_re, d_Sbus_im, ctx.sbus, n_bus, batch_size, ctx.sbus_stride);

    std::vector<float> h_V_re(bus_count);
    std::vector<float> h_V_im(bus_count);
    std::vector<float> h_Va(bus_count);
    std::vector<float> h_Vm(bus_count);
    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t dst_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n_bus);
        const std::size_t src_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(ctx.V0_stride);
        for (int32_t bus = 0; bus < n_bus; ++bus) {
            const auto& v0  = ctx.V0[src_base + static_cast<std::size_t>(bus)];
            const std::size_t dst = dst_base + static_cast<std::size_t>(bus);
            h_V_re[dst] = static_cast<float>(v0.real());
            h_V_im[dst] = static_cast<float>(v0.imag());
            h_Va[dst]   = static_cast<float>(std::arg(v0));
            h_Vm[dst]   = static_cast<float>(std::abs(v0));
        }
    }
    d_V_re.assign(h_V_re.data(), h_V_re.size());
    d_V_im.assign(h_V_im.data(), h_V_im.size());
    d_Va.assign(h_Va.data(), h_Va.size());
    d_Vm.assign(h_Vm.data(), h_Vm.size());

    d_F.memsetZero();
    d_normF.memsetZero();
    d_dx.memsetZero();
    d_J_values.memsetZero();
    d_Ibus_re.memsetZero();
    d_Ibus_im.memsetZero();
}


void CudaFp32Buffers::download(NRResult& result) const
{
    if (batch_size != 1) {
        throw std::runtime_error("CudaFp32Buffers::download: use download_batch for batch_size > 1");
    }

    std::vector<float> h_re(static_cast<std::size_t>(n_bus));
    std::vector<float> h_im(static_cast<std::size_t>(n_bus));

    d_V_re.copyTo(h_re.data(), h_re.size());
    d_V_im.copyTo(h_im.data(), h_im.size());

    result.V.resize(static_cast<std::size_t>(n_bus));
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        result.V[static_cast<std::size_t>(bus)] = std::complex<double>(
            static_cast<double>(h_re[static_cast<std::size_t>(bus)]),
            static_cast<double>(h_im[static_cast<std::size_t>(bus)]));
    }
}


void CudaFp32Buffers::download_batch(NRBatchResult& result) const
{
    const std::size_t total = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(n_bus);
    std::vector<float> h_re(total);
    std::vector<float> h_im(total);

    d_V_re.copyTo(h_re.data(), h_re.size());
    d_V_im.copyTo(h_im.data(), h_im.size());

    result.n_bus      = n_bus;
    result.batch_size = batch_size;
    result.V.resize(total);

    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n_bus);
        for (int32_t bus = 0; bus < n_bus; ++bus) {
            const std::size_t idx = base + static_cast<std::size_t>(bus);
            result.V[idx] = std::complex<double>(
                static_cast<double>(h_re[idx]),
                static_cast<double>(h_im[idx]));
        }
    }

    if (!d_normF.empty()) {
        std::vector<float> h_norm(static_cast<std::size_t>(batch_size));
        d_normF.copyTo(h_norm.data(), h_norm.size());
        result.final_mismatch.resize(static_cast<std::size_t>(batch_size));
        for (int32_t b = 0; b < batch_size; ++b) {
            result.final_mismatch[static_cast<std::size_t>(b)] =
                static_cast<double>(h_norm[static_cast<std::size_t>(b)]);
        }
    }
}

#endif  // CUPF_WITH_CUDA
