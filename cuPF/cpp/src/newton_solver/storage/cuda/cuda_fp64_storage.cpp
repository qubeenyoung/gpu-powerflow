#ifdef CUPF_WITH_CUDA

#include "cuda_fp64_storage.hpp"

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

void upload_complex_components(DeviceBuffer<double>& dst_re,
                               DeviceBuffer<double>& dst_im,
                               const std::complex<double>* src,
                               int32_t count)
{
    std::vector<double> h_re(static_cast<std::size_t>(count));
    std::vector<double> h_im(static_cast<std::size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        h_re[static_cast<std::size_t>(i)] = src[i].real();
        h_im[static_cast<std::size_t>(i)] = src[i].imag();
    }
    dst_re.assign(h_re.data(), h_re.size());
    dst_im.assign(h_im.data(), h_im.size());
}

}  // namespace


void CudaFp64Buffers::prepare(const InitializeContext& ctx)
{
    require_pointer(ctx.ybus.indptr,  "InitializeContext.ybus.indptr",  ctx.ybus.rows + 1);
    require_pointer(ctx.ybus.indices, "InitializeContext.ybus.indices", ctx.ybus.nnz);
    require_pointer(ctx.pv, "InitializeContext.pv", ctx.n_pv);
    require_pointer(ctx.pq, "InitializeContext.pq", ctx.n_pq);

    n_bus  = ctx.n_bus;
    n_pvpq = ctx.maps.n_pvpq;
    n_pq   = ctx.maps.n_pq;
    dimF   = n_pvpq + n_pq;

    const int32_t nnz_ybus = ctx.ybus.nnz;
    const int32_t nnz_J    = ctx.J.nnz;

    d_Ybus_re.resize(nnz_ybus);
    d_Ybus_im.resize(nnz_ybus);
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


void CudaFp64Buffers::upload(const SolveContext& ctx)
{
    if (ctx.ybus == nullptr || ctx.sbus == nullptr || ctx.V0 == nullptr) {
        throw std::invalid_argument("CudaFp64Buffers::upload: solve context is incomplete");
    }

    const YbusView& ybus = *ctx.ybus;
    if (ybus.rows != n_bus || ybus.cols != n_bus ||
        ybus.nnz != static_cast<int32_t>(d_Ybus_re.size())) {
        throw std::runtime_error("CudaFp64Buffers::upload: Ybus dimensions do not match initialize()");
    }

    require_pointer(ybus.data,  "SolveContext.ybus->data", ybus.nnz);
    require_pointer(ctx.sbus,   "SolveContext.sbus",       n_bus);
    require_pointer(ctx.V0,     "SolveContext.V0",         n_bus);

    upload_complex_components(d_Ybus_re, d_Ybus_im, ybus.data, ybus.nnz);
    upload_complex_components(d_Sbus_re, d_Sbus_im, ctx.sbus,  n_bus);
    upload_complex_components(d_V_re,    d_V_im,    ctx.V0,    n_bus);

    std::vector<double> h_Va(static_cast<std::size_t>(n_bus));
    std::vector<double> h_Vm(static_cast<std::size_t>(n_bus));
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        h_Va[static_cast<std::size_t>(bus)] = std::arg(ctx.V0[bus]);
        h_Vm[static_cast<std::size_t>(bus)] = std::abs(ctx.V0[bus]);
    }
    d_Va.assign(h_Va.data(), h_Va.size());
    d_Vm.assign(h_Vm.data(), h_Vm.size());

    d_F.memsetZero();
    d_normF.memsetZero();
    d_dx.memsetZero();
    d_Ibus_re.memsetZero();
    d_Ibus_im.memsetZero();
}


void CudaFp64Buffers::download(NRResult& result) const
{
    std::vector<double> h_re(static_cast<std::size_t>(n_bus));
    std::vector<double> h_im(static_cast<std::size_t>(n_bus));

    d_V_re.copyTo(h_re.data(), h_re.size());
    d_V_im.copyTo(h_im.data(), h_im.size());

    result.V.resize(static_cast<std::size_t>(n_bus));
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        result.V[static_cast<std::size_t>(bus)] = std::complex<double>(
            h_re[static_cast<std::size_t>(bus)],
            h_im[static_cast<std::size_t>(bus)]);
    }
}

#endif  // CUPF_WITH_CUDA
