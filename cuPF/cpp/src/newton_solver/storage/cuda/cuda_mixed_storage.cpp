// ---------------------------------------------------------------------------
// cuda_mixed_storage.cpp — CUDA Mixed 정밀도 Storage 구현
//
// cuda_fp64_storage.cpp 와 같은 public FP64 API를 받지만, optimized mixed
// profile 내부 dtype과 batch-major runtime buffers를 관리한다.
//
// 주요 차이점:
//   - d_Ybus_re/im : FP32 data
//   - d_V_re/im : FP64 derived voltage cache
//   - d_Sbus_re/im, d_Ibus_re/im : FP64 specified power/current cache
//   - d_J_values, d_dx : FP32 Jacobian/linear-solve data
//   - d_Va/Vm, d_F : FP64 authoritative voltage state and residual
//
// upload_complex_components<DeviceScalar>():
//   public complex<double>[] 입력을 batch-major SoA device buffer로 변환한다.
//   DeviceScalar = float  : Ybus
//   DeviceScalar = double : Sbus 업로드
//
// prepare() / upload() / download_result() 의 역할은 cuda_fp64_storage.cpp 와 동일.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_mixed_storage.hpp"

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

// CSR indptr → 행 번호 배열 생성 (cuda_fp64_storage.cpp 와 동일).
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

// std::complex<double>[] → batch-major 실수·허수 분리 후 device 업로드.
template <typename DeviceScalar>
void upload_complex_components(DeviceBuffer<DeviceScalar>& dst_re,
                               DeviceBuffer<DeviceScalar>& dst_im,
                               const std::complex<double>* src,
                               int32_t logical_count,
                               int32_t batch_size,
                               int64_t stride)
{
    const std::size_t total_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(logical_count);
    std::vector<DeviceScalar> h_re(total_count);
    std::vector<DeviceScalar> h_im(total_count);

    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t dst_batch =
            static_cast<std::size_t>(b) * static_cast<std::size_t>(logical_count);
        const std::size_t src_batch = static_cast<std::size_t>(b) * static_cast<std::size_t>(stride);
        for (int32_t i = 0; i < logical_count; ++i) {
            const std::complex<double>& value = src[src_batch + static_cast<std::size_t>(i)];
            h_re[dst_batch + static_cast<std::size_t>(i)] = static_cast<DeviceScalar>(value.real());
            h_im[dst_batch + static_cast<std::size_t>(i)] = static_cast<DeviceScalar>(value.imag());
        }
    }
    dst_re.assign(h_re.data(), h_re.size());
    dst_im.assign(h_im.data(), h_im.size());
}

// Ybus 값은 batch 공통이 기본이다. batch별 값 경로는 같은 buffer 이름을
// [B * nnz_ybus] layout으로 확장한다.
template <typename DeviceScalar>
void upload_ybus_components(DeviceBuffer<DeviceScalar>& dst_re,
                            DeviceBuffer<DeviceScalar>& dst_im,
                            const std::complex<double>* src,
                            int32_t nnz_ybus,
                            int32_t batch_size,
                            int64_t stride,
                            bool values_batched)
{
    const int32_t stored_batches = values_batched ? batch_size : 1;
    const std::size_t total_count =
        static_cast<std::size_t>(stored_batches) * static_cast<std::size_t>(nnz_ybus);
    std::vector<DeviceScalar> h_re(total_count);
    std::vector<DeviceScalar> h_im(total_count);

    for (int32_t b = 0; b < stored_batches; ++b) {
        const std::size_t dst_batch =
            static_cast<std::size_t>(b) * static_cast<std::size_t>(nnz_ybus);
        const std::size_t src_batch =
            values_batched ? static_cast<std::size_t>(b) * static_cast<std::size_t>(stride) : 0;
        for (int32_t k = 0; k < nnz_ybus; ++k) {
            const std::complex<double>& value = src[src_batch + static_cast<std::size_t>(k)];
            h_re[dst_batch + static_cast<std::size_t>(k)] = static_cast<DeviceScalar>(value.real());
            h_im[dst_batch + static_cast<std::size_t>(k)] = static_cast<DeviceScalar>(value.imag());
        }
    }

    dst_re.assign(h_re.data(), h_re.size());
    dst_im.assign(h_im.data(), h_im.size());
}

}  // namespace


CudaMixedStorage::CudaMixedStorage()  = default;
CudaMixedStorage::~CudaMixedStorage() = default;


void CudaMixedStorage::prepare(const AnalyzeContext& ctx)
{
    require_pointer(ctx.ybus.indptr, "AnalyzeContext.ybus.indptr", ctx.ybus.rows + 1);
    require_pointer(ctx.ybus.indices, "AnalyzeContext.ybus.indices", ctx.ybus.nnz);
    require_pointer(ctx.pv, "AnalyzeContext.pv", ctx.n_pv);
    require_pointer(ctx.pq, "AnalyzeContext.pq", ctx.n_pq);

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
    d_Ybus_indptr.assign(ctx.ybus.indptr, static_cast<std::size_t>(ctx.ybus.rows + 1));
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


void CudaMixedStorage::upload(const SolveContext& ctx)
{
    if (ctx.ybus == nullptr || ctx.sbus == nullptr || ctx.V0 == nullptr) {
        throw std::invalid_argument("CudaMixedStorage::upload: solve context is incomplete");
    }

    const YbusViewF64& ybus = *ctx.ybus;
    if (ybus.rows != n_bus || ybus.cols != n_bus ||
        ybus.nnz != nnz_ybus) {
        throw std::runtime_error("CudaMixedStorage::upload: Ybus dimensions do not match analyze()");
    }

    require_pointer(ybus.data, "SolveContext.ybus->data", ybus.nnz);
    require_pointer(ctx.sbus, "SolveContext.sbus", n_bus);
    require_pointer(ctx.V0, "SolveContext.V0", n_bus);
    if (ctx.batch_size <= 0) {
        throw std::invalid_argument("CudaMixedStorage::upload: batch_size must be positive");
    }
    if (ctx.sbus_stride < n_bus || ctx.V0_stride < n_bus) {
        throw std::invalid_argument("CudaMixedStorage::upload: batch strides must be at least n_bus");
    }
    if (ctx.ybus_values_batched && ctx.ybus_value_stride < ybus.nnz) {
        throw std::invalid_argument("CudaMixedStorage::upload: ybus_value_stride must be at least nnz");
    }

    batch_size = ctx.batch_size;
    ybus_values_batched = ctx.ybus_values_batched;

    const std::size_t bus_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(n_bus);
    const std::size_t residual_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(dimF);
    const std::size_t jacobian_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(nnz_J);

    const auto ensure_size = [](auto& buffer, std::size_t count) {
        if (buffer.size() != count) {
            buffer.resize(count);
        }
    };

    ensure_size(d_J_values, jacobian_count);
    ensure_size(d_F, residual_count);
    ensure_size(d_normF, static_cast<std::size_t>(batch_size));
    ensure_size(d_dx, residual_count);
    ensure_size(d_Va, bus_count);
    ensure_size(d_Vm, bus_count);
    ensure_size(d_V_re, bus_count);
    ensure_size(d_V_im, bus_count);
    ensure_size(d_Sbus_re, bus_count);
    ensure_size(d_Sbus_im, bus_count);
    ensure_size(d_Ibus_re, bus_count);
    ensure_size(d_Ibus_im, bus_count);

    upload_ybus_components(
        d_Ybus_re,
        d_Ybus_im,
        ybus.data,
        ybus.nnz,
        batch_size,
        ctx.ybus_value_stride,
        ybus_values_batched);
    upload_complex_components(d_Sbus_re, d_Sbus_im, ctx.sbus, n_bus, batch_size, ctx.sbus_stride);

    std::vector<double> h_V_re(bus_count);
    std::vector<double> h_V_im(bus_count);
    std::vector<double> h_Va(bus_count);
    std::vector<double> h_Vm(bus_count);
    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t dst_batch =
            static_cast<std::size_t>(b) * static_cast<std::size_t>(n_bus);
        const std::size_t src_batch =
            static_cast<std::size_t>(b) * static_cast<std::size_t>(ctx.V0_stride);
        for (int32_t bus = 0; bus < n_bus; ++bus) {
            const auto& v0 = ctx.V0[src_batch + static_cast<std::size_t>(bus)];
            const std::size_t dst = dst_batch + static_cast<std::size_t>(bus);
            h_V_re[dst] = v0.real();
            h_V_im[dst] = v0.imag();
            h_Va[dst] = std::arg(v0);
            h_Vm[dst] = std::abs(v0);
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


void CudaMixedStorage::download_result(NRResultF64& result) const
{
    if (batch_size != 1) {
        throw std::runtime_error("CudaMixedStorage::download_result: use batch result download for batch_size > 1");
    }

    std::vector<double> h_Va(static_cast<std::size_t>(n_bus));
    std::vector<double> h_Vm(static_cast<std::size_t>(n_bus));

    d_Va.copyTo(h_Va.data(), h_Va.size());
    d_Vm.copyTo(h_Vm.data(), h_Vm.size());

    result.V.resize(static_cast<std::size_t>(n_bus));
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const double va = h_Va[static_cast<std::size_t>(bus)];
        const double vm = h_Vm[static_cast<std::size_t>(bus)];
        result.V[static_cast<std::size_t>(bus)] = std::complex<double>(
            vm * std::cos(va),
            vm * std::sin(va));
    }
}


void CudaMixedStorage::download_batch_result(NRBatchResultF64& result) const
{
    std::vector<double> h_Va(static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(n_bus));
    std::vector<double> h_Vm(static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(n_bus));

    d_Va.copyTo(h_Va.data(), h_Va.size());
    d_Vm.copyTo(h_Vm.data(), h_Vm.size());

    result.n_bus = n_bus;
    result.batch_size = batch_size;
    result.V.resize(static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(n_bus));

    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t batch_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n_bus);
        for (int32_t bus = 0; bus < n_bus; ++bus) {
            const std::size_t idx = batch_base + static_cast<std::size_t>(bus);
            const double va = h_Va[idx];
            const double vm = h_Vm[idx];
            result.V[idx] = std::complex<double>(vm * std::cos(va), vm * std::sin(va));
        }
    }

    if (!d_normF.empty()) {
        result.final_mismatch.resize(static_cast<std::size_t>(batch_size));
        d_normF.copyTo(result.final_mismatch.data(), result.final_mismatch.size());
    }
}

#endif  // CUPF_WITH_CUDA
