// ---------------------------------------------------------------------------
// cuda_mixed_storage.cpp — CUDA Mixed 정밀도 Storage 구현
//
// cuda_fp64_storage.cpp 와 동일한 구조이며, Jacobian·solve 관련 버퍼만 float.
//
// 주요 차이점:
//   - d_J_values : DeviceBuffer<float>  (FP32 Jacobian fill + cuDSS FP32 입력)
//   - d_dx       : DeviceBuffer<float>  (cuDSS FP32 출력 → VoltageUpdate에서 FP64 캐스트)
//   - d_Ybus_re/im, d_V_re/im, d_F, d_Va/Vm, d_Sbus_re/im : 모두 FP64(double)
//
// upload_complex_components<DeviceScalar>():
//   FP64 버전과 달리 템플릿으로 구현되어 double 과 float 양쪽에 사용된다.
//     - upload(Ybus): DeviceScalar = double  (Mismatch는 FP64 필요)
//     - upload(Sbus): DeviceScalar = double
//     - upload(V0):   DeviceScalar = double
//   static_cast<DeviceScalar>(src[i].real()) 로 타입 변환을 통일.
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

// std::complex<double>[] → 실수·허수 분리 후 DeviceScalar 타입으로 device 업로드.
// DeviceScalar = double : Ybus/Sbus/V 버퍼 (FP64 유지)
// DeviceScalar = float  : 필요 시 확장 가능 (현재 Mixed에서 float 버퍼에는 직접 사용 안 함)
template <typename DeviceScalar>
void upload_complex_components(DeviceBuffer<DeviceScalar>& dst_re,
                               DeviceBuffer<DeviceScalar>& dst_im,
                               const std::complex<double>* src,
                               int32_t count)
{
    std::vector<DeviceScalar> h_re(static_cast<std::size_t>(count));
    std::vector<DeviceScalar> h_im(static_cast<std::size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        h_re[static_cast<std::size_t>(i)] = static_cast<DeviceScalar>(src[i].real());
        h_im[static_cast<std::size_t>(i)] = static_cast<DeviceScalar>(src[i].imag());
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

    const int32_t nnz_ybus = ctx.ybus.nnz;
    const int32_t nnz_J    = ctx.J.nnz;

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
    d_dx.resize(dimF);

    d_Va.resize(n_bus);
    d_Vm.resize(n_bus);
    d_V_re.resize(n_bus);
    d_V_im.resize(n_bus);

    d_Sbus_re.resize(n_bus);
    d_Sbus_im.resize(n_bus);

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
    d_dx.memsetZero();
    d_J_values.memsetZero();
}


void CudaMixedStorage::upload(const SolveContext& ctx)
{
    if (ctx.ybus == nullptr || ctx.sbus == nullptr || ctx.V0 == nullptr) {
        throw std::invalid_argument("CudaMixedStorage::upload: solve context is incomplete");
    }

    const YbusViewF64& ybus = *ctx.ybus;
    if (ybus.rows != n_bus || ybus.cols != n_bus ||
        ybus.nnz != static_cast<int32_t>(d_Ybus_re.size())) {
        throw std::runtime_error("CudaMixedStorage::upload: Ybus dimensions do not match analyze()");
    }

    require_pointer(ybus.data, "SolveContext.ybus->data", ybus.nnz);
    require_pointer(ctx.sbus, "SolveContext.sbus", n_bus);
    require_pointer(ctx.V0, "SolveContext.V0", n_bus);

    upload_complex_components(d_Ybus_re, d_Ybus_im, ybus.data, ybus.nnz);
    upload_complex_components(d_Sbus_re, d_Sbus_im, ctx.sbus, n_bus);
    upload_complex_components(d_V_re, d_V_im, ctx.V0, n_bus);

    std::vector<double> h_Va(static_cast<std::size_t>(n_bus));
    std::vector<double> h_Vm(static_cast<std::size_t>(n_bus));
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        h_Va[static_cast<std::size_t>(bus)] = std::arg(ctx.V0[bus]);
        h_Vm[static_cast<std::size_t>(bus)] = std::abs(ctx.V0[bus]);
    }
    d_Va.assign(h_Va.data(), h_Va.size());
    d_Vm.assign(h_Vm.data(), h_Vm.size());

    d_F.memsetZero();
    d_dx.memsetZero();
}


void CudaMixedStorage::download_result(NRResultF64& result) const
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
