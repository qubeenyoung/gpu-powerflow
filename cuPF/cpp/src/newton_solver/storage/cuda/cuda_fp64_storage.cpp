// ---------------------------------------------------------------------------
// cuda_fp64_storage.cpp — CUDA FP64 Storage 구현
//
// CUDA FP64 경로의 device 버퍼와 Ybus/Jacobian 희소 구조를 관리한다.
//
// prepare() — analyze() 단계에서 한 번만 호출:
//   1. 위상 메타데이터(n_bus, n_pvpq, n_pq, dimF) 설정
//   2. Ybus CSR 구조(indptr/indices) device 업로드
//   3. build_ybus_row_index() : CSR indptr → 행 번호 배열(d_Y_row) 생성
//      edge-based Jacobian 커널이 y_row[k] = i 로 행을 즉시 읽기 위해 필요.
//   4. Jacobian CSR 구조(row_ptr/col_idx) device 업로드
//   5. JacobianMaps (mapJ**, diagJ**) device 업로드
//   6. 가변 버퍼(d_F, d_dx, d_J_values, d_Va, d_Vm, d_V_re/im, d_Sbus_re/im) 할당
//
// upload() — solve() 마다 호출:
//   1. Ybus 값 device 업로드 (upload_complex_components: 실수·허수 분리)
//   2. Sbus, V0 device 업로드
//   3. V0 로부터 Va, Vm 초기화 (std::arg, std::abs → host → device)
//   4. d_F, d_dx 초기화 (cudaMemset 0)
//
// upload_complex_components():
//   std::complex<double>[] → 실수 배열 + 허수 배열 분리 후 device 업로드.
//   CUDA 커널이 실수·허수를 별도 포인터로 받기 때문에 분리가 필요.
// ---------------------------------------------------------------------------

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

// CSR indptr → 행 번호 배열 생성.
// d_Y_row[k] = i 가 되도록 indptr[i]..indptr[i+1) 구간에 i 를 기록.
// edge-based 커널이 엣지 k 에서 행(i)을 O(1) 로 조회하기 위해 사용.
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

// std::complex<double>[] 를 실수·허수 배열로 분리해 device 업로드.
// CUDA 커널은 실수·허수를 별도 포인터로 받으므로 분리가 필요.
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


void CudaFp64Storage::prepare(const AnalyzeContext& ctx)
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


void CudaFp64Storage::upload(const SolveContext& ctx)
{
    if (ctx.ybus == nullptr || ctx.sbus == nullptr || ctx.V0 == nullptr) {
        throw std::invalid_argument("CudaFp64Storage::upload: solve context is incomplete");
    }

    const YbusViewF64& ybus = *ctx.ybus;
    if (ybus.rows != n_bus || ybus.cols != n_bus ||
        ybus.nnz != static_cast<int32_t>(d_Ybus_re.size())) {
        throw std::runtime_error("CudaFp64Storage::upload: Ybus dimensions do not match analyze()");
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


void CudaFp64Storage::download_result(NRResultF64& result) const
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
