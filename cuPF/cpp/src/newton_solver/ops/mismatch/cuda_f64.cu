// ---------------------------------------------------------------------------
// cuda_f64.cu — CUDA FP64 Mismatch 연산자 구현
//
// NR 각 반복에서 F 벡터(미스매치)를 계산하고 수렴 판정한다.
//
// 구성:
//   Mixed path:
//     launch_compute_ibus()
//     launch_compute_mismatch()
//     launch_reduce_norm()
//   FP64 path:
//   mismatch_pack_kernel       : CUDA 커널 — F 벡터 계산
//   run_mismatch_typed()       : storage/value dtype별 실행 로직
//   CudaMismatchOpF64::run()   : storage 유형에 따라 dtype 분기 호출
//
// mismatch_pack_f64_kernel 스레드 할당:
//   스레드 tid 가 F 벡터의 tid번째 원소를 담당.
//   F 벡터 패킹 순서:
//     [0,       n_pv)           → ΔP_pv  : Re(mis_i), bus = pv[tid]
//     [n_pv,    n_pv+n_pq)      → ΔP_pq  : Re(mis_i), bus = pq[tid - n_pv]
//     [n_pv+n_pq, dimF)         → ΔQ_pq  : Im(mis_i), bus = pq[tid - n_pv - n_pq]
//
//   mis_i = V_i · conj(I_i) - S_spec_i,   I_i = Σ_j Y_ij · V_j
//   스레드 당 해당 버스의 CSR 행 전체를 순회해 I_bus 를 inline SpMV로 계산.
//
// Mixed storage는 Ibus64 custom CSR kernel과 residual F64 kernel로 분리한다.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_f64.hpp"

#include "cuda_mismatch_kernels.hpp"
#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/dump.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>


namespace {

// 스레드 tid 가 F[tid] 를 계산한다.
// 각 스레드가 자신이 담당하는 버스의 CSR 행을 전체 순회(inline SpMV)해
// I_bus = Σ_j Y_ij · V_j 를 구하고, mis = V · conj(I) - Sbus 의
// 실수(ΔP) 또는 허수(ΔQ) 부분을 F[tid] 에 기록한다.
template <typename ValueScalar, typename AccScalar>
__global__ void mismatch_pack_kernel(
    const ValueScalar* __restrict__ y_re,
    const ValueScalar* __restrict__ y_im,
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const ValueScalar* __restrict__ v_re,
    const ValueScalar* __restrict__ v_im,
    const ValueScalar* __restrict__ sbus_re,
    const ValueScalar* __restrict__ sbus_im,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv,
    int32_t n_pq,
    double* __restrict__ F)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dimF = n_pv + 2 * n_pq;
    if (tid >= dimF) {
        return;
    }

    // F 벡터 패킹: tid → (버스 번호, ΔP 또는 ΔQ)
    int32_t bus = 0;
    bool take_imag = false;  // false=ΔP(Re), true=ΔQ(Im)
    if (tid < n_pv) {
        bus = pv[tid];                    // ΔP_pv
    } else if (tid < n_pv + n_pq) {
        bus = pq[tid - n_pv];             // ΔP_pq
    } else {
        bus = pq[tid - n_pv - n_pq];     // ΔQ_pq
        take_imag = true;
    }

    // I_bus = Σ_j Y_ij · V_j  (inline SpMV, 스레드 당 CSR 행 전체 순회)
    AccScalar i_re = static_cast<AccScalar>(0.0);
    AccScalar i_im = static_cast<AccScalar>(0.0);
    for (int32_t k = y_row_ptr[bus]; k < y_row_ptr[bus + 1]; ++k) {
        const int32_t col = y_col[k];
        const AccScalar yr = static_cast<AccScalar>(y_re[k]);
        const AccScalar yi = static_cast<AccScalar>(y_im[k]);
        const AccScalar vr = static_cast<AccScalar>(v_re[col]);
        const AccScalar vi = static_cast<AccScalar>(v_im[col]);
        i_re += yr * vr - yi * vi;  // Re(Y_ij · V_j)
        i_im += yr * vi + yi * vr;  // Im(Y_ij · V_j)
    }

    // mis_i = V_i · conj(I_i) - Sbus_i
    //   S_calc = V · conj(I) = (Vr+jVi)·(Ir-jIi)
    //          = (Vr·Ir + Vi·Ii) + j(Vi·Ir - Vr·Ii)
    const double vr = static_cast<double>(v_re[bus]);
    const double vi = static_cast<double>(v_im[bus]);
    const double ir = static_cast<double>(i_re);
    const double ii = static_cast<double>(i_im);
    const double mis_re = vr * ir + vi * ii - static_cast<double>(sbus_re[bus]);  // ΔP
    const double mis_im = vi * ir - vr * ii - static_cast<double>(sbus_im[bus]);  // ΔQ
    F[tid] = take_imag ? mis_im : mis_re;
}

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

// CudaFp64Storage 및 CudaMixedStorage 공용 실행 로직.
// 두 storage 모두 d_F 가 double* 이지만 입력 dtype은 profile별로 다르다.
// 커널 완료 후 d_F 를 host로 내려 L∞ normF 를 계산하고 수렴 여부를 판정한다.
template <typename Storage, typename ValueScalar, typename AccScalar>
void run_mismatch_typed(Storage& storage, IterationContext& ctx, const char* op_name)
{
    if (storage.n_bus <= 0 || storage.dimF <= 0) {
        throw std::runtime_error(std::string(op_name) + "::run: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t grid = (storage.dimF + block - 1) / block;

    mismatch_pack_kernel<ValueScalar, AccScalar><<<grid, block>>>(
        storage.d_Ybus_re.data(),
        storage.d_Ybus_im.data(),
        storage.d_Ybus_indptr.data(),
        storage.d_Ybus_indices.data(),
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_Sbus_re.data(),
        storage.d_Sbus_im.data(),
        storage.d_pv.data(),
        storage.d_pq.data(),
        storage.n_pvpq - storage.n_pq,  // n_pv = n_pvpq - n_pq
        storage.n_pq,
        storage.d_F.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();

    // d_F 를 host로 내려 L∞ normF 계산 (수렴 판정은 host에서 수행)
    std::vector<double> h_F(static_cast<std::size_t>(storage.dimF));
    storage.d_F.copyTo(h_F.data(), h_F.size());

    ctx.normF = 0.0;
    for (double value : h_F) {
        ctx.normF = std::max(ctx.normF, std::abs(value));
    }

    if (!std::isfinite(ctx.normF)) {
        throw std::runtime_error(std::string(op_name) + "::run: mismatch norm is not finite");
    }

    newton_solver::utils::dumpVector("residual", ctx.iter, h_F);
    newton_solver::utils::dumpVector("residual_before_update", ctx.iter, h_F);
    ctx.converged = (ctx.normF <= ctx.config.tolerance);
}

void run_mismatch_mixed(CudaMixedStorage& storage, IterationContext& ctx, const char* op_name)
{
    if (storage.n_bus <= 0 || storage.dimF <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error(std::string(op_name) + "::run: storage is not prepared");
    }

    launch_compute_ibus_batch_fp32(storage);
    launch_compute_mismatch_batch_f64(storage);
    launch_reduce_norm_batch_f64(storage);

    std::vector<double> h_norm(static_cast<std::size_t>(storage.batch_size));
    storage.d_normF.copyTo(h_norm.data(), h_norm.size());

    ctx.normF = 0.0;
    for (double value : h_norm) {
        ctx.normF = std::max(ctx.normF, value);
    }

    if (!std::isfinite(ctx.normF)) {
        throw std::runtime_error(std::string(op_name) + "::run: mismatch norm is not finite");
    }

    if (newton_solver::utils::isDumpEnabled()) {
        std::vector<double> h_F(static_cast<std::size_t>(storage.batch_size) *
                                static_cast<std::size_t>(storage.dimF));
        storage.d_F.copyTo(h_F.data(), h_F.size());
        newton_solver::utils::dumpVector("residual", ctx.iter, h_F);
        newton_solver::utils::dumpVector("residual_before_update", ctx.iter, h_F);
    }

    ctx.converged = (ctx.normF <= ctx.config.tolerance);
}

}  // namespace


CudaMismatchOpF64::CudaMismatchOpF64(IStorage& storage)
    : storage_(storage) {}


void CudaMismatchOpF64::run(IterationContext& ctx)
{
    if (storage_.compute() == ComputePolicy::Mixed) {
        run_mismatch_mixed(static_cast<CudaMixedStorage&>(storage_), ctx, "CudaMismatchOpF64");
        return;
    }

    run_mismatch_typed<CudaFp64Storage, double, double>(
        static_cast<CudaFp64Storage&>(storage_),
        ctx,
        "CudaMismatchOpF64");
}

#endif  // CUPF_WITH_CUDA
