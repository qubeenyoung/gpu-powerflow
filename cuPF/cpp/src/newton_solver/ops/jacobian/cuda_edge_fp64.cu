#ifdef CUPF_WITH_CUDA

#include "cuda_edge_fp64.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"

#include <cmath>
#include <stdexcept>


namespace {

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

// ---------------------------------------------------------------------------
// atomic_add_f64_compat: FP64 atomicAdd 호환 래퍼.
//
// sm_60(Pascal) 이상에서는 하드웨어 FP64 atomicAdd를 직접 사용한다.
// 이전 아키텍처에서는 CAS 루프로 에뮬레이션한다.
// ---------------------------------------------------------------------------
__device__ inline double atomic_add_f64_compat(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
#endif
}

// ---------------------------------------------------------------------------
// update_jacobian_fp64_kernel: Edge-based FP64 Jacobian fill 커널.
//
// ■ 병렬화 전략 (Edge-based)
//   스레드 하나가 Ybus 비영 원소(엣지) 하나를 담당한다.
//   인덱스 k = blockIdx.x * blockDim.x + threadIdx.x
//   → Ybus의 k번째 원소 (행 i, 열 j, 값 Y_ij)를 처리.
//
// ■ 계산하는 값 (오프 대각 기여, i ≠ j):
//
//   curr = Y_ij · V_j                          (행 전류 기여)
//
//   term_va = -j · V_i · conj(curr)
//     = Re: (-j·V_i) · conj(curr) 의 실수부 → ∂P_i/∂θ_j
//     = Im: (-j·V_i) · conj(curr) 의 허수부 → ∂Q_i/∂θ_j
//
//     -j · V_i = V_im - j·V_re  이므로:
//       neg_j_vi_re = vi_im,  neg_j_vi_im = -vi_re
//     conj(curr) = curr_re - j·curr_im 이므로 곱:
//       term_va_re = neg_j_vi_re·curr_re + neg_j_vi_im·curr_im
//       term_va_im = neg_j_vi_im·curr_re - neg_j_vi_re·curr_im
//
//   term_vm = V_i · conj(curr / |V_j|)
//     = Vi · conj(Y_ij · V̂_j)
//     → ∂P_i/∂|V_j| (실수부), ∂Q_i/∂|V_j| (허수부)
//
// ■ 대각 기여 (오프 대각의 음수로 누산):
//   ∂S_i/∂θ_i  의 이웃 기여 = -term_va  (오프 대각 기여의 합산이 대각에 영향)
//   ∂S_i/∂|V_i| 의 이웃 기여 = V̂_i · conj(curr)
//
// ■ Atomic add 필요 이유:
//   대각 원소 diag[i]에는 여러 스레드(j=0,...,n-1)가 동시에 기여하므로
//   경쟁 조건을 막기 위해 atomic_add_f64_compat을 사용한다.
//   오프 대각(map**)는 각 k마다 고유한 위치이므로 atomic 불필요.
//   그러나 코드 일관성과 잠재적 중복 패턴 방지를 위해 atomic을 유지한다.
// ---------------------------------------------------------------------------
__global__ void update_jacobian_fp64_kernel(
    int32_t n_elements,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
    const int32_t* __restrict__ y_row,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    double* __restrict__ J_values)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_elements) {
        return;
    }

    const int32_t i = y_row[k];
    const int32_t j = y_col[k];

    const double yr    = y_re[k];
    const double yi    = y_im[k];
    const double vi_re = v_re[i];
    const double vi_im = v_im[i];
    const double vj_re = v_re[j];
    const double vj_im = v_im[j];

    // curr = Y_ij · V_j (복소 행렬-벡터 곱의 단일 원소 기여)
    const double curr_re = yr * vj_re - yi * vj_im;
    const double curr_im = yr * vj_im + yi * vj_re;

    // term_va = -j · V_i · conj(curr)
    //   -j · V_i = (vi_im) + j·(-vi_re)
    const double neg_j_vi_re = vi_im;
    const double neg_j_vi_im = -vi_re;
    const double term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;  // ∂P_i/∂θ_j
    const double term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;  // ∂Q_i/∂θ_j

    // term_vm = V_i · conj(curr / |V_j|) = V_i · conj(Y_ij · V̂_j)
    const double vj_abs = hypot(vj_re, vj_im);
    double term_vm_re = 0.0;
    double term_vm_im = 0.0;
    if (vj_abs > 1e-12) {
        const double scaled_re = curr_re / vj_abs;   // Re(Y_ij · V̂_j)
        const double scaled_im = curr_im / vj_abs;   // Im(Y_ij · V̂_j)
        // V_i · conj(scaled) = (vi_re + j·vi_im)(scaled_re - j·scaled_im)
        term_vm_re = vi_re * scaled_re + vi_im * scaled_im;  // ∂P_i/∂|V_j|
        term_vm_im = vi_im * scaled_re - vi_re * scaled_im;  // ∂Q_i/∂|V_j|
    }

    // 오프 대각 scatter: map**[k]가 가리키는 J.values 위치에 기여
    if (map11[k] >= 0) atomic_add_f64_compat(&J_values[map11[k]], term_va_re);
    if (map21[k] >= 0) atomic_add_f64_compat(&J_values[map21[k]], term_va_im);
    if (map12[k] >= 0) atomic_add_f64_compat(&J_values[map12[k]], term_vm_re);
    if (map22[k] >= 0) atomic_add_f64_compat(&J_values[map22[k]], term_vm_im);

    // 대각 ∂S_i/∂θ_i 이웃 기여: 오프 대각 term_va의 음수를 누산
    //   ∂P_i/∂θ_i += -term_va_re,   ∂Q_i/∂θ_i += -term_va_im
    if (diag11[i] >= 0) atomic_add_f64_compat(&J_values[diag11[i]], -term_va_re);
    if (diag21[i] >= 0) atomic_add_f64_compat(&J_values[diag21[i]], -term_va_im);

    // 대각 ∂S_i/∂|V_i| 이웃 기여: V̂_i · conj(curr) 를 누산
    const double vi_abs = hypot(vi_re, vi_im);
    if (vi_abs > 1e-12) {
        const double vi_norm_re = vi_re / vi_abs;
        const double vi_norm_im = vi_im / vi_abs;
        // V̂_i · conj(curr) = (vi_norm_re + j·vi_norm_im)(curr_re - j·curr_im)
        const double term_vm2_re = vi_norm_re * curr_re + vi_norm_im * curr_im;  // ∂P_i/∂|V_i|
        const double term_vm2_im = vi_norm_im * curr_re - vi_norm_re * curr_im;  // ∂Q_i/∂|V_i|
        if (diag12[i] >= 0) atomic_add_f64_compat(&J_values[diag12[i]], term_vm2_re);
        if (diag22[i] >= 0) atomic_add_f64_compat(&J_values[diag22[i]], term_vm2_im);
    }
}

}  // namespace


CudaJacobianOpEdgeFp64::CudaJacobianOpEdgeFp64(IStorage& storage)
    : storage_(storage) {}


void CudaJacobianOpEdgeFp64::run(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaFp64Storage&>(storage_);

    if (storage.d_Y_row.empty() || storage.d_J_values.empty()) {
        throw std::runtime_error("CudaJacobianOpEdgeFp64::run: storage is not prepared");
    }

    storage.d_J_values.memsetZero();

    constexpr int32_t block = 256;
    const int32_t y_nnz = static_cast<int32_t>(storage.d_Y_row.size());
    const int32_t grid = (y_nnz + block - 1) / block;

    update_jacobian_fp64_kernel<<<grid, block>>>(
        y_nnz,
        storage.d_Ybus_re.data(),
        storage.d_Ybus_im.data(),
        storage.d_Y_row.data(),
        storage.d_Ybus_indices.data(),
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_mapJ11.data(),
        storage.d_mapJ21.data(),
        storage.d_mapJ12.data(),
        storage.d_mapJ22.data(),
        storage.d_diagJ11.data(),
        storage.d_diagJ21.data(),
        storage.d_diagJ12.data(),
        storage.d_diagJ22.data(),
        storage.d_J_values.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
