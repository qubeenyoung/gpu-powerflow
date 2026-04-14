// ---------------------------------------------------------------------------
// cuda_edge_fp32.cu — FP32 edge-based Jacobian fill 커널 (Mixed 전용)
//
// cuda_edge_fp64.cu 와 완전히 동일한 알고리즘이며, 연산 타입만 float.
//
// FP32 차이점:
//   - d_V_re/im (FP64) → static_cast<float> 후 레지스터에 저장
//   - y_re/im   (FP64) → static_cast<float>
//   - J_values  : float* (CudaMixedStorage)
//   - atomicAdd : float 버전, sm_20+ 에서 하드웨어 지원 (CAS 에뮬레이션 불필요)
//   - 영전압 임계값 : 1e-6f (FP64 커널의 1e-12 에 해당하는 FP32 수준)
//
// 수학 배경: docs/math.md §3 "Jacobian 편미분 공식" 참조.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_edge_fp32.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"

#include <stdexcept>


namespace {

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

// 스레드 하나가 Ybus 비영 원소(엣지) k 하나를 담당.
// FP64 버퍼에서 값을 읽어 float로 캐스트 후 Jacobian 원소를 계산한다.
//
// 오프 대각 기여 (i ≠ j):
//   curr    = Y_ij · V_j                            (복소 곱)
//   term_va = -j · V_i · conj(curr)                 → J11, J21 오프 대각
//   term_vm = V_i · conj(curr / |V_j|)              → J12, J22 오프 대각
//   atomicAdd: mapJ{11,21,12,22}[k] 위치에 기여값 scatter
//
// 대각 기여 (항상, 자기 열 Y_ii 포함):
//   대각 보정 = 각 엣지에서의 -term_va 누산
//   atomicAdd: diagJ{11,21}[i] 위치에 기여
//   diagJ{12,22}[i]에는 V̂_i 기반 term_vm2 기여
__global__ void update_jacobian_edge_fp32_kernel(
    int32_t n_elements,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
    const int32_t* __restrict__ y_row,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ v_re_f64,
    const double* __restrict__ v_im_f64,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    float* __restrict__ J_values)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_elements) {
        return;
    }

    const int32_t i = y_row[k];
    const int32_t j = y_col[k];

    // FP64 버퍼에서 float으로 캐스트 (Mixed 모드: 전압·Ybus는 FP64 저장)
    const float yr = static_cast<float>(y_re[k]);
    const float yi = static_cast<float>(y_im[k]);
    const float vi_re = static_cast<float>(v_re_f64[i]);
    const float vi_im = static_cast<float>(v_im_f64[i]);
    const float vj_re = static_cast<float>(v_re_f64[j]);
    const float vj_im = static_cast<float>(v_im_f64[j]);

    // curr = Y_ij · V_j  (복소 곱)
    const float curr_re = yr * vj_re - yi * vj_im;
    const float curr_im = yr * vj_im + yi * vj_re;

    // term_va = -j · V_i · conj(curr)
    //   -j · V_i = (V_im, -V_re)
    //   (a+jb)·conj(c+jd) = (ac+bd) + j(bc-ad)
    const float neg_j_vi_re = vi_im;
    const float neg_j_vi_im = -vi_re;
    const float term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;  // J11 오프 대각
    const float term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;  // J21 오프 대각

    // term_vm = V_i · conj(curr / |V_j|)
    //   FP32 임계값 1e-6f 로 영전압 방어
    const float vj_abs = hypotf(vj_re, vj_im);
    float term_vm_re = 0.0f;
    float term_vm_im = 0.0f;
    if (vj_abs > 1e-6f) {
        const float scaled_re = curr_re / vj_abs;
        const float scaled_im = curr_im / vj_abs;
        term_vm_re = vi_re * scaled_re + vi_im * scaled_im;  // J12 오프 대각
        term_vm_im = vi_im * scaled_re - vi_re * scaled_im;  // J22 오프 대각
    }

    // 오프 대각 scatter: map**[k] → J_values
    // float atomicAdd: sm_20+ 하드웨어 지원, CAS 에뮬레이션 불필요
    if (map11[k] >= 0) atomicAdd(&J_values[map11[k]], term_va_re);
    if (map21[k] >= 0) atomicAdd(&J_values[map21[k]], term_va_im);
    if (map12[k] >= 0) atomicAdd(&J_values[map12[k]], term_vm_re);
    if (map22[k] >= 0) atomicAdd(&J_values[map22[k]], term_vm_im);

    // 대각 보정 J11/J21: 모든 이웃 엣지의 -term_va 누산
    if (diag11[i] >= 0) atomicAdd(&J_values[diag11[i]], -term_va_re);
    if (diag21[i] >= 0) atomicAdd(&J_values[diag21[i]], -term_va_im);

    // 대각 보정 J12/J22: V̂_i 기반 term_vm2 = V̂_i · conj(curr)
    const float vi_abs = hypotf(vi_re, vi_im);
    if (vi_abs > 1e-6f) {
        const float vi_norm_re = vi_re / vi_abs;
        const float vi_norm_im = vi_im / vi_abs;
        const float term_vm2_re = vi_norm_re * curr_re + vi_norm_im * curr_im;
        const float term_vm2_im = vi_norm_im * curr_re - vi_norm_re * curr_im;
        if (diag12[i] >= 0) atomicAdd(&J_values[diag12[i]], term_vm2_re);
        if (diag22[i] >= 0) atomicAdd(&J_values[diag22[i]], term_vm2_im);
    }
}

}  // namespace


CudaJacobianOpEdgeFp32::CudaJacobianOpEdgeFp32(IStorage& storage)
    : storage_(storage) {}


void CudaJacobianOpEdgeFp32::run(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaMixedStorage&>(storage_);

    if (storage.d_Y_row.empty() || storage.d_J_values.empty()) {
        throw std::runtime_error("CudaJacobianOpEdgeFp32::run: storage is not prepared");
    }

    storage.d_J_values.memsetZero();

    constexpr int32_t block = 256;
    const int32_t y_nnz = static_cast<int32_t>(storage.d_Y_row.size());
    const int32_t grid = (y_nnz + block - 1) / block;

    update_jacobian_edge_fp32_kernel<<<grid, block>>>(
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
