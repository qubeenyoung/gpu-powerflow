// ---------------------------------------------------------------------------
// cuda_vertex_fp32.cu — FP32 vertex-based Jacobian fill 커널 (Mixed 전용)
//
// cuda_vertex_fp64.cu 와 완전히 동일한 알고리즘이며, 연산 타입만 float.
//
// FP32 차이점:
//   - d_V_re/im (FP64) → static_cast<float>
//   - y_re/im   (FP64) → static_cast<float>
//   - J_values  : float* (CudaMixedStorage)
//   - warp_sum  : __shfl_down_sync 으로 float 리덕션
//   - 영전압 임계값 : 1e-6f
//   - 오프 대각 직접 write (atomic 불필요): warp 단위 분리 구조 덕분
//
// 알고리즘 요약:
//   warp 하나(32 레인)가 버스 하나(pvpq 인덱스)를 처리.
//   레인이 행의 원소를 warp_size 스트라이드로 분담.
//   대각 기여는 레지스터에 누산 → warp_sum → lane 0 단일 write.
//
// 수학 배경: docs/math.md §3 참조.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_vertex_fp32.hpp"

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

// butterfly reduction: 32 레인의 float 값을 lane 0 에 합산.
__device__ inline float warp_sum(float value)
{
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// warp 하나가 버스 하나를 처리.
// FP64 전압·Ybus를 float으로 캐스트 후 edge-based와 동일한 Jacobian 공식 적용.
__global__ void update_jacobian_vertex_fp32_kernel(
    int32_t n_active_buses,
    const int32_t* __restrict__ pvpq,
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
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
    constexpr int32_t warp_size = 32;

    const int32_t warp_id_in_block = threadIdx.x / warp_size;
    const int32_t lane = threadIdx.x & (warp_size - 1);
    const int32_t warps_per_block = blockDim.x / warp_size;
    const int32_t bus_slot = blockIdx.x * warps_per_block + warp_id_in_block;

    if (bus_slot >= n_active_buses) {
        return;
    }

    const int32_t i = pvpq[bus_slot];
    const int32_t row_begin = y_row_ptr[i];
    const int32_t row_end = y_row_ptr[i + 1];

    // FP64 버퍼에서 float으로 캐스트 (Mixed 모드)
    const float vi_re = static_cast<float>(v_re_f64[i]);
    const float vi_im = static_cast<float>(v_im_f64[i]);
    const float vi_abs = hypotf(vi_re, vi_im);
    const bool have_vi_norm = vi_abs > 1e-6f;  // 대각 J12/J22 계산용 V̂_i
    const float vi_norm_re = have_vi_norm ? vi_re / vi_abs : 0.0f;
    const float vi_norm_im = have_vi_norm ? vi_im / vi_abs : 0.0f;

    // 대각 누산기: 모든 이웃 엣지의 기여를 레인별로 독립적으로 합산
    // warp_sum 후 lane 0 이 최종값을 J_values에 단일 write
    float diag11_acc = 0.0f;
    float diag21_acc = 0.0f;
    float diag12_acc = 0.0f;
    float diag22_acc = 0.0f;

    // 자기 엣지(j == i) term 누산 (warp_sum 후 대각 최종값에 더해짐)
    float self11 = 0.0f;
    float self21 = 0.0f;
    float self12 = 0.0f;
    float self22 = 0.0f;

    // warp 내 레인이 행의 원소를 warp_size 스트라이드로 분담
    for (int32_t k = row_begin + lane; k < row_end; k += warp_size) {
        const int32_t j = y_col[k];

        const float yr = static_cast<float>(y_re[k]);
        const float yi = static_cast<float>(y_im[k]);
        const float vj_re = static_cast<float>(v_re_f64[j]);
        const float vj_im = static_cast<float>(v_im_f64[j]);

        // curr = Y_ij · V_j
        const float curr_re = yr * vj_re - yi * vj_im;
        const float curr_im = yr * vj_im + yi * vj_re;

        // term_va = -j · V_i · conj(curr)
        const float neg_j_vi_re = vi_im;
        const float neg_j_vi_im = -vi_re;
        const float term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
        const float term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

        // term_vm = V_i · conj(curr / |V_j|)
        const float vj_abs = hypotf(vj_re, vj_im);
        float term_vm_re = 0.0f;
        float term_vm_im = 0.0f;
        if (vj_abs > 1e-6f) {
            const float scaled_re = curr_re / vj_abs;
            const float scaled_im = curr_im / vj_abs;
            term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
            term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
        }

        // 대각 보정 누산: J11/J21 는 -term_va 합산
        diag11_acc += -term_va_re;
        diag21_acc += -term_va_im;

        // 대각 보정 누산: J12/J22 는 V̂_i · conj(curr) 합산
        if (have_vi_norm) {
            diag12_acc += vi_norm_re * curr_re + vi_norm_im * curr_im;
            diag22_acc += vi_norm_im * curr_re - vi_norm_re * curr_im;
        }

        if (j == i) {
            // 자기 엣지(j == i) term은 오프 대각에 scatter하지 않고 별도 누산
            self11 = term_va_re;
            self21 = term_va_im;
            self12 = term_vm_re;
            self22 = term_vm_im;
            continue;
        }

        // 오프 대각 직접 write (vertex-based는 레인이 겹치지 않으므로 atomic 불필요)
        if (map11[k] >= 0) J_values[map11[k]] = term_va_re;
        if (map21[k] >= 0) J_values[map21[k]] = term_va_im;
        if (map12[k] >= 0) J_values[map12[k]] = term_vm_re;
        if (map22[k] >= 0) J_values[map22[k]] = term_vm_im;
    }

    // warp butterfly reduction: 모든 레인의 누산기를 lane 0 에 합산
    diag11_acc = warp_sum(diag11_acc);
    diag21_acc = warp_sum(diag21_acc);
    diag12_acc = warp_sum(diag12_acc);
    diag22_acc = warp_sum(diag22_acc);
    self11 = warp_sum(self11);
    self21 = warp_sum(self21);
    self12 = warp_sum(self12);
    self22 = warp_sum(self22);

    // lane 0 만이 대각 원소를 J_values에 최종 write (단일 write, atomic 불필요)
    if (lane == 0) {
        if (diag11[i] >= 0) J_values[diag11[i]] = self11 + diag11_acc;
        if (diag21[i] >= 0) J_values[diag21[i]] = self21 + diag21_acc;
        if (diag12[i] >= 0) J_values[diag12[i]] = self12 + diag12_acc;
        if (diag22[i] >= 0) J_values[diag22[i]] = self22 + diag22_acc;
    }
}

}  // namespace


CudaJacobianOpVertexFp32::CudaJacobianOpVertexFp32(IStorage& storage)
    : storage_(storage) {}


void CudaJacobianOpVertexFp32::run(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaMixedStorage&>(storage_);

    if (storage.d_pvpq.empty() || storage.d_J_values.empty()) {
        throw std::runtime_error("CudaJacobianOpVertexFp32::run: storage is not prepared");
    }

    storage.d_J_values.memsetZero();

    constexpr int32_t block = 256;
    constexpr int32_t warp_size = 32;
    const int32_t warps_per_block = block / warp_size;
    const int32_t grid = (storage.n_pvpq + warps_per_block - 1) / warps_per_block;

    update_jacobian_vertex_fp32_kernel<<<grid, block>>>(
        storage.n_pvpq,
        storage.d_pvpq.data(),
        storage.d_Ybus_indptr.data(),
        storage.d_Ybus_indices.data(),
        storage.d_Ybus_re.data(),
        storage.d_Ybus_im.data(),
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
