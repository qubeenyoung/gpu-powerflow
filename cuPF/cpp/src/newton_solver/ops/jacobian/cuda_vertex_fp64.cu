#ifdef CUPF_WITH_CUDA

#include "cuda_vertex_fp64.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"

#include <stdexcept>


namespace {

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

// ---------------------------------------------------------------------------
// warp_sum: warp 내 32개 레인의 값을 butterfly reduction으로 합산한다.
// __shfl_down_sync를 사용하므로 레지스터 간 통신만 일어나고 공유 메모리 불필요.
// ---------------------------------------------------------------------------
__device__ inline double warp_sum(double value)
{
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// ---------------------------------------------------------------------------
// update_jacobian_vertex_fp64_kernel: Vertex-based FP64 Jacobian fill 커널.
//
// ■ 병렬화 전략 (Vertex-based)
//   warp 하나가 pvpq 버스 하나(정점 i)를 담당한다.
//   warp의 32개 레인이 버스 i의 행 [row_begin, row_end)를 스트라이드로 분담한다.
//     lane l → k = row_begin + l, row_begin + l + 32, ...
//
// ■ Edge-based 대비 장점
//   - 오프 대각 write에 atomic이 불필요 (한 레인만 각 J 위치를 씀).
//   - 대각 누산을 warp_sum으로 처리해 atomic_add 없이 레인 0이 한 번에 write.
//   - 대용량 네트워크(행이 긴 버스)에서 warp 효율이 높다.
//
// ■ 계산 흐름
//   각 레인이 담당하는 k(엣지)에 대해:
//     curr = Y_ik · V_k
//     term_va = -j · V_i · conj(curr)   → 오프 대각 ∂S_i/∂θ_k
//     term_vm = V_i · conj(Y_ik · V̂_k) → 오프 대각 ∂S_i/∂|V_k|
//
//   대각 누산 (레지스터 level):
//     diag**_acc += 이웃 k의 대각 기여  (모든 k에 대해 누산)
//     self** = 자기 자신(k=i)의 term_va/vm (대각 자체 원소)
//
//   warp 내 reduction → lane 0이 최종 대각값 write:
//     diag[i] = self + diag_acc  (= 자기 어드미턴스 + Σ 이웃 기여)
//
// ■ self vs diag_acc 분리 이유
//   Ybus의 대각 원소 (i,i)도 루프에서 만나는데, 이 원소는 term_va 공식
//   (-j·V_i·conj(Y_ii·V_i))를 그대로 적용하면 자기 어드미턴스 항이 된다.
//   이를 self**에 따로 보관하고, 이웃(k≠i)으로부터의 대각 기여(diag_acc)와
//   합산해 최종 대각값을 구한다.
// ---------------------------------------------------------------------------
__global__ void update_jacobian_vertex_fp64_kernel(
    int32_t n_active_buses,
    const int32_t* __restrict__ pvpq,
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
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
    constexpr int32_t warp_size = 32;

    const int32_t warp_id_in_block = threadIdx.x / warp_size;
    const int32_t lane             = threadIdx.x & (warp_size - 1);
    const int32_t warps_per_block  = blockDim.x / warp_size;
    const int32_t bus_slot         = blockIdx.x * warps_per_block + warp_id_in_block;

    if (bus_slot >= n_active_buses) {
        return;
    }

    const int32_t i         = pvpq[bus_slot];
    const int32_t row_begin = y_row_ptr[i];
    const int32_t row_end   = y_row_ptr[i + 1];

    // V_i 정보를 warp 공유 (모든 레인이 같은 V_i를 사용)
    const double vi_re = v_re[i];
    const double vi_im = v_im[i];
    const double vi_abs = hypot(vi_re, vi_im);
    const bool   have_vi_norm = vi_abs > 1e-12;
    const double vi_norm_re   = have_vi_norm ? vi_re / vi_abs : 0.0;
    const double vi_norm_im   = have_vi_norm ? vi_im / vi_abs : 0.0;

    // 대각 누산 레지스터 (warp_sum 전까지 각 레인이 독립적으로 누산)
    double diag11_acc = 0.0;
    double diag21_acc = 0.0;
    double diag12_acc = 0.0;
    double diag22_acc = 0.0;

    // 자기 자신(k=i) 원소의 term_va/vm (warp_sum 후 lane 0이 합산)
    double self11 = 0.0;
    double self21 = 0.0;
    double self12 = 0.0;
    double self22 = 0.0;

    // warp 레인이 스트라이드로 버스 i의 행 원소를 분담
    for (int32_t k = row_begin + lane; k < row_end; k += warp_size) {
        const int32_t j    = y_col[k];
        const double yr    = y_re[k];
        const double yi    = y_im[k];
        const double vj_re = v_re[j];
        const double vj_im = v_im[j];

        // curr = Y_ij · V_j
        const double curr_re = yr * vj_re - yi * vj_im;
        const double curr_im = yr * vj_im + yi * vj_re;

        // term_va = -j · V_i · conj(curr)
        const double neg_j_vi_re = vi_im;
        const double neg_j_vi_im = -vi_re;
        const double term_va_re  = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
        const double term_va_im  = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

        // term_vm = V_i · conj(Y_ij · V̂_j)
        const double vj_abs = hypot(vj_re, vj_im);
        double term_vm_re = 0.0;
        double term_vm_im = 0.0;
        if (vj_abs > 1e-12) {
            const double scaled_re = curr_re / vj_abs;
            const double scaled_im = curr_im / vj_abs;
            term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
            term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
        }

        // 대각 이웃 기여 누산 (j=i 포함, 아래에서 self로 분리)
        diag11_acc += -term_va_re;
        diag21_acc += -term_va_im;

        if (have_vi_norm) {
            // ∂|S_i|/∂|V_i| 이웃 기여: V̂_i · conj(curr)
            diag12_acc += vi_norm_re * curr_re + vi_norm_im * curr_im;
            diag22_acc += vi_norm_im * curr_re - vi_norm_re * curr_im;
        }

        // 자기 자신(j=i)의 경우: 오프 대각 write 생략, self에 보관
        if (j == i) {
            self11 = term_va_re;
            self21 = term_va_im;
            self12 = term_vm_re;
            self22 = term_vm_im;
            continue;
        }

        // 오프 대각 write (atomic 불필요: 이 레인만 이 위치를 씀)
        if (map11[k] >= 0) J_values[map11[k]] = term_va_re;
        if (map21[k] >= 0) J_values[map21[k]] = term_va_im;
        if (map12[k] >= 0) J_values[map12[k]] = term_vm_re;
        if (map22[k] >= 0) J_values[map22[k]] = term_vm_im;
    }

    // warp 내 butterfly reduction: 각 레인의 누산값을 lane 0으로 합산
    diag11_acc = warp_sum(diag11_acc);
    diag21_acc = warp_sum(diag21_acc);
    diag12_acc = warp_sum(diag12_acc);
    diag22_acc = warp_sum(diag22_acc);
    self11     = warp_sum(self11);
    self21     = warp_sum(self21);
    self12     = warp_sum(self12);
    self22     = warp_sum(self22);

    // lane 0만 최종 대각값을 write
    //   diag = self(자기 어드미턴스 항) + diag_acc(이웃 기여의 합)
    if (lane == 0) {
        if (diag11[i] >= 0) J_values[diag11[i]] = self11 + diag11_acc;
        if (diag21[i] >= 0) J_values[diag21[i]] = self21 + diag21_acc;
        if (diag12[i] >= 0) J_values[diag12[i]] = self12 + diag12_acc;
        if (diag22[i] >= 0) J_values[diag22[i]] = self22 + diag22_acc;
    }
}

}  // namespace


CudaJacobianOpVertexFp64::CudaJacobianOpVertexFp64(IStorage& storage)
    : storage_(storage) {}


void CudaJacobianOpVertexFp64::run(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaFp64Storage&>(storage_);

    if (storage.d_pvpq.empty() || storage.d_J_values.empty()) {
        throw std::runtime_error("CudaJacobianOpVertexFp64::run: storage is not prepared");
    }

    storage.d_J_values.memsetZero();

    constexpr int32_t block = 256;
    constexpr int32_t warp_size = 32;
    const int32_t warps_per_block = block / warp_size;
    const int32_t grid = (storage.n_pvpq + warps_per_block - 1) / warps_per_block;

    update_jacobian_vertex_fp64_kernel<<<grid, block>>>(
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
