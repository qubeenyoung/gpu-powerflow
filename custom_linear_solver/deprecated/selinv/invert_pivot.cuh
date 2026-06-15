#pragma once

// Deprecated — selinv (selective inversion) pre-computation kernel.
//
// Original behavior: factorize end 에서 각 front 의 nc×nc pivot block 을 미리 역행렬화해서
// solve 의 triangular solve (trsv, sequential) 를 GEMV (parallel) 로 바꾸기. NR loop / contingency
// 같은 *1 factor + 다수 solve* 시나리오에 이득. 하지만 power-grid 의 *1 factor + 1 solve* 패턴에선
// invert_pivot 비용 > solve 가속 효과 라 default OFF 로 운용되다 최종적으로 삭제됨.
//
// 코드 한 줄 요약: inverse 계산은 nc×nc 의 작은 dense 문제지만 *내부 산술이 FP64* (`double Ui/Li`)
// 라서 FP32 mode 라도 RTX 3090 의 FP64 1/64 throughput 에 묶임 — selinv 켤 때 factor wall 의
// 33-48% 가 이 단일 커널이 됐었던 원인.

#include <cuda_runtime.h>

namespace custom_linear_solver::deprecated::selinv {
namespace {

constexpr int MF_REG_NC = 32;

template <typename FT>
__global__ void mf_invert_pivot_b(int npanels, const int* __restrict__ front_ptr,
                                  const int* __restrict__ front_off,
                                  const int* __restrict__ ncols,
                                  FT* frontB, long front_total)
{
    const int p = blockIdx.x;
    if (p >= npanels) return;
    FT* front = frontB + (long)blockIdx.y * front_total;
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int j = threadIdx.x;
    __shared__ double Ui[MF_REG_NC * MF_REG_NC];
    __shared__ double Li[MF_REG_NC * MF_REG_NC];
    if (j < nc) {
        Ui[j * nc + j] = 1.0 / static_cast<double>(F[(long)j * fsz + j]);
        for (int i = j - 1; i >= 0; --i) {
            double v = 0.0;
            for (int k = i + 1; k <= j; ++k)
                v -= static_cast<double>(F[(long)i * fsz + k]) * Ui[k * nc + j];
            Ui[i * nc + j] = v / static_cast<double>(F[(long)i * fsz + i]);
        }
        Li[j * nc + j] = 1.0;
        for (int i = j + 1; i < nc; ++i) {
            double v = 0.0;
            for (int k = j; k < i; ++k)
                v -= static_cast<double>(F[(long)i * fsz + k]) * Li[k * nc + j];
            Li[i * nc + j] = v;
        }
    }
    __syncthreads();
    if (j < nc) {
        for (int i = 0; i <= j; ++i) F[(long)i * fsz + j] = static_cast<FT>(Ui[i * nc + j]);
        for (int i = j + 1; i < nc; ++i) F[(long)i * fsz + j] = static_cast<FT>(Li[i * nc + j]);
    }
}

}  // namespace
}  // namespace custom_linear_solver::deprecated::selinv
