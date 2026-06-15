#pragma once

// Deprecated — Mixed precision factor kernels.
//
// Mixed: FP64 master front (정확한 assembly/solve) + FP32 working LU. RTX 3090 FP64 1/64 throughput
// 한도를 피하려고 LU 만 FP32 로 가는데, master 와 working 의 *double 메모리 점유* 라 메모리 1.5×.
// 정확도 ~1e-5. 결국 pure FP32 의 ~1e-4 가 NR loop 에는 충분하고, 메모리 절감이 더 컸기에 삭제.
//
// 두 변형:
//   mf_factor_extend_mixed_b      — scalar FP32 trailing
//   mf_factor_extend_mixed_tc_b   — FP16 WMMA tensor-core trailing (옛 TC mode)

#include <cuda_runtime.h>
#include <mma.h>

#include "batched/lu_device.cuh"

namespace custom_linear_solver::deprecated::precision_mixed {
namespace {

using namespace custom_linear_solver::batched;  // lu_panel_factor, trailing_update_scalar 등

// ---- batched mixed factor: FP64 master assembly, FP32 working LU --------------------
__global__ void mf_factor_extend_mixed_b(int lbegin, int lend, const int* __restrict__ plcols,
                                         const int* __restrict__ front_off,
                                         const int* __restrict__ front_ptr,
                                         const int* __restrict__ ncols,
                                         const int* __restrict__ panel_parent,
                                         const int* __restrict__ asm_ptr,
                                         const int* __restrict__ asm_local, double* masterB,
                                         float* workingB, long front_total, int* sing,
                                         int do_extend)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const long boff = (long)blockIdx.y * front_total;
    double* master = masterB + boff;
    float* working = workingB + boff;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    double* M = master + front_off[p];
    float* W = working + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const long fsz2 = (long)fsz * fsz;
    cast_copy<float, double>(W, M, fsz2, t, nt);
    __syncthreads();
    if (fsz <= 48) {
        lu_small_front<float>(W, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(W, fsz, nc, t, nt, sing);
        u_panel_solve<float>(W, fsz, nc, uc, t, nt);
        trailing_update_scalar<float>(W, fsz, nc, uc, t, nt);
    }
    __syncthreads();
    writeback_factored<double, float>(M, W, fsz, nc, uc, t, nt);
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    double* Mp = master + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<double, float>(Mp, pfsz, W, fsz, nc, uc, asm_local, abase, t, nt);
}

// ---- batched MIXED factor with FP16 TENSOR-CORE (WMMA) trailing update --------------
// Same FP64-master / FP32-working design as mf_factor_extend_mixed_b, but the dense rank-nc
// trailing update is a half-precision WMMA GEMM (C(uc x uc) -= L(uc x nc) * U(nc x uc)),
// with nc(<=32) zero-padded to 16x16x16 tiles. L/U staged FP16 in shared, FP32 accumulate.
__global__ void mf_factor_extend_mixed_tc_b(int lbegin, int lend, const int* __restrict__ plcols,
                                            const int* __restrict__ front_off,
                                            const int* __restrict__ front_ptr,
                                            const int* __restrict__ ncols,
                                            const int* __restrict__ panel_parent,
                                            const int* __restrict__ asm_ptr,
                                            const int* __restrict__ asm_local, double* masterB,
                                            float* workingB, long front_total, int* sing,
                                            int do_extend, int ucp_max)
{
    namespace wmma = nvcuda::wmma;
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const long boff = (long)blockIdx.y * front_total;
    double* master = masterB + boff;
    float* working = workingB + boff;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    double* M = master + front_off[p];
    float* W = working + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const long fsz2 = (long)fsz * fsz;
    cast_copy<float, double>(W, M, fsz2, t, nt);
    __syncthreads();

    if (fsz <= 48) {
        lu_small_front<float>(W, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(W, fsz, nc, t, nt, sing);
        u_panel_solve<float>(W, fsz, nc, uc, t, nt);
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(W, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem[];
            __half* Lh = reinterpret_cast<__half*>(smem);
            __half* Uh = Lh + (long)ucp_max * 32;
            float* Csc = reinterpret_cast<float*>(Uh + (long)32 * ucp_max);
            const int UCP = ((uc + 15) / 16) * 16;
            const int KP = ((nc + 15) / 16) * 16;
            for (int e = t; e < UCP * KP; e += nt) {
                const int i = e / KP, k = e % KP;
                Lh[e] = (i < uc && k < nc) ? __float2half(W[(long)(nc + i) * fsz + k])
                                           : __float2half(0.0f);
            }
            for (int e = t; e < KP * UCP; e += nt) {
                const int k = e / UCP, j = e % UCP;
                Uh[k * ucp_max + j] = (k < nc && j < uc) ? __float2half(W[(long)k * fsz + (nc + j)])
                                                         : __float2half(0.0f);
            }
            __syncthreads();
            const int ntj = UCP / 16, nks = KP / 16;
            const int warp = t >> 5, nwarp = nt >> 5, lane = t & 31;
            for (int ti = warp; ti < ntj; ti += nwarp) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
                for (int kc = 0; kc < nks; ++kc)
                    wmma::load_matrix_sync(af[kc], &Lh[(ti * 16) * KP + kc * 16], KP);
                for (int tj = 0; tj < ntj; ++tj) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
                    wmma::fill_fragment(cf, 0.0f);
                    for (int kc = 0; kc < nks; ++kc) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
                        wmma::load_matrix_sync(bf, &Uh[(kc * 16) * ucp_max + tj * 16], ucp_max);
                        wmma::mma_sync(cf, af[kc], bf, cf);
                    }
                    wmma::store_matrix_sync(&Csc[warp * 256], cf, 16, wmma::mem_row_major);
                    __syncwarp();
                    for (int e = lane; e < 256; e += 32) {
                        const int r = e >> 4, c = e & 15;
                        const int ii = ti * 16 + r, jj = tj * 16 + c;
                        if (ii < uc && jj < uc)
                            W[(long)(nc + ii) * fsz + (nc + jj)] -= Csc[warp * 256 + e];
                    }
                    __syncwarp();
                }
            }
        }
    }
    __syncthreads();
    writeback_factored<double, float>(M, W, fsz, nc, uc, t, nt);
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    double* Mp = master + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<double, float>(Mp, pfsz, W, fsz, nc, uc, asm_local, abase, t, nt);
}

}  // namespace
}  // namespace custom_linear_solver::deprecated::precision_mixed
