#pragma once

// Internal — included only by multifrontal.cu (single TU).
//
// Big-front factor kernels. These run the front in global memory (no shared-resident staging)
// and are used for fronts larger than MID_THRESH, where the front can't fit in shared
// alongside the L / U staging panels. Three precision variants:
//
//   factor_big<T>      – scalar trailing. T = float or double.
//   factor_big_tc      – FP16 WMMA tensor-core trailing, FP32 front. Used for the TC precision.
//
// One block per (front, batch). The factor phases (panel LU, U solve, trailing, extend-add)
// are the same as the mid kernels — only the staging strategy differs.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "factorize/primitives.cuh"
#include "factorize/mid.cuh"  // trailing_update_wmma_f32

namespace custom_linear_solver {
namespace {

// Big-front factor with scalar trailing. T = float (FP32 mode) or double (FP64 mode).
template <typename T>
__global__ void factor_big(int lbegin, int lend, const int* __restrict__ plcols,
                           const int* __restrict__ front_off,
                           const int* __restrict__ front_ptr,
                           const int* __restrict__ ncols,
                           const int* __restrict__ panel_parent,
                           const int* __restrict__ asm_ptr,
                           const int* __restrict__ asm_local, T* frontB,
                           long front_total, int* sing, int do_extend)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;

    // The main factor body — phases 1–3 (panel LU + U solve + trailing). Scalar trailing.
    factorize_front<T>(F, fsz, nc, uc, t, nt, sing,
        [&] { trailing_update_scalar<T>(F, fsz, nc, uc, t, nt); });

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<T, T>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// Big-front factor with FP16 WMMA tensor-core trailing. FP32 front. The trailing GEMM stages
// L / U into FP16 in shared and accumulates back into the FP32 front. Falls back to scalar
// trailing for fronts not eligible for WMMA (nc > 32 or uc > 256).
__global__ void factor_big_tc(int lbegin, int lend, const int* __restrict__ plcols,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ panel_parent,
                              const int* __restrict__ asm_ptr,
                              const int* __restrict__ asm_local, float* frontB,
                              long front_total, int* sing, int do_extend, int ucp_max)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;

    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_tc[];
            __half* Lh  = reinterpret_cast<__half*>(smem_big_tc);
            __half* Uh  = Lh + (long)ucp_max * 32;
            float*  Csc = reinterpret_cast<float*>(Uh + (long)32 * ucp_max);
            trailing_update_wmma_f32(F, fsz, nc, uc, Lh, Uh, Csc, ucp_max, t, nt);
        }
    });

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

}  // namespace
}  // namespace custom_linear_solver
