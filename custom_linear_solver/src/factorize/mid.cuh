#pragma once

// Internal — included only by multifrontal.cu (single TU).
//
// Mid-front factor kernels. The whole front (fsz × fsz) is staged into dynamic shared once,
// then the four factor phases (panel LU, U solve, trailing update, extend-add) run against
// shared instead of re-reading global on each of the ~nc sequential passes that dominate the
// global-memory path. Two variants differ only in the trailing update:
//
//   factor_mid<T>    – tiled scalar trailing. T = float or double. The default for non-TC
//                      precisions; only requires enough shared to hold the front + the L / U
//                      staging panels.
//   factor_mid_tc    – FP16 WMMA tensor-core trailing. FP32 front. Used for the TC precision.
//
// Shared-memory layout matters because load_matrix_sync requires 128-bit alignment. The TC
// kernel places the FP16 staging buffers (Lh / Uh) at the shared base; the tiled kernel
// places the front Fs at the base. The dispatcher sizes the shared budget per launch and
// falls through to factor_big<T> when the front + staging would exceed 96 KB.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "factorize/primitives.cuh"

namespace custom_linear_solver {
namespace {

// ---------------------------------------------------------------------------------------
// Shared-staged scalar trailing GEMM C(uc x uc) -= L(uc x nc) * U(nc x uc).
//
// Stages L and U into compact contiguous shared buffers so the inner GEMM hits stride-1 /
// stride-uc accesses with no bank conflicts. The front layout is row-major in F (ld = fsz):
//   L is at F[(nc+i) * fsz + k]    (uc rows × nc cols)
//   U is at F[k * fsz + (nc+j)]    (nc rows × uc cols)
//   C is at F[(nc+i) * fsz + (nc+j)]
template <typename T>
__device__ __forceinline__ void trailing_update_staged(T* F, int fsz, int nc, int uc, int t,
                                                        int nt, T* sh_L, T* sh_U)
{
    // Stage L into sh_L (row-major, uc x nc, ld = nc).
    for (int e = t; e < uc * nc; e += nt) {
        const int i = e / nc, k = e % nc;
        sh_L[e] = F[(long)(nc + i) * fsz + k];
    }
    // Stage U into sh_U (row-major, nc x uc, ld = uc).
    for (int e = t; e < nc * uc; e += nt) {
        const int k = e / uc, j = e % uc;
        sh_U[e] = F[(long)k * fsz + (nc + j)];
    }
    __syncthreads();
    // C -= L * U, both panels read from shared.
    for (int e = t; e < uc * uc; e += nt) {
        const int i = e / uc, j = e % uc;
        T acc = T(0);
        for (int k = 0; k < nc; ++k) acc += sh_L[i * nc + k] * sh_U[k * uc + j];
        F[(long)(nc + i) * fsz + (nc + j)] -= acc;
    }
}

// ---------------------------------------------------------------------------------------
// Shared-staged FP16 WMMA trailing on an FP32 front. nc (= K) is zero-padded to a 16-multiple,
// and the contraction streams KP/16 = 1 or 2 16-deep K tiles per output tile, amortizing
// the per-tile store + subtract over the WMMA instructions.
//
// Requires nc <= 32 (KP <= 32, two A fragments) and uc <= ucp_stride.
__device__ __forceinline__ void trailing_update_wmma_f32(float* F, int fsz, int nc, int uc,
                                                          __half* Lh, __half* Uh, float* Csc,
                                                          int ucp_stride, int t, int nt)
{
    namespace wmma = nvcuda::wmma;
    const int UCP = ((uc + 15) / 16) * 16;
    const int KP = ((nc + 15) / 16) * 16;
    for (int e = t; e < UCP * KP; e += nt) {
        const int i = e / KP, k = e % KP;
        Lh[e] = (i < uc && k < nc) ? __float2half(F[(long)(nc + i) * fsz + k]) : __float2half(0.0f);
    }
    for (int e = t; e < KP * UCP; e += nt) {
        const int k = e / UCP, j = e % UCP;
        Uh[k * ucp_stride + j] =
            (k < nc && j < uc) ? __float2half(F[(long)k * fsz + (nc + j)]) : __float2half(0.0f);
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
                wmma::load_matrix_sync(bf, &Uh[(kc * 16) * ucp_stride + tj * 16], ucp_stride);
                wmma::mma_sync(cf, af[kc], bf, cf);
            }
            wmma::store_matrix_sync(&Csc[warp * 256], cf, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = e >> 4, c = e & 15;
                const int ii = ti * 16 + r, jj = tj * 16 + c;
                if (ii < uc && jj < uc) F[(long)(nc + ii) * fsz + (nc + jj)] -= Csc[warp * 256 + e];
            }
            __syncwarp();
        }
    }
}

// ---------------------------------------------------------------------------------------
// factor_mid<T> — tiled mid-front factor for FP32 (T = float) and FP64 (T = double).
//
// One block per (front, batch). Shared layout: Fs[fsz_cap²] then sh_L[uc*nc] then sh_U[nc*uc].
// fsz_cap = the level's max fsz; level_max_nc / level_max_uc bound the staging panels. The
// uc x uc contribution block remains in Fs after the trailing update and is fed into the
// parent via extend_add without an intervening global writeback.
template <typename T>
__global__ void factor_mid(int lbegin, int lend, const int* __restrict__ plcols,
                           const int* __restrict__ front_off,
                           const int* __restrict__ front_ptr,
                           const int* __restrict__ ncols,
                           const int* __restrict__ panel_parent,
                           const int* __restrict__ asm_ptr,
                           const int* __restrict__ asm_local, T* frontB,
                           long front_total, int* sing, int do_extend, int fsz_cap,
                           int level_max_nc, int level_max_uc)
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
    const int fsz2 = fsz * fsz;

    extern __shared__ char smem_mid[];
    T* Fs   = reinterpret_cast<T*>(smem_mid);
    T* sh_L = Fs + (long)fsz_cap * fsz_cap;
    T* sh_U = sh_L + (long)level_max_uc * level_max_nc;

    // Stage the front into shared.
    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    // The main factor body — phases 1–3 (panel LU + U solve + trailing). Shared-staged trailing.
    factorize_front<T>(Fs, fsz, nc, uc, t, nt, sing,
        [&] { trailing_update_staged<T>(Fs, fsz, nc, uc, t, nt, sh_L, sh_U); });
    __syncthreads();

    // Write back only the factored L/U the solve will read; the uc x uc contribution block
    // stays in Fs for the extend-add.
    writeback_factored<T, T>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<T, T>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
}

// ---------------------------------------------------------------------------------------
// factor_mid_tc — TC variant. FP32 front, FP16 WMMA trailing with FP32 accumulate.
//
// Shared layout: Lh (FP16) and Uh (FP16) come first so load_matrix_sync sees 128-bit
// alignment; then the per-warp FP32 WMMA scratch Csc; then the front Fs. Caller computes the
// shared size to match.
__global__ void factor_mid_tc(int lbegin, int lend, const int* __restrict__ plcols,
                              const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols,
                              const int* __restrict__ panel_parent,
                              const int* __restrict__ asm_ptr,
                              const int* __restrict__ asm_local, float* frontB,
                              long front_total, int* sing, int do_extend, int ucp_max,
                              int fsz_cap)
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
    const int fsz2 = fsz * fsz;

    extern __shared__ char smem_mid_tc[];
    __half* Lh  = reinterpret_cast<__half*>(smem_mid_tc);
    __half* Uh  = Lh + (long)ucp_max * 32;
    float*  Csc = reinterpret_cast<float*>(Uh + (long)32 * ucp_max);
    float*  Fs  = Csc + 4 * 256;

    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing, [&] {
        if (nc <= 32 && uc <= 256) {
            trailing_update_wmma_f32(Fs, fsz, nc, uc, Lh, Uh, Csc, ucp_max, t, nt);
        } else {
            trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt);
        }
    });
    __syncthreads();
    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
}

}  // namespace
}  // namespace custom_linear_solver
