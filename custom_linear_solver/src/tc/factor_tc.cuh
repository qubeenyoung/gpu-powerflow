#pragma once

// Internal — included only by batched/multifrontal_batched.cu (single TU; the build uses
// CUDA_SEPARABLE_COMPILATION OFF, so every batched kernel must be defined in the TU that launches
// it). This is the TENSOR-CORE factor module, promoted out of batched/factor_kernels.cuh.
//
// Two WMMA factor kernels share the FP16 tensor-core trailing-update design (the rank-nc trailing
// C(uc x uc) -= L(uc x nc) * U(nc x uc) is the compute bulk of the big fronts; nc(=K) is zero-
// padded to the 16x16x16 tile and streamed KP/16 deep, so the per-tile store+subtract is amortized
// over KP/16 mma_sync). They differ only in where the front lives:
//
//   mf_factor_extend_mixed_tc_b  - FP64 MASTER + FP32 working. Reference-accurate assembly/extend-
//        add (double), FP32 working LU, FP16 trailing. Accuracy ~1e-3..1e-1 (recovered by IR if
//        the app needs < FP16 error). Heavier: per-front cast_copy(FP64->FP32) + writeback +
//        double-atomic extend-add + 2x front memory.
//
//   mf_factor_extend_tc32_b      - FP32-NATIVE (NO master). The front IS the FP32 arena: assembly,
//        panel LU, U-solve, trailing and extend-add all FP32; only the trailing GEMM inputs are
//        rounded to FP16 (FP32 accumulate). Same storage/traffic as the pure-FP32 path, so it
//        carries none of the FP64-master overhead that makes Mixed/TC lose to pure FP32 on these
//        latency-bound power-grid fronts -- the tensor cores are pure upside on the big fronts.
//        Accuracy tracks pure FP32 except for the FP16 rounding of the trailing contributions
//        (panel/pivot/U-solve stay FP32), which is the regime the caller asked for (~FP32, not <).

#include <cuda_runtime.h>
#include <mma.h>

#include "batched/lu_device.cuh"

namespace custom_linear_solver::batched {
namespace {

// Device helper: the shared-staged FP16 WMMA rank-nc trailing update on an FP32 front F (ld=fsz).
// Stages L(uc x nc) and U(nc x uc) into the caller-provided FP16 shared buffers (Lh: UCP*KP row-
// major with ld=KP; Uh: KP*UCP row-major with ld=ucp_stride), runs the multi-k-step WMMA GEMM, and
// subtracts the FP32 result back into the trailing block of F. Csc is a per-warp 16x16 FP32 scratch
// (nwarp*256 floats). Requires nc<=32 (KP<=32, two A-fragments) and uc<=ucp_stride.
__device__ __forceinline__ void tc_trailing_wmma_f32(float* F, int fsz, int nc, int uc,
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

constexpr int TC_REG_NC = 32;

// ---- FP32-NATIVE factor with FP16 TENSOR-CORE trailing (no FP64 master). One block per
// (front, batch), blockDim=128 (4 warps cooperate on the output tiles). The front F is the FP32
// arena frontBf; everything is FP32 except the trailing GEMM inputs (FP16, FP32 accumulate).
__global__ void mf_factor_extend_tc32_b(int lbegin, int lend, const int* __restrict__ plcols,
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

    if (fsz <= 48) {
        lu_small_front<float>(F, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(F, fsz, nc, t, nt, sing);
        u_panel_solve<float>(F, fsz, nc, uc, t, nt);
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_tc32[];
            __half* Lh = reinterpret_cast<__half*>(smem_tc32);
            __half* Uh = Lh + (long)ucp_max * 32;
            float* Csc = reinterpret_cast<float*>(Uh + (long)32 * ucp_max);
            tc_trailing_wmma_f32(F, fsz, nc, uc, Lh, Uh, Csc, ucp_max, t, nt);
        }
    }
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// ---- FP32-native SHARED-RESIDENT medium/large-front factor (block per (front,batch), 128
// threads). The whole front is staged into dynamic shared (sized per level to the level's max
// fsz^2), so the panel LU, U-panel solve, trailing update and extend-add all run against low-
// latency shared instead of re-reading the front from global on every one of the ~nc sequential
// passes (the global block kernel runs these levels at ~50% DRAM and <50% compute = latency/traffic
// bound). The trailing update is the FP16 tensor-core GEMM when USE_TC (and the front is tile-
// eligible), else the FP32 scalar loop. Layout in shared: Fs[fsz_cap^2] then (USE_TC) the FP16 L/U
// staging + the per-warp FP32 WMMA scratch.
template <bool USE_TC>
__global__ void mf_factor_mid_tc32_b(int lbegin, int lend, const int* __restrict__ plcols,
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

    // Shared layout: the FP16 WMMA staging (Lh/Uh) and the FP32 warp scratch (Csc) come FIRST so
    // Lh sits at the 16-byte-aligned shared base (load_matrix_sync needs 128-bit alignment); the
    // front Fs follows (each prior region is a multiple of 16 bytes, so Fs stays aligned too). For
    // the non-TC path there is no staging and Fs is at the base.
    extern __shared__ char smem_mid[];
    __half* Lh = reinterpret_cast<__half*>(smem_mid);
    __half* Uh = Lh + (long)ucp_max * 32;
    float* Csc = reinterpret_cast<float*>(Uh + (long)32 * ucp_max);
    float* Fs = USE_TC ? (Csc + 4 * 256) : reinterpret_cast<float*>(smem_mid);
    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];  // coalesced stage-in
    __syncthreads();

    if (fsz <= 48) {
        lu_small_front<float>(Fs, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
        if (USE_TC && nc <= 32 && uc <= 256) {
            tc_trailing_wmma_f32(Fs, fsz, nc, uc, Lh, Uh, Csc, ucp_max, t, nt);
        } else {
            trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt);
        }
    }
    __syncthreads();
    // Write back ONLY the factored L/U the solve reads: the nc pivot-column rows (pivot + U panel)
    // and the L block of the trailing rows. The uc x uc contribution block stays in shared for the
    // extend-add below and is never read again from global -> skip it (the dominant write traffic on
    // the DRAM-bound mid levels). writeback_factored does exactly this split.
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
}  // namespace custom_linear_solver::batched
