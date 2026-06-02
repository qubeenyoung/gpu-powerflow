#pragma once

// Internal — included only by batched/multifrontal_batched.cu (see lu_device.cuh for why the
// batched kernels live in headers folded into a single TU).
//
// Batched FACTOR kernels (all front-major: gridDim.y = batch; arena = B * front_total). One
// block per (front, batch); the dense per-front LU + extend-add building blocks are shared via
// lu_device.cuh so the three precision variants differ only in their element types and trailing
// update:
//   mf_factor_extend_level_b<FT> - FP64 (FT=double) and pure-FP32 (FT=float) dense LU + extend
//   mf_factor_extend_mixed_b     - Mixed: FP64 master + FP32 working LU (scalar trailing)
//   mf_factor_extend_mixed_tc_b  - TC:    FP64 master + FP16 tensor-core (WMMA) trailing
//   mf_invert_pivot_b<FT>        - optional selinv (invert each pivot block for a GEMV solve)

#include <cuda_runtime.h>
#include <mma.h>

#include "batched/lu_device.cuh"

namespace custom_linear_solver::batched {
namespace {

// ---- batched fused factor + extend-add. One block per (front, batch). FT = front type
// (double = FP64 mode, float = pure-FP32 mode). The dense no-pivot LU + rank-nc trailing
// update run in FT; the CB is extend-added into the parent front (atomicAdd). ----------
template <typename FT>
__global__ void mf_factor_extend_level_b(int lbegin, int lend, const int* __restrict__ plcols,
                                         const int* __restrict__ front_off,
                                         const int* __restrict__ front_ptr,
                                         const int* __restrict__ ncols,
                                         const int* __restrict__ panel_parent,
                                         const int* __restrict__ asm_ptr,
                                         const int* __restrict__ asm_local, FT* frontB,
                                         long front_total, int* sing, int do_extend)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    FT* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;

    if (fsz <= 48) {
        lu_small_front<FT>(F, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<FT>(F, fsz, nc, t, nt, sing);
        u_panel_solve<FT>(F, fsz, nc, uc, t, nt);
        trailing_update_scalar<FT>(F, fsz, nc, uc, t, nt);
    }
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    FT* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<FT, FT>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

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

constexpr int MF_REG_NC = 32;

// ---- batched MIXED factor with FP16 TENSOR-CORE (WMMA) trailing update --------------
// Same FP64-master / FP32-working design as mf_factor_extend_mixed_b, but the dense
// rank-nc trailing update (the compute bulk of the big fronts) is a half-precision WMMA
// GEMM: C(uc x uc) -= L(uc x nc) * U(nc x uc), with the nc(<=16) contraction zero-padded to
// the 16x16x16 tensor-core tile (K=16). L/U are staged FP16 in shared (padded to UCP =
// ceil(uc/16)*16); the FP32 accumulate is subtracted back into the working front. The FP16
// inputs make the trailing ~1e-3 accurate -> recovered by batched iterative refinement. Small
// fronts (fsz<=48, no full 16x16 tile) keep the FP32 path. One block per (front, batch),
// blockDim=128 (4 warps cooperate on the output tiles).
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

    if (fsz <= 48) {  // small front: plain FP32 fused LU (no tensor-core tile)
        lu_small_front<float>(W, fsz, nc, t, nt, sing);
    } else {
        // Phase 1: factor the nc-wide panel (FP32, full height).
        lu_panel_factor<float>(W, fsz, nc, t, nt, sing);
        // Phase 2: U panel triangular solve (FP32).
        u_panel_solve<float>(W, fsz, nc, uc, t, nt);
        // Phase 3: FP16 tensor-core trailing  C -= L * U with MULTI-K-STEP. nc (= K) up to 32 is
        // zero-padded to KP = ceil(nc/16)*16 and the WMMA contraction streams KP/16 16-deep
        // k-tiles, so the per-output-tile store+subtract overhead is amortized over KP/16 mma_sync
        // (deep K = the point of growing nc by blocked amalgamation). L/U are staged FP16 in shared
        // (aligned ld). Each warp owns a tile-row (fixed ti), reusing its A fragments across tj.
        // Fronts too big for the shared staging (nc>32 or uc>256) fall back to FP32.
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(W, fsz, nc, uc, t, nt);
        } else {
            // Dynamic shared sized per level (by the level's max uc): Lh[ucp_max*32], Uh[32*ucp_max]
            // halfs, then Csc. Smaller-front levels get far less shared -> the occupancy that the
            // earlier static 36KB version destroyed (1 block/SM) is restored.
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

// Invert each front's nc x nc pivot block (Uinv upper incl. diag, Linv unit-lower) so the solve
// applies them as parallel GEMVs (selinv). FT = front type; the inverse is computed in double for
// stability and stored back in FT. One block per (front,batch), one thread per inverse column.
template <typename FT>
__global__ void mf_invert_pivot_b(int npanels, const int* __restrict__ front_ptr,
                                  const int* __restrict__ front_off, const int* __restrict__ ncols,
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
            for (int k = i + 1; k <= j; ++k) v -= static_cast<double>(F[(long)i * fsz + k]) * Ui[k * nc + j];
            Ui[i * nc + j] = v / static_cast<double>(F[(long)i * fsz + i]);
        }
        Li[j * nc + j] = 1.0;
        for (int i = j + 1; i < nc; ++i) {
            double v = 0.0;
            for (int k = j; k < i; ++k) v -= static_cast<double>(F[(long)i * fsz + k]) * Li[k * nc + j];
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
}  // namespace custom_linear_solver::batched
