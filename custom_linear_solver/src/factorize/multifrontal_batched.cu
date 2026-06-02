#include "factorize/multifrontal_batched.hpp"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <mma.h>

namespace custom_linear_solver::factorize {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// ---- batched numeric scatter: front_b[a_pos[q]] += values_b[o2c[q]] ----------------
// Templated on the front element type FT (double for FP64/Mixed/TC master, float for FP32).
template <typename FT>
__global__ void scatter_batched(int nnz_a, long front_total, const int* __restrict__ o2c,
                                const int* __restrict__ a_pos, const double* __restrict__ valuesB,
                                FT* __restrict__ frontB)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnz_a) return;
    const int pos = a_pos[q];
    if (pos < 0) return;
    const long b = blockIdx.y;
    atomicAdd(&frontB[b * front_total + pos], static_cast<FT>(valuesB[b * (long)nnz_a + o2c[q]]));
}

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
        for (int k = 0; k < nc; ++k) {
            FT piv = F[(long)k * fsz + k];
            if (piv == FT(0)) { if (t == 0) *sing = 1; piv = FT(1); }
            for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
            __syncthreads();
            const int m = fsz - k - 1;
            for (int e = t; e < m * m; e += nt) {
                const int ii = k + 1 + e / m, jj = k + 1 + e % m;
                F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
            }
            __syncthreads();
        }
    } else {
        for (int k = 0; k < nc; ++k) {
            FT piv = F[(long)k * fsz + k];
            if (piv == FT(0)) { if (t == 0) *sing = 1; piv = FT(1); }
            for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
            __syncthreads();
            const int pc = nc - 1 - k;
            for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
                const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
                F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
            }
            if (pc > 0) __syncthreads();
        }
        for (int k = 1; k < nc; ++k) {
            for (int e = t; e < uc; e += nt) {
                const int jj = nc + e;
                FT v = F[(long)k * fsz + jj];
                for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
                F[(long)k * fsz + jj] = v;
            }
            __syncthreads();
        }
        for (int e = t; e < uc * uc; e += nt) {
            const int ii = nc + e / uc, jj = nc + e % uc;
            FT acc = FT(0);
            for (int k = 0; k < nc; ++k) acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
            F[(long)ii * fsz + jj] -= acc;
        }
    }
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    FT* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = t; e < uc * uc; e += nt) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  F[(long)(nc + a) * fsz + (nc + b)]);
    }
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
    for (long e = t; e < fsz2; e += nt) W[e] = (float)M[e];
    __syncthreads();
    if (fsz <= 48) {
        for (int k = 0; k < nc; ++k) {
            float piv = W[(long)k * fsz + k];
            if (piv == 0.0f) { if (t == 0) *sing = 1; piv = 1.0f; }
            for (int i = k + 1 + t; i < fsz; i += nt) W[(long)i * fsz + k] /= piv;
            __syncthreads();
            const int m = fsz - k - 1;
            for (int e = t; e < m * m; e += nt) {
                const int ii = k + 1 + e / m, jj = k + 1 + e % m;
                W[(long)ii * fsz + jj] -= W[(long)ii * fsz + k] * W[(long)k * fsz + jj];
            }
            __syncthreads();
        }
    } else {
        for (int k = 0; k < nc; ++k) {
            float piv = W[(long)k * fsz + k];
            if (piv == 0.0f) { if (t == 0) *sing = 1; piv = 1.0f; }
            for (int i = k + 1 + t; i < fsz; i += nt) W[(long)i * fsz + k] /= piv;
            __syncthreads();
            const int pc = nc - 1 - k;
            for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
                const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
                W[(long)ii * fsz + jj] -= W[(long)ii * fsz + k] * W[(long)k * fsz + jj];
            }
            if (pc > 0) __syncthreads();
        }
        for (int k = 1; k < nc; ++k) {
            for (int e = t; e < uc; e += nt) {
                const int jj = nc + e;
                float v = W[(long)k * fsz + jj];
                for (int i = 0; i < k; ++i) v -= W[(long)k * fsz + i] * W[(long)i * fsz + jj];
                W[(long)k * fsz + jj] = v;
            }
            __syncthreads();
        }
        for (int e = t; e < uc * uc; e += nt) {
            const int ii = nc + e / uc, jj = nc + e % uc;
            float acc = 0.0f;
            for (int k = 0; k < nc; ++k) acc += W[(long)ii * fsz + k] * W[(long)k * fsz + jj];
            W[(long)ii * fsz + jj] -= acc;
        }
    }
    __syncthreads();
    for (long e = t; e < (long)nc * fsz; e += nt) M[e] = (double)W[e];
    for (int e = t; e < uc * nc; e += nt) {
        const long id2 = (long)(nc + e / nc) * fsz + (e % nc);
        M[id2] = (double)W[id2];
    }
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    double* Mp = master + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = t; e < uc * uc; e += nt) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Mp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  (double)W[(long)(nc + a) * fsz + (nc + b)]);
    }
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
    for (long e = t; e < fsz2; e += nt) W[e] = (float)M[e];
    __syncthreads();

    if (fsz <= 48) {  // small front: plain FP32 fused LU (no tensor-core tile)
        for (int k = 0; k < nc; ++k) {
            float piv = W[(long)k * fsz + k];
            if (piv == 0.0f) { if (t == 0) *sing = 1; piv = 1.0f; }
            for (int i = k + 1 + t; i < fsz; i += nt) W[(long)i * fsz + k] /= piv;
            __syncthreads();
            const int m = fsz - k - 1;
            for (int e = t; e < m * m; e += nt) {
                const int ii = k + 1 + e / m, jj = k + 1 + e % m;
                W[(long)ii * fsz + jj] -= W[(long)ii * fsz + k] * W[(long)k * fsz + jj];
            }
            __syncthreads();
        }
    } else {
        // Phase 1: factor the nc-wide panel (FP32, full height).
        for (int k = 0; k < nc; ++k) {
            float piv = W[(long)k * fsz + k];
            if (piv == 0.0f) { if (t == 0) *sing = 1; piv = 1.0f; }
            for (int i = k + 1 + t; i < fsz; i += nt) W[(long)i * fsz + k] /= piv;
            __syncthreads();
            const int pc = nc - 1 - k;
            for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
                const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
                W[(long)ii * fsz + jj] -= W[(long)ii * fsz + k] * W[(long)k * fsz + jj];
            }
            if (pc > 0) __syncthreads();
        }
        // Phase 2: U panel triangular solve (FP32).
        for (int k = 1; k < nc; ++k) {
            for (int e = t; e < uc; e += nt) {
                const int jj = nc + e;
                float v = W[(long)k * fsz + jj];
                for (int i = 0; i < k; ++i) v -= W[(long)k * fsz + i] * W[(long)i * fsz + jj];
                W[(long)k * fsz + jj] = v;
            }
            __syncthreads();
        }
        // Phase 3: FP16 tensor-core trailing  C -= L * U with MULTI-K-STEP. nc (= K) up to 32 is
        // zero-padded to KP = ceil(nc/16)*16 and the WMMA contraction streams KP/16 16-deep
        // k-tiles, so the per-output-tile store+subtract overhead is amortized over KP/16 mma_sync
        // (deep K = the point of growing nc by blocked amalgamation). L/U are staged FP16 in shared
        // (aligned ld). Each warp owns a tile-row (fixed ti), reusing its A fragments across tj.
        // Fronts too big for the shared staging (nc>32 or uc>256) fall back to FP32.
        if (nc > 32 || uc > 256) {
            for (int e = t; e < uc * uc; e += nt) {
                const int ii = nc + e / uc, jj = nc + e % uc;
                float acc = 0.0f;
                for (int k = 0; k < nc; ++k) acc += W[(long)ii * fsz + k] * W[(long)k * fsz + jj];
                W[(long)ii * fsz + jj] -= acc;
            }
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
    for (long e = t; e < (long)nc * fsz; e += nt) M[e] = (double)W[e];
    for (int e = t; e < uc * nc; e += nt) {
        const long id2 = (long)(nc + e / nc) * fsz + (e % nc);
        M[id2] = (double)W[id2];
    }
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    double* Mp = master + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = t; e < uc * uc; e += nt) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Mp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  (double)W[(long)(nc + a) * fsz + (nc + b)]);
    }
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

// ---- batched solve (selinv GEMV form) -----------------------------------------------
__global__ void gather_rhs_b(int n, const double* __restrict__ rhsB, const int* __restrict__ perm,
                             double* __restrict__ yB)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const long b = blockIdx.y;
    yB[b * n + k] = rhsB[b * (long)n + perm[k]];
}
__global__ void scatter_sol_b(int n, const double* __restrict__ yB, const int* __restrict__ perm,
                             double* __restrict__ solB)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const long b = blockIdx.y;
    solB[b * (long)n + perm[k]] = yB[b * n + k];
}

// Forward solve level (L y = b). FT = front type (read in FT, accumulated in double; the y
// working vector stays double). selinv -> apply L_pp^-1 as a GEMV; else warp-parallel substitution.
template <typename FT>
__global__ void mf_fwd_level_b(int lbegin, int lend, const int* __restrict__ plcols,
                              const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols, const int* __restrict__ front_rows,
                              const FT* frontB, double* yB, long front_total, int n, int selinv)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const FT* front = frontB + (long)blockIdx.y * front_total;
    double* y = yB + (long)blockIdx.y * n;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const FT* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    __shared__ double sh_piv[64];
    if (selinv) {
        for (int k = t; k < nc; k += nt) {
            double v = y[fr[k]];
            for (int i = 0; i < k; ++i) v += F[(long)k * fsz + i] * y[fr[i]];
            sh_piv[k] = v;
        }
        __syncthreads();
        for (int k = t; k < nc; k += nt) y[fr[k]] = sh_piv[k];
    } else {
        // WARP-PARALLEL forward substitution L_pp * sh_piv = rhs (nc<=32). Replaces the serial
        // thread-0 O(nc^2) loop that bottlenecked the amalgamated (big-nc) solve: lane k finalizes
        // sh_piv[k] = rhs[k] - sum_{i<k} L[k][i] sh_piv[i], broadcasts it, and every lane j>k folds
        // -L[j][k]*sh_piv[k] into its running partial -> O(nc) steps of O(1) parallel work.
        if (t < 32) {
            const int lane = t;
            const unsigned mask = 0xffffffffu;
            double part = 0.0, sk = 0.0;
            for (int k = 0; k < nc; ++k) {
                if (lane == k) { sk = y[fr[k]] + part; sh_piv[k] = sk; y[fr[k]] = sk; }
                sk = __shfl_sync(mask, sk, k);
                if (lane > k && lane < nc) part -= F[(long)lane * fsz + k] * sk;
            }
        }
        __syncthreads();
    }
    for (int i = nc + t; i < fsz; i += nt) {
        double upd = 0.0;
        for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
        atomicAdd(&y[fr[i]], -upd);
    }
}

constexpr int MF_MAX_NC = 64;

// Backward solve level (U x = y). FT = front type (read in FT, accumulated in double; y stays
// double). selinv -> apply U_pp^-1 as a GEMV; else warp-parallel substitution.
template <typename FT>
__global__ void mf_bwd_level_b(int lbegin, int lend, const int* __restrict__ plcols,
                              const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols, const int* __restrict__ front_rows,
                              const FT* frontB, double* yB, long front_total, int n, int selinv)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const FT* front = frontB + (long)blockIdx.y * front_total;
    double* y = yB + (long)blockIdx.y * n;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const FT* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    const int cb = fsz - nc;
    // CB contribution rhs[k] -= sum_j U[k][nc+j]*x[nc+j]. PARALLEL-OVER-K: cache x in dynamic shared
    // (coalesced gather), then lane k owns ONE output (pk) -> no per-thread part[nc] register array
    // and no nc-way warp reduction (the register pressure + reduction that made bwd the solve
    // bottleneck). Higher occupancy; the per-lane front-row reads hit L1/L2.
    extern __shared__ double xsh[];  // size >= cb (set per level at launch)
    __shared__ double rhs[MF_MAX_NC];
    for (int k = t; k < nc; k += nt) rhs[k] = y[fr[k]];
    for (int j = t; j < cb; j += nt) xsh[j] = y[fr[nc + j]];
    __syncthreads();
    if (t < nc) {
        double pk = 0.0;
        for (int j = 0; j < cb; ++j) pk += F[(long)t * fsz + (nc + j)] * xsh[j];
        rhs[t] -= pk;
    }
    __syncthreads();
    if (selinv) {
        for (int k = t; k < nc; k += nt) {
            double v = 0.0;
            for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
            y[fr[k]] = v;
        }
    } else if (t < 32) {
        // WARP-PARALLEL backward substitution U_pp * x = rhs (nc<=32): lane k (high->low) finalizes
        // x[k] = (rhs[k] - sum_{j>k} U[k][j] x[j]) / U[k][k], broadcasts it, lanes i<k fold
        // -U[i][k]*x[k] into their partial. Replaces the serial thread-0 O(nc^2) loop.
        const int lane = t;
        const unsigned mask = 0xffffffffu;
        double part = 0.0, xk = 0.0;
        for (int k = nc - 1; k >= 0; --k) {
            if (lane == k) { xk = (rhs[k] + part) / F[(long)k * fsz + k]; y[fr[k]] = xk; }
            xk = __shfl_sync(mask, xk, k);
            if (lane < k) part -= F[(long)lane * fsz + k] * xk;
        }
    }
}

}  // namespace

BatchedState::~BatchedState()
{
    if (factor_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(factor_graph_exec));
    if (solve_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(solve_graph_exec));
    if (d_frontB) cudaFree(d_frontB);
    if (d_frontBf) cudaFree(d_frontBf);
    if (d_yB) cudaFree(d_yB);
    if (d_sing) cudaFree(d_sing);
    if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

bool batched_setup(const MultifrontalPlan& plan, int B, BatchPrecision prec, BatchedState& st)
{
    if (plan.num_panels == 0 || B <= 0) return false;
    st.B = B;
    st.front_total = plan.front_total;
    st.n = plan.n;
    st.prec = prec;
    st.selinv = std::getenv("MF_NO_SELINV") == nullptr;
    const bool pure_fp32 = (prec == BatchPrecision::FP32);
    const bool need_double = (prec != BatchPrecision::FP32);              // FP64 front or Mixed/TC master
    const bool need_float = (prec == BatchPrecision::FP32 || prec == BatchPrecision::Mixed ||
                             prec == BatchPrecision::TC);                 // FP32 front or Mixed/TC working
    const long fe = (long)B * plan.front_total;
    if (need_double && cudaMalloc(&st.d_frontB, fe * sizeof(double)) != cudaSuccess) return false;
    if (need_float && cudaMalloc(&st.d_frontBf, fe * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&st.d_yB, (long)B * plan.n * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(&st.d_sing, sizeof(int)) != cudaSuccess) return false;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    st.stream = stream;

    const int T = 128;
    const int do_extend = 1;
    // Capture batched factor graph: per level, gridDim=(level_size, B); kernel chosen by precision.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        dim3 grid(e - b, B);
        switch (prec) {
            case BatchPrecision::FP64:
                mf_factor_extend_level_b<double><<<grid, T, 0, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB,
                    plan.front_total, st.d_sing, do_extend);
                break;
            case BatchPrecision::FP32:
                mf_factor_extend_level_b<float><<<grid, T, 0, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend);
                break;
            case BatchPrecision::Mixed:
                mf_factor_extend_mixed_b<<<grid, T, 0, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend);
                break;
            case BatchPrecision::TC: {
                // per-level max uc (= dynamic-shared Uh stride) so small-front levels use little shared.
                int max_uc = 1;  // only WMMA-eligible fronts (uc<=256, nc<=32) use the shared staging
                for (int q = b; q < e; ++q) {
                    const int pp = plan.h_plcols[q];
                    const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
                    const int nc = plan.h_ncols[pp];
                    const int uc = fsz - nc;
                    if (uc > max_uc && uc <= 256 && nc <= 32) max_uc = uc;
                }
                const int ucp_max = ((max_uc + 15) / 16) * 16;
                const size_t shbytes =
                    (size_t)2 * ucp_max * 32 * sizeof(__half) + 4 * 256 * sizeof(float);
                mf_factor_extend_mixed_tc_b<<<grid, 128, shbytes, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend, ucp_max);
                break;
            }
        }
    }
    if (st.selinv) {
        const dim3 ig(plan.num_panels, B);
        if (pure_fp32)
            mf_invert_pivot_b<float><<<ig, 32, 0, stream>>>(
                plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, st.d_frontBf,
                plan.front_total);
        else
            mf_invert_pivot_b<double><<<ig, 32, 0, stream>>>(
                plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, st.d_frontB,
                plan.front_total);
    }
    cudaGraph_t g;
    cudaStreamEndCapture(stream, &g);
    cudaGraphExec_t ge;
    cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphDestroy(g);
    st.factor_graph_exec = ge;

    // Capture batched solve graph (gather -> fwd levels -> bwd levels -> scatter is done
    // outside the graph in batched_solve; here only the level kernels, like the single path).
    const int sel = st.selinv ? 1 : 0;
    // 32 threads (1 warp) per (front,batch): the warp-parallel pivot solve uses one warp anyway and
    // smaller blocks pack more per SM -> higher occupancy across the many B*fronts blocks (swept).
    const int TS = 32;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    // The solve reads the front the factor produced: the FP32 front (pure-FP32 mode) or the FP64
    // front / master (all other modes).
    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        const dim3 fg(e - b, B);
        if (pure_fp32)
            mf_fwd_level_b<float><<<fg, TS, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yB, plan.front_total, plan.n, sel);
        else
            mf_fwd_level_b<double><<<fg, TS, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n, sel);
    }
    for (int L = plan.num_plevels - 1; L >= 0; --L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        int max_cb = 1;  // dynamic shared for the bwd x-cache, sized by the level's max CB rows
        for (int q = b; q < e; ++q) {
            const int pp = plan.h_plcols[q];
            const int cbq = (plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp]) - plan.h_ncols[pp];
            if (cbq > max_cb) max_cb = cbq;
        }
        const dim3 bg(e - b, B);
        const size_t bsh = (size_t)max_cb * sizeof(double);
        if (pure_fp32)
            mf_bwd_level_b<float><<<bg, TS, bsh, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yB, plan.front_total, plan.n, sel);
        else
            mf_bwd_level_b<double><<<bg, TS, bsh, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n, sel);
    }
    cudaGraph_t sg;
    cudaStreamEndCapture(stream, &sg);
    cudaGraphExec_t sge;
    cudaGraphInstantiate(&sge, sg, nullptr, nullptr, 0);
    cudaGraphDestroy(sg);
    st.solve_graph_exec = sge;
    return cudaGetLastError() == cudaSuccess;
}

bool batched_factorize(const MultifrontalPlan& plan, BatchedState& st, const double* d_valuesB,
                       const int* d_o2c, double* kernel_ms)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const long fe = (long)st.B * plan.front_total;
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
    // Zero + scatter A into the front the factor graph consumes: the FP32 front (pure-FP32) or the
    // FP64 front / master (all other modes; the FP32 working arena is overwritten per front).
    cudaMemsetAsync(st.d_sing, 0, sizeof(int), stream);
    const int T = 256;
    const dim3 sgrid((plan.nnz_a + T - 1) / T, st.B);
    if (st.prec == BatchPrecision::FP32) {
        cudaMemsetAsync(st.d_frontBf, 0, fe * sizeof(float), stream);
        scatter_batched<float><<<sgrid, T, 0, stream>>>(plan.nnz_a, plan.front_total, d_o2c,
                                                        plan.d_a_pos, d_valuesB, st.d_frontBf);
    } else {
        cudaMemsetAsync(st.d_frontB, 0, fe * sizeof(double), stream);
        scatter_batched<double><<<sgrid, T, 0, stream>>>(plan.nnz_a, plan.front_total, d_o2c,
                                                         plan.d_a_pos, d_valuesB, st.d_frontB);
    }
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.factor_graph_exec), stream);
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) { float ms = 0; cudaEventElapsedTime(&ms, k0, k1); *kernel_ms = ms; }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
    return cudaGetLastError() == cudaSuccess;
}

bool batched_solve(const MultifrontalPlan& plan, BatchedState& st, const double* d_rhsB,
                   double* d_solB, const int* d_perm, double* kernel_ms)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const int n = plan.n;
    const int T = 256;
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
    gather_rhs_b<<<dim3((n + T - 1) / T, st.B), T, 0, stream>>>(n, d_rhsB, d_perm, st.d_yB);
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.solve_graph_exec), stream);
    scatter_sol_b<<<dim3((n + T - 1) / T, st.B), T, 0, stream>>>(n, st.d_yB, d_perm, d_solB);
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) { float ms = 0; cudaEventElapsedTime(&ms, k0, k1); *kernel_ms = ms; }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace custom_linear_solver::factorize
