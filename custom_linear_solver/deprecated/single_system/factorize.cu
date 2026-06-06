#include "factorize/multifrontal.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "symbolic/multifrontal.hpp"
#include "symbolic/supernode.hpp"
#include "solve/multifrontal.hpp"

namespace custom_linear_solver::factorize {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// One block per front: BLOCKED (rank-nc) dense no-pivot LU. Instead of nc rank-1
// passes that re-read/write the whole trailing block and sync nc times, factor the
// nc-wide panel (the part with the sequential pivot dependency), then do a SINGLE
// rank-nc trailing update -- the embarrassingly-parallel bulk -- in one pass. This
// cuts trailing-block traffic and __syncthreads ~nc x, speeding up the few big
// critical-path fronts (cy85: the big fronts are latency/sync bound, not bandwidth).
// The trailing (fsz-nc) block is left as the contribution block for the parent.
// FUSED factor + extend-add, one block per front, one kernel launch per level
// (was two: factor then extend). After a block factors its front the CB is ready,
// so it immediately extend-adds into the parent front -- the parent is at a strictly
// higher level (not factored this launch), so the atomicAdd is race-free. Halves the
// graph nodes and removes one inter-kernel sync per level (cuts the level
// serialization on the deep power-grid etrees, ~72 levels on SyntheticUSA).
template <typename FT>
__global__ void mf_factor_extend_level(int lbegin, int lend, const int* __restrict__ plcols,
                                       const int* __restrict__ front_off,
                                       const int* __restrict__ front_ptr,
                                       const int* __restrict__ ncols,
                                       const int* __restrict__ panel_parent,
                                       const int* __restrict__ asm_ptr,
                                       const int* __restrict__ asm_local, FT* front,
                                       int* sing, int do_extend)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;  // trailing / U-panel column count

    if (fsz <= 48) {
        // Small fronts: per-pivot rank-1 loop (the blocked 3-phase overhead only
        // pays off on the big critical-path fronts).
        for (int k = 0; k < nc; ++k) {
            FT piv = F[(long)k * fsz + k];
            if (piv == FT(0)) {
                if (t == 0) *sing = 1;
                piv = FT(1);
            }
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
        // Big fronts: BLOCKED rank-nc. Phase 1: factor the nc-wide panel (full
        // height), trailing untouched.
        for (int k = 0; k < nc; ++k) {
            FT piv = F[(long)k * fsz + k];
            if (piv == FT(0)) {
                if (t == 0) *sing = 1;
                piv = FT(1);
            }
            for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
            __syncthreads();
            const int pc = nc - 1 - k;  // panel columns to the right of k
            for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
                const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
                F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
            }
            if (pc > 0) __syncthreads();
        }
        // Phase 2: U panel (rows 0..nc-1, cols >= nc), triangular solve, seq in k.
        for (int k = 1; k < nc; ++k) {
            for (int e = t; e < uc; e += nt) {
                const int jj = nc + e;
                FT v = F[(long)k * fsz + jj];
                for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
                F[(long)k * fsz + jj] = v;
            }
            __syncthreads();
        }
        // Phase 3: single rank-nc trailing update (the embarrassingly-parallel bulk).
        for (int e = t; e < uc * uc; e += nt) {
            const int ii = nc + e / uc, jj = nc + e % uc;
            FT acc = 0;
            for (int k = 0; k < nc; ++k) acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
            F[(long)ii * fsz + jj] -= acc;
        }
    }

    // Extend-add this front's CB (trailing uc x uc) into the parent front. Sync so
    // the whole CB is visible before any thread scatters it.
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;  // do_extend=0 = diagnostic (breaks correctness)
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

// MIXED-PRECISION factor (cy132): FP64 "master" front for the assembly (precise, no FP32
// cancellation), FP32 "working" copy for the bandwidth-bound dense LU. Per front: narrow
// master->working, FP32 LU on working, write the L/U+CB back to master (FP64), then
// extend-add the working CB into the PARENT's FP64 master (precise accumulation). Pivots
// derive from the precise FP64 master (narrowed) so they stay nonzero -> stable; the
// within-front FP32 LU is ~1e-6 (recovered by iterative refinement). Gives the FP32
// bandwidth win on the LU bulk while keeping the assembly in FP64.
__global__ void mf_factor_extend_mixed(int lbegin, int lend, const int* __restrict__ plcols,
                                       const int* __restrict__ front_off,
                                       const int* __restrict__ front_ptr,
                                       const int* __restrict__ ncols,
                                       const int* __restrict__ panel_parent,
                                       const int* __restrict__ asm_ptr,
                                       const int* __restrict__ asm_local, double* master,
                                       float* working, int* sing, int do_extend)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    double* M = master + front_off[p];
    float* W = working + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const long fsz2 = (long)fsz * fsz;
    for (long e = t; e < fsz2; e += nt) W[e] = (float)M[e];  // narrow master -> working
    __syncthreads();
    if (fsz <= 48) {
        for (int k = 0; k < nc; ++k) {
            float piv = W[(long)k * fsz + k];
            if (piv == 0.0f) {
                if (t == 0) *sing = 1;
                piv = 1.0f;
            }
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
            if (piv == 0.0f) {
                if (t == 0) *sing = 1;
                piv = 1.0f;
            }
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
    // Writeback only the L/U (the nc pivot rows + the nc-wide L panel) to master -- the
    // CB/trailing (uc x uc, most of the front) is never read from master (emit/solve use
    // only L/U; the parent's extend-add reads the working buffer below). Skipping it cuts
    // the writeback from fsz^2 to ~2*nc*fsz (~6x fewer global writes).
    for (long e = t; e < (long)nc * fsz; e += nt) M[e] = (double)W[e];  // rows 0..nc (U+pivots)
    for (int e = t; e < uc * nc; e += nt) {                             // L panel (rows>=nc,col<nc)
        const long idx2 = (long)(nc + e / nc) * fsz + (e % nc);
        M[idx2] = (double)W[idx2];
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

// ---- cy147: multi-block big-front factor ------------------------------------------
// onetone2's 4.5x gap is its big fronts (>=257, 68.8% of work) run one-block-each ->
// a handful of blocks on 82 SMs (massive under-utilization). Split each big-front level
// into 3 graph-ordered kernels so the embarrassingly-parallel trailing update (the bulk)
// spreads many blocks per front across the GPU. The default fused path is untouched;
// this triple runs only for big-front levels. FP64 master only.
// (A) panel (Phase 1) + U panel (Phase 2): one block per front, sequential phases.
__global__ void mf_bigA_panelU(int lbegin, int lend, const int* __restrict__ plcols,
                               const int* __restrict__ front_off,
                               const int* __restrict__ front_ptr,
                               const int* __restrict__ ncols, double* front, int* sing)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    double* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    for (int k = 0; k < nc; ++k) {  // Phase 1: factor the nc-wide panel (full height)
        double piv = F[(long)k * fsz + k];
        if (piv == 0.0) { if (t == 0) *sing = 1; piv = 1.0; }
        for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
        __syncthreads();
        const int pc = nc - 1 - k;
        for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
            const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        if (pc > 0) __syncthreads();
    }
    for (int k = 1; k < nc; ++k) {  // Phase 2: U panel triangular solve
        for (int e = t; e < uc; e += nt) {
            const int jj = nc + e;
            double v = F[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
            F[(long)k * fsz + jj] = v;
        }
        __syncthreads();
    }
}
// (B) Phase 3 trailing rank-nc update: grid (gridDim.x=tiles, gridDim.y=front-in-level).
// Each front's uc*uc outputs are strided across gridDim.x blocks -> many blocks/front.
// cy270: shared-mem TILED rank-nc trailing update. The naive version re-reads the L-row
// F[ii][0..nc] and U-col F[0..nc][jj] from global for every trailing element (rank-nc GEMM,
// arithmetic intensity ~nc/(2*nc) reads -> BW/L2-bound). Tiling stages a TSxnc L-tile and an
// ncxTS U-tile into shared so each panel value is reused TS times. One block per (TSxTS output
// tile, front); grid.x = ceil(maxfsz/TS)^2 (out-of-range tiles early-return). blockDim = TS*TS.
constexpr int MF_BT_TS = 16;
constexpr int kFactorTinyThreads = 128;
constexpr int kFactorTinyFrontCount = 1024;
constexpr int kFactorBigThreads = 768;
constexpr int kFactorDefaultThreads = 384;
constexpr int kBigMultiFrontThreshold = 81;
constexpr int kBigExtendTiles = 16;
__global__ void mf_bigB_trailing(int lbegin, int lend, const int* __restrict__ plcols,
                                 const int* __restrict__ front_off,
                                 const int* __restrict__ front_ptr,
                                 const int* __restrict__ ncols, double* front)
{
    const int idx = lbegin + blockIdx.y;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    double* F = front + front_off[p];
    const int uc = fsz - nc;
    const int ntc = (uc + MF_BT_TS - 1) / MF_BT_TS;
    if (ntc <= 0) return;
    const int ti = blockIdx.x / ntc, tj = blockIdx.x % ntc;
    if (ti >= ntc) return;  // this front has fewer tiles than grid.x provides
    // cy271: DYNAMIC shared sized by the level's max nc (not a static 64 bound) -> for the
    // common nc<=8 the L-/U-tile is ~2KB not 16KB, lifting occupancy from ~3 to the
    // thread-limited ~6-8 blocks/SM. Layout: Lsh[TS][nc] (row-major), then Ush[nc][TS].
    extern __shared__ double sh[];
    double* Lsh = sh;                       // Lsh[rr*nc + k]
    double* Ush = sh + MF_BT_TS * nc;       // Ush[k*TS + cc]
    const int t = threadIdx.x;
    for (int e = t; e < MF_BT_TS * nc; e += MF_BT_TS * MF_BT_TS) {
        const int rr = e / nc, k = e % nc;
        const int ii = nc + ti * MF_BT_TS + rr;
        Lsh[e] = (ii < fsz) ? F[(long)ii * fsz + k] : 0.0;
    }
    for (int e = t; e < nc * MF_BT_TS; e += MF_BT_TS * MF_BT_TS) {
        const int k = e / MF_BT_TS, cc = e % MF_BT_TS;
        const int jj = nc + tj * MF_BT_TS + cc;
        Ush[e] = (jj < fsz) ? F[(long)k * fsz + jj] : 0.0;
    }
    __syncthreads();
    const int r = t / MF_BT_TS, c = t % MF_BT_TS;
    const int ii = nc + ti * MF_BT_TS + r, jj = nc + tj * MF_BT_TS + c;
    if (ii < fsz && jj < fsz) {
        double acc = 0.0;
        for (int k = 0; k < nc; ++k) acc += Lsh[r * nc + k] * Ush[k * MF_BT_TS + c];
        F[(long)ii * fsz + jj] -= acc;
    }
}
// (C) extend-add the CB into the parent. cy273: MULTI-BLOCK (gridDim.x tiles x gridDim.y
// fronts) -- the uc*uc atomicAdds are independent (distinct parent slots, low contention),
// so spreading them across tiles uses more SMs on spine levels with few big fronts (the
// extend was the only big-front kernel still one-block-per-front). Each block strides its
// share of uc*uc; out-of-range (small uc) blocks no-op.
__global__ void mf_bigC_extend(int lbegin, int lend, const int* __restrict__ plcols,
                               const int* __restrict__ front_off,
                               const int* __restrict__ front_ptr,
                               const int* __restrict__ ncols,
                               const int* __restrict__ panel_parent,
                               const int* __restrict__ asm_ptr,
                               const int* __restrict__ asm_local, double* front)
{
    const int idx = lbegin + blockIdx.y;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int par = panel_parent[p];
    if (par < 0) return;
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const int uc = fsz - nc;
    double* F = front + front_off[p];
    double* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    const int t = threadIdx.x, nt = blockDim.x;
    for (long e = (long)blockIdx.x * nt + t; e < (long)uc * uc; e += (long)gridDim.x * nt) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  F[(long)(nc + a) * fsz + (nc + b)]);
    }
}

template <typename FT, typename VT>
__global__ void mf_scatter_csr_values(int nnz_a, const int* __restrict__ ordered_value_to_csr,
                                      const int* __restrict__ a_pos,
                                      const VT* __restrict__ csr_values, FT* front)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnz_a) return;
    const int pos = a_pos[q];
    if (pos >= 0) atomicAdd(&front[pos], static_cast<FT>(csr_values[ordered_value_to_csr[q]]));
}

__device__ int front_lower_bound(const int* front_rows, int begin, int end, int value)
{
    int lo = begin;
    int hi = end;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (front_rows[mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return lo - begin;
}

__global__ void build_a_pos_device(int n, const int* __restrict__ Ap,
                                   const int* __restrict__ Ai,
                                   const int* __restrict__ panel_of,
                                   const int* __restrict__ panel_first,
                                   const int* __restrict__ panel_ncols,
                                   const int* __restrict__ front_off,
                                   const int* __restrict__ front_ptr,
                                   const int* __restrict__ front_rows,
                                   int* __restrict__ a_pos)
{
    const int j = blockIdx.x;
    if (j >= n) return;
    for (int q = Ap[j] + threadIdx.x; q < Ap[j + 1]; q += blockDim.x) {
        const int i = Ai[q];
        const int mn = i < j ? i : j;
        const int mx = i < j ? j : i;
        const int owner = panel_of[mn];
        const int first = panel_first[owner];
        const int ncols = panel_ncols[owner];
        const int fbegin = front_ptr[owner];
        const int fend = front_ptr[owner + 1];
        const int fsz = fend - fbegin;
        const int mn_local = mn - first;
        const int mx_local = (mx < first + ncols)
                                 ? (mx - first)
                                 : front_lower_bound(front_rows, fbegin, fend, mx);
        const int ri = (i < j) ? mn_local : mx_local;
        const int ci = (i < j) ? mx_local : mn_local;
        a_pos[q] = front_off[owner] + ri * fsz + ci;
    }
}

constexpr int MF_REG_NC = 16;    // register-partial bound; nc <= panel_cap (cy338: 8->16 to test bigger cap)

// cy335/336 partitioned-inverse: invert each front's nc x nc pivot block in place -- U_pp (upper,
// incl diagonal) AND L_pp (unit-lower, strict-lower; unit diagonal implicit). Runs after all factor
// levels. The backward solve then uses a parallel GEMV x = Uinv @ rhs, and the forward uses
// sh_piv = Linv @ rhs, both replacing sequential triangular
// solves. Only the nc x nc block is touched (the L/U panels at rows/cols >= nc are unchanged). One
// block per front, one thread per inverse column (nc<=8); shared scratch so all reads of the original
// L/U finish before the in-place write-back. Adds a little FACTOR time (user OK'd F dropping as long
// as it still beats cuDSS); the win is on S.
template <typename FT>
__global__ void mf_invert_pivot(int npanels, const int* __restrict__ front_ptr,
                                const int* __restrict__ front_off, const int* __restrict__ ncols,
                                FT* front)
{
    const int p = blockIdx.x;
    if (p >= npanels) return;
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int j = threadIdx.x;  // one thread per inverse column j (blockDim >= nc)
    __shared__ FT Ui[MF_REG_NC * MF_REG_NC];  // Uinv[i][j], upper (i<=j)
    __shared__ FT Li[MF_REG_NC * MF_REG_NC];  // Linv[i][j], unit-lower (i>=j)
    if (j < nc) {
        // Uinv column j: back-substitution up the column (needs Uinv[k][j], k>i)
        Ui[j * nc + j] = FT(1) / F[(long)j * fsz + j];
        for (int i = j - 1; i >= 0; --i) {
            FT v = FT(0);
            for (int k = i + 1; k <= j; ++k) v -= F[(long)i * fsz + k] * Ui[k * nc + j];
            Ui[i * nc + j] = v / F[(long)i * fsz + i];
        }
        // Linv column j: forward-substitution down the column (unit diagonal; needs Linv[k][j], k<i)
        Li[j * nc + j] = FT(1);
        for (int i = j + 1; i < nc; ++i) {
            FT v = FT(0);
            for (int k = j; k < i; ++k) v -= F[(long)i * fsz + k] * Li[k * nc + j];
            Li[i * nc + j] = v;
        }
    }
    __syncthreads();  // all reads of the original L/U done before any in-place write-back
    if (j < nc) {
        for (int i = 0; i <= j; ++i) F[(long)i * fsz + j] = Ui[i * nc + j];      // upper incl diag <- Uinv
        for (int i = j + 1; i < nc; ++i) F[(long)i * fsz + j] = Li[i * nc + j];  // strict lower <- Linv
    }
}

// Warp-per-front factor for tiny-front levels (cy* ported from batched/factor_small.cuh into the
// single-batch path). The block kernel above is 1 block / front with 128–384 threads even when fsz≤16
// (256 elem fits in 8 lanes), so SM% on the wide-base levels (L0–L2 in case8387) is 1–4 %. Packing W
// warps per block, one warp per front, runs the no-pivot LU lane-parallel with __syncwarp() (cheaper
// than a multi-warp __syncthreads) and stages the front in per-warp shared. Used only when the level's
// maxfsz fits comfortably in one warp's working set.
constexpr int kFactorSmallWarpsPerBlock = 8;       // 8 fronts packed per block (256 threads)
constexpr int kFactorSmallWarpFszCap = 32;         // fsz>32 levels hand back to the block kernel
constexpr int kFactorSmallWarpMinCnt = 32;         // need enough fronts to amortize block start
constexpr int kFactorSmallWarpOptInShared = 96 * 1024;  // sm_86 max dyn-shared per block

template <typename FT>
__device__ __forceinline__ void lu_small_warp_inplace(FT* F, int fsz, int nc, int lane, int* sing)
{
    for (int k = 0; k < nc; ++k) {
        FT piv = F[(long)k * fsz + k];
        if (piv == FT(0)) {
            if (lane == 0) *sing = 1;
            piv = FT(1);
        }
        for (int i = k + 1 + lane; i < fsz; i += 32) F[(long)i * fsz + k] /= piv;
        __syncwarp();
        const int m = fsz - k - 1;
        for (int e = lane; e < m * m; e += 32) {
            const int ii = k + 1 + e / m, jj = k + 1 + e % m;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        __syncwarp();
    }
}

template <typename FT>
__device__ __forceinline__ void writeback_lu_warp(FT* M, const FT* W, int fsz, int nc, int uc,
                                                  int lane)
{
    // Pivot rows (rows 0..nc, full width) + L panel (rows >= nc, cols < nc). The CB
    // (trailing uc x uc) stays in W and is extend-added into the parent below.
    for (long e = lane; e < (long)nc * fsz; e += 32) M[e] = W[e];
    for (int e = lane; e < uc * nc; e += 32) {
        const long id2 = (long)(nc + e / nc) * fsz + (e % nc);
        M[id2] = W[id2];
    }
}

// Shared-staged variant of mf_factor_extend_level for mid fronts (fsz roughly 33-72). Stages the
// whole front into block-shared at start, runs the fused factor+extend in shared, writes back L/U
// at the end, and lets the parent's extend-add read CB from shared. Targets the same code path as
// the existing block kernel but with all rank-1 updates resolved against shared memory rather than
// global, killing the strided F[ii*fsz+k] reads. Dynamic shared = fsz_cap² doubles.
template <typename FT>
__global__ void mf_factor_extend_level_shared(int lbegin, int lend, int fsz2cap,
                                              const int* __restrict__ plcols,
                                              const int* __restrict__ front_off,
                                              const int* __restrict__ front_ptr,
                                              const int* __restrict__ ncols,
                                              const int* __restrict__ panel_parent,
                                              const int* __restrict__ asm_ptr,
                                              const int* __restrict__ asm_local, FT* front,
                                              int* sing, int do_extend)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;

    extern __shared__ unsigned char ext_sh_raw[];
    FT* Fs = reinterpret_cast<FT*>(ext_sh_raw);

    // Stage-in (coalesced)
    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    // Rank-1 LU in shared
    for (int k = 0; k < nc; ++k) {
        FT piv = Fs[(long)k * fsz + k];
        if (piv == FT(0)) {
            if (t == 0) *sing = 1;
            piv = FT(1);
        }
        for (int i = k + 1 + t; i < fsz; i += nt) Fs[(long)i * fsz + k] /= piv;
        __syncthreads();
        const int m = fsz - k - 1;
        for (int e = t; e < m * m; e += nt) {
            const int ii = k + 1 + e / m, jj = k + 1 + e % m;
            Fs[(long)ii * fsz + jj] -= Fs[(long)ii * fsz + k] * Fs[(long)k * fsz + jj];
        }
        __syncthreads();
    }

    // Writeback only the L/U (CB stays in shared for the extend-add below).
    for (long e = t; e < (long)nc * fsz; e += nt) F[e] = Fs[e];
    for (int e = t; e < uc * nc; e += nt) {
        const long id2 = (long)(nc + e / nc) * fsz + (e % nc);
        F[id2] = Fs[id2];
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
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}

constexpr int kFactorSharedFszMin = 33;            // overlap with warp upper bound (32)
constexpr int kFactorSharedFszMax = 72;            // 72^2 * 8 = 41KB, fits default shared

bool use_shared_factor(int maxfsz, int cnt)
{
    if (const char* off = std::getenv("CLS_NO_SHARED_FACTOR")) {
        if (off[0] && off[0] != '0') return false;
    }
    return maxfsz >= kFactorSharedFszMin && maxfsz <= kFactorSharedFszMax && cnt >= 2;
}

template <typename FT>
__global__ void mf_factor_small_warp(int lbegin, int lend, int fsz2cap,
                                     const int* __restrict__ plcols,
                                     const int* __restrict__ front_off,
                                     const int* __restrict__ front_ptr,
                                     const int* __restrict__ ncols,
                                     const int* __restrict__ panel_parent,
                                     const int* __restrict__ asm_ptr,
                                     const int* __restrict__ asm_local, FT* front, int* sing,
                                     int do_extend)
{
    extern __shared__ unsigned char smem_sw_raw[];
    FT* smem_sw = reinterpret_cast<FT*>(smem_sw_raw);
    const int warp_in_blk = threadIdx.x >> 5;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int level_size = lend - lbegin;
    if (warp_global >= level_size) return;
    const int p = plcols[lbegin + warp_global];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    FT* Fs = smem_sw + (long)warp_in_blk * fsz2cap;

    for (int e = lane; e < fsz2; e += 32) Fs[e] = F[e];
    __syncwarp();
    lu_small_warp_inplace<FT>(Fs, fsz, nc, lane, sing);
    __syncwarp();
    writeback_lu_warp<FT>(F, Fs, fsz, nc, uc, lane);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    FT* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = lane; e < uc * uc; e += 32) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}

int factor_threads_for_level(const MultifrontalPlan& plan, int L)
{
    const int cnt = plan.plptr[L + 1] - plan.plptr[L];
    int mx = 0;
    for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
        const int pp = plan.h_plcols[q];
        mx = std::max(mx, plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp]);
    }
    if (mx >= 257) return kFactorBigThreads;
    if (mx <= 48 && cnt >= kFactorTinyFrontCount) return kFactorTinyThreads;
    return kFactorDefaultThreads;
}

// Warp-per-front routing for tiny-fsz levels. OFF by default in single-batch: 8-trial sweep on
// case8387 (FP64) showed +3.4% factor regression vs the block kernel for the levels that match
// (L0,L1: cnt ~ 4000/1500, fsz ≤ 20). Reason: the block kernel at 128 threads already saturates
// the grid (>1000 blocks → SM 38%, occupancy 78%), so warp packing reduces total threads in flight
// without freeing meaningful resources. Kept as opt-in (CLS_USE_SMALL_WARP=1) for research and
// for matrices with different shape where it might pay off.
bool use_small_warp_factor(int maxfsz, int cnt)
{
    const char* on = std::getenv("CLS_USE_SMALL_WARP");
    if (!on || !on[0] || on[0] == '0') return false;
    return maxfsz > 0 && maxfsz <= kFactorSmallWarpFszCap && cnt >= kFactorSmallWarpMinCnt;
}

[[maybe_unused]] void issue_factor_levels(MultifrontalPlan& plan, cudaStream_t stream)
{
    constexpr int do_extend = 1;

    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L];
        const int e = plan.plptr[L + 1];
        if (e <= b) continue;
        const int T = factor_threads_for_level(plan, L);
        int mxf = 0;
        for (int q = b; q < e; ++q)
            mxf = std::max(mxf, plan.h_front_ptr[plan.h_plcols[q] + 1] -
                                    plan.h_front_ptr[plan.h_plcols[q]]);
        if (!plan.fp32 && !plan.pure_fp32 && mxf >= kBigMultiFrontThreshold) {
            mf_bigA_panelU<<<e - b, T, 0, stream>>>(b, e, plan.d_plcols, plan.d_front_off,
                plan.d_front_ptr, plan.d_ncols, plan.d_front, plan.d_sing);
            const int mxtc = (mxf + MF_BT_TS - 1) / MF_BT_TS;
            int mxnc = 1;
            for (int q = b; q < e; ++q) mxnc = std::max(mxnc, plan.h_ncols[plan.h_plcols[q]]);
            const size_t bt_sh = (size_t)2 * MF_BT_TS * mxnc * sizeof(double);
            mf_bigB_trailing<<<dim3(mxtc * mxtc, e - b), MF_BT_TS * MF_BT_TS, bt_sh, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front);
            mf_bigC_extend<<<dim3(kBigExtendTiles, e - b), T, 0, stream>>>(b, e,
                plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_front);
        } else if (plan.pure_fp32) {
            mf_factor_extend_level<float><<<e - b, T, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_frontf,
                plan.d_sing, do_extend);
        } else if (plan.fp32) {
            mf_factor_extend_mixed<<<e - b, T, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_front,
                plan.d_frontf, plan.d_sing, do_extend);
        } else {
            mf_factor_extend_level<double><<<e - b, T, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_front,
                plan.d_sing, do_extend);
        }
    }

    if (plan.pure_fp32) {
        mf_invert_pivot<float><<<plan.num_panels, 32, 0, stream>>>(
            plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, plan.d_frontf);
    } else {
        mf_invert_pivot<double><<<plan.num_panels, 32, 0, stream>>>(
            plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, plan.d_front);
    }
}

}  // namespace

MultifrontalPlan analyze_multifrontal(int n, int nnz_a, const int* d_Ap, const int* d_Ai,
                                      const std::vector<int>& Lp,
                                      const std::vector<int>& Li,
                                      const std::vector<int>& parent, int panel_cap, bool fp32,
                                      const custom_linear_solver::symbolic::PanelPartition* forced_panels,
                                      bool pure_fp32)
{
    namespace sym = custom_linear_solver::symbolic;
    constexpr bool solve_f32 = false;
    auto lap = [](const char*) {};
    MultifrontalPlan plan;
    plan.n = n;
    plan.nnz_a = nnz_a;
    if (n <= 0) return plan;

    std::vector<int> colcount(n);
    for (int j = 0; j < n; ++j) colcount[j] = Lp[j + 1] - Lp[j];
    // cy169: A bigger cap merges longer etree
    // chains into one panel -> fewer fronts -> fewer SOLVE levels (the dominant S cost:
    // 2*num_plevels serialized kernels). Trades padded fill (F flops) for S latency --
    // viable where we have F-margin. Clamp to MF_MAX_NC=64 (shared pivot-buffer bound;
    // nc<=cap). cap>MF_REG_NC overflows the register-partial path's part[]/piv[].
    // cy338: post-partitioned-inverse (cy335/336) the per-front solve is a cheap GEMV, so the
    // S benefit of a SHALLOWER panel-etree (fewer serialized solve levels) now outweighs the
    // padded-fill cost on the deep/large matrices, AND the user OK'd F dropping as long as it
    // still beats cuDSS. ADAPTIVE cap by size: small matrices keep cap8 (thin F margin, no S
    // gain from amalgamation); larger ones amalgamate more (they have F headroom + deeper
    // spines). Swept (cy338): cap12 cuts S -4..-10% on the 16k-80k band with F still < cuDSS;
    // cap16 on the >80k band (huge F margin). cap16 breaks F on small power-grid / onetone2, so
    // it is gated to the largest.
    // cy338 adaptive cap, retuned for the shared-resident mid/big-front batched kernels (which now
    // handle wider fronts efficiently): the largest matrices amalgamate harder (cap20) -- fewer,
    // denser fronts and fewer solve levels outweigh the extra padded fill now that big fronts run
    // well -- while the mid band stays at 12 (cap16+ regressed it) and small at panel_cap.
    int eff_cap = n >= 80000 ? 20 : (n >= 16000 ? 12 : panel_cap);
    if (const char* ce = std::getenv("CLS_CAP")) eff_cap = std::atoi(ce);  // research override
    if (eff_cap < 1) eff_cap = 1;
    if (eff_cap > 64) eff_cap = 64;
    const sym::PanelPartition panels =
        forced_panels ? *forced_panels : sym::relaxed_panels(n, parent, colcount, eff_cap);
    lap("relaxed_panels");
    const sym::MultifrontalSymbolic mf = sym::multifrontal_symbolic(n, Lp, Li, panels);
    lap("multifrontal_symbolic");
    const int P = panels.num_panels;
    plan.num_panels = P;
    plan.h_front_ptr = mf.front_ptr;   // host copies for batched dispatch / TC shared sizing
    plan.h_ncols = panels.ncols;
    // h_plcols filled after plcols is built below.

    // Per-panel front offset (in doubles) into the front arena = prefix sum of fsz².
    std::vector<int> front_off(P + 1, 0);
    long total = 0;
    for (int p = 0; p < P; ++p) {
        const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
        front_off[p] = static_cast<int>(total);
        total += fsz * fsz;
        if (total > (1L << 30)) {  // > 1G doubles (8GB) -> bail out, keep cy71 path
            return MultifrontalPlan{};
        }
    }
    front_off[P] = static_cast<int>(total);
    plan.front_total = total;
    plan.h_front_off = front_off;  // host mirror for per-panel cuBLAS dispatch (Phase Σ.6)

    // Phase Σ.8: per-panel pivot offsets for within-panel partial pivoting (CLS_USE_PIVOTING=1).
    plan.h_pivot_offset.assign(P + 1, 0);
    for (int p = 0; p < P; ++p) plan.h_pivot_offset[p + 1] = plan.h_pivot_offset[p] + panels.ncols[p];
    plan.total_pivots = plan.h_pivot_offset[P];  // = n
    if (cudaMalloc(&plan.d_pivot_offset, (size_t)(P + 1) * sizeof(int)) == cudaSuccess) {
        cudaMemcpy(plan.d_pivot_offset, plan.h_pivot_offset.data(),
                   (size_t)(P + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Panel-etree levels (parent id > child in postorder -> single forward pass).
    std::vector<int> plevel(P, 0);
    int num_plevels = 0;
    for (int p = 0; p < P; ++p) {
        const int par = mf.panel_parent[p];
        if (par != -1) plevel[par] = std::max(plevel[par], plevel[p] + 1);
    }
    for (int p = 0; p < P; ++p) num_plevels = std::max(num_plevels, plevel[p] + 1);
    plan.num_plevels = num_plevels;
    plan.plptr.assign(num_plevels + 1, 0);
    for (int p = 0; p < P; ++p) ++plan.plptr[plevel[p] + 1];
    for (int L = 0; L < num_plevels; ++L) plan.plptr[L + 1] += plan.plptr[L];
    std::vector<int> plcols(P);

    if (std::getenv("CLS_PANEL_DUMP")) {  // per-panel (fsz, nc) for GEMM-fraction analysis
        fprintf(stderr, "[CLS_PANEL_DUMP] panel,fsz,nc,uc\n");
        for (int p = 0; p < P; ++p) {
            const int fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
            const int nc = panels.ncols[p];
            fprintf(stderr, "%d,%d,%d,%d\n", p, fsz, nc, fsz - nc);
        }
    }
    if (std::getenv("CLS_DUMP")) {  // research: front-size distribution + per-level structure
        const int NB = 7;
        const int edges[NB] = {16, 32, 48, 64, 96, 160, 1 << 30};
        long cnt[NB] = {0}, sf2[NB] = {0};
        double sf3[NB] = {0};
        long tot2 = 0; double tot3 = 0;
        for (int p = 0; p < P; ++p) {
            const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
            int bck = 0; while (fsz > edges[bck]) ++bck;
            cnt[bck]++; sf2[bck] += fsz * fsz; sf3[bck] += (double)fsz * fsz * fsz;
            tot2 += fsz * fsz; tot3 += (double)fsz * fsz * fsz;
        }
        fprintf(stderr, "[CLS_DUMP] n=%d P=%d levels=%d cap=%d front_total(MB f32)=%.1f\n",
                n, P, num_plevels, eff_cap, total * 4.0 / 1e6);
        const int lo[NB] = {1, 17, 33, 49, 65, 97, 161};
        for (int bk = 0; bk < NB; ++bk)
            fprintf(stderr, "  fsz[%4d..%-9d] cnt=%-8ld f2%%=%5.1f f3%%=%5.1f\n", lo[bk],
                    edges[bk] == (1 << 30) ? 999999 : edges[bk], cnt[bk],
                    100.0 * sf2[bk] / std::max(1L, tot2), 100.0 * sf3[bk] / std::max(1e-9, tot3));
        for (int L = 0; L < num_plevels && L < 40; ++L) {
            long c = 0, m = 0, s2 = 0;
            for (int p = 0; p < P; ++p)
                if (plevel[p] == L) {
                    const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
                    ++c; m = std::max(m, fsz); s2 += fsz * fsz;
                }
            fprintf(stderr, "  L%-2d cnt=%-7ld maxfsz=%-4ld f2=%ld\n", L, c, m, s2);
        }
    }
    {
        std::vector<int> next(plan.plptr.begin(), plan.plptr.end());
        for (int p = 0; p < P; ++p) plcols[next[plevel[p]]++] = p;
    }
    plan.h_plcols = plcols;

    // --- Spine + K-subtree partition (Phase 1 of tree-restructuring research) ---
    // Spine = contiguous "cnt=1 chain" at the top of the panel etree.
    // Below the spine: K subtree roots = panels at the level just below the spine bottom.
    // Each subtree is the closure of one of those roots (all descendants by panel_parent).
    {
        // Compute cnt-per-level (we already have plptr which gives cnt = plptr[L+1] - plptr[L]).
        auto level_cnt = [&](int L) { return plan.plptr[L + 1] - plan.plptr[L]; };

        // Spine: walk down from the top level while cnt == 1.
        int spine_top = num_plevels - 1;
        int spine_bot = spine_top;
        while (spine_bot >= 0 && level_cnt(spine_bot) == 1) --spine_bot;
        ++spine_bot;  // now spine_bot = lowest L with cnt=1 in the chain (start of spine)
        if (spine_bot <= spine_top) {
            plan.spine_start_level = spine_bot;
            // Collect spine panels in factor order (bottom -> top).
            plan.h_spine_panels.clear();
            for (int L = spine_bot; L <= spine_top; ++L) {
                for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
                    plan.h_spine_panels.push_back(plan.h_plcols[q]);
                }
            }
        } else {
            // No spine at all (root level has cnt > 1).
            plan.spine_start_level = -1;
            plan.h_spine_panels.clear();
        }

        // Phase Σ.11 — correct subtree partition. The previous version assumed every below-
        // spine panel is reachable from some panel at level (spine_start_level - 1). But the
        // panel etree can have a panel at L<spine_start_level whose parent is in the spine
        // *directly* (skipping the "expected root level"). Those panels were stranded,
        // causing multi-stream dispatch to lose them and producing relres ~ 1e+28 garbage.
        //
        // Correct definition: a panel p's subtree root is the highest ancestor whose
        // panel_parent is in the spine (or is -1, i.e. p is its own root). All panels with
        // the same subtree-root form one subtree. Subtree roots = the set of such anchors.
        plan.h_subtree_of_panel.assign(P, -1);
        // Mark spine panels (those with subtree_of remaining -1 == spine).
        std::vector<char> is_spine(P, 0);
        if (plan.spine_start_level >= 0) {
            for (int sp : plan.h_spine_panels) is_spine[sp] = 1;
        }
        // For each non-spine panel, find its subtree root: walk up panel_parent until we
        // hit a panel whose parent is in spine (or -1). Topological order (parent id > child id)
        // means we can dynamic-program in increasing id with one pass.
        std::vector<int> subtree_root_of(P, -1);
        for (int p = 0; p < P; ++p) {
            if (is_spine[p]) continue;
            const int par = mf.panel_parent[p];
            if (par < 0 || is_spine[par]) {
                subtree_root_of[p] = p;  // p itself is the subtree root
            } else {
                subtree_root_of[p] = subtree_root_of[par];  // inherit (parent already processed)
            }
        }
        // (Note: parent id > child id holds for postorder panel ids, so subtree_root_of[par]
        // is set before we read it... wait, parent id > child id means par > p, so when we
        // process p we haven't processed par yet! Fix: iterate in REVERSE id order. Parent
        // comes before child in REVERSE, so subtree_root_of[par] is set first.)
        std::fill(subtree_root_of.begin(), subtree_root_of.end(), -1);
        for (int p = P - 1; p >= 0; --p) {
            if (is_spine[p]) continue;
            const int par = mf.panel_parent[p];
            if (par < 0 || is_spine[par]) {
                subtree_root_of[p] = p;
            } else {
                subtree_root_of[p] = subtree_root_of[par];
            }
        }
        // Collect distinct subtree roots with their member counts.
        std::vector<int> root_member_count(P, 0);
        for (int p = 0; p < P; ++p) {
            if (subtree_root_of[p] >= 0) ++root_member_count[subtree_root_of[p]];
        }
        std::vector<int> distinct_roots;
        for (int p = 0; p < P; ++p) {
            if (root_member_count[p] > 0) distinct_roots.push_back(p);
        }
        // Sort by member count desc.
        std::sort(distinct_roots.begin(), distinct_roots.end(),
                  [&](int a, int b) { return root_member_count[a] > root_member_count[b]; });

        // Cap to MAX_SUBTREES (matches TCState.subtree_streams[8]). When more distinct roots
        // exist (case8387 fix: K=29 of which only 2 are large), keep the top (K_cap-1)
        // largest as separate subtrees and merge ALL remaining roots' members into one
        // "spillover" subtree (id K_cap-1). The spillover subtree is processed on its own
        // stream level-by-level; correctness holds because within one stream the level
        // dispatch order respects panel-etree dependencies.
        constexpr int MAX_SUBTREES = 8;
        plan.h_subtree_roots.clear();
        std::vector<int> root_to_id(P, -1);
        const int kept = std::min((int)distinct_roots.size(), MAX_SUBTREES - 1);
        for (int i = 0; i < kept; ++i) {
            root_to_id[distinct_roots[i]] = i;
            plan.h_subtree_roots.push_back(distinct_roots[i]);
        }
        // If there are more roots, all "extra" roots map to id = kept (spillover bin).
        if ((int)distinct_roots.size() > kept) {
            for (int i = kept; i < (int)distinct_roots.size(); ++i) {
                root_to_id[distinct_roots[i]] = kept;
            }
            // The spillover bin needs a representative root for h_subtree_roots[kept]; use
            // the first "extra" root (member count smaller but valid panel id).
            plan.h_subtree_roots.push_back(distinct_roots[kept]);
        }
        plan.num_subtrees = (int)plan.h_subtree_roots.size();
        // Assign subtree id per panel via its subtree_root_of.
        for (int p = 0; p < P; ++p) {
            const int r = subtree_root_of[p];
            if (r < 0) continue;
            plan.h_subtree_of_panel[p] = root_to_id[r];
        }

        // Phase 3 — Re-sort plcols within each level so panels of the same subtree are
        // contiguous (subtree 0 first, then subtree 1, ..., then -1=spine). After this,
        // for subtree k at level L the range is [off, off+cnt) inside plcols.
        if (plan.num_subtrees > 0) {
            plan.h_subtree_level_off.assign((long)plan.num_subtrees * num_plevels, 0);
            plan.h_subtree_level_cnt.assign((long)plan.num_subtrees * num_plevels, 0);
            // For each level: bucket-sort panels by subtree_of_panel (spine=-1 goes last)
            for (int L = 0; L < num_plevels; ++L) {
                const int lo = plan.plptr[L], hi = plan.plptr[L + 1];
                std::vector<int> bucketed;
                bucketed.reserve(hi - lo);
                // First, subtree 0..K-1
                for (int k = 0; k < plan.num_subtrees; ++k) {
                    plan.h_subtree_level_off[(long)k * num_plevels + L] = (int)bucketed.size() + lo;
                    int cnt = 0;
                    for (int q = lo; q < hi; ++q) {
                        const int p = plcols[q];
                        if (plan.h_subtree_of_panel[p] == k) {
                            bucketed.push_back(p);
                            ++cnt;
                        }
                    }
                    plan.h_subtree_level_cnt[(long)k * num_plevels + L] = cnt;
                }
                // Then spine panels (subtree_of_panel == -1) at the end
                for (int q = lo; q < hi; ++q) {
                    const int p = plcols[q];
                    if (plan.h_subtree_of_panel[p] == -1) bucketed.push_back(p);
                }
                // Write back to plcols
                for (size_t i = 0; i < bucketed.size(); ++i) plcols[lo + i] = bucketed[i];
            }
            // h_plcols was already assigned from plcols above; refresh.
            plan.h_plcols = plcols;
        }

        if (std::getenv("CLS_TREE_INFO")) {
            fprintf(stderr, "[CLS_TREE_INFO] P=%d levels=%d  spine=[L%d..L%d] (%zu panels)  K=%d\n",
                    P, num_plevels, plan.spine_start_level, num_plevels - 1,
                    plan.h_spine_panels.size(), plan.num_subtrees);
            for (int k = 0; k < plan.num_subtrees; ++k) {
                int sz = 0;
                for (int p = 0; p < P; ++p) if (plan.h_subtree_of_panel[p] == k) ++sz;
                fprintf(stderr, "  subtree %d: root panel %d, %d panels\n",
                        k, plan.h_subtree_roots[k], sz);
                // Per-level breakdown for this subtree
                for (int L = 0; L < num_plevels && L < 12; ++L) {
                    const int c = plan.h_subtree_level_cnt[(long)k * num_plevels + L];
                    if (c > 0) fprintf(stderr, "    sub%d L%d cnt=%d\n", k, L, c);
                }
            }
        }
    }

    // Parent-local extend-add map: asm_idx is a global front_rows index; subtract
    // the parent front's start so the kernel indexes into the parent's fsz x fsz.
    std::vector<int> asm_local(mf.asm_idx.size());
    for (int p = 0; p < P; ++p) {
        const int par = mf.panel_parent[p];
        const int base = (par >= 0) ? mf.front_ptr[par] : 0;
        for (int a = mf.asm_ptr[p]; a < mf.asm_ptr[p + 1]; ++a)
            asm_local[a] = mf.asm_idx[a] - base;
    }
    plan.asm_total = static_cast<int>(asm_local.size());

    // One arena, 256-byte-aligned sub-arrays (avoids the L2 straddle from cycle 39).
    auto al = [](long b) { return (b + 255) & ~static_cast<long>(255); };
    long off = 0;
    const long o_front = off; off = al(off + total * sizeof(double));
    const long o_foff = off;  off = al(off + (long)(P + 1) * sizeof(int));
    const long o_fptr = off;  off = al(off + (long)(P + 1) * sizeof(int));
    const long o_nc = off;    off = al(off + (long)P * sizeof(int));
    const long o_plc = off;   off = al(off + (long)P * sizeof(int));
    const long o_par = off;   off = al(off + (long)P * sizeof(int));
    const long o_aptr = off;  off = al(off + (long)(P + 1) * sizeof(int));
    const long o_aloc = off;  off = al(off + (long)std::max(1, plan.asm_total) * sizeof(int));
    const long o_apos = off;  off = al(off + (long)std::max(1, plan.nnz_a) * sizeof(int));
    const int front_store = mf.front_ptr[P];
    plan.front_store = front_store;
    const long o_fr = off;    off = al(off + (long)std::max(1, front_store) * sizeof(int));
    const long o_pf = off;    off = al(off + (long)std::max(1, P) * sizeof(int));
    const long o_pof = off;   off = al(off + (long)std::max(1, n) * sizeof(int));
    const long o_y = off;     off = al(off + (long)n * sizeof(double));
    const long o_sing = off;  off = al(off + sizeof(int));
    cudaMalloc(&plan.arena, off);
    char* base = static_cast<char*>(plan.arena);
    plan.d_front = reinterpret_cast<double*>(base + o_front);
    // `fp32` means single-system mixed factor (FP64 master + FP32 working). `pure_fp32`
    // means the factor and solve front are float.
    plan.fp32 = fp32;
    plan.pure_fp32 = pure_fp32;
    // FP32 "working" arena for the mixed factor (double-master design): a separate
    // allocation (front_total floats), narrowed/written-back per front during factor.
    // The FP64 master (d_front) holds the assembly + final L/U used by solve.
    if (fp32 || pure_fp32 || solve_f32) cudaMalloc(&plan.d_frontf, (long)total * sizeof(float));
    plan.d_front_off = reinterpret_cast<int*>(base + o_foff);
    plan.d_front_ptr = reinterpret_cast<int*>(base + o_fptr);
    plan.d_ncols = reinterpret_cast<int*>(base + o_nc);
    plan.d_plcols = reinterpret_cast<int*>(base + o_plc);
    plan.d_panel_parent = reinterpret_cast<int*>(base + o_par);
    plan.d_asm_ptr = reinterpret_cast<int*>(base + o_aptr);
    plan.d_asm_local = reinterpret_cast<int*>(base + o_aloc);
    plan.d_a_pos = reinterpret_cast<int*>(base + o_apos);
    plan.d_front_rows = reinterpret_cast<int*>(base + o_fr);
    plan.d_y = reinterpret_cast<double*>(base + o_y);
    plan.d_sing = reinterpret_cast<int*>(base + o_sing);
    if (pure_fp32) cudaMalloc(&plan.d_yf, (long)n * sizeof(float));
    int* d_panel_first = reinterpret_cast<int*>(base + o_pf);
    int* d_panel_of = reinterpret_cast<int*>(base + o_pof);

    auto H2D = [](int* d, const std::vector<int>& v) {
        if (!v.empty()) cudaMemcpy(d, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice);
    };
    H2D(plan.d_front_off, front_off);
    H2D(plan.d_front_ptr, mf.front_ptr);
    H2D(plan.d_ncols, panels.ncols);
    H2D(plan.d_plcols, plcols);
    H2D(plan.d_panel_parent, mf.panel_parent);
    H2D(plan.d_asm_ptr, mf.asm_ptr);
    H2D(plan.d_asm_local, asm_local);
    H2D(plan.d_front_rows, mf.front_rows);
    // Upload spine panel list (Phase 4). Separate allocation so it survives plan moves.
    if (!plan.h_spine_panels.empty()) {
        cudaMalloc(&plan.d_spine_panels, plan.h_spine_panels.size() * sizeof(int));
        cudaMemcpy(plan.d_spine_panels, plan.h_spine_panels.data(),
                   plan.h_spine_panels.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    lap("arena_malloc+H2D");
    H2D(d_panel_first, panels.first);
    H2D(d_panel_of, panels.panel_of);
    build_a_pos_device<<<n, 128>>>(n, d_Ap, d_Ai, d_panel_of, d_panel_first, plan.d_ncols,
                                   plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                                   plan.d_a_pos);
    const cudaError_t apos_err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (apos_err != cudaSuccess) return MultifrontalPlan{};
    lap("a_pos_device");

    // Capture the level-scheduled schedule (factor+extend per level) in
    // one CUDA graph; replayed each factorize after the value-dependent A scatter.
#ifdef CLS_INTERNAL_GRAPH
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    plan.stream = stream;
    plan.owns_stream = true;
    // Per-level factor block size (cy140, measured under a HOST-locked 1395MHz clock ->
    // cross-process variance <0.1%, so these <5% deltas are real, not the cy125 artifact).
    // Medium fronts (fsz<=256: ALL power-grid + rajat27/memplus/rajat15) prefer 384
    // threads/block: -1.3..-5.8% factor vs the old flat 512 (512 over-allocates -> fewer
    // blocks/SM for the medium hotspot). BUT onetone2's work is 68.8% in fronts >=257
    // (deep serial critical path) which want MORE threads: a finer sweep (cy141, locked
    // clock) gives medium fronts 384 (beats 320/448/512 on every power-grid + rajat27/
    // memplus matrix) and onetone2's big fronts 768 (45.02@512 -> 44.33@768 = -1.5%;
    // 1024 over-allocates). onetone2 is the ONLY matrix with fronts >=257, so the 768
    // branch touches nothing else. So: max front >=257 -> 768, else 384.
    // Tiny-front leaf levels (max fsz <=48) hold ~99% of the fronts but ~25% of work
    // (cy143 profiler). At the medium-front 384 they badly under-occupy (a fsz<=16 front
    // has <=256 elems but gets 384 threads -> ~5 blocks/SM). A 128-thread block packs
    // many more blocks/SM for these embarrassingly-parallel leaves -> -5..-7% factor on
    // every matrix. BUT only when the level has MANY tiny fronts (>=1024) to fill the GPU
    // with 128-blocks; a sparse tiny level under-utilizes at 128, so it keeps 384 (cy143
    // locked-clock sweep: a count guard flips the small-matrix regression into a win).
    auto level_ft = [&](int L) -> int {
        const int cnt = plan.plptr[L + 1] - plan.plptr[L];
        int mx = 0;
        for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
            const int pp = plcols[q];
            mx = std::max(mx, mf.front_ptr[pp + 1] - mf.front_ptr[pp]);
        }
        if (mx >= 257) return kFactorBigThreads;   // onetone2 big serial fronts
        if (mx <= 48 && cnt >= kFactorTinyFrontCount)
            return kFactorTinyThreads;  // MANY tiny fronts -> occupancy
        return kFactorDefaultThreads;  // medium hotspot 49-256
    };
    constexpr int do_extend = 1;
    // cy148: multi-block ALL levels whose max front >=81 (the medium-large 81-256 fronts are
    // FEW-per-level near the root -> one-block-each under-uses the 82 SMs). cy270: the trailing
    // is now a SHARED-MEM TILED rank-nc GEMM (mf_bigB_trailing) -- grid.x = ceil(maxfsz/TS)^2
    // output tiles per front, each block staging an L-/U-tile (reuse TS x). Widens the power F
    // win (SyntheticUSA 4.10->4.05, ACTIVSg25k 1.78->1.76); onetone2's gap is deep-etree (plev
    // 110) serialization not trailing-BW, so it stays ~neutral there.
    // Opt-in to >48KB dynamic shared once: only the FP64 warp-per-front kernel may exceed it
    // (maxfsz=32 with W=8 -> 64KB). Safe to call before capture; applies for graph launch.
    {
        cudaError_t aerr = cudaFuncSetAttribute(mf_factor_small_warp<double>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kFactorSmallWarpOptInShared);
        (void)aerr;
        // Shared-staged block kernel uses fsz_cap² doubles per block (up to 72² = ~41KB, default OK).
        // Bump the carveout anyway so wider sweeps work.
        aerr = cudaFuncSetAttribute(mf_factor_extend_level_shared<double>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kFactorSmallWarpOptInShared);
        (void)aerr;
    }
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int L = 0; L < num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        const int T = level_ft(L);
        int mxf = 0;
        for (int q = b; q < e; ++q)
            mxf = std::max(mxf, mf.front_ptr[plcols[q] + 1] - mf.front_ptr[plcols[q]]);
        if (!fp32 && !pure_fp32 && mxf >= kBigMultiFrontThreshold) {
            mf_bigA_panelU<<<e - b, T, 0, stream>>>(b, e, plan.d_plcols, plan.d_front_off,
                plan.d_front_ptr, plan.d_ncols, plan.d_front, plan.d_sing);
            // cy270/271: TILED trailing -- grid.x = ceil(maxfsz/TS)^2 tiles per front, blockDim=TS*TS,
            // DYNAMIC shared = 2*TS*maxnc doubles (sized by the level's max nc for occupancy).
            const int mxtc = (mxf + MF_BT_TS - 1) / MF_BT_TS;
            int mxnc = 1;
            for (int q = b; q < e; ++q) mxnc = std::max(mxnc, panels.ncols[plcols[q]]);
            const size_t bt_sh = (size_t)2 * MF_BT_TS * mxnc * sizeof(double);
            mf_bigB_trailing<<<dim3(mxtc * mxtc, e - b), MF_BT_TS * MF_BT_TS, bt_sh, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols, plan.d_front);
            // cy273: multi-block the extend (independent atomics) -> grid.x tiles x fronts.
            // cy275: swept tile count -> 16 optimal.
            mf_bigC_extend<<<dim3(kBigExtendTiles, e - b), T, 0, stream>>>(b, e,
                plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_front);
        } else if (!fp32 && !pure_fp32 && use_small_warp_factor(mxf, e - b)) {
            // Warp-per-front: W warps per block, dynamic shared = W * mxf^2 doubles. W shrinks as
            // fsz grows so that the per-block shared stays under the sm_86 opt-in limit while still
            // packing enough warps for SM occupancy.
            int W;
            if (mxf <= 24) W = 8;        // 25KB shared at fsz=24, default-shared OK
            else if (mxf <= 32) W = 8;   // 64KB shared at fsz=32, needs opt-in
            else W = 4;                  // 73KB at fsz=48, needs opt-in
            const int n_blocks = ((e - b) + W - 1) / W;
            const int fsz2cap = mxf * mxf;
            const size_t sh_bytes = (size_t)W * fsz2cap * sizeof(double);
            if (std::getenv("CLS_WARP_DBG"))
                fprintf(stderr, "[WARP] L=%d cnt=%d mxf=%d W=%d nb=%d sh=%zuB\n",
                        L, e - b, mxf, W, n_blocks, sh_bytes);
            mf_factor_small_warp<double><<<n_blocks, W * 32, sh_bytes, stream>>>(
                b, e, fsz2cap, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local,
                plan.d_front, plan.d_sing, do_extend);
        } else if (!fp32 && !pure_fp32 && use_shared_factor(mxf, e - b)) {
            // Shared-staged mid-front kernel (one block / front but front resident in shared).
            const int fsz2cap = mxf * mxf;
            const size_t sh_bytes = (size_t)fsz2cap * sizeof(double);
            if (std::getenv("CLS_WARP_DBG"))
                fprintf(stderr, "[SHARED] L=%d cnt=%d mxf=%d T=%d sh=%zuB\n",
                        L, e - b, mxf, T, sh_bytes);
            mf_factor_extend_level_shared<double><<<e - b, T, sh_bytes, stream>>>(
                b, e, fsz2cap, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local,
                plan.d_front, plan.d_sing, do_extend);
        } else if (pure_fp32)
            mf_factor_extend_level<float><<<e - b, T, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_frontf,
                plan.d_sing, do_extend);
        else if (fp32)  // double-master: FP64 master assembly + FP32 working LU
            mf_factor_extend_mixed<<<e - b, T, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_front,
                plan.d_frontf, plan.d_sing, do_extend);
        else
            mf_factor_extend_level<double><<<e - b, T, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_front, plan.d_sing,
                do_extend);
    }
    // cy335 partitioned-inverse: invert each front's U_pp pivot block. The backward solve then uses
    // a parallel GEMV instead of a sequential triangular
    // back-solve. Appended to the factor graph -> counted in factor time.
    constexpr bool selinv = true;
    if (pure_fp32)
        mf_invert_pivot<float><<<plan.num_panels, 32, 0, stream>>>(
            plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, plan.d_frontf);
    else
        mf_invert_pivot<double><<<plan.num_panels, 32, 0, stream>>>(
            plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, plan.d_front);
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    plan.graph_exec = exec;

    custom_linear_solver::solve::capture_multifrontal_solve_graph(plan, mf.front_ptr, plcols,
                                                                  solve_f32 || pure_fp32,
                                                                  pure_fp32,
                                                                  selinv);
    lap("solve_graph_capture");
#else
    (void)solve_f32;
#endif
    return plan;
}

template <typename VT>
bool factorize_multifrontal_device_T(MultifrontalPlan& plan, const VT* d_csr_values,
                                     const int* d_ordered_value_to_csr, double* kernel_ms)
{
    const int n = plan.n;
    if (n <= 0 || plan.num_panels == 0 || d_csr_values == nullptr ||
        d_ordered_value_to_csr == nullptr)
        return false;

    cudaStream_t stream = static_cast<cudaStream_t>(plan.stream);

    const int T = 128;
#ifdef CLS_INTERNAL_GRAPH
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
#else
    (void)kernel_ms;
#endif
    // FP64 master holds the assembly (both modes); the FP32 working arena is overwritten
    // per front by the mixed kernel's narrow, so it needs no memset.
    if (plan.pure_fp32) {
        cudaMemsetAsync(plan.d_frontf, 0, plan.front_total * sizeof(float), stream);
    } else {
        cudaMemsetAsync(plan.d_front, 0, plan.front_total * sizeof(double), stream);
    }
    cudaMemsetAsync(plan.d_sing, 0, sizeof(int), stream);
    const int sb = (plan.nnz_a + T - 1) / T;  // scatter A into the FP64 master (both modes)
    if (plan.pure_fp32) {
        mf_scatter_csr_values<float, VT><<<sb, T, 0, stream>>>(
            plan.nnz_a, d_ordered_value_to_csr, plan.d_a_pos, d_csr_values, plan.d_frontf);
    } else {
        mf_scatter_csr_values<double, VT><<<sb, T, 0, stream>>>(
            plan.nnz_a, d_ordered_value_to_csr, plan.d_a_pos, d_csr_values, plan.d_front);
    }
#ifdef CLS_INTERNAL_GRAPH
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(plan.graph_exec), stream);
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, k0, k1);
        *kernel_ms = ms;
    }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
#else
    issue_factor_levels(plan, stream);
#endif

    return cudaGetLastError() == cudaSuccess;
}

bool factorize_multifrontal_device(MultifrontalPlan& plan, const double* d_csr_values,
                                   const int* d_ordered_value_to_csr, double* kernel_ms)
{
    return factorize_multifrontal_device_T(plan, d_csr_values, d_ordered_value_to_csr, kernel_ms);
}

bool factorize_multifrontal_device(MultifrontalPlan& plan, const float* d_csr_values,
                                   const int* d_ordered_value_to_csr, double* kernel_ms)
{
    return factorize_multifrontal_device_T(plan, d_csr_values, d_ordered_value_to_csr, kernel_ms);
}

}  // namespace custom_linear_solver::factorize
