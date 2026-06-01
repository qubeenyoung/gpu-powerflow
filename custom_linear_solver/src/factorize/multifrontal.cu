#include "factorize/multifrontal.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <thread>
#include <utility>
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
                double v = F[(long)k * fsz + jj];
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

// ---- cy147: opt-in MULTI-BLOCK big-front factor (MF_BIGMULTI) ---------------------
// onetone2's 4.5x gap is its big fronts (>=257, 68.8% of work) run one-block-each ->
// a handful of blocks on 82 SMs (massive under-utilization). Split each big-front level
// into 3 graph-ordered kernels so the embarrassingly-parallel trailing update (the bulk)
// spreads many blocks per front across the GPU. The default fused path is untouched;
// this triple runs only for big-front levels when MF_BIGMULTI is set. FP64 master only.
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

__global__ void mf_scatter_csr_values(int nnz_a, const int* __restrict__ ordered_value_to_csr,
                                      const int* __restrict__ a_pos,
                                      const double* __restrict__ csr_values, double* front)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnz_a) return;
    const int pos = a_pos[q];
    if (pos >= 0) atomicAdd(&front[pos], csr_values[ordered_value_to_csr[q]]);
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
__global__ void mf_invert_pivot(int npanels, const int* __restrict__ front_ptr,
                                const int* __restrict__ front_off, const int* __restrict__ ncols,
                                double* front)
{
    const int p = blockIdx.x;
    if (p >= npanels) return;
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    double* F = front + front_off[p];
    const int j = threadIdx.x;  // one thread per inverse column j (blockDim >= nc)
    __shared__ double Ui[MF_REG_NC * MF_REG_NC];  // Uinv[i][j], upper (i<=j)
    __shared__ double Li[MF_REG_NC * MF_REG_NC];  // Linv[i][j], unit-lower (i>=j)
    if (j < nc) {
        // Uinv column j: back-substitution up the column (needs Uinv[k][j], k>i)
        Ui[j * nc + j] = 1.0 / F[(long)j * fsz + j];
        for (int i = j - 1; i >= 0; --i) {
            double v = 0.0;
            for (int k = i + 1; k <= j; ++k) v -= F[(long)i * fsz + k] * Ui[k * nc + j];
            Ui[i * nc + j] = v / F[(long)i * fsz + i];
        }
        // Linv column j: forward-substitution down the column (unit diagonal; needs Linv[k][j], k<i)
        Li[j * nc + j] = 1.0;
        for (int i = j + 1; i < nc; ++i) {
            double v = 0.0;
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

}  // namespace

MultifrontalPlan analyze_multifrontal(int n, int nnz_a, const int* d_Ap, const int* d_Ai,
                                      const std::vector<int>& Lp,
                                      const std::vector<int>& Li,
                                      const std::vector<int>& parent, int panel_cap, bool fp32)
{
    namespace sym = custom_linear_solver::symbolic;
    const bool tm = std::getenv("MF_TIME") != nullptr;  // analysis sub-phase profiling
    const bool solve_f32 = std::getenv("MF_SOLVE_F32") != nullptr;  // cy170: FP32 solve A/B
    if (std::getenv("MF_MIXED") != nullptr) fp32 = true;  // force FP64-master/FP32-LU mixed factor
    auto tclk = std::chrono::steady_clock::now();
    auto lap = [&](const char* nm) {
        if (tm) {
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "  [analyze] %-22s %.2f ms\n", nm,
                         std::chrono::duration<double, std::milli>(now - tclk).count());
            tclk = now;
        }
    };
    MultifrontalPlan plan;
    plan.n = n;
    plan.nnz_a = nnz_a;
    if (n <= 0) return plan;

    std::vector<int> colcount(n);
    for (int j = 0; j < n; ++j) colcount[j] = Lp[j + 1] - Lp[j];
    // cy169: MF_CAP sweeps the amalgamation level. A bigger cap merges longer etree
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
    // it is gated to the largest. MF_CAP env overrides for sweeps.
    const char* cap_s = std::getenv("MF_CAP");
    int eff_cap = cap_s ? std::atoi(cap_s)
                        : (n >= 80000 ? 16 : (n >= 16000 ? 12 : panel_cap));
    if (eff_cap < 1) eff_cap = 1;
    if (eff_cap > 64) eff_cap = 64;
    const sym::PanelPartition panels = sym::relaxed_panels(n, parent, colcount, eff_cap);
    if (tm) {
        long tf = Lp[n];
        std::fprintf(stderr, "  [amalg] cap=%d panels=%d padded_fill=%ld pad_ratio=%.2fx n/panels=%.1f\n",
                     eff_cap, panels.num_panels, panels.padded_fill,
                     tf > 0 ? (double)panels.padded_fill / tf : 0.0,
                     panels.num_panels > 0 ? (double)n / panels.num_panels : 0.0);
    }
    lap("relaxed_panels");
    const sym::MultifrontalSymbolic mf = sym::multifrontal_symbolic(n, Lp, Li, panels);
    lap("multifrontal_symbolic");
    const int P = panels.num_panels;
    plan.num_panels = P;

    // Front-size-band profile (env MF_FRONTPROF): where the factor FLOPS live. Dense
    // front work ~ fsz^2 * nc (panel factor) + uc^2 * nc (trailing). Buckets by fsz so
    // tuning targets the real hotspot (cy123 plan target #1). Read-only diagnostic.
    if (std::getenv("MF_FRONTPROF")) {
        const int NB = 5;
        const int lo[NB] = {0, 17, 49, 129, 257};
        const int hi[NB] = {16, 48, 128, 256, 1 << 30};
        long cnt[NB] = {0}, flops[NB] = {0};
        long tot_flops = 0;
        for (int p = 0; p < P; ++p) {
            const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
            const long nc = panels.ncols[p];
            const long f = fsz * fsz * nc;  // ~dense LU work for this front
            tot_flops += f;
            for (int b = 0; b < NB; ++b)
                if (fsz >= lo[b] && fsz <= hi[b]) { cnt[b] += 1; flops[b] += f; break; }
        }
        std::fprintf(stderr, "  MF_FRONTPROF P=%d tot_work=%.3gG\n", P, (double)tot_flops / 1e9);
        for (int b = 0; b < NB; ++b)
            std::fprintf(stderr, "    fsz[%4d..%-7d] cnt=%-8ld work=%-7.3gG (%.1f%%)\n", lo[b],
                         hi[b] == (1 << 30) ? 999999 : hi[b], cnt[b], (double)flops[b] / 1e9,
                         tot_flops ? 100.0 * flops[b] / tot_flops : 0.0);
    }

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
    {
        std::vector<int> next(plan.plptr.begin(), plan.plptr.end());
        for (int p = 0; p < P; ++p) plcols[next[plevel[p]]++] = p;
    }
    if (tm) {
        // cy171: solve is latency-bound (low BW utilization) -> profile per-level
        // parallelism. A level with < ~82 fronts (RTX3090 SM count) under-fills the GPU.
        // Report how much solve-work (Σ fsz*nc) sits in those occupancy-starved levels:
        // that work is the upside a finer cross-level (DAG) schedule could overlap.
        const int SM = 82;
        int narrow_levels = 0;
        long work_narrow = 0, work_total = 0, crit = 0;
        for (int L = 0; L < num_plevels; ++L) {
            const int cnt = plan.plptr[L + 1] - plan.plptr[L];
            long wl = 0, wmax = 0;
            for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
                const int p = plcols[q];
                const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
                const long w = fsz * (long)panels.ncols[p];
                wl += w;
                if (w > wmax) wmax = w;
            }
            work_total += wl;
            crit += wmax;  // critical-path proxy: each level contributes its biggest front
            if (cnt < SM) { ++narrow_levels; work_narrow += wl; }
        }
        std::fprintf(stderr,
                     "  [solve-prof] levels=%d narrow(<%dSM)=%d work_in_narrow=%.0f%% "
                     "crit_path/total=%.0f%% (finer-schedule floor)\n",
                     num_plevels, SM, narrow_levels,
                     work_total ? 100.0 * work_narrow / work_total : 0.0,
                     work_total ? 100.0 * crit / work_total : 0.0);
    }
    if (std::getenv("MF_MS_PROFILE")) {
        // cy173: size the IMPLEMENTABLE multi-subtree-solve prize. Partition the panel etree
        // into independent subtrees (concurrent across streams) + a shared top of separators
        // above them (sequential). The shared-top work fraction s bounds the single-solve
        // speedup: makespan ~ s + (1-s)/G  ->  speedup ~ 1/(s+(1-s)/G). Greedy: split the
        // heaviest frontier node into its children until the frontier has >= G subtrees.
        std::vector<std::vector<int>> kids(P);
        std::vector<int> roots;
        for (int p = 0; p < P; ++p) {
            const int par = mf.panel_parent[p];
            if (par >= 0) kids[par].push_back(p); else roots.push_back(p);
        }
        std::vector<long> sw(P, 0), own(P, 0);
        long W = 0;
        for (int p = 0; p < P; ++p) {
            const long w = (long)(mf.front_ptr[p + 1] - mf.front_ptr[p]) * panels.ncols[p];
            own[p] = w; sw[p] = w; W += w;
        }
        for (int L = 0; L < num_plevels; ++L)  // children-before-parent (level order)
            for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
                const int p = plcols[q], par = mf.panel_parent[p];
                if (par >= 0) sw[par] += sw[p];
            }
        for (int G : {2, 4, 8}) {
            std::priority_queue<std::pair<long, int>> heap;
            for (int r : roots) heap.push({sw[r], r});
            long shared = 0;
            while ((int)heap.size() < G && !heap.empty()) {
                auto [w, u] = heap.top();
                if (kids[u].empty()) break;
                heap.pop(); shared += own[u];
                for (int c : kids[u]) heap.push({sw[c], c});
            }
            long maxsub = 0, sumsub = 0;
            int nsub = heap.size();
            while (!heap.empty()) { auto [w, u] = heap.top(); heap.pop(); maxsub = std::max(maxsub, w); sumsub += w; }
            const double s = W ? (double)shared / W : 0.0;
            const double sp = (s + (W ? (double)maxsub / W : 0.0));  // makespan frac ~ shared + slowest stream
            std::fprintf(stderr,
                         "  [ms-prof G=%d] subtrees=%d shared_top=%.0f%% max_subtree=%.0f%% "
                         "balance=%.2f est_speedup~%.1fx\n",
                         G, nsub, 100.0 * s, W ? 100.0 * maxsub / W : 0.0,
                         sumsub ? (double)maxsub * nsub / sumsub : 0.0, sp > 0 ? 1.0 / sp : 0.0);
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
    // FP32 view of the same arena region: the double front reserves total*8 bytes, the
    // FP32 front needs only total*4, so it fits in-place (no extra alloc). Only one of
    // d_front/d_frontf is used per plan, chosen by fp32.
    plan.fp32 = fp32;
    // FP32 "working" arena for the mixed factor (double-master design): a separate
    // allocation (front_total floats), narrowed/written-back per front during factor.
    // The FP64 master (d_front) holds the assembly + final L/U used by solve.
    if (fp32 || solve_f32) cudaMalloc(&plan.d_frontf, (long)total * sizeof(float));
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
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    plan.stream = stream;
    // 512 threads/block: the big fronts near the root (up to 214x214) sit on the
    // sequential critical path and underutilize the GPU with one small block each;
    // more threads/front cuts their dense-LU time. Swept 128/256/512/1024 ->
    // 512 best (1024 over-allocates: fewer blocks/SM). 2.2-2.6x on large matrices.
    const char* ft_s = std::getenv("MF_FT");  // factor block-size override (force/sweep)
    const int ft_force = ft_s ? std::atoi(ft_s) : 0;
    // Per-level factor block size (cy140, measured under a HOST-locked 1395MHz clock ->
    // cross-process variance <0.1%, so these <5% deltas are real, not the cy125 artifact).
    // Medium fronts (fsz<=256: ALL power-grid + rajat27/memplus/rajat15) prefer 384
    // threads/block: -1.3..-5.8% factor vs the old flat 512 (512 over-allocates -> fewer
    // blocks/SM for the medium hotspot). BUT onetone2's work is 68.8% in fronts >=257
    // (deep serial critical path) which want MORE threads: a finer sweep (cy141, locked
    // clock) gives medium fronts 384 (beats 320/448/512 on every power-grid + rajat27/
    // memplus matrix) and onetone2's big fronts 768 (45.02@512 -> 44.33@768 = -1.5%;
    // 1024 over-allocates). onetone2 is the ONLY matrix with fronts >=257, so the 768
    // branch touches nothing else. So: max front >=257 -> 768, else 384. MF_FT forces.
    // Tiny-front leaf levels (max fsz <=48) hold ~99% of the fronts but ~25% of work
    // (cy143 profiler). At the medium-front 384 they badly under-occupy (a fsz<=16 front
    // has <=256 elems but gets 384 threads -> ~5 blocks/SM). A 128-thread block packs
    // many more blocks/SM for these embarrassingly-parallel leaves -> -5..-7% factor on
    // every matrix. BUT only when the level has MANY tiny fronts (>=1024) to fill the GPU
    // with 128-blocks; a sparse tiny level under-utilizes at 128, so it keeps 384 (cy143
    // locked-clock sweep: a count guard flips the small-matrix regression into a win).
    const char* fts_s = std::getenv("MF_FT_TINY");  // tiny-front-level block size (sweep)
    const int ft_tiny = fts_s ? std::atoi(fts_s) : 128;
    const char* ftc_s = std::getenv("MF_FT_TINY_CNT");  // min level front-count for tiny block
    const int ft_tiny_cnt = ftc_s ? std::atoi(ftc_s) : 1024;
    auto level_ft = [&](int L) -> int {
        if (ft_force) return ft_force;
        const int cnt = plan.plptr[L + 1] - plan.plptr[L];
        int mx = 0;
        for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
            const int pp = plcols[q];
            mx = std::max(mx, mf.front_ptr[pp + 1] - mf.front_ptr[pp]);
        }
        if (mx >= 257) return 768;   // onetone2 big serial fronts
        if (mx <= 48 && cnt >= ft_tiny_cnt) return ft_tiny;  // MANY tiny fronts -> occupancy
        return 384;  // medium hotspot 49-256 (cy146: splitting 49-128 off regresses; 384 best)
    };
    const int do_extend = std::getenv("MF_SKIP_EXTEND") == nullptr ? 1 : 0;  // diagnostic
    const char* flo_s = std::getenv("MF_FLO");  // diagnostic: factor only levels [FLO,FHI)
    const char* fhi_s = std::getenv("MF_FHI");
    const int flo = flo_s ? std::atoi(flo_s) : 0;
    const int fhi = fhi_s ? std::atoi(fhi_s) : num_plevels;
    const bool bigmulti = std::getenv("MF_NO_BIGMULTI") == nullptr;  // cy147 DEFAULT ON
    const char* mbt_s = std::getenv("MB_THRESH");  // cy148: multi-block max-front trigger
    // cy148: multi-block ALL levels whose max front >=81 (the medium-large 81-256 fronts are
    // FEW-per-level near the root -> one-block-each under-uses the 82 SMs). cy270: the trailing
    // is now a SHARED-MEM TILED rank-nc GEMM (mf_bigB_trailing) -- grid.x = ceil(maxfsz/TS)^2
    // output tiles per front, each block staging an L-/U-tile (reuse TS x). Widens the power F
    // win (SyntheticUSA 4.10->4.05, ACTIVSg25k 1.78->1.76); onetone2's gap is deep-etree (plev
    // 110) serialization not trailing-BW, so it stays ~neutral there.
    const int mb_thresh = mbt_s ? std::atoi(mbt_s) : 81;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int L = flo; L < fhi; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        const int T = level_ft(L);
        int mxf = 0;
        for (int q = b; q < e; ++q)
            mxf = std::max(mxf, mf.front_ptr[plcols[q] + 1] - mf.front_ptr[plcols[q]]);
        if (bigmulti && !fp32 && mxf >= mb_thresh) {  // big-front level -> multi-block triple
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
            if (do_extend) {
                // cy273: multi-block the extend (independent atomics) -> grid.x tiles x fronts.
                // cy275: swept tile count -> 16 optimal (64/128 over-spread = atomic contention +
                // launch overhead; <16 under-parallelizes). 16 beats cy273's 64/128 on all (onetone2
                // 8.31->7.84, SyntheticUSA 3.19->3.10). MF_XTILES env overrides.
                const char* xt_s = std::getenv("MF_XTILES");
                const int xtiles = xt_s ? std::atoi(xt_s) : 16;
                mf_bigC_extend<<<dim3(xtiles, e - b), T, 0, stream>>>(b, e, plan.d_plcols,
                    plan.d_front_off, plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent,
                    plan.d_asm_ptr, plan.d_asm_local, plan.d_front);
            }
        } else if (fp32)  // double-master: FP64 master assembly + FP32 working LU
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
    // back-solve. Appended to the factor graph -> counted in factor time. Opt-out MF_NO_SELINV.
    const bool selinv = std::getenv("MF_NO_SELINV") == nullptr;
    if (selinv)
        mf_invert_pivot<<<plan.num_panels, 32, 0, stream>>>(
            plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, plan.d_front);
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    plan.graph_exec = exec;

    custom_linear_solver::solve::capture_multifrontal_solve_graph(plan, mf.front_ptr, plcols,
                                                                  solve_f32, selinv);
    lap("solve_graph_capture");
    return plan;
}

bool factorize_multifrontal_device(MultifrontalPlan& plan, const double* d_csr_values,
                                   const int* d_ordered_value_to_csr, double* kernel_ms)
{
    const int n = plan.n;
    if (n <= 0 || plan.num_panels == 0 || d_csr_values == nullptr ||
        d_ordered_value_to_csr == nullptr)
        return false;

    cudaStream_t stream = static_cast<cudaStream_t>(plan.stream);
    const bool split = std::getenv("CLS_FACTOR_SPLIT") != nullptr;

    const int T = 128;
    cudaEvent_t k0, k1, km, ks;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    if (split) { cudaEventCreate(&km); cudaEventCreate(&ks); }
    cudaEventRecord(k0, stream);
    // FP64 master holds the assembly (both modes); the FP32 working arena is overwritten
    // per front by the mixed kernel's narrow, so it needs no memset.
    cudaMemsetAsync(plan.d_front, 0, plan.front_total * sizeof(double), stream);
    cudaMemsetAsync(plan.d_sing, 0, sizeof(int), stream);
    if (split) cudaEventRecord(km, stream);
    const int sb = (plan.nnz_a + T - 1) / T;  // scatter A into the FP64 master (both modes)
    mf_scatter_csr_values<<<sb, T, 0, stream>>>(plan.nnz_a, d_ordered_value_to_csr,
                                                plan.d_a_pos, d_csr_values, plan.d_front);
    if (split) cudaEventRecord(ks, stream);
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(plan.graph_exec), stream);
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, k0, k1);
        *kernel_ms = ms;
    }
    if (split) {
        float mms = 0, sms = 0, gms = 0;
        cudaEventElapsedTime(&mms, k0, km);
        cudaEventElapsedTime(&sms, km, ks);
        cudaEventElapsedTime(&gms, ks, k1);
        std::fprintf(stderr, "  [factor-split] memset=%.3f scatter=%.3f graph=%.3f ms\n", mms, sms, gms);
        cudaEventDestroy(km); cudaEventDestroy(ks);
    }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);

    return cudaGetLastError() == cudaSuccess;
}

}  // namespace custom_linear_solver::factorize
