#include "mysolver/gpu/gpu_mf.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "mysolver/symbolic/multifrontal.hpp"
#include "mysolver/symbolic/supernode.hpp"

namespace mysolver::gpu {

namespace {

void build_symmetric_filled(int n, const std::vector<int>& Lp, const std::vector<int>& Li,
                            std::vector<int>& Sp, std::vector<int>& Si)
{
    // S = fill(L) + fill(L)^T (CSC, diagonal kept). cy264: NO SORT/DEDUP. column k of S is
    // {rows of L in col k} (all >= k: diagonal + below) UNION {j<k : k is a row of L col j}
    // (the mirror) -- two DISJOINT ranges (>=k vs <k), so the union is already duplicate-free.
    // And the only consumers are order-INDEPENDENT: emit_map (per-entry owner-front mapping,
    // no binary search into Si) and numeric::solve (tests i>j / i==j / i<j per entry, not
    // sorted order). So the flat counting fill IS Sp/Si directly -- the old per-slice sort
    // (cy259) + dedup were pure overhead. (Removed ~10ms on SyntheticUSA.)
    Sp.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j)
        for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
            const int i = Li[p];
            ++Sp[j + 1];                 // col j gets row i (includes diagonal i==j)
            if (i != j) ++Sp[i + 1];     // symmetric mirror: col i gets row j
        }
    for (int j = 0; j < n; ++j) Sp[j + 1] += Sp[j];
    Si.resize(Sp[n]);
    std::vector<int> next(Sp.begin(), Sp.end());
    for (int j = 0; j < n; ++j)
        for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
            const int i = Li[p];
            Si[next[j]++] = i;
            if (i != j) Si[next[i]++] = j;
        }
}

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

template <typename FT>
__global__ void mf_scatterA(int nnz_a, const int* __restrict__ a_pos,
                            const double* __restrict__ Ax, FT* front)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnz_a) return;
    const int pos = a_pos[q];
    if (pos >= 0) atomicAdd(&front[pos], static_cast<FT>(Ax[q]));
}

template <typename FT>
__global__ void mf_emit(int total, const int* __restrict__ emit_front,
                        const FT* __restrict__ front, double* Sx)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= total) return;
    // cy266: emit_sx was a vestigial identity map (emit_sx[e]==e always) -> write Sx[e] directly,
    // removing a per-entry global-memory indirection + the whole d_emit_sx array (nnz_s ints) + its H2D.
    Sx[e] = static_cast<double>(front[emit_front[e]]);
}

// MULTIFRONTAL forward solve L y = b, one block per front, per panel level (low->
// high). Step 1: solve the nc x nc unit-lower pivot block in place (sequential,
// nc<=panel_cap). Step 2: apply the L panel to the CB rows -> atomicAdd into the
// global y (those rows are pivots of ancestors; several fronts contribute).
template <typename FT>
__global__ void mf_fwd_level(int lbegin, int lend, const int* __restrict__ plcols,
                             const int* __restrict__ front_off,
                             const int* __restrict__ front_ptr, const int* __restrict__ ncols,
                             const int* __restrict__ front_rows, const FT* __restrict__ front,
                             double* y, int selinv)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const FT* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    // cy189: stage the nc solved pivots in shared so the apply reads them from shared instead of
    // re-gathering y[fr[k]] (scattered global) for every CB row (nc <= panel_cap <= MF_MAX_NC=64).
    __shared__ double sh_piv[64];
    if (selinv) {
        // cy336 partitioned-inverse: L_pp inverted at factor time -> parallel GEMV sh_piv = Linv @ rhs
        // (one thread per pivot row; no loop-carried dependency, no thread-0 serialization). The y
        // writes are DEFERRED past the __syncthreads so the GEMV reads the original rhs (no
        // read-after-write race) and NO extra shared buffer is needed (reuses sh_piv) -> no occupancy
        // hit, unlike the cy334 register-hoist that regressed the occupancy-good forward leaf levels.
        for (int k = t; k < nc; k += nt) {
            double v = y[fr[k]];                                  // Linv[k][k] = 1 (implicit unit diag)
            for (int i = 0; i < k; ++i) v += F[(long)k * fsz + i] * y[fr[i]];  // Linv[k][i], i<k
            sh_piv[k] = v;
        }
        __syncthreads();
        for (int k = t; k < nc; k += nt) y[fr[k]] = sh_piv[k];
    } else {
        if (t == 0) {
            for (int k = 0; k < nc; ++k) {
                double v = y[fr[k]];
                for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * sh_piv[i];
                sh_piv[k] = v;
                y[fr[k]] = v;
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

// MULTIFRONTAL backward solve U x = y, one block per front, per panel level (high->
// low). x[pivots] = U_pp^{-1} (y[pivots] - U_pc x[CB cols]). The CB-column product
// (the bulk -- up to ~200 cols) is a PARALLEL reduction into shared rhs; the small
// nc x nc upper back-solve then runs on thread 0. CB columns are final (ancestors,
// processed at higher levels); this front writes only its own pivots (no atomics).
constexpr int MF_MAX_NC = 64;   // panel_cap bound for the shared pivot buffers
constexpr int MF_CB_TILE = 256;  // CB columns staged in shared per tile (fixed)
constexpr int MF_REG_NC = 16;    // register-partial bound; nc <= panel_cap (cy338: 8->16 to test bigger cap)
template <typename FT>
__global__ void mf_bwd_level(int lbegin, int lend, const int* __restrict__ plcols,
                             const int* __restrict__ front_off,
                             const int* __restrict__ front_ptr, const int* __restrict__ ncols,
                             const int* __restrict__ front_rows, const FT* __restrict__ front,
                             double* y, int selinv)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const FT* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    const int cb = fsz - nc;
    __shared__ double rhs[MF_MAX_NC];
    __shared__ double wsum[(256 / 32) * MF_REG_NC];  // per-warp partials (nt<=256)
    // cy188: __ldg the scattered y gathers (read-only data cache). These read ancestor pivots
    // (CB cols) and this front's own pivots that are FINAL at this point (written by lower levels
    // / the forward) -> safe + the read-only cache hides the scattered-gather latency (the
    // backward's latency bottleneck, cy171). Not coherent with same-kernel writes, but no block
    // writes these locations during this level (fronts independent; CB cols are done ancestors).
    for (int k = t; k < nc; k += nt) rhs[k] = __ldg(&y[fr[k]]);
    // rhs[k] -= sum_j U[k][nc+j]*x[fr[nc+j]], TILED over the CB columns so any front
    // size uses the fast coalesced reduction (circuit fronts exceed the old 256 cap;
    // tiling keeps shared fixed -> no occupancy hit for power-grid). nc register
    // partials accumulate across tiles; one warp-shuffle reduction at the end.
    double part[MF_REG_NC];
    for (int k = 0; k < nc; ++k) part[k] = 0.0;
    // cy332 (ncu): each thread consumed only the xcb[] entries it had itself written (the read
    // index set == the write index set per thread, both `j = t, t+nt, ...`), so the shared staging
    // plus its two __syncthreads were dead weight -- and ncu pinned the dominant backward stall
    // (~35.6%, shared-memory short-scoreboard dependency) right here. Read y straight into a
    // register: no shared bounce, no sync, no tiling needed (the tile loop only bounded the old
    // shared buffer to MF_CB_TILE). F's column access order is unchanged.
    for (int j = t; j < cb; j += nt) {
        const double xj = __ldg(&y[fr[nc + j]]);
        for (int k = 0; k < nc; ++k) part[k] += F[(long)k * fsz + (nc + j)] * xj;
    }
    const int lane = t & 31, warp = t >> 5;
    if (nt <= 32) {
        // cy333 single-warp fast path: with one warp the shuffle reduction already lands the full
        // sum in lane 0, so the shared wsum staging and BOTH __syncthreads block barriers (which
        // exist only to combine per-warp partials) are pure overhead. level_ts routes the narrow
        // few-front spine levels here (ncu cy332: those run ~10 active threads/warp, so a 2nd warp
        // is wasted). __syncwarp keeps the per-lane rhs[] loads from the top visible to lane 0.
        __syncwarp();
        for (int k = 0; k < nc; ++k) {
            double v = part[k];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) rhs[k] -= v;
        }
        __syncwarp();  // rhs[] now final; make lane-0's writes visible to all lanes for the GEMV
        if (selinv) {
            // cy335 partitioned-inverse: U_pp was inverted at factor time (mf_invert_upper), so the
            // sequential triangular back-solve becomes a parallel GEMV x = Uinv @ rhs (one lane per
            // pivot row -- no loop-carried dependency, no thread-0 serialization).
            for (int k = t; k < nc; k += nt) {
                double v = 0.0;
                for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
                y[fr[k]] = v;
            }
        } else if (lane == 0) {
            for (int k = nc - 1; k >= 0; --k) {
                double v = rhs[k];
                for (int j = k + 1; j < nc; ++j) v -= F[(long)k * fsz + j] * y[fr[j]];
                y[fr[k]] = v / F[(long)k * fsz + k];
            }
        }
        return;
    }
    for (int k = 0; k < nc; ++k) {
        double v = part[k];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) wsum[warp * nc + k] = v;
    }
    __syncthreads();
    if (t == 0) {
        const int nw = (nt + 31) / 32;
        for (int k = 0; k < nc; ++k) {
            double sm = 0.0;
            for (int w = 0; w < nw; ++w) sm += wsum[w * nc + k];
            rhs[k] -= sm;
        }
    }
    __syncthreads();
    if (selinv) {
        // cy335 partitioned-inverse: U_pp inverted at factor time -> parallel GEMV x = Uinv @ rhs
        // (one thread per pivot row) replaces the thread-0 sequential triangular back-solve.
        for (int k = t; k < nc; k += nt) {
            double v = 0.0;
            for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
            y[fr[k]] = v;
        }
    } else if (t == 0) {  // back-solve the nc x nc upper block (tiny, nc <= panel_cap)
        // cy190: staging x[j] in shared (like the cy189 forward) REGRESSED here (+1%): the back-solve
        // is tiny (nc<=8, thread-0) so y[fr[j]] is already cache-hot, and the extra shared writes cost
        // more than the indirection saved. Kept the direct read. (The forward apply's loop was the
        // asymmetry -- it re-gathered per CB row; the backward's bulk (CB reduction) was already staged.)
        for (int k = nc - 1; k >= 0; --k) {
            double v = rhs[k];
            for (int j = k + 1; j < nc; ++j) v -= F[(long)k * fsz + j] * y[fr[j]];
            y[fr[k]] = v / F[(long)k * fsz + k];
        }
    }
}

// cy335/336 partitioned-inverse: invert each front's nc x nc pivot block in place -- U_pp (upper,
// incl diagonal) AND L_pp (unit-lower, strict-lower; unit diagonal implicit). Runs AFTER emit (which
// extracts the true L/U into Sx) and after all factor levels. The backward solve then uses a parallel
// GEMV x = Uinv @ rhs, and the forward uses sh_piv = Linv @ rhs, both replacing sequential triangular
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

// cy170: narrow the FP64 master front arena to an FP32 copy ONCE per factor (amortized),
// so the solve reads half the front bytes. Mixed: FP32 storage, FP64 accumulate (the
// kernels are <float> -> double). Accuracy relaxes to ~1e-6 (cuDSS-level OK per the
// user's cy169 relaxation); the berr gate / refinement recovers if needed.
__global__ void mf_f64_to_f32(const double* __restrict__ src, float* __restrict__ dst, long ne)
{
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < ne;
         i += (long)gridDim.x * blockDim.x)
        dst[i] = static_cast<float>(src[i]);
}

}  // namespace

GpuMfPlan::~GpuMfPlan()
{
    if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
    if (solve_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(solve_graph_exec));
    if (solve_graph) cudaGraphDestroy(static_cast<cudaGraph_t>(solve_graph));
    if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    if (d_frontf) cudaFree(d_frontf);  // separate FP32 working arena (mixed factor)
    if (arena) cudaFree(arena);
}

GpuMfPlan::GpuMfPlan(GpuMfPlan&& o) noexcept { *this = std::move(o); }

GpuMfPlan& GpuMfPlan::operator=(GpuMfPlan&& o) noexcept
{
    if (this != &o) {
        if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
        if (solve_graph_exec)
            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(solve_graph_exec));
        if (solve_graph) cudaGraphDestroy(static_cast<cudaGraph_t>(solve_graph));
        if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        if (d_frontf) cudaFree(d_frontf);
        if (arena) cudaFree(arena);
        n = o.n; num_panels = o.num_panels; num_plevels = o.num_plevels;
        nnz_a = o.nnz_a; nnz_s = o.nnz_s; asm_total = o.asm_total; emit_total = o.emit_total;
        front_total = o.front_total; arena = o.arena;
        d_front = o.d_front; d_Ax = o.d_Ax; d_Sx = o.d_Sx; d_frontf = o.d_frontf; fp32 = o.fp32;
        d_front_off = o.d_front_off; d_front_ptr = o.d_front_ptr; d_ncols = o.d_ncols;
        d_plcols = o.d_plcols; d_panel_parent = o.d_panel_parent;
        d_asm_ptr = o.d_asm_ptr; d_asm_local = o.d_asm_local;
        d_a_pos = o.d_a_pos; d_emit_front = o.d_emit_front;
        d_sing = o.d_sing; d_front_rows = o.d_front_rows; d_y = o.d_y;
        front_store = o.front_store;
        Sp = std::move(o.Sp); Si = std::move(o.Si); plptr = std::move(o.plptr);
        stream = o.stream; graph_exec = o.graph_exec; solve_graph_exec = o.solve_graph_exec;
        solve_graph = o.solve_graph;
        o.arena = nullptr; o.stream = nullptr; o.graph_exec = nullptr;
        o.solve_graph_exec = nullptr; o.d_frontf = nullptr; o.solve_graph = nullptr;
    }
    return *this;
}

GpuMfPlan gpu_mf_analyze(int n, const int* Ap, const int* Ai, const std::vector<int>& Lp,
                         const std::vector<int>& Li, const std::vector<int>& parent,
                         int panel_cap, bool fp32)
{
    namespace sym = mysolver::symbolic;
    const bool tm = std::getenv("MF_TIME") != nullptr;  // analysis sub-phase profiling
    const bool solve_f32 = std::getenv("MF_SOLVE_F32") != nullptr;  // cy170: FP32 solve A/B
    auto tclk = std::chrono::steady_clock::now();
    auto lap = [&](const char* nm) {
        if (tm) {
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "  [analyze] %-22s %.2f ms\n", nm,
                         std::chrono::duration<double, std::milli>(now - tclk).count());
            tclk = now;
        }
    };
    GpuMfPlan plan;
    plan.n = n;
    build_symmetric_filled(n, Lp, Li, plan.Sp, plan.Si);
    const std::vector<int>& Sp = plan.Sp;
    const std::vector<int>& Si = plan.Si;
    plan.nnz_s = static_cast<int>(Si.size());
    plan.nnz_a = Ap[n];
    if (n <= 0) return plan;
    lap("build_symmetric_filled");

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
            return GpuMfPlan{};
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

    // Local index of a global row within panel p's front_rows slice.
    auto fidx = [&](int p, int g) {
        const auto b = mf.front_rows.begin() + mf.front_ptr[p];
        const auto e = mf.front_rows.begin() + mf.front_ptr[p + 1];
        return static_cast<int>(std::lower_bound(b, e, g) - b);
    };

    // A-entry -> front arena position (owner = panel_of[min(i,j)]). cy186: flat per-entry loop
    // (each a_pos[q] independent, fidx read-only) -> parallelize over columns like emit_map (cy185),
    // gated on nnz_a work. byte-identical.
    std::vector<int> a_pos(plan.nnz_a, -1);
    auto apos_chunk = [&](int clo, int chi) {
        for (int j = clo; j < chi; ++j)
            for (int q = Ap[j]; q < Ap[j + 1]; ++q) {
                const int i = Ai[q];
                const int owner = panels.panel_of[i < j ? i : j];
                const long fsz = mf.front_ptr[owner + 1] - mf.front_ptr[owner];
                a_pos[q] = front_off[owner] + static_cast<int>(fidx(owner, i) * fsz + fidx(owner, j));
            }
    };
    {
        unsigned hw = std::thread::hardware_concurrency();
        const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
        if (plan.nnz_a >= 200000 && nth > 1) {
            std::vector<std::thread> th;
            const int chunk = (n + nth - 1) / nth;
            for (int t = 0; t < nth; ++t) {
                const int a = t * chunk, b = std::min(n, a + chunk);
                if (a < b) th.emplace_back(apos_chunk, a, b);
            }
            for (auto& x : th) x.join();
        } else {
            apos_chunk(0, n);
        }
    }
    lap("a_pos");

    // Emit map: each Sx entry -> its front position. Iterate the SPARSE pattern
    // (nnz_S) and send each entry to its OWNER front -- O(nnz_S) vs the old dense
    // O(Sum fsz^2) scan (the cy107 analysis bottleneck: 644ms on SyntheticUSA).
    // Owner of (R,C): L (R>=C) lives in column C's panel = panel_of[C] (C pivot col,
    // R any front row); U (R<C) lives in row R's panel = panel_of[R] (R pivot row,
    // C any front col). Each Sx entry has exactly one owner -> no double-emit.
    std::vector<int> emit_front(plan.nnz_s);
    plan.emit_total = plan.nnz_s;
    // cy185: parallelize the emit map (~27ms single-threaded on onetone2's GPU analysis). Each S
    // nonzero is INDEPENDENT (flat arrays, fidx is a read-only binary search) -> byte-identical
    // output. Unlike build_symmetric's vector<vector> (cy183, didn't parallelize), this is flat.
    auto emit_chunk = [&](int clo, int chi) {
        for (int C = clo; C < chi; ++C)
            for (int p = Sp[C]; p < Sp[C + 1]; ++p) {
                const int R = Si[p];
                int owner, ri, ci;
                if (R >= C) {
                    owner = panels.panel_of[C];
                    ci = C - panels.first[owner];
                    ri = fidx(owner, R);
                } else {
                    owner = panels.panel_of[R];
                    ri = R - panels.first[owner];
                    ci = fidx(owner, C);
                }
                const long fsz = mf.front_ptr[owner + 1] - mf.front_ptr[owner];
                emit_front[p] = front_off[owner] + static_cast<int>(ri * fsz + ci);
            }
    };
    {
        // Gate on emit WORK (nnz_s), not n: parallel helps only dense-fill (onetone2 1.08M ->
        // 27->17ms); sparse-fill (Synth ~0.3M) is small + the thread overhead nets a slight loss.
        unsigned hw = std::thread::hardware_concurrency();
        const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
        if (plan.nnz_s >= 800000 && nth > 1) {
            std::vector<std::thread> th;
            const int chunk = (n + nth - 1) / nth;
            for (int t = 0; t < nth; ++t) {
                const int a = t * chunk, b = std::min(n, a + chunk);
                if (a < b) th.emplace_back(emit_chunk, a, b);
            }
            for (auto& x : th) x.join();
        } else {
            emit_chunk(0, n);
        }
    }
    lap("emit_map");

    // One arena, 256-byte-aligned sub-arrays (avoids the L2 straddle from cycle 39).
    auto al = [](long b) { return (b + 255) & ~static_cast<long>(255); };
    long off = 0;
    const long o_front = off; off = al(off + total * sizeof(double));
    const long o_Ax = off;    off = al(off + (long)plan.nnz_a * sizeof(double));
    const long o_Sx = off;    off = al(off + (long)plan.nnz_s * sizeof(double));
    const long o_foff = off;  off = al(off + (long)(P + 1) * sizeof(int));
    const long o_fptr = off;  off = al(off + (long)(P + 1) * sizeof(int));
    const long o_nc = off;    off = al(off + (long)P * sizeof(int));
    const long o_plc = off;   off = al(off + (long)P * sizeof(int));
    const long o_par = off;   off = al(off + (long)P * sizeof(int));
    const long o_aptr = off;  off = al(off + (long)(P + 1) * sizeof(int));
    const long o_aloc = off;  off = al(off + (long)std::max(1, plan.asm_total) * sizeof(int));
    const long o_apos = off;  off = al(off + (long)std::max(1, plan.nnz_a) * sizeof(int));
    const long o_ef = off;    off = al(off + (long)std::max(1, plan.emit_total) * sizeof(int));
    const int front_store = mf.front_ptr[P];
    plan.front_store = front_store;
    const long o_fr = off;    off = al(off + (long)std::max(1, front_store) * sizeof(int));
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
    // The FP64 master (d_front) holds the assembly + final L/U (solve/emit read it).
    if (fp32 || solve_f32) cudaMalloc(&plan.d_frontf, (long)total * sizeof(float));
    plan.d_Ax = reinterpret_cast<double*>(base + o_Ax);
    plan.d_Sx = reinterpret_cast<double*>(base + o_Sx);
    plan.d_front_off = reinterpret_cast<int*>(base + o_foff);
    plan.d_front_ptr = reinterpret_cast<int*>(base + o_fptr);
    plan.d_ncols = reinterpret_cast<int*>(base + o_nc);
    plan.d_plcols = reinterpret_cast<int*>(base + o_plc);
    plan.d_panel_parent = reinterpret_cast<int*>(base + o_par);
    plan.d_asm_ptr = reinterpret_cast<int*>(base + o_aptr);
    plan.d_asm_local = reinterpret_cast<int*>(base + o_aloc);
    plan.d_a_pos = reinterpret_cast<int*>(base + o_apos);
    plan.d_emit_front = reinterpret_cast<int*>(base + o_ef);
    plan.d_front_rows = reinterpret_cast<int*>(base + o_fr);
    plan.d_y = reinterpret_cast<double*>(base + o_y);
    plan.d_sing = reinterpret_cast<int*>(base + o_sing);

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
    H2D(plan.d_a_pos, a_pos);
    H2D(plan.d_emit_front, emit_front);
    lap("arena_malloc+H2D");

    // Capture the level-scheduled schedule (factor+extend per level, then emit) in
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
    const int T_emit = ft_force ? ft_force : 512;  // emit is a flat scatter; size-insensitive
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
    if (plan.emit_total > 0 && std::getenv("MF_SKIP_EMIT") == nullptr) {  // skip = diagnostic
        const int nb = (plan.emit_total + T_emit - 1) / T_emit;  // emit from FP64 master
        mf_emit<double><<<nb, T_emit, 0, stream>>>(plan.emit_total, plan.d_emit_front,
                                              plan.d_front, plan.d_Sx);
    }
    // cy335 partitioned-inverse: invert each front's U_pp pivot block (after emit, so Sx keeps the
    // true L/U). The backward solve then uses a parallel GEMV instead of a sequential triangular
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

    // Capture the multifrontal solve schedule: forward over panel levels low->high,
    // then backward high->low, operating in place on d_y. Value-independent (uses
    // the front pointers); replayed per solve on the latest factored fronts.
    // Smaller blocks than the factor: the per-front solve work is tiny (nc<=8
    // pivots), so 512-thread blocks waste occupancy on the ~75k mostly-tiny fronts.
    const char* st_s = std::getenv("MF_ST");  // solve block-size override (force/sweep)
    const int st_force = st_s ? std::atoi(st_s) : 0;
    // Per-level solve block size. Power-grid fronts are tiny (max ~214) and prefer 64
    // threads (128 over-subscribes the many small fronts); circuit big-front levels
    // (fsz up to thousands) finish the CB reduction faster with 128. Pick per level by
    // its max front size -> picks the better of {64,128} per matrix: onetone2 (big
    // circuit fronts) solve -9% (warm 3.84->3.50), ZERO change to power-grid (its
    // levels never reach the threshold -> stay 64) and to the other circuits (their
    // big levels are few / 128-neutral). Capped at 128: the bwd wsum shared buffer is
    // sized for nt<=256 (8 warps), and 128 is the circuit sweet spot (256 regresses).
    // MF_ST forces a constant for sweeps. (Initial cold-clock A/B showed larger deltas
    // -- they were P8->P2 warmup, not real; warm steady-state is the -9% above.)
    // cy144: the cy143 count-guard generalizes to the SOLVE. High-count tiny-front levels
    // (max fsz<=48 AND >=512 fronts) use 32-thread blocks -> more blocks/SM for the many
    // tiny per-front solves -> power-grid solve -3..-5% (case6468/case8387/ACTIV/Synth),
    // circuits neutral (no high-count tiny solve levels). Sparse tiny levels keep 64.
    // (Supersedes cy142's "solve at limit": that tested GLOBAL MF_ST, which forced the
    // medium/big levels to 32 too and regressed the large -- the per-level count guard
    // avoids that.) MF_ST_TINY / MF_ST_TINY_CNT retained for sweeps.
    const char* sts_s = std::getenv("MF_ST_TINY");
    const int st_tiny = sts_s ? std::atoi(sts_s) : 32;
    const char* stc_s = std::getenv("MF_ST_TINY_CNT");
    const int st_tiny_cnt = stc_s ? std::atoi(stc_s) : 512;
    // cy174: tested more threads/block for big-front (mx>=256) deep occupancy-starved levels
    // to hide their scattered-read latency (cy171) -> flat (onetone2 -1.6% at 256, within
    // noise; ACTIV/Synth/rajat15 unchanged). The latency is the scattered gather + etree
    // dependency, NOT occupancy -> more warps don't help. Kept at the proven 128.
    const char* tsb_s = std::getenv("MF_TS_BIG");    // cy302 sweep: block for mx>=256 (default 128)
    const int ts_big = tsb_s ? std::atoi(tsb_s) : 192;  // cy337: re-swept post-GEMV (was 128)
    const char* tssp_s = std::getenv("MF_TS_SPINE");  // cy302 sweep: block for mid-size narrow spine (default 128)
    const int ts_spine = tssp_s ? std::atoi(tssp_s) : 96;  // cy337: re-swept post-GEMV (was 128)
    const char* tspc_s = std::getenv("MF_TS_SPINE_CNT");  // cy303 sweep: narrow-level cnt threshold (default 82=SM)
    const int ts_spine_cnt = tspc_s ? std::atoi(tspc_s) : 82;
    const char* tssw_s = std::getenv("MF_TS_SW_MX");  // cy333 sweep: max front size routed to single-warp (32) (default 40)
    const int ts_sw_mx = tssw_s ? std::atoi(tssw_s) : 40;
    auto level_ts = [&](int L) -> int {
        if (st_force) return st_force;
        const int cnt = plan.plptr[L + 1] - plan.plptr[L];
        int mx = 0;
        for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
            const int pp = plcols[q];
            mx = std::max(mx, mf.front_ptr[pp + 1] - mf.front_ptr[pp]);
        }
        if (mx >= 256) return ts_big;
        if (mx <= 48 && cnt >= st_tiny_cnt) return st_tiny;
        // cy301: narrow mid-size-front levels (the latency-exposed spine; cnt<SM, front big
        // enough to use the warps) -> ts_spine (default 128, was 64/2 warps) to hide the
        // scattered-gather latency. cy174 tested mx>=256 (flat) but NOT this mid-size-narrow case.
        // Gated mx>=64 so small-front narrow levels (e.g. ACTIVSg25k) don't waste threads. Opt-out MF_NO_TS_SPINE.
        if (std::getenv("MF_NO_TS_SPINE") == nullptr && cnt < ts_spine_cnt && mx >= 64) return ts_spine;
        // cy333: few-front narrow spine levels (mx small, cnt<SM) -> single warp (32). The front is
        // too narrow to fill 2 warps (ncu cy332: ~10/32 active threads/warp), and 32 threads trigger
        // the kernel's single-warp fast path (no wsum, no block barrier). Opt-out MF_NO_TS_SW.
        if (std::getenv("MF_NO_TS_SW") == nullptr && cnt < ts_spine_cnt && mx <= ts_sw_mx) return 32;
        return 64;
    };
    const bool skip_fwd = std::getenv("MF_SKIP_FWD") != nullptr;  // diagnostic only
    const bool skip_bwd = std::getenv("MF_SKIP_BWD") != nullptr;
    // cy170: solve_f32 -> kernels read the FP32 front copy (plan.d_frontf, narrowed once
    // per factor in gpu_mf_solve before replay). Default reads the FP64 master.
    const float* ff = plan.d_frontf;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int L = 0; L < num_plevels && !skip_fwd; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        if (solve_f32)
            mf_fwd_level<float><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, ff, plan.d_y, selinv ? 1 : 0);
        else
            mf_fwd_level<double><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, plan.d_front, plan.d_y, selinv ? 1 : 0);
    }
    for (int L = num_plevels - 1; L >= 0 && !skip_bwd; --L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        if (solve_f32)
            mf_bwd_level<float><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, ff, plan.d_y, selinv ? 1 : 0);
        else
            mf_bwd_level<double><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, plan.d_front, plan.d_y, selinv ? 1 : 0);
    }
    cudaGraph_t sgraph;
    cudaStreamEndCapture(stream, &sgraph);
    cudaGraphExec_t sexec;
    cudaGraphInstantiate(&sexec, sgraph, nullptr, nullptr, 0);
    if (std::getenv("MF_SOLVE_CC")) plan.solve_graph = sgraph;  // cy172: keep for concurrency probe
    else cudaGraphDestroy(sgraph);
    plan.solve_graph_exec = sexec;
    lap("graph_capture");
    return plan;
}

void gpu_mf_solve(GpuMfPlan& plan, const std::vector<double>& b, std::vector<double>& x_out,
                  double* kernel_ms)
{
    const int n = plan.n;
    x_out.assign(n, 0.0);
    if (n <= 0 || plan.num_panels == 0) return;
    cudaStream_t stream = static_cast<cudaStream_t>(plan.stream);
    cudaMemcpyAsync(plan.d_y, b.data(), (long)n * sizeof(double), cudaMemcpyHostToDevice, stream);
    // cy170: narrow the factored FP64 front to FP32 once before the solve (amortized per
    // factor). Outside the timed region -> kernel_ms is the pure FP32-solve ceiling.
    if (std::getenv("MF_SOLVE_F32") && plan.d_frontf)
        mf_f64_to_f32<<<256, 256, 0, stream>>>(plan.d_front, plan.d_frontf, plan.front_total);
    // cy172: concurrency probe. Launch K independent copies of the solve graph on K streams.
    // If T_K << K*T_1, the serial-level solve leaves idle GPU that a finer (cross-level /
    // multi-subtree) schedule could fill -> sizes the prize before the big rewrite.
    if (const char* ccs = std::getenv("MF_SOLVE_CC")) {
        const int K = std::max(1, std::atoi(ccs));
        if (plan.solve_graph && K >= 1) {
            cudaGraph_t g = static_cast<cudaGraph_t>(plan.solve_graph);
            std::vector<cudaGraphExec_t> execs(K);
            std::vector<cudaStream_t> sts(K);
            execs[0] = static_cast<cudaGraphExec_t>(plan.solve_graph_exec);
            sts[0] = stream;
            for (int i = 1; i < K; ++i) {
                cudaGraphInstantiate(&execs[i], g, nullptr, nullptr, 0);
                cudaStreamCreate(&sts[i]);
            }
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < K; ++i) cudaGraphLaunch(execs[i], sts[i]);
            cudaDeviceSynchronize();  // wait for ALL K streams
            const auto t1 = std::chrono::steady_clock::now();
            const double tk = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (kernel_ms) *kernel_ms = tk;  // wall time for K concurrent solves
            for (int i = 1; i < K; ++i) { cudaGraphExecDestroy(execs[i]); cudaStreamDestroy(sts[i]); }
            cudaMemcpy(x_out.data(), plan.d_y, (long)n * sizeof(double), cudaMemcpyDeviceToHost);
            return;
        }
    }
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(plan.solve_graph_exec), stream);
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, k0, k1);
        *kernel_ms = ms;
    }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
    cudaMemcpy(x_out.data(), plan.d_y, (long)n * sizeof(double), cudaMemcpyDeviceToHost);
}

bool gpu_mf_factorize(GpuMfPlan& plan, const double* Ax, numeric::SparseLU& out,
                      double* kernel_ms)
{
    const int n = plan.n;
    out.n = n;
    out.Sp = plan.Sp;
    out.Si = plan.Si;
    out.x.assign(plan.nnz_s, 0.0);
    if (n <= 0 || plan.num_panels == 0) return false;

    cudaStream_t stream = static_cast<cudaStream_t>(plan.stream);
    cudaMemcpyAsync(plan.d_Ax, Ax, (long)plan.nnz_a * sizeof(double), cudaMemcpyHostToDevice,
                    stream);
    // FP64 master holds the assembly (both modes); the FP32 working arena is overwritten
    // per front by the mixed kernel's narrow, so it needs no memset.
    cudaMemsetAsync(plan.d_front, 0, plan.front_total * sizeof(double), stream);
    cudaMemsetAsync(plan.d_sing, 0, sizeof(int), stream);

    const int T = 128;
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
    const int sb = (plan.nnz_a + T - 1) / T;  // scatter A into the FP64 master (both modes)
    mf_scatterA<double><<<sb, T, 0, stream>>>(plan.nnz_a, plan.d_a_pos, plan.d_Ax, plan.d_front);
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

    cudaMemcpy(out.x.data(), plan.d_Sx, (long)plan.nnz_s * sizeof(double),
               cudaMemcpyDeviceToHost);
    int sing = 0;
    cudaMemcpy(&sing, plan.d_sing, sizeof(int), cudaMemcpyDeviceToHost);
    // cy180: static-pivoting attempt (opt-in MF_STATIC_PIVOT). A zero pivot is boosted to 1 by
    // the kernels; the idea was to PROCEED (not hard-fail) and let the caller's refinement+gate
    // recover. MEASURED-NEGATIVE for onetone2: its parND zero pivot is STRUCTURAL (not a tiny
    // near-singular value), so the boost is a relatively-infinite perturbation that cascades to
    // OVERFLOW (refinement rnorm=inf) -> gate rejects -> CPU anyway, just with a wasted attempt.
    // A structural zero needs real row/col PIVOTING (the no-pivot GPU factor lacks it) or a
    // zero-avoiding ordering. Default stays the old hard-fail (no wasted attempt). Kept opt-in.
    if (sing != 0 && std::getenv("MF_STATIC_PIVOT") == nullptr) return false;
    return true;
}

}  // namespace mysolver::gpu
