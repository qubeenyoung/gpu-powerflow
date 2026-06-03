#include "factorize/multifrontal.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "symbolic/multifrontal.hpp"
#include "symbolic/supernode.hpp"
#include "solve/multifrontal.hpp"

// ============================================================================
// Single-system multifrontal numeric factorization (LU, no pivoting).
//
// A "front" is one etree node's small dense matrix, row-major (leading dim = fsz)
// in a per-front slot of one big device arena. The first `nc` columns are the
// PIVOT columns eliminated at this node; the remaining `uc = fsz - nc` form the
// CONTRIBUTION BLOCK (CB) assembled (extend-added) into the parent front.
//
//        nc            uc                front F (fsz x fsz, row-major):
//     <------>  <--------------->          U_pp : pivot block (rows<nc, cols<nc)
//    +--------+-----------------+          U_pc : U panel     (rows<nc, cols>=nc)
//  nc|  U_pp  |      U_pc       | rows<nc  L_cp : L panel     (rows>=nc, cols<nc)
//    +--------+-----------------+          CB   : trailing    (rows>=nc, cols>=nc)
//    |        |                 |                 -> extend-added into the parent
//  uc|  L_cp  |   CB (uc x uc)  | rows>=nc
//    +--------+-----------------+   solve reads U_pp,U_pc,L_cp; CB is parent-only.
//
// Per level (leaves -> root, one kernel launch per level):
//   1. dense LU of the front: factor the nc-wide panel, U-solve U_pc, then the
//      rank-nc trailing update  CB -= L_cp * U_pc.
//   2. extend-add CB into the parent (atomicAdd; the parent is a higher,
//      not-yet-factored level this launch -> race-free).
// After all levels: invert each pivot block (mf_invert_pivot) so the SOLVE phase
// is a GEMV instead of a sequential triangular solve.
//
// One front is factored by ONE kernel chosen per level by analyze (gate = max
// front size / count in the level):
//   tiny leaf, many fronts  -> mf_factor_small_warp   (warp/front; B=1, off by default)
//   medium spine, few fronts-> mf_factor_shared       (front staged in shared; FP32)
//   large separator fronts  -> mf_bigA_panelU + mf_bigB_trailing + mf_bigC_extend (FP64)
//   default                 -> mf_factor_extend_level (one block/front)
//   mixed precision         -> mf_factor_extend_mixed (FP64 master + FP32 working)
// ============================================================================

namespace custom_linear_solver::factorize {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// Default per-front factor: one block factors one front, then extend-adds its CB
// into the parent. One launch per etree level. Generic over FT (double / float).
//   In : front[] assembled values, level range [lbegin,lend), front metadata.
//   Out: front[] holds this front's L/U in place; parent fronts get this CB.
//   sing: set to 1 on a zero pivot (no pivoting -- caller may diagonal-shift retry).
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

// Tiny-front leaf-level factor: ONE WARP per front (vs one block), many warps per block.
//   In/Out: same as mf_factor_extend_level. FT = double (fp64) or float (pure-fp32).
// Rationale: leaf levels are tens of thousands of fronts of fsz<=~48. A block-per-front
// wastes most threads and pays a block __syncthreads per pivot; a warp uses __syncwarp
// (far cheaper) and packs many fronts per block -> more fronts in flight per SM.
// (Off by default for B=1: regressed because the gate also caught few-front spine levels;
// kept for reference. The batched path uses a shared-staged variant.)
template <typename FT>
__global__ void mf_factor_small_warp(int lbegin, int lend, const int* __restrict__ plcols,
                                     const int* __restrict__ front_off,
                                     const int* __restrict__ front_ptr,
                                     const int* __restrict__ ncols,
                                     const int* __restrict__ panel_parent,
                                     const int* __restrict__ asm_ptr,
                                     const int* __restrict__ asm_local, FT* front, int* sing,
                                     int do_extend)
{
    // One warp owns front fi; lane = thread within the warp.
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int fi = lbegin + warp_global;
    if (fi >= lend) return;

    const int p = plcols[fi];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int uc = fsz - nc;

    // In-place rank-1 LU: per pivot k, scale the column below k, then rank-1 update
    // the trailing submatrix. Lane-parallel with a warp barrier between steps.
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

    // Extend-add this front's CB into the parent (race-free: parent is a higher level).
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    FT* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = lane; e < uc * uc; e += 32) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  F[(long)(nc + a) * fsz + (nc + b)]);
    }
}

// Medium/spine-level factor: stage the whole front into dynamic SHARED, factor there,
// write back only L/U. One block per front. Used for FP32 spine levels.
//   In/Out: same as mf_factor_extend_level, plus dyn shared = fsz2cap * sizeof(FT).
// Rationale: root-ward levels run on 1-5 blocks at ~0.3% SM/DRAM yet cost 23-42us each --
// a single block whose sequential nc-step panel LU pays a ~500-cycle GLOBAL round-trip per
// access (latency-bound, 81 SMs idle). Shared turns those into ~30-cycle accesses. The CB is
// extend-added straight from shared and never written back (solve/invert read only L/U).
template <typename FT>
__global__ void mf_factor_shared(int lbegin, int lend, long fsz2cap,
                                 const int* __restrict__ plcols,
                                 const int* __restrict__ front_off,
                                 const int* __restrict__ front_ptr,
                                 const int* __restrict__ ncols,
                                 const int* __restrict__ panel_parent,
                                 const int* __restrict__ asm_ptr,
                                 const int* __restrict__ asm_local, FT* front, int* sing,
                                 int do_extend)
{
    (void)fsz2cap;
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const long fsz2 = (long)fsz * fsz;
    extern __shared__ unsigned char smem_raw[];
    FT* Fs = reinterpret_cast<FT*>(smem_raw);
    for (long e = t; e < fsz2; e += nt) Fs[e] = F[e];  // stage front into shared
    __syncthreads();
    // Phase 1: factor the nc-wide panel (full height).
    for (int k = 0; k < nc; ++k) {
        FT piv = Fs[(long)k * fsz + k];
        if (piv == FT(0)) {
            if (t == 0) *sing = 1;
            piv = FT(1);
        }
        for (int i = k + 1 + t; i < fsz; i += nt) Fs[(long)i * fsz + k] /= piv;
        __syncthreads();
        const int pc = nc - 1 - k;
        for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
            const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
            Fs[(long)ii * fsz + jj] -= Fs[(long)ii * fsz + k] * Fs[(long)k * fsz + jj];
        }
        if (pc > 0) __syncthreads();
    }
    // Phase 2: U panel triangular solve (rows 0..nc-1, cols >= nc).
    for (int k = 1; k < nc; ++k) {
        for (int e = t; e < uc; e += nt) {
            const int jj = nc + e;
            FT v = Fs[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) v -= Fs[(long)k * fsz + i] * Fs[(long)i * fsz + jj];
            Fs[(long)k * fsz + jj] = v;
        }
        __syncthreads();
    }
    // Phase 3: single rank-nc trailing update (the parallel bulk).
    for (int e = t; e < uc * uc; e += nt) {
        const int ii = nc + e / uc, jj = nc + e % uc;
        FT acc = 0;
        for (int k = 0; k < nc; ++k) acc += Fs[(long)ii * fsz + k] * Fs[(long)k * fsz + jj];
        Fs[(long)ii * fsz + jj] -= acc;
    }
    __syncthreads();
    // Write back only L/U (solve + invert never read the trailing CB from global).
    for (long e = t; e < (long)nc * fsz; e += nt) F[e] = Fs[e];           // U + pivot rows
    for (int e = t; e < uc * nc; e += nt) {                                // L panel (rows>=nc,col<nc)
        const long id2 = (long)(nc + e / nc) * fsz + (e % nc);
        F[id2] = Fs[id2];
    }
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    FT* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = t; e < uc * uc; e += nt) {  // extend-add CB straight from shared
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}

// Mixed-precision factor: FP64 "master" front for the assembly, FP32 "working" copy for
// the bandwidth-bound dense LU.
//   In : master[] (FP64 assembled values).   Out: master[] holds L/U; parent master gets CB.
//   working[]: scratch, narrowed from master per front (no memset needed).
// Per front: narrow master->working, FP32 LU on working, write L/U back to master, extend-add
// the working CB into the parent's FP64 master (precise accumulation). Pivots come from the
// FP64 master so they stay nonzero; the within-front FP32 LU is ~1e-6 accuracy.
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

// ---- Multi-block big-front factor (FP64) -------------------------------------------
// A level of large separator fronts has too few fronts to fill the GPU one-block-each, so
// each such front is split across 3 graph-ordered kernels (A panel -> B trailing -> C
// extend) that spread the embarrassingly-parallel work across many blocks/SMs. Used only
// for levels whose max front >= kBigMultiFrontThreshold; the default path is untouched.
//
// (A) panel factor + U-solve: one block per front, the two sequential phases.
//   In : front[] assembled values.   Out: front[] panel L/U + U_pc done; trailing CB untouched.
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
// (B) trailing rank-nc update CB -= L_cp * U_pc, shared-mem TILED.
//   grid = (tiles, fronts-in-level): each front's uc*uc outputs are split into TSxTS tiles
//   across gridDim.x, so many blocks work one front. Each block stages a TSxnc L-tile and an
//   ncxTS U-tile into shared (each panel value reused TS times -> not BW-bound). blockDim=TS*TS.
constexpr int MF_BT_TS = 16;
constexpr int kFactorTinyThreads = 128;
constexpr int kFactorTinyFrontCount = 1024;
constexpr int kFactorBigThreads = 768;
constexpr int kFactorDefaultThreads = 384;
constexpr int kBigMultiFrontThreshold = 81;
constexpr int kBigExtendTiles = 16;
constexpr int kWarpFrontMax = 0;         // warp-per-front disabled for B=1 (regressed: catches few-front
                                         // spine levels, 32 threads too few for medium fronts + uncoalesced)
constexpr int kSharedFrontMin = 33;      // levels with maxfsz in [min,max] stage the front into shared
constexpr int kSharedFrontMax = 63;      // fsz>=64 keeps the multi-block path (needs trailing parallelism)
constexpr int kSharedFrontThreads = 256; // block size for the shared-front kernel
constexpr int kSharedFrontMaxBytes = 101376;  // 99KB opt-in dynamic-shared ceiling (sm_86)
constexpr int kFactorWarpsPerBlock = 8;  // warps packed per block in the warp-per-front kernel
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
// (C) extend-add the CB into the parent, MULTI-BLOCK (grid = tiles x fronts).
//   The uc*uc atomicAdds hit distinct parent slots (independent), so they are spread across
//   gridDim.x tiles to use more SMs on spine levels with few big fronts. Each block strides
//   its share of uc*uc.
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

// Scatter the current CSR values into the (zeroed) front arena: each nonzero q lands at the
// precomputed slot a_pos[q] (atomicAdd because duplicate map targets accumulate). VT = input
// value type (double/float), FT = arena type. a_pos[q] < 0 means "not stored in any front".
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

// Index of `value` within front `owner`'s sorted row list [begin,end) (returns local offset).
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

// Analyze-time: for each CSR nonzero (i,j) of the upper triangle, precompute the linear slot
// a_pos[] in the front arena where its value will be scattered each factorize. The owner front
// of min(i,j) holds the entry; row/col map to local front coordinates (pivot cols are contiguous,
// trailing rows via binary search). One block per column j.
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

// Partitioned inverse: invert each front's nc x nc pivot block IN PLACE -- U_pp (upper incl
// diagonal) and L_pp (unit-lower). Runs once after all factor levels.
//   Why: it turns the per-front SOLVE from a sequential triangular solve into a parallel GEMV
//   (forward sh_piv = Linv*rhs, backward x = Uinv*rhs) -- the main solve speedup. Costs a little
//   factor time, pays off on solve. Only the nc x nc block changes; L/U panels are untouched.
//   One block per front, one thread per inverse column (nc <= MF_REG_NC).
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

[[maybe_unused]] void issue_factor_levels(MultifrontalPlan& plan, cudaStream_t stream)
{
    constexpr int do_extend = 1;
    cudaFuncSetAttribute(mf_factor_shared<double>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSharedFrontMaxBytes);
    cudaFuncSetAttribute(mf_factor_shared<float>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSharedFrontMaxBytes);

    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L];
        const int e = plan.plptr[L + 1];
        if (e <= b) continue;
        const int T = factor_threads_for_level(plan, L);
        int mxf = 0;
        for (int q = b; q < e; ++q)
            mxf = std::max(mxf, plan.h_front_ptr[plan.h_plcols[q] + 1] -
                                    plan.h_front_ptr[plan.h_plcols[q]]);
        const long sh_elt = (long)mxf * mxf;
        const long sh_bytes = sh_elt * (plan.pure_fp32 ? sizeof(float) : sizeof(double));
        const bool use_shared = plan.pure_fp32 && mxf >= 49 && mxf <= 120 &&
                                sh_bytes <= kSharedFrontMaxBytes && (e - b) <= 64;
        if (!plan.fp32 && mxf <= kWarpFrontMax) {
            const int grid = (e - b + kFactorWarpsPerBlock - 1) / kFactorWarpsPerBlock;
            const int bs = kFactorWarpsPerBlock * 32;
            if (plan.pure_fp32)
                mf_factor_small_warp<float><<<grid, bs, 0, stream>>>(b, e, plan.d_plcols,
                    plan.d_front_off, plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent,
                    plan.d_asm_ptr, plan.d_asm_local, plan.d_frontf, plan.d_sing, do_extend);
            else
                mf_factor_small_warp<double><<<grid, bs, 0, stream>>>(b, e, plan.d_plcols,
                    plan.d_front_off, plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent,
                    plan.d_asm_ptr, plan.d_asm_local, plan.d_front, plan.d_sing, do_extend);
        } else if (use_shared) {
            if (plan.pure_fp32)
                mf_factor_shared<float><<<e - b, kSharedFrontThreads, sh_bytes, stream>>>(
                    b, e, sh_elt, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_frontf,
                    plan.d_sing, do_extend);
            else
                mf_factor_shared<double><<<e - b, kSharedFrontThreads, sh_bytes, stream>>>(
                    b, e, sh_elt, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_front,
                    plan.d_sing, do_extend);
        } else if (!plan.fp32 && !plan.pure_fp32 && mxf >= kBigMultiFrontThreshold) {
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
    // Amalgamation cap: a bigger cap merges more etree chain into one panel -> fewer fronts ->
    // fewer (serialized) solve levels, traded against more padded fill. Adaptive by size: small
    // matrices have no fill margin (keep panel_cap); larger ones amalgamate harder (deeper spines
    // + fill headroom). Swept values; cap16+ regresses the mid band. Bounded by MF_MAX_NC.
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
    const long o_plp = off;   off = al(off + (long)(num_plevels + 1) * sizeof(int));
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
    plan.d_plptr = reinterpret_cast<int*>(base + o_plp);
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
    H2D(plan.d_plptr, plan.plptr);
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
#ifdef CLS_INTERNAL_GRAPH
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    plan.stream = stream;
    plan.owns_stream = true;
    // Per-level factor block size, picked from the level's max front size and front count:
    //   huge serial fronts (>=257)      -> 768 threads (hide the deep per-front critical path)
    //   many tiny fronts (<=48, >=1024) -> 128 threads (pack more blocks/SM for the leaves)
    //   medium hotspot (49..256)        -> 384 threads (best occupancy for the mid band)
    // (Swept under a locked clock; <5% deltas but reproducible.)
    const char* ft_env = std::getenv("CLS_FT");
    auto level_ft = [&](int L) -> int {
        const int cnt = plan.plptr[L + 1] - plan.plptr[L];
        int mx = 0;
        for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
            const int pp = plcols[q];
            mx = std::max(mx, mf.front_ptr[pp + 1] - mf.front_ptr[pp]);
        }
        if (ft_env && mx < 257 && mx > 48) return std::atoi(ft_env);  // research: mid-spine thread sweep
        if (mx >= 257) return kFactorBigThreads;   // onetone2 big serial fronts
        if (mx <= 48 && cnt >= kFactorTinyFrontCount)
            return kFactorTinyThreads;  // MANY tiny fronts -> occupancy
        return kFactorDefaultThreads;  // medium hotspot 49-256
    };
    constexpr int do_extend = 1;
    // Build the per-level factor schedule into one CUDA graph (replayed each factorize). Each
    // level picks a kernel by its max front size / front count (see the gate in the loop below).
    // Opt in to the >48KB dynamic-shared cap so the shared-front kernel can stage fronts up to fsz~111.
    cudaFuncSetAttribute(mf_factor_shared<double>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSharedFrontMaxBytes);
    cudaFuncSetAttribute(mf_factor_shared<float>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSharedFrontMaxBytes);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int L = 0; L < num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        const int T = level_ft(L);
        int mxf = 0;
        for (int q = b; q < e; ++q)
            mxf = std::max(mxf, mf.front_ptr[plcols[q] + 1] - mf.front_ptr[plcols[q]]);
        const long sh_elt = (long)mxf * mxf;
        const long sh_bytes = sh_elt * (pure_fp32 ? sizeof(float) : sizeof(double));
        int sh_cnt_max = 64;
        if (const char* sc = std::getenv("CLS_SHCNT")) sh_cnt_max = std::atoi(sc);
        // FP32-only shared-front: float halves the shared footprint vs FP64 (occupancy preserved on
        // medium fronts where the FP64 version regressed) and fp32 never uses the multi-block path,
        // so its spine fronts are the most latency-starved. Min 49 = where the original kernel also
        // switches to the blocked 3-phase LU, so rounding (relres) matches bit-for-bit.
        const bool use_shared = pure_fp32 && mxf >= 49 && mxf <= 120 &&
                                sh_bytes <= kSharedFrontMaxBytes && (e - b) <= sh_cnt_max;
        if (!fp32 && mxf <= kWarpFrontMax) {
            // Tiny-front leaf level: one warp per front, many warps per block (pure_fp32 or fp64).
            const int grid = (e - b + kFactorWarpsPerBlock - 1) / kFactorWarpsPerBlock;
            const int bs = kFactorWarpsPerBlock * 32;
            if (pure_fp32)
                mf_factor_small_warp<float><<<grid, bs, 0, stream>>>(b, e, plan.d_plcols,
                    plan.d_front_off, plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent,
                    plan.d_asm_ptr, plan.d_asm_local, plan.d_frontf, plan.d_sing, do_extend);
            else
                mf_factor_small_warp<double><<<grid, bs, 0, stream>>>(b, e, plan.d_plcols,
                    plan.d_front_off, plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent,
                    plan.d_asm_ptr, plan.d_asm_local, plan.d_front, plan.d_sing, do_extend);
        } else if (use_shared) {
            // Medium/spine level: stage each front into shared (kills the latency-bound global LU).
            if (pure_fp32)
                mf_factor_shared<float><<<e - b, kSharedFrontThreads, sh_bytes, stream>>>(
                    b, e, sh_elt, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_frontf,
                    plan.d_sing, do_extend);
            else
                mf_factor_shared<double><<<e - b, kSharedFrontThreads, sh_bytes, stream>>>(
                    b, e, sh_elt, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, plan.d_front,
                    plan.d_sing, do_extend);
        } else if (!fp32 && !pure_fp32 && mxf >= kBigMultiFrontThreshold) {
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
