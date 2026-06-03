#pragma once

// Internal — included only by batched/multifrontal_batched.cu (see lu_device.cuh for why the
// batched kernels live in headers folded into a single TU).
//
// Batched SOLVE kernels (front-major; gridDim.y = batch):
//   gather_rhs_b / scatter_sol_b              - permuted RHS gather / solution scatter
//   mf_fwd_level_b<FT> / mf_bwd_level_b<FT>   - forward/backward triangular solve per etree level

#include <cuda_runtime.h>

namespace custom_linear_solver::batched {
namespace {

// ---- batched solve helpers --------------------------------------------------------------
// Gather the permuted RHS into the working vector y (per batch): y[k] = rhs[perm[k]].
// RT = RHS element type (double or float). YT = solve working-vector type (double for
// FP64/Mixed/TC, float for the pure-FP32 solve).
template <typename RT, typename YT>
__global__ void gather_rhs_b(int n, const RT* __restrict__ rhsB, const int* __restrict__ perm,
                             YT* __restrict__ yB)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const long b = blockIdx.y;
    yB[b * n + k] = static_cast<YT>(rhsB[b * (long)n + perm[k]]);
}
// Scatter the working vector y back to the solution in original order: sol[perm[k]] = y[k].
// YT = working-vector type, ST = solution element type (cuPF's Mixed step buffer is float).
template <typename YT, typename ST>
__global__ void scatter_sol_b(int n, const YT* __restrict__ yB, const int* __restrict__ perm,
                             ST* __restrict__ solB)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const long b = blockIdx.y;
    solB[b * (long)n + perm[k]] = static_cast<ST>(yB[b * n + k]);
}

// Forward solve level (L y = b). FT = front type AND working/accumulation type: FP32 mode runs
// the whole solve (front, y vector, accumulators) in float -> a true single-precision linear
// solve; FP64/Mixed/TC read the FP64 master front and keep y in double. selinv -> apply L_pp^-1
// as a GEMV; else warp-parallel substitution.
template <typename FT>
__global__ void mf_fwd_level_b(int lbegin, int lend, const int* __restrict__ plcols,
                              const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols, const int* __restrict__ front_rows,
                              const FT* frontB, FT* yB, long front_total, int n, int selinv)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const FT* front = frontB + (long)blockIdx.y * front_total;
    FT* y = yB + (long)blockIdx.y * n;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const FT* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    __shared__ FT sh_piv[64];  // resolved pivot values, shared with the CB-update loop below

    // Resolve this front's nc pivot rows into sh_piv (selinv: GEMV with L_pp^-1; else substitution).
    if (selinv) {
        for (int k = t; k < nc; k += nt) {
            FT v = y[fr[k]];
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
            FT part = FT(0), sk = FT(0);
            for (int k = 0; k < nc; ++k) {
                if (lane == k) { sk = y[fr[k]] + part; sh_piv[k] = sk; y[fr[k]] = sk; }
                sk = __shfl_sync(mask, sk, k);
                if (lane > k && lane < nc) part -= F[(long)lane * fsz + k] * sk;
            }
        }
        __syncthreads();
    }

    // Subtract the pivots' effect (L panel) from the CB / ancestor rows: y[anc] -= L * sh_piv.
    for (int i = nc + t; i < fsz; i += nt) {
        FT upd = FT(0);
        for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
        atomicAdd(&y[fr[i]], -upd);
    }
}

constexpr int MF_MAX_NC = 64;

// Backward solve level (U x = y). FT = front type AND working/accumulation type (see fwd: FP32
// runs the solve in float, FP64/Mixed/TC in double). selinv -> apply U_pp^-1 as a GEMV; else
// warp-parallel substitution.
template <typename FT>
__global__ void mf_bwd_level_b(int lbegin, int lend, const int* __restrict__ plcols,
                              const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols, const int* __restrict__ front_rows,
                              const FT* frontB, FT* yB, long front_total, int n, int selinv)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const FT* front = frontB + (long)blockIdx.y * front_total;
    FT* y = yB + (long)blockIdx.y * n;
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
    // bottleneck). Higher occupancy; the per-lane front-row reads hit L1/L2. The dynamic shared is
    // typed as raw bytes then viewed as FT so the float/double instantiations don't clash.
    extern __shared__ unsigned char xsh_raw[];  // size >= cb*sizeof(FT) (set per level at launch)
    FT* xsh = reinterpret_cast<FT*>(xsh_raw);
    __shared__ FT rhs[MF_MAX_NC];
    for (int k = t; k < nc; k += nt) rhs[k] = y[fr[k]];
    for (int j = t; j < cb; j += nt) xsh[j] = y[fr[nc + j]];
    __syncthreads();
    if (t < nc) {
        FT pk = FT(0);
        for (int j = 0; j < cb; ++j) pk += F[(long)t * fsz + (nc + j)] * xsh[j];
        rhs[t] -= pk;
    }
    __syncthreads();

    // Solve the nc pivot rows (selinv: GEMV with U_pp^-1; else warp-parallel back-substitution).
    if (selinv) {
        for (int k = t; k < nc; k += nt) {
            FT v = FT(0);
            for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
            y[fr[k]] = v;
        }
    } else if (t < 32) {
        // WARP-PARALLEL backward substitution U_pp * x = rhs (nc<=32): lane k (high->low) finalizes
        // x[k] = (rhs[k] - sum_{j>k} U[k][j] x[j]) / U[k][k], broadcasts it, lanes i<k fold
        // -U[i][k]*x[k] into their partial. Replaces the serial thread-0 O(nc^2) loop.
        const int lane = t;
        const unsigned mask = 0xffffffffu;
        FT part = FT(0), xk = FT(0);
        for (int k = nc - 1; k >= 0; --k) {
            if (lane == k) { xk = (rhs[k] + part) / F[(long)k * fsz + k]; y[fr[k]] = xk; }
            xk = __shfl_sync(mask, xk, k);
            if (lane < k) part -= F[(long)lane * fsz + k] * xk;
        }
    }
}

}  // namespace
}  // namespace custom_linear_solver::batched
