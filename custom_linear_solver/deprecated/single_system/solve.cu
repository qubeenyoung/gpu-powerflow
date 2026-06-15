#include "solve/multifrontal.hpp"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================================
// Single-system multifrontal triangular solve  A x = b  (A = L*U from factor).
//
// Two sweeps over the etree levels, reusing the factored fronts (one CUDA-graph
// node per level per sweep):
//   forward  (leaves -> root):  L y = b  -- each front resolves its nc pivot rows,
//             then subtracts their effect from the ancestor rows it touches (the CB
//             rows) via atomicAdd into y.
//   backward (root -> leaves):  U x = y  -- each front reads the already-solved
//             ancestor rows, then solves its nc pivot rows.
//
// `selinv` path (always on for B=1): the factor's partitioned inverse made each
// pivot block an explicit inverse, so the per-front pivot solve is a GEMV
// (sh_piv = Linv*rhs forward, x = Uinv*rhs backward) instead of a sequential
// triangular solve.
//
// y[] = permuted RHS/solution; fr[] = front_rows maps a front's local rows to
// global y indices (pivot rows 0..nc-1, then the CB / ancestor rows).
// ============================================================================

namespace custom_linear_solver::solve {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// Forward sweep, one block per front (one launch per level, leaves->root).
//   In : y[] = permuted RHS so far, factored front[].   Out: y[] updated in place.
//   FT = front type, YT = y type. selinv: use the inverted pivot block (GEMV).
template <typename FT, typename YT>
__global__ void mf_fwd_level(int lbegin, int lend, const int* __restrict__ plcols,
                             const int* __restrict__ front_off,
                             const int* __restrict__ front_ptr, const int* __restrict__ ncols,
                             const int* __restrict__ front_rows, const FT* __restrict__ front,
                             YT* y, int selinv)
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
    __shared__ YT sh_piv[64];  // resolved pivot values, shared with the CB-update loop below

    // Resolve this front's nc pivot rows into sh_piv (and write them back to y).
    if (selinv) {
        for (int k = t; k < nc; k += nt) {
            YT v = y[fr[k]];
            for (int i = 0; i < k; ++i) v += F[(long)k * fsz + i] * y[fr[i]];
            sh_piv[k] = v;
        }
        __syncthreads();
        for (int k = t; k < nc; k += nt) y[fr[k]] = sh_piv[k];
    } else {
        if (t == 0) {
            for (int k = 0; k < nc; ++k) {
                YT v = y[fr[k]];
                for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * sh_piv[i];
                sh_piv[k] = v;
                y[fr[k]] = v;
            }
        }
        __syncthreads();
    }

    // Subtract the pivots' effect (L panel) from the CB / ancestor rows: y[anc] -= L*sh_piv.
    for (int i = nc + t; i < fsz; i += nt) {
        YT upd = YT(0);
        for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
        atomicAdd(&y[fr[i]], -upd);
    }
}

constexpr int MF_MAX_NC = 64;
constexpr int MF_REG_NC = 16;
constexpr int kSolveTinyThreads = 32;
constexpr int kSolveTinyFrontCount = 512;
constexpr int kSolveBigThreads = 192;
constexpr int kSolveSpineThreads = 96;
constexpr int kSolveSpineFrontCount = 82;
constexpr int kSolveSmallWaveMaxFront = 40;

// Backward sweep, one block per front (one launch per level, root->leaves).
//   In : y[] = forward result with ancestor rows already solved.   Out: y[] pivot rows solved.
// Per front: rhs = y[pivot] - U_pc * x[ancestors], then solve the nc pivot rows. The
// U_pc*x reduction is split across threads (register partials) and combined by warp shuffle;
// nt<=32 uses a single-warp fast path, else a cross-warp shared reduction (wsum).
template <typename FT, typename YT>
__global__ void mf_bwd_level(int lbegin, int lend, const int* __restrict__ plcols,
                             const int* __restrict__ front_off,
                             const int* __restrict__ front_ptr, const int* __restrict__ ncols,
                             const int* __restrict__ front_rows, const FT* __restrict__ front,
                             YT* y, int selinv)
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
    __shared__ YT rhs[MF_MAX_NC];                  // pivot RHS being reduced
    __shared__ YT wsum[(256 / 32) * MF_REG_NC];    // per-warp partials for the cross-warp reduce

    // Load pivot RHS and accumulate this thread's slice of -U_pc * x[ancestors] into registers.
    for (int k = t; k < nc; k += nt) rhs[k] = __ldg(&y[fr[k]]);

    YT part[MF_REG_NC];
    for (int k = 0; k < nc; ++k) part[k] = YT(0);
    for (int j = t; j < cb; j += nt) {
        const YT xj = __ldg(&y[fr[nc + j]]);
        for (int k = 0; k < nc; ++k) part[k] += F[(long)k * fsz + (nc + j)] * xj;
    }

    const int lane = t & 31, warp = t >> 5;

    // Single-warp fast path: warp-reduce the partials into rhs, then solve the pivots.
    if (nt <= 32) {
        __syncwarp();
        for (int k = 0; k < nc; ++k) {
            YT v = part[k];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) rhs[k] -= v;
        }
        __syncwarp();
        if (selinv) {
            for (int k = t; k < nc; k += nt) {
                YT v = YT(0);
                for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
                y[fr[k]] = v;
            }
        } else if (lane == 0) {
            for (int k = nc - 1; k >= 0; --k) {
                YT v = rhs[k];
                for (int j = k + 1; j < nc; ++j) v -= F[(long)k * fsz + j] * y[fr[j]];
                y[fr[k]] = v / F[(long)k * fsz + k];
            }
        }
        return;
    }

    // Multi-warp path: warp-reduce partials to wsum, then thread 0 sums warps into rhs.
    for (int k = 0; k < nc; ++k) {
        YT v = part[k];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) wsum[warp * nc + k] = v;
    }
    __syncthreads();
    if (t == 0) {
        const int nw = (nt + 31) / 32;
        for (int k = 0; k < nc; ++k) {
            YT sm = YT(0);
            for (int w = 0; w < nw; ++w) sm += wsum[w * nc + k];
            rhs[k] -= sm;
        }
    }
    __syncthreads();

    // Solve the nc pivot rows: GEMV with the inverted U block (selinv) or a back-substitution.
    if (selinv) {
        for (int k = t; k < nc; k += nt) {
            YT v = YT(0);
            for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
            y[fr[k]] = v;
        }
    } else if (t == 0) {
        for (int k = nc - 1; k >= 0; --k) {
            YT v = rhs[k];
            for (int j = k + 1; j < nc; ++j) v -= F[(long)k * fsz + j] * y[fr[j]];
            y[fr[k]] = v / F[(long)k * fsz + k];
        }
    }
}

// Gather user RHS into the solver's fill-reducing order: y[k] = rhs[perm[k]] (RT->YT cast).
template <typename RT, typename YT>
__global__ void gather_permuted_rhs(int n, const RT* __restrict__ rhs,
                                    const int* __restrict__ perm, YT* __restrict__ y)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) y[k] = static_cast<YT>(rhs[perm[k]]);
}

// Scatter the solved vector back to user order: solution[perm[k]] = y[k] (YT->ST cast).
template <typename YT, typename ST>
__global__ void scatter_permuted_solution(int n, const YT* __restrict__ y,
                                          const int* __restrict__ perm,
                                          ST* __restrict__ solution)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) solution[perm[k]] = static_cast<ST>(y[k]);
}

// ---- B=1 cooperative fused solve megakernel ---------------------------------------------
// For a single system the per-level solve work is tiny, so the old path (2*num_plevels separate
// kernel launches) is dominated by per-launch latency, not arithmetic (measured: every fwd/bwd
// level launch costs a near-constant ~floor regardless of front count). This fuses ALL forward
// levels and ALL backward levels into ONE cooperative launch: blocks grid-stride over the fronts
// of a level, then cg::grid_group::sync() acts as the inter-level barrier (replacing the implicit
// stream barrier between separate launches). gridDim is capped to the resident-block count so the
// grid barrier cannot deadlock. selected-inverse only (the single-case path always uses it).
template <typename FT, typename YT>
__global__ void mf_solve_coop(int num_plevels, const int* __restrict__ plptr,
                              const int* __restrict__ plcols, const int* __restrict__ front_off,
                              const int* __restrict__ front_ptr, const int* __restrict__ ncols,
                              const int* __restrict__ front_rows, const FT* __restrict__ front,
                              YT* y)
{
    cg::grid_group grid = cg::this_grid();
    const int t = threadIdx.x, nt = blockDim.x;
    const int lane = t & 31, warp = t >> 5;
    const int nw = (nt + 31) >> 5;
    __shared__ YT sh_piv[MF_MAX_NC];
    __shared__ YT rhs[MF_MAX_NC];
    __shared__ YT wsum[(256 / 32) * MF_REG_NC];

    // FORWARD sweep: leaves -> root.
    for (int L = 0; L < num_plevels; ++L) {
        const int b = plptr[L], e = plptr[L + 1];
        for (int fi = b + blockIdx.x; fi < e; fi += gridDim.x) {
            const int p = plcols[fi];
            const int s = front_ptr[p];
            const int fsz = front_ptr[p + 1] - s;
            const int nc = ncols[p];
            const FT* F = front + front_off[p];
            const int* fr = front_rows + s;
            __syncthreads();
            for (int k = t; k < nc; k += nt) {
                YT v = y[fr[k]];
                for (int i = 0; i < k; ++i) v += F[(long)k * fsz + i] * y[fr[i]];
                sh_piv[k] = v;
            }
            __syncthreads();
            for (int k = t; k < nc; k += nt) y[fr[k]] = sh_piv[k];
            for (int i = nc + t; i < fsz; i += nt) {
                YT upd = YT(0);
                for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
                atomicAdd(&y[fr[i]], -upd);
            }
        }
        grid.sync();
    }

    // BACKWARD sweep: root -> leaves.
    for (int L = num_plevels - 1; L >= 0; --L) {
        const int b = plptr[L], e = plptr[L + 1];
        for (int fi = b + blockIdx.x; fi < e; fi += gridDim.x) {
            const int p = plcols[fi];
            const int s = front_ptr[p];
            const int fsz = front_ptr[p + 1] - s;
            const int nc = ncols[p];
            const FT* F = front + front_off[p];
            const int* fr = front_rows + s;
            const int cb = fsz - nc;
            __syncthreads();
            for (int k = t; k < nc; k += nt) rhs[k] = __ldg(&y[fr[k]]);
            YT part[MF_REG_NC];
            for (int k = 0; k < nc; ++k) part[k] = YT(0);
            for (int j = t; j < cb; j += nt) {
                const YT xj = __ldg(&y[fr[nc + j]]);
                for (int k = 0; k < nc; ++k) part[k] += F[(long)k * fsz + (nc + j)] * xj;
            }
            for (int k = 0; k < nc; ++k) {
                YT v = part[k];
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
                if (lane == 0) wsum[warp * nc + k] = v;
            }
            __syncthreads();
            if (t == 0) {
                for (int k = 0; k < nc; ++k) {
                    YT sm = YT(0);
                    for (int w = 0; w < nw; ++w) sm += wsum[w * nc + k];
                    rhs[k] -= sm;
                }
            }
            __syncthreads();
            for (int k = t; k < nc; k += nt) {
                YT v = YT(0);
                for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
                y[fr[k]] = v;
            }
        }
        grid.sync();
    }
}

// Resident-block count for the cooperative solve kernel (gridDim cap so grid.sync can't deadlock).
template <typename FT, typename YT>
static int coop_solve_grid(int blockDim)
{
    int dev = 0;
    cudaGetDevice(&dev);
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
    int bpsm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, mf_solve_coop<FT, YT>, blockDim, 0);
    if (bpsm < 1) bpsm = 1;
    return bpsm * sm;
}

static int coop_solve_block()
{
    if (const char* b = std::getenv("CLS_COOP_BLK")) return std::atoi(b);
    return 64;
}

template <typename FT, typename YT>
static void launch_coop_solve(MultifrontalPlan& plan, cudaStream_t stream, FT* d_front, YT* d_y)
{
    const int kCoopSolveBlock = coop_solve_block();
    int grid = plan.coop_grid_solve;
    if (grid <= 0) grid = coop_solve_grid<FT, YT>(kCoopSolveBlock);
    if (const char* g = std::getenv("CLS_COOP_GRID")) {
        const int cap = std::atoi(g);
        if (cap > 0 && grid > cap) grid = cap;
    }
    void* args[] = {&plan.num_plevels, &plan.d_plptr, &plan.d_plcols, &plan.d_front_off,
                    &plan.d_front_ptr, &plan.d_ncols, &plan.d_front_rows, &d_front, &d_y};
    cudaLaunchCooperativeKernel(reinterpret_cast<void*>(mf_solve_coop<FT, YT>), grid,
                                kCoopSolveBlock, args, 0, stream);
}

void issue_solve_levels_perlevel(MultifrontalPlan& plan, cudaStream_t stream, bool front_f32,
                                 bool y_f32, bool use_selected_inverse);

void issue_solve_levels(MultifrontalPlan& plan, cudaStream_t stream, bool front_f32, bool y_f32,
                        bool use_selected_inverse)
{
    // Fused cooperative megakernel: single launch replaces 2*num_plevels per-level launches.
    // Gated to the selected-inverse single-case path (always true here).
    // NOTE: measured slower than per-level launches in a graph (grid.sync cost + lower occupancy
    // outweigh the saved launches), so it is opt-in via CLS_COOP_SOLVE for experiments only.
    if (use_selected_inverse && std::getenv("CLS_COOP_SOLVE")) {
        if (front_f32 && y_f32)
            launch_coop_solve<float, float>(plan, stream, plan.d_frontf, plan.d_yf);
        else if (!front_f32)
            launch_coop_solve<double, double>(plan, stream, plan.d_front, plan.d_y);
        else
            launch_coop_solve<float, double>(plan, stream, plan.d_frontf, plan.d_y);
        return;
    }
    issue_solve_levels_perlevel(plan, stream, front_f32, y_f32, use_selected_inverse);
}

void issue_solve_levels_perlevel(MultifrontalPlan& plan, cudaStream_t stream, bool front_f32,
                                 bool y_f32, bool use_selected_inverse)
{
    const int num_plevels = plan.num_plevels;

    auto level_ts = [&](int L) -> int {
        const int cnt = plan.plptr[L + 1] - plan.plptr[L];
        int mx = 0;
        for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
            const int pp = plan.h_plcols[q];
            mx = std::max(mx, plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp]);
        }
        if (mx >= 256) return kSolveBigThreads;
        if (mx <= 48 && cnt >= kSolveTinyFrontCount) return kSolveTinyThreads;
        if (cnt < kSolveSpineFrontCount && mx >= 64)
            return kSolveSpineThreads;
        if (cnt < kSolveSpineFrontCount && mx <= kSolveSmallWaveMaxFront)
            return 32;
        return 64;
    };

    const int selinv = use_selected_inverse ? 1 : 0;

    for (int L = 0; L < num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        if (front_f32 && y_f32)
            mf_fwd_level<float, float><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, plan.d_frontf, plan.d_yf, selinv);
        else if (front_f32)
            mf_fwd_level<float, double><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, plan.d_frontf, plan.d_y, selinv);
        else
            mf_fwd_level<double, double><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, plan.d_front, plan.d_y, selinv);
    }
    for (int L = num_plevels - 1; L >= 0; --L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        if (front_f32 && y_f32)
            mf_bwd_level<float, float><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, plan.d_frontf, plan.d_yf, selinv);
        else if (front_f32)
            mf_bwd_level<float, double><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, plan.d_frontf, plan.d_y, selinv);
        else
            mf_bwd_level<double, double><<<e - b, level_ts(L), 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, plan.d_front, plan.d_y, selinv);
    }
}

}  // namespace

void capture_multifrontal_solve_graph(MultifrontalPlan& plan, const std::vector<int>& front_ptr,
                                      const std::vector<int>& plcols, bool front_f32,
                                      bool y_f32,
                                      bool use_selected_inverse)
{
    (void)front_ptr;
    (void)plcols;
#ifdef CLS_INTERNAL_GRAPH
    cudaStream_t stream = static_cast<cudaStream_t>(plan.stream);

    // Precompute the cooperative grid size (resident-block cap) before capture so the captured
    // launch uses a fixed, deadlock-safe gridDim.
    if (use_selected_inverse && std::getenv("CLS_COOP_SOLVE")) {
        const int blk = coop_solve_block();
        if (front_f32 && y_f32) plan.coop_grid_solve = coop_solve_grid<float, float>(blk);
        else if (!front_f32) plan.coop_grid_solve = coop_solve_grid<double, double>(blk);
        else plan.coop_grid_solve = coop_solve_grid<float, double>(blk);
    }
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    issue_solve_levels(plan, stream, front_f32, y_f32, use_selected_inverse);
    cudaGraph_t sgraph;
    cudaStreamEndCapture(stream, &sgraph);
    cudaGraphExec_t sexec;
    cudaGraphInstantiate(&sexec, sgraph, nullptr, nullptr, 0);
    cudaGraphDestroy(sgraph);
    plan.solve_graph_exec = sexec;
#else
    (void)front_f32;
    (void)y_f32;
    (void)use_selected_inverse;
#endif
}

template <typename RT, typename ST>
bool solve_multifrontal_device_T(MultifrontalPlan& plan, const RT* d_rhs, ST* d_solution,
                                 const int* d_perm, double* kernel_ms)
{
    const int n = plan.n;
    if (n <= 0 || plan.num_panels == 0 || d_rhs == nullptr || d_solution == nullptr ||
        d_perm == nullptr)
        return false;

    cudaStream_t stream = static_cast<cudaStream_t>(plan.stream);
    constexpr int T = 256;
    const int nb = (n + T - 1) / T;

#ifdef CLS_INTERNAL_GRAPH
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
#else
    (void)kernel_ms;
#endif
    if (plan.pure_fp32)
        gather_permuted_rhs<RT, float><<<nb, T, 0, stream>>>(n, d_rhs, d_perm, plan.d_yf);
    else
        gather_permuted_rhs<RT, double><<<nb, T, 0, stream>>>(n, d_rhs, d_perm, plan.d_y);
#ifdef CLS_INTERNAL_GRAPH
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(plan.solve_graph_exec), stream);
#else
    issue_solve_levels(plan, stream, plan.pure_fp32, plan.pure_fp32, true);
#endif
    if (plan.pure_fp32)
        scatter_permuted_solution<float, ST><<<nb, T, 0, stream>>>(n, plan.d_yf, d_perm,
                                                                   d_solution);
    else
        scatter_permuted_solution<double, ST><<<nb, T, 0, stream>>>(n, plan.d_y, d_perm,
                                                                    d_solution);
#ifdef CLS_INTERNAL_GRAPH
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, k0, k1);
        *kernel_ms = ms;
    }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
#endif
    return cudaGetLastError() == cudaSuccess;
}

bool solve_multifrontal_device(MultifrontalPlan& plan, const double* d_rhs, double* d_solution,
                               const int* d_perm, double* kernel_ms)
{
    return solve_multifrontal_device_T(plan, d_rhs, d_solution, d_perm, kernel_ms);
}

bool solve_multifrontal_device(MultifrontalPlan& plan, const float* d_rhs, float* d_solution,
                               const int* d_perm, double* kernel_ms)
{
    return solve_multifrontal_device_T(plan, d_rhs, d_solution, d_perm, kernel_ms);
}

bool solve_multifrontal_device(MultifrontalPlan& plan, const double* d_rhs, float* d_solution,
                               const int* d_perm, double* kernel_ms)
{
    return solve_multifrontal_device_T(plan, d_rhs, d_solution, d_perm, kernel_ms);
}

}  // namespace custom_linear_solver::solve
