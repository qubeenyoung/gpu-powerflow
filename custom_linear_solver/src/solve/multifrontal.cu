#include "solve/multifrontal.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

namespace custom_linear_solver::solve {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

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
    __shared__ YT sh_piv[64];

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

    for (int i = nc + t; i < fsz; i += nt) {
        YT upd = YT(0);
        for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
        atomicAdd(&y[fr[i]], -upd);
    }
}

constexpr int MF_MAX_NC = 64;
constexpr int MF_REG_NC = 16;

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
    __shared__ YT rhs[MF_MAX_NC];
    __shared__ YT wsum[(256 / 32) * MF_REG_NC];

    for (int k = t; k < nc; k += nt) rhs[k] = __ldg(&y[fr[k]]);

    YT part[MF_REG_NC];
    for (int k = 0; k < nc; ++k) part[k] = YT(0);
    for (int j = t; j < cb; j += nt) {
        const YT xj = __ldg(&y[fr[nc + j]]);
        for (int k = 0; k < nc; ++k) part[k] += F[(long)k * fsz + (nc + j)] * xj;
    }

    const int lane = t & 31, warp = t >> 5;
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

__global__ void mf_f64_to_f32(const double* __restrict__ src, float* __restrict__ dst, long ne)
{
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < ne;
         i += (long)gridDim.x * blockDim.x)
        dst[i] = static_cast<float>(src[i]);
}

template <typename RT, typename YT>
__global__ void gather_permuted_rhs(int n, const RT* __restrict__ rhs,
                                    const int* __restrict__ perm, YT* __restrict__ y)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) y[k] = static_cast<YT>(rhs[perm[k]]);
}

template <typename YT, typename ST>
__global__ void scatter_permuted_solution(int n, const YT* __restrict__ y,
                                          const int* __restrict__ perm,
                                          ST* __restrict__ solution)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) solution[perm[k]] = static_cast<ST>(y[k]);
}

void issue_solve_levels(MultifrontalPlan& plan, cudaStream_t stream, bool front_f32, bool y_f32,
                        bool use_selected_inverse)
{
    const int num_plevels = plan.num_plevels;
    const char* st_s = std::getenv("MF_ST");
    const int st_force = st_s ? std::atoi(st_s) : 0;
    const char* sts_s = std::getenv("MF_ST_TINY");
    const int st_tiny = sts_s ? std::atoi(sts_s) : 32;
    const char* stc_s = std::getenv("MF_ST_TINY_CNT");
    const int st_tiny_cnt = stc_s ? std::atoi(stc_s) : 512;
    const char* tsb_s = std::getenv("MF_TS_BIG");
    const int ts_big = tsb_s ? std::atoi(tsb_s) : 192;
    const char* tssp_s = std::getenv("MF_TS_SPINE");
    const int ts_spine = tssp_s ? std::atoi(tssp_s) : 96;
    const char* tspc_s = std::getenv("MF_TS_SPINE_CNT");
    const int ts_spine_cnt = tspc_s ? std::atoi(tspc_s) : 82;
    const char* tssw_s = std::getenv("MF_TS_SW_MX");
    const int ts_sw_mx = tssw_s ? std::atoi(tssw_s) : 40;

    auto level_ts = [&](int L) -> int {
        if (st_force) return st_force;
        const int cnt = plan.plptr[L + 1] - plan.plptr[L];
        int mx = 0;
        for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
            const int pp = plan.h_plcols[q];
            mx = std::max(mx, plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp]);
        }
        if (mx >= 256) return ts_big;
        if (mx <= 48 && cnt >= st_tiny_cnt) return st_tiny;
        if (std::getenv("MF_NO_TS_SPINE") == nullptr && cnt < ts_spine_cnt && mx >= 64)
            return ts_spine;
        if (std::getenv("MF_NO_TS_SW") == nullptr && cnt < ts_spine_cnt && mx <= ts_sw_mx)
            return 32;
        return 64;
    };

    const bool skip_fwd = std::getenv("MF_SKIP_FWD") != nullptr;
    const bool skip_bwd = std::getenv("MF_SKIP_BWD") != nullptr;
    const int selinv = use_selected_inverse ? 1 : 0;

    for (int L = 0; L < num_plevels && !skip_fwd; ++L) {
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
    for (int L = num_plevels - 1; L >= 0 && !skip_bwd; --L) {
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

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    issue_solve_levels(plan, stream, front_f32, y_f32, use_selected_inverse);
    cudaGraph_t sgraph;
    cudaStreamEndCapture(stream, &sgraph);
    cudaGraphExec_t sexec;
    cudaGraphInstantiate(&sexec, sgraph, nullptr, nullptr, 0);
    if (std::getenv("MF_SOLVE_CC"))
        plan.solve_graph = sgraph;
    else
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
    if (!plan.pure_fp32 && std::getenv("MF_SOLVE_F32") && plan.d_frontf)
        mf_f64_to_f32<<<256, 256, 0, stream>>>(plan.d_front, plan.d_frontf, plan.front_total);

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
    issue_solve_levels(plan, stream, plan.pure_fp32, plan.pure_fp32,
                       std::getenv("MF_NO_SELINV") == nullptr);
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
