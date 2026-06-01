#include "factorize/multifrontal_batched.hpp"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace custom_linear_solver::factorize {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// ---- batched numeric scatter: front_b[a_pos[q]] += values_b[o2c[q]] ----------------
__global__ void scatter_batched(int nnz_a, long front_total, const int* __restrict__ o2c,
                                const int* __restrict__ a_pos, const double* __restrict__ valuesB,
                                double* __restrict__ frontB)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnz_a) return;
    const int pos = a_pos[q];
    if (pos < 0) return;
    const long b = blockIdx.y;
    atomicAdd(&frontB[b * front_total + pos], valuesB[b * (long)nnz_a + o2c[q]]);
}

// ---- batched fused factor + extend-add (FP64). One block per (front, batch). ----------
__global__ void mf_factor_extend_level_b(int lbegin, int lend, const int* __restrict__ plcols,
                                         const int* __restrict__ front_off,
                                         const int* __restrict__ front_ptr,
                                         const int* __restrict__ ncols,
                                         const int* __restrict__ panel_parent,
                                         const int* __restrict__ asm_ptr,
                                         const int* __restrict__ asm_local, double* frontB,
                                         long front_total, int* sing, int do_extend)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    double* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    double* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;

    if (fsz <= 48) {
        for (int k = 0; k < nc; ++k) {
            double piv = F[(long)k * fsz + k];
            if (piv == 0.0) { if (t == 0) *sing = 1; piv = 1.0; }
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
        for (int k = 1; k < nc; ++k) {
            for (int e = t; e < uc; e += nt) {
                const int jj = nc + e;
                double v = F[(long)k * fsz + jj];
                for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
                F[(long)k * fsz + jj] = v;
            }
            __syncthreads();
        }
        for (int e = t; e < uc * uc; e += nt) {
            const int ii = nc + e / uc, jj = nc + e % uc;
            double acc = 0;
            for (int k = 0; k < nc; ++k) acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
            F[(long)ii * fsz + jj] -= acc;
        }
    }
    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    double* Fp = front + front_off[par];
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

constexpr int MF_REG_NC = 16;

__global__ void mf_invert_pivot_b(int npanels, const int* __restrict__ front_ptr,
                                  const int* __restrict__ front_off, const int* __restrict__ ncols,
                                  double* frontB, long front_total)
{
    const int p = blockIdx.x;
    if (p >= npanels) return;
    double* front = frontB + (long)blockIdx.y * front_total;
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    double* F = front + front_off[p];
    const int j = threadIdx.x;
    __shared__ double Ui[MF_REG_NC * MF_REG_NC];
    __shared__ double Li[MF_REG_NC * MF_REG_NC];
    if (j < nc) {
        Ui[j * nc + j] = 1.0 / F[(long)j * fsz + j];
        for (int i = j - 1; i >= 0; --i) {
            double v = 0.0;
            for (int k = i + 1; k <= j; ++k) v -= F[(long)i * fsz + k] * Ui[k * nc + j];
            Ui[i * nc + j] = v / F[(long)i * fsz + i];
        }
        Li[j * nc + j] = 1.0;
        for (int i = j + 1; i < nc; ++i) {
            double v = 0.0;
            for (int k = j; k < i; ++k) v -= F[(long)i * fsz + k] * Li[k * nc + j];
            Li[i * nc + j] = v;
        }
    }
    __syncthreads();
    if (j < nc) {
        for (int i = 0; i <= j; ++i) F[(long)i * fsz + j] = Ui[i * nc + j];
        for (int i = j + 1; i < nc; ++i) F[(long)i * fsz + j] = Li[i * nc + j];
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

__global__ void mf_fwd_level_b(int lbegin, int lend, const int* __restrict__ plcols,
                              const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols, const int* __restrict__ front_rows,
                              const double* frontB, double* yB, long front_total, int n, int selinv)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const double* front = frontB + (long)blockIdx.y * front_total;
    double* y = yB + (long)blockIdx.y * n;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const double* F = front + front_off[p];
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

constexpr int MF_MAX_NC = 64;

__global__ void mf_bwd_level_b(int lbegin, int lend, const int* __restrict__ plcols,
                              const int* __restrict__ front_off, const int* __restrict__ front_ptr,
                              const int* __restrict__ ncols, const int* __restrict__ front_rows,
                              const double* frontB, double* yB, long front_total, int n, int selinv)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const double* front = frontB + (long)blockIdx.y * front_total;
    double* y = yB + (long)blockIdx.y * n;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const double* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int t = threadIdx.x, nt = blockDim.x;
    const int cb = fsz - nc;
    __shared__ double rhs[MF_MAX_NC];
    __shared__ double wsum[(256 / 32) * MF_REG_NC];
    for (int k = t; k < nc; k += nt) rhs[k] = y[fr[k]];
    double part[MF_REG_NC];
    for (int k = 0; k < nc; ++k) part[k] = 0.0;
    for (int j = t; j < cb; j += nt) {
        const double xj = y[fr[nc + j]];
        for (int k = 0; k < nc; ++k) part[k] += F[(long)k * fsz + (nc + j)] * xj;
    }
    const int lane = t & 31, warp = t >> 5;
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
        for (int k = t; k < nc; k += nt) {
            double v = 0.0;
            for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
            y[fr[k]] = v;
        }
    } else if (t == 0) {
        for (int k = nc - 1; k >= 0; --k) {
            double v = rhs[k];
            for (int j = k + 1; j < nc; ++j) v -= F[(long)k * fsz + j] * y[fr[j]];
            y[fr[k]] = v / F[(long)k * fsz + k];
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

bool batched_setup(const MultifrontalPlan& plan, int B, bool fp32, BatchedState& st)
{
    if (plan.num_panels == 0 || B <= 0) return false;
    st.B = B;
    st.front_total = plan.front_total;
    st.n = plan.n;
    st.fp32 = fp32;
    st.selinv = std::getenv("MF_NO_SELINV") == nullptr;
    const long fe = (long)B * plan.front_total;
    if (cudaMalloc(&st.d_frontB, fe * sizeof(double)) != cudaSuccess) return false;
    if (fp32 && cudaMalloc(&st.d_frontBf, fe * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&st.d_yB, (long)B * plan.n * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(&st.d_sing, sizeof(int)) != cudaSuccess) return false;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    st.stream = stream;

    const int T = 128;
    const int do_extend = 1;
    // Capture batched factor graph: per level, gridDim=(level_size, B).
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        dim3 grid(e - b, B);
        if (fp32)
            mf_factor_extend_mixed_b<<<grid, T, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB, st.d_frontBf,
                plan.front_total, st.d_sing, do_extend);
        else
            mf_factor_extend_level_b<<<grid, T, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontB,
                plan.front_total, st.d_sing, do_extend);
    }
    if (st.selinv)
        mf_invert_pivot_b<<<dim3(plan.num_panels, B), 32, 0, stream>>>(
            plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, st.d_frontB,
            plan.front_total);
    cudaGraph_t g;
    cudaStreamEndCapture(stream, &g);
    cudaGraphExec_t ge;
    cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphDestroy(g);
    st.factor_graph_exec = ge;

    // Capture batched solve graph (gather -> fwd levels -> bwd levels -> scatter is done
    // outside the graph in batched_solve; here only the level kernels, like the single path).
    const int sel = st.selinv ? 1 : 0;
    const int TS = 64;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        mf_fwd_level_b<<<dim3(e - b, B), TS, 0, stream>>>(
            b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_front_rows, st.d_frontB, st.d_yB, plan.front_total, plan.n, sel);
    }
    for (int L = plan.num_plevels - 1; L >= 0; --L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        mf_bwd_level_b<<<dim3(e - b, B), TS, 0, stream>>>(
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
    cudaMemsetAsync(st.d_frontB, 0, fe * sizeof(double), stream);
    cudaMemsetAsync(st.d_sing, 0, sizeof(int), stream);
    const int T = 256;
    scatter_batched<<<dim3((plan.nnz_a + T - 1) / T, st.B), T, 0, stream>>>(
        plan.nnz_a, plan.front_total, d_o2c, plan.d_a_pos, d_valuesB, st.d_frontB);
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
