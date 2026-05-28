#include "mysolver/gpu/gpu_solve.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace mysolver::gpu {

namespace {

int host_find(const std::vector<int>& Sp, const std::vector<int>& Si, int j, int i)
{
    const auto first = Si.begin() + Sp[j];
    const auto last = Si.begin() + Sp[j + 1];
    const auto it = std::lower_bound(first, last, i);
    return (it != last && *it == i) ? static_cast<int>(it - Si.begin()) : -1;
}

// Forward L y = b: column j (final) pushes -L(i,j)*y[j] to y[i], i>j (atomic).
__global__ void fwd_level(int begin, int end, const int* lcols, const int* Sp,
                          const int* Si, const double* Sx, double* x)
{
    const int idx = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end) return;
    const int j = lcols[idx];
    const double xj = x[j];
    for (int p = Sp[j]; p < Sp[j + 1]; ++p) {
        const int i = Si[p];
        if (i > j) atomicAdd(&x[i], -Sx[p] * xj);
    }
}

// Backward U x = y: divide by pivot, push -U(i,j)*x[j] to x[i], i<j (atomic).
__global__ void bwd_level(int begin, int end, const int* lcols, const int* Sp,
                          const int* Si, const double* Sx, const int* dpos, double* x)
{
    const int idx = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end) return;
    const int j = lcols[idx];
    const double xj = x[j] / Sx[dpos[j]];
    x[j] = xj;
    for (int p = Sp[j]; p < Sp[j + 1]; ++p) {
        const int i = Si[p];
        if (i < j) atomicAdd(&x[i], -Sx[p] * xj);
    }
}

}  // namespace

void gpu_solve(const numeric::SparseLU& lu, const std::vector<int>& parent,
               const std::vector<double>& b, std::vector<double>& x_out)
{
    const int n = lu.n;
    const int nnz = static_cast<int>(lu.Si.size());

    std::vector<int> dpos(n);
    for (int j = 0; j < n; ++j) dpos[j] = host_find(lu.Sp, lu.Si, j, j);

    std::vector<int> level(n, 0);
    int num_levels = 0;
    for (int j = 0; j < n; ++j)
        if (parent[j] != -1) level[parent[j]] = std::max(level[parent[j]], level[j] + 1);
    for (int j = 0; j < n; ++j) num_levels = std::max(num_levels, level[j] + 1);
    std::vector<int> lptr(num_levels + 1, 0), lcols(n);
    for (int j = 0; j < n; ++j) ++lptr[level[j] + 1];
    for (int L = 0; L < num_levels; ++L) lptr[L + 1] += lptr[L];
    {
        std::vector<int> next(lptr.begin(), lptr.end());
        for (int j = 0; j < n; ++j) lcols[next[level[j]]++] = j;
    }

    int *d_Sp, *d_Si, *d_lc, *d_dp;
    double *d_Sx, *d_x;
    cudaMalloc(&d_Sp, (n + 1) * sizeof(int));
    cudaMalloc(&d_Si, nnz * sizeof(int));
    cudaMalloc(&d_Sx, nnz * sizeof(double));
    cudaMalloc(&d_lc, n * sizeof(int));
    cudaMalloc(&d_dp, n * sizeof(int));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMemcpy(d_Sp, lu.Sp.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Si, lu.Si.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sx, lu.x.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lc, lcols.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dp, dpos.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    const int threads = 128;
    for (int L = 0; L < num_levels; ++L) {  // forward: low -> high
        const int cols = lptr[L + 1] - lptr[L];
        if (cols > 0)
            fwd_level<<<(cols + threads - 1) / threads, threads>>>(lptr[L], lptr[L + 1], d_lc,
                                                                   d_Sp, d_Si, d_Sx, d_x);
    }
    for (int L = num_levels - 1; L >= 0; --L) {  // backward: high -> low
        const int cols = lptr[L + 1] - lptr[L];
        if (cols > 0)
            bwd_level<<<(cols + threads - 1) / threads, threads>>>(lptr[L], lptr[L + 1], d_lc,
                                                                   d_Sp, d_Si, d_Sx, d_dp, d_x);
    }

    x_out.assign(n, 0.0);
    cudaMemcpy(x_out.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_Sp); cudaFree(d_Si); cudaFree(d_Sx); cudaFree(d_lc); cudaFree(d_dp); cudaFree(d_x);
}

GpuSolvePlan::~GpuSolvePlan()
{
    if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
    if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    if (arena) cudaFree(arena);
}

GpuSolvePlan::GpuSolvePlan(GpuSolvePlan&& o) noexcept { *this = std::move(o); }

GpuSolvePlan& GpuSolvePlan::operator=(GpuSolvePlan&& o) noexcept
{
    if (this != &o) {
        if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
        if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        if (arena) cudaFree(arena);
        n = o.n; nnz = o.nnz; num_levels = o.num_levels; arena = o.arena;
        d_Sp = o.d_Sp; d_Si = o.d_Si; d_lc = o.d_lc; d_dp = o.d_dp;
        d_Sx = o.d_Sx; d_x = o.d_x;
        stream = o.stream; graph_exec = o.graph_exec;
        o.arena = nullptr; o.stream = nullptr; o.graph_exec = nullptr;
    }
    return *this;
}

GpuSolvePlan gpu_solve_analyze(const numeric::SparseLU& lu, const std::vector<int>& parent)
{
    GpuSolvePlan plan;
    const int n = lu.n;
    const int nnz = static_cast<int>(lu.Si.size());
    plan.n = n;
    plan.nnz = nnz;
    if (n <= 0) return plan;

    std::vector<int> dpos(n);
    for (int j = 0; j < n; ++j) dpos[j] = host_find(lu.Sp, lu.Si, j, j);

    std::vector<int> level(n, 0);
    int num_levels = 0;
    for (int j = 0; j < n; ++j)
        if (parent[j] != -1) level[parent[j]] = std::max(level[parent[j]], level[j] + 1);
    for (int j = 0; j < n; ++j) num_levels = std::max(num_levels, level[j] + 1);
    plan.num_levels = num_levels;
    std::vector<int> lptr(num_levels + 1, 0), lcols(n);
    for (int j = 0; j < n; ++j) ++lptr[level[j] + 1];
    for (int L = 0; L < num_levels; ++L) lptr[L + 1] += lptr[L];
    {
        std::vector<int> next(lptr.begin(), lptr.end());
        for (int j = 0; j < n; ++j) lcols[next[level[j]]++] = j;
    }

    // One 256-byte-aligned arena (avoids the L2 straddle from the factor path).
    auto al = [](long b) { return (b + 255) & ~static_cast<long>(255); };
    long off = 0;
    const long o_Sp = off; off = al(off + (long)(n + 1) * sizeof(int));
    const long o_Si = off; off = al(off + (long)nnz * sizeof(int));
    const long o_lc = off; off = al(off + (long)n * sizeof(int));
    const long o_dp = off; off = al(off + (long)n * sizeof(int));
    const long o_Sx = off; off = al(off + (long)nnz * sizeof(double));
    const long o_x = off;  off = al(off + (long)n * sizeof(double));
    cudaMalloc(&plan.arena, off);
    char* base = static_cast<char*>(plan.arena);
    plan.d_Sp = reinterpret_cast<int*>(base + o_Sp);
    plan.d_Si = reinterpret_cast<int*>(base + o_Si);
    plan.d_lc = reinterpret_cast<int*>(base + o_lc);
    plan.d_dp = reinterpret_cast<int*>(base + o_dp);
    plan.d_Sx = reinterpret_cast<double*>(base + o_Sx);
    plan.d_x = reinterpret_cast<double*>(base + o_x);
    cudaMemcpy(plan.d_Sp, lu.Sp.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(plan.d_Si, lu.Si.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(plan.d_Sx, lu.x.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(plan.d_lc, lcols.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(plan.d_dp, dpos.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Capture the whole fwd (low->high) + bwd (high->low) level schedule in one
    // graph; replayed per solve so the 2*num_levels launches cost no CPU overhead.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    plan.stream = stream;
    const int threads = 128;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int L = 0; L < num_levels; ++L) {
        const int cols = lptr[L + 1] - lptr[L];
        if (cols > 0)
            fwd_level<<<(cols + threads - 1) / threads, threads, 0, stream>>>(
                lptr[L], lptr[L + 1], plan.d_lc, plan.d_Sp, plan.d_Si, plan.d_Sx, plan.d_x);
    }
    for (int L = num_levels - 1; L >= 0; --L) {
        const int cols = lptr[L + 1] - lptr[L];
        if (cols > 0)
            bwd_level<<<(cols + threads - 1) / threads, threads, 0, stream>>>(
                lptr[L], lptr[L + 1], plan.d_lc, plan.d_Sp, plan.d_Si, plan.d_Sx, plan.d_dp,
                plan.d_x);
    }
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    plan.graph_exec = exec;
    return plan;
}

void gpu_solve_apply(GpuSolvePlan& plan, const std::vector<double>& b,
                     std::vector<double>& x_out, double* kernel_ms)
{
    const int n = plan.n;
    x_out.assign(n, 0.0);
    if (n <= 0) return;
    cudaStream_t stream = static_cast<cudaStream_t>(plan.stream);
    cudaMemcpyAsync(plan.d_x, b.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
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
    cudaMemcpy(x_out.data(), plan.d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
}

}  // namespace mysolver::gpu
