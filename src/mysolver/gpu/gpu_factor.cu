#include "mysolver/gpu/gpu_factor.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace mysolver::gpu {

namespace {

void build_symmetric_filled(int n, const std::vector<int>& Lp, const std::vector<int>& Li,
                            std::vector<int>& Sp, std::vector<int>& Si)
{
    std::vector<std::vector<int>> cols(n);
    for (int j = 0; j < n; ++j) {
        for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
            const int i = Li[p];
            cols[j].push_back(i);
            if (i != j) cols[i].push_back(j);
        }
    }
    Sp.assign(n + 1, 0);
    Si.clear();
    for (int j = 0; j < n; ++j) {
        std::sort(cols[j].begin(), cols[j].end());
        cols[j].erase(std::unique(cols[j].begin(), cols[j].end()), cols[j].end());
        for (int i : cols[j]) Si.push_back(i);
        Sp[j + 1] = static_cast<int>(Si.size());
    }
}

int host_find(const std::vector<int>& Sp, const std::vector<int>& Si, int j, int i)
{
    const auto first = Si.begin() + Sp[j];
    const auto last = Si.begin() + Sp[j + 1];
    const auto it = std::lower_bound(first, last, i);
    return (it != last && *it == i) ? static_cast<int>(it - Si.begin()) : -1;
}

// Search-free numeric kernel (PLAN dependency_map): each column's Schur updates
// are precomputed as (ukj_pos, src_pos, tgt_pos) triples ordered so accumulation
// is correct; the kernel just multiplies-and-subtracts with direct indices.
__global__ void factor_level(int begin, int end, const int* __restrict__ level_cols,
                             const int* __restrict__ op_ptr, const int* __restrict__ op_ukj,
                             const int* __restrict__ op_src, const int* __restrict__ op_tgt,
                             const int* __restrict__ Sp, const int* __restrict__ Si,
                             const int* __restrict__ dpos, double* Sx, int* singular)
{
    const int idx = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end) return;
    const int j = level_cols[idx];
    for (int o = op_ptr[j]; o < op_ptr[j + 1]; ++o) {
        // Sx[src] is L from an already-factored (descendant) column -> read-only
        // this level -> route through the read-only data cache (__ldg). Sx[ukj]
        // and Sx[tgt] are in column j (written this level), so plain access.
        Sx[op_tgt[o]] -= __ldg(&Sx[op_src[o]]) * Sx[op_ukj[o]];
    }
    const double piv = Sx[dpos[j]];
    if (piv == 0.0) { *singular = 1; return; }
    for (int p = Sp[j]; p < Sp[j + 1]; ++p) {
        if (Si[p] > j) Sx[p] /= piv;
    }
}

// RIGHT-LOOKING kernels. Instead of one thread gathering a column's updates
// (left-looking, latency-bound on narrow levels), we expose ALL of a level's
// Schur ops as independent threads that scatter via atomicAdd -> thousands of
// concurrent ops hide the memory latency. Correctness: process etree levels
// low->high; a level's columns are finalized (all updates from lower levels
// already scattered in), then their ops scatter to higher columns.

// Finalize level L's columns: divide the below-diagonal entries by the pivot
// (the diagonal + entries are fully accumulated by scatters from lower levels).
__global__ void finalize_level(int begin, int end, const int* __restrict__ level_cols,
                               const int* __restrict__ Sp, const int* __restrict__ Si,
                               const int* __restrict__ dpos, double* Sx, int* singular)
{
    const int idx = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end) return;
    const int j = level_cols[idx];
    const double piv = Sx[dpos[j]];
    if (piv == 0.0) { *singular = 1; return; }
    for (int p = Sp[j]; p < Sp[j + 1]; ++p)
        if (Si[p] > j) Sx[p] /= piv;
}

// Scatter all ops whose updater is in the current level: one thread per op,
// atomicAdd into the target. Sx[src]=L(i,k) and Sx[ukj]=U(k,j) are finalized.
__global__ void scatter_level(int begin, int end, const int* __restrict__ op_ukj,
                              const int* __restrict__ op_src, const int* __restrict__ op_tgt,
                              double* Sx)
{
    const int o = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= end) return;
    atomicAdd(&Sx[op_tgt[o]], -__ldg(&Sx[op_src[o]]) * __ldg(&Sx[op_ukj[o]]));
}

}  // namespace

GpuFactorPlan::~GpuFactorPlan()
{
    if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
    if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    if (arena) cudaFree(arena);
}

GpuFactorPlan::GpuFactorPlan(GpuFactorPlan&& o) noexcept { *this = std::move(o); }

GpuFactorPlan& GpuFactorPlan::operator=(GpuFactorPlan&& o) noexcept
{
    if (this != &o) {
        if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
        if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        if (arena) cudaFree(arena);
        n = o.n; nnz = o.nnz; num_levels = o.num_levels;
        arena = o.arena; d_Sx = o.d_Sx; d_Sp = o.d_Sp; d_Si = o.d_Si; d_lc = o.d_lc;
        d_optr = o.d_optr; d_ouk = o.d_ouk; d_osr = o.d_osr; d_otg = o.d_otg;
        d_dp = o.d_dp; d_sing = o.d_sing;
        stream = o.stream; graph_exec = o.graph_exec;
        Sp = std::move(o.Sp); Si = std::move(o.Si); lptr = std::move(o.lptr);
        op_rl_ptr = std::move(o.op_rl_ptr); scatter = std::move(o.scatter);
        o.arena = nullptr; o.stream = nullptr; o.graph_exec = nullptr;
    }
    return *this;
}

GpuFactorPlan gpu_analyze(int n, const int* Ap, const int* Ai, const std::vector<int>& Lp,
                          const std::vector<int>& Li, const std::vector<int>& parent)
{
    GpuFactorPlan plan;
    plan.n = n;
    build_symmetric_filled(n, Lp, Li, plan.Sp, plan.Si);
    const std::vector<int>& Sp = plan.Sp;
    const std::vector<int>& Si = plan.Si;

    // A-entry -> position in Sx (value-independent; applied each factorize).
    plan.scatter.assign(Ap[n], -1);
    for (int j = 0; j < n; ++j)
        for (int q = Ap[j]; q < Ap[j + 1]; ++q) plan.scatter[q] = host_find(Sp, Si, j, Ai[q]);

    // Per-column etree levels (descendants done first) — computed first so the
    // right-looking ops can be bucketed by their updater's level.
    std::vector<int> level(n, 0);
    int num_levels = 0;
    for (int j = 0; j < n; ++j)
        if (parent[j] != -1) level[parent[j]] = std::max(level[parent[j]], level[j] + 1);
    for (int j = 0; j < n; ++j) num_levels = std::max(num_levels, level[j] + 1);
    plan.num_levels = num_levels;
    plan.lptr.assign(num_levels + 1, 0);
    std::vector<int> lcols(n);
    for (int j = 0; j < n; ++j) ++plan.lptr[level[j] + 1];
    for (int L = 0; L < num_levels; ++L) plan.lptr[L + 1] += plan.lptr[L];
    {
        std::vector<int> next(plan.lptr.begin(), plan.lptr.end());
        for (int j = 0; j < n; ++j) lcols[next[level[j]]++] = j;
    }

    // Dependency map (Schur ops: ukj=U(k,j) pos, src=L(i,k) pos, tgt=(i,j) pos),
    // bucketed by the UPDATER's level for the right-looking scatter. All binary
    // searches happen here (symbolic, amortized once per pattern). Two passes:
    // count ops per updater-level, then place — so op_rl_ptr[L]..op_rl_ptr[L+1]
    // are exactly the ops whose updater k is in level L.
    std::vector<int> dpos(n, -1);
    plan.op_rl_ptr.assign(num_levels + 1, 0);
    for (int j = 0; j < n; ++j) {
        dpos[j] = host_find(Sp, Si, j, j);
        for (int p = Sp[j]; p < Sp[j + 1]; ++p) {
            const int k = Si[p];
            if (k >= j) break;
            for (int q = Sp[k]; q < Sp[k + 1]; ++q) {
                const int i = Si[q];
                if (i > k && host_find(Sp, Si, j, i) >= 0) ++plan.op_rl_ptr[level[k] + 1];
            }
        }
    }
    for (int L = 0; L < num_levels; ++L) plan.op_rl_ptr[L + 1] += plan.op_rl_ptr[L];
    const int nops_total = plan.op_rl_ptr[num_levels];
    std::vector<int> op_ukj(nops_total), op_src(nops_total), op_tgt(nops_total);
    {
        std::vector<int> cur(plan.op_rl_ptr.begin(), plan.op_rl_ptr.end());
        for (int j = 0; j < n; ++j)
            for (int p = Sp[j]; p < Sp[j + 1]; ++p) {
                const int k = Si[p];
                if (k >= j) break;
                for (int q = Sp[k]; q < Sp[k + 1]; ++q) {
                    const int i = Si[q];
                    if (i <= k) continue;
                    const int tgt = host_find(Sp, Si, j, i);
                    if (tgt < 0) continue;
                    const int slot = cur[level[k]]++;
                    op_ukj[slot] = p;
                    op_src[slot] = q;
                    op_tgt[slot] = tgt;
                }
            }
    }

    // Single device arena, but each sub-array 256-byte aligned (matching what
    // separate cudaMalloc gives). Misaligned base pointers make each warp's loads
    // straddle extra L2 cache lines -> a packed (4/8-byte-offset) arena measured
    // ~35% slower on the op-array reads; aligning recovers it while keeping the
    // single allocation. Carve byte offsets with 256-byte rounding.
    const int nnz = static_cast<int>(Si.size());
    const int nops = static_cast<int>(op_ukj.size());
    const int nop1 = std::max(1, nops);
    plan.nnz = nnz;
    auto al = [](size_t b) { return (b + 255) & ~static_cast<size_t>(255); };
    size_t off = 0;
    const size_t o_Sx = off;   off = al(off + static_cast<size_t>(nnz) * sizeof(double));
    const size_t o_Sp = off;   off = al(off + static_cast<size_t>(n + 1) * sizeof(int));
    const size_t o_Si = off;   off = al(off + static_cast<size_t>(nnz) * sizeof(int));
    const size_t o_lc = off;   off = al(off + static_cast<size_t>(n) * sizeof(int));
    const size_t o_ouk = off;  off = al(off + static_cast<size_t>(nop1) * sizeof(int));
    const size_t o_osr = off;  off = al(off + static_cast<size_t>(nop1) * sizeof(int));
    const size_t o_otg = off;  off = al(off + static_cast<size_t>(nop1) * sizeof(int));
    const size_t o_dp = off;   off = al(off + static_cast<size_t>(n) * sizeof(int));
    const size_t o_sing = off; off = al(off + sizeof(int));
    cudaMalloc(&plan.arena, off);
    char* base = static_cast<char*>(plan.arena);
    plan.d_Sx = reinterpret_cast<double*>(base + o_Sx);
    plan.d_Sp = reinterpret_cast<int*>(base + o_Sp);
    plan.d_Si = reinterpret_cast<int*>(base + o_Si);
    plan.d_lc = reinterpret_cast<int*>(base + o_lc);
    plan.d_ouk = reinterpret_cast<int*>(base + o_ouk);
    plan.d_osr = reinterpret_cast<int*>(base + o_osr);
    plan.d_otg = reinterpret_cast<int*>(base + o_otg);
    plan.d_dp = reinterpret_cast<int*>(base + o_dp);
    plan.d_sing = reinterpret_cast<int*>(base + o_sing);
    cudaMemcpy(plan.d_Sp, Sp.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(plan.d_Si, Si.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(plan.d_lc, lcols.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    if (nops > 0) {
        cudaMemcpy(plan.d_ouk, op_ukj.data(), nops * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(plan.d_osr, op_src.data(), nops * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(plan.d_otg, op_tgt.data(), nops * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(plan.d_dp, dpos.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Capture the whole right-looking schedule (2*num_levels tiny finalize/scatter
    // launches) into one CUDA graph. Now that the kernels are fast (cycle 70), the
    // per-launch CPU overhead dominates; replaying one graph removes it.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    plan.stream = stream;
    const int threads = 256;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int L = 0; L < num_levels; ++L) {
        const int cols = plan.lptr[L + 1] - plan.lptr[L];
        if (cols > 0)
            finalize_level<<<(cols + threads - 1) / threads, threads, 0, stream>>>(
                plan.lptr[L], plan.lptr[L + 1], plan.d_lc, plan.d_Sp, plan.d_Si, plan.d_dp,
                plan.d_Sx, plan.d_sing);
        const int ops = plan.op_rl_ptr[L + 1] - plan.op_rl_ptr[L];
        if (ops > 0)
            scatter_level<<<(ops + threads - 1) / threads, threads, 0, stream>>>(
                plan.op_rl_ptr[L], plan.op_rl_ptr[L + 1], plan.d_ouk, plan.d_osr, plan.d_otg,
                plan.d_Sx);
    }
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    plan.graph_exec = exec;
    return plan;
}

bool gpu_factorize(GpuFactorPlan& plan, const double* Ax, numeric::SparseLU& out,
                   double* kernel_ms)
{
    const int n = plan.n, nnz = plan.nnz;
    out.n = n;
    out.Sp = plan.Sp;
    out.Si = plan.Si;
    out.x.assign(nnz, 0.0);
    for (size_t q = 0; q < plan.scatter.size(); ++q)
        if (plan.scatter[q] >= 0) out.x[plan.scatter[q]] = Ax[q];

    cudaStream_t stream = static_cast<cudaStream_t>(plan.stream);
    cudaMemcpyAsync(plan.d_Sx, out.x.data(), nnz * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(plan.d_sing, 0, sizeof(int), stream);

    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
    // Replay the captured right-looking schedule (finalize+scatter per level) as
    // one graph: the GPU runs the 2*num_levels nodes back-to-back, no CPU launch.
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(plan.graph_exec), stream);
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) {
        float kms = 0.0f;
        cudaEventElapsedTime(&kms, k0, k1);
        *kernel_ms = kms;
    }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
    int singular = 0;
    cudaMemcpy(&singular, plan.d_sing, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.x.data(), plan.d_Sx, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    return singular == 0;
}

bool gpu_factor(int n, const int* Ap, const int* Ai, const double* Ax,
                const std::vector<int>& Lp, const std::vector<int>& Li,
                const std::vector<int>& parent, numeric::SparseLU& out, double* kernel_ms)
{
    GpuFactorPlan plan = gpu_analyze(n, Ap, Ai, Lp, Li, parent);
    return gpu_factorize(plan, Ax, out, kernel_ms);
}

}  // namespace mysolver::gpu
