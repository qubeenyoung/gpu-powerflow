#include "mysolver/reordering/gpu_nd.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include <metis.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace mysolver::reordering {

namespace {

// Level-synchronous BFS expansion: each vertex on the current frontier (level==d) marks its
// unvisited neighbors as level d+1. Races are idempotent (all writers set the same d+1), so no
// atomics needed; `changed` flags whether the frontier advanced.
__global__ void bfs_expand(int n, const int* __restrict__ xadj, const int* __restrict__ adj,
                           int* level, int d, int* changed)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || level[v] != d) return;
    for (int p = xadj[v]; p < xadj[v + 1]; ++p) {
        const int u = adj[p];
        if (level[u] < 0) { level[u] = d + 1; *changed = 1; }
    }
}

// cy212: GPU parallel matching (piece 1 of a resident-GPU multilevel coarsening, the only lever
// left for an A-win over cuDSS -- CPU multilevel only ties METIS, cy211). Each unmatched vertex
// proposes its lowest-degree unmatched neighbor (deterministic tie-break by id); a mutual pair
// (v->u and u->v) matches. Idempotent: both endpoints write the same pairing. Iterate until no
// new match; leftover vertices stay singletons. Mirrors the CPU low-degree-first heuristic so fill
// stays at parity. (Contraction stays CPU for now -> resident benefit comes when it is also GPU.)
__global__ void match_propose(int n, const int* __restrict__ xadj, const int* __restrict__ adj,
                              const int* __restrict__ matched, int* __restrict__ proposal, int deg_thr)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || matched[v] >= 0) return;
    // cy218: degree-bucketed -- only vertices with own-degree <= deg_thr propose in this phase, so
    // LOW-degree vertices match first (approximates the CPU sequential low-degree-first heuristic;
    // resident all-level GPU matching compounded a per-level quality loss -> fill drift, cy213).
    if (xadj[v + 1] - xadj[v] > deg_thr) return;
    int best = -1, bd = 0x7fffffff;
    for (int p = xadj[v]; p < xadj[v + 1]; ++p) {
        const int u = adj[p];
        if (u == v || matched[u] >= 0) continue;
        const int d = xadj[u + 1] - xadj[u];
        if (d < bd || (d == bd && u < best)) { bd = d; best = u; }
    }
    proposal[v] = best;
}

__global__ void match_commit(int n, int* __restrict__ matched, const int* __restrict__ proposal,
                             int* __restrict__ changed)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || matched[v] >= 0) return;
    const int u = proposal[v];
    if (u >= 0 && proposal[u] == v) { matched[v] = u; if (v < u) *changed = 1; }
}

// cy213: kernels for device-resident contraction (the coarse-CSR build, the coarsening bottleneck).
__global__ void k_isrep(int gn, const int* __restrict__ matched, int* __restrict__ isrep)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= gn) return;
    isrep[v] = (matched[v] < 0 || v < matched[v]) ? 1 : 0;  // pair-rep = the lower id; singletons are reps
}
__global__ void k_cmap(int gn, const int* __restrict__ matched, const int* __restrict__ cid,
                       int* __restrict__ cmap)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= gn) return;
    const int rep = (matched[v] < 0 || v < matched[v]) ? v : matched[v];
    cmap[v] = cid[rep];  // both members of a pair share the rep's coarse id
}
// One key per directed fine edge: (coarse_src*cn + coarse_dst), or sentinel cn*cn for intra-coarse.
__global__ void k_emit(int gn, const int* __restrict__ xadj, const int* __restrict__ adj,
                       const int* __restrict__ cmap, long cn, long* __restrict__ keys)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= gn) return;
    const int cv = cmap[v];
    for (int p = xadj[v]; p < xadj[v + 1]; ++p) {
        const int cu = cmap[adj[p]];
        keys[p] = (cu != cv) ? ((long)cv * cn + cu) : (cn * cn);
    }
}
__global__ void k_deg(long num, const long* __restrict__ keys, long cn, int* __restrict__ deg)
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= num) return;
    atomicAdd(&deg[keys[i] / cn], 1);  // src = key/cn
}
__global__ void k_dst(long num, const long* __restrict__ keys, long cn, int* __restrict__ ca)
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= num) return;
    ca[i] = (int)(keys[i] % cn);  // dst = key%cn; keys sorted -> already grouped by src (CSR order)
}
__global__ void k_degbuf(int gn, const int* __restrict__ xadj, int* __restrict__ deg)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < gn) deg[v] = xadj[v + 1] - xadj[v];
}

// ---------------------------------------------------------------------------------------------
// cy208: CPU MULTILEVEL nested-dissection (env GPU_ND_ML). The single-level BFS-bisection cut is
// fill-limited (cy207: ~1.0-2.6x METIS, the level-structure cut is the ceiling). Multilevel is the
// real source of METIS quality: coarsen the graph (greedy matching contraction) down to a tiny
// graph, cut THAT (a small graph -> a good cut), then project the cut back up refining with FM at
// each level. These graph-parameterized CPU helpers mirror the single-level GPU lambdas.

// CPU level-synchronous BFS from `start`; level[v] = -1 if unreached. Returns the max level.
int cpu_bfs(int gn, const int* gx, const int* ga, int start, std::vector<int>& level)
{
    std::fill(level.begin(), level.end(), -1);
    level[start] = 0;
    std::vector<int> frontier{start}, next;
    int maxl = 0;
    while (!frontier.empty()) {
        next.clear();
        for (int v : frontier)
            for (int p = gx[v]; p < gx[v + 1]; ++p) {
                const int u = ga[p];
                if (level[u] < 0) { level[u] = level[v] + 1; maxl = level[u]; next.push_back(u); }
            }
        frontier.swap(next);
    }
    return maxl;
}

// Cut-position search + Koenig min-vertex-cover separator on graph (gn,gx,ga) given level `lvl`.
// Writes cand[] in {0,1,2}; returns separator size or -1 if no balanced cut. (Mirrors build_part.)
int build_part_cpu(int gn, const int* gx, const int* ga, const std::vector<int>& lvl,
                   std::vector<int>& cand)
{
    int maxl = 0;
    for (int v = 0; v < gn; ++v) if (lvl[v] > maxl) maxl = lvl[v];
    std::vector<int> L(lvl);
    for (int v = 0; v < gn; ++v) if (L[v] < 0) L[v] = maxl + 1;
    std::vector<int> cnt(static_cast<std::size_t>(maxl) + 2, 0);
    for (int v = 0; v < gn; ++v) cnt[L[v]]++;
    const int side_min = (gn * 35) / 100;
    int cum = 0, bestL = -1, bestscore = gn + 1;
    for (int l = 1; l <= maxl; ++l) {
        cum += cnt[l - 1];
        const int p0 = cum, p1 = gn - cum;
        const int proxy = std::min(cnt[l - 1], cnt[l]);
        if (p0 >= side_min && p1 >= side_min && proxy < bestscore) { bestscore = proxy; bestL = l; }
    }
    if (bestL < 0) return -1;
    cand.assign(gn, 0);
    for (int v = 0; v < gn; ++v) cand[v] = (L[v] < bestL) ? 0 : 1;
    std::vector<int> ridx(gn, -1), rightv, leftv;
    for (int v = 0; v < gn; ++v)
        if (cand[v] == 1)
            for (int p = gx[v]; p < gx[v + 1]; ++p)
                if (cand[ga[p]] == 0) { ridx[v] = (int)rightv.size(); rightv.push_back(v); break; }
    std::vector<std::vector<int>> adjL;
    for (int v = 0; v < gn; ++v) {
        if (cand[v] != 0) continue;
        std::vector<int> nbr;
        for (int p = gx[v]; p < gx[v + 1]; ++p)
            if (cand[ga[p]] == 1) nbr.push_back(ridx[ga[p]]);
        if (!nbr.empty()) { leftv.push_back(v); adjL.push_back(std::move(nbr)); }
    }
    const int nl = (int)leftv.size(), nr = (int)rightv.size();
    if (nl == 0 || nr == 0) return 0;
    std::vector<int> matchR(nr, -1), matchL(nl, -1);
    std::function<bool(int, std::vector<char>&)> aug = [&](int u, std::vector<char>& vis) -> bool {
        for (int j : adjL[u])
            if (!vis[j]) {
                vis[j] = 1;
                if (matchR[j] < 0 || aug(matchR[j], vis)) { matchR[j] = u; matchL[u] = j; return true; }
            }
        return false;
    };
    for (int i = 0; i < nl; ++i) { std::vector<char> vis(nr, 0); aug(i, vis); }
    std::vector<char> visL(nl, 0), visR(nr, 0);
    std::function<void(int)> dfs = [&](int u) {
        visL[u] = 1;
        for (int j : adjL[u])
            if (!visR[j]) { visR[j] = 1; if (matchR[j] >= 0 && !visL[matchR[j]]) dfs(matchR[j]); }
    };
    for (int i = 0; i < nl; ++i) if (matchL[i] < 0 && !visL[i]) dfs(i);
    int sep = 0;
    for (int i = 0; i < nl; ++i) if (!visL[i]) { cand[leftv[i]] = 2; ++sep; }
    for (int j = 0; j < nr; ++j) if (visR[j]) { cand[rightv[j]] = 2; ++sep; }
    return sep;
}

// Greedy delta<=0 FM separator refinement on graph (gn,gx,ga). (Mirrors fm_refine.)
void fm_refine_cpu(int gn, const int* gx, const int* ga, std::vector<int>& p)
{
    int s0 = 0, s1 = 0;
    for (int v = 0; v < gn; ++v) { if (p[v] == 0) ++s0; else if (p[v] == 1) ++s1; }
    const int cap = (gn * 65) / 100;
    std::vector<char> locked(gn, 0);
    std::vector<int> work;
    for (int v = 0; v < gn; ++v) if (p[v] == 2) work.push_back(v);
    for (int guard = 0; guard < gn + 64; ++guard) {
        int bv = -1, ba = -1, bdelta = 1;
        int w = 0;
        for (int idx = 0; idx < (int)work.size(); ++idx) {
            const int v = work[idx];
            if (p[v] != 2 || locked[v]) continue;
            work[w++] = v;
            int c0 = 0, c1 = 0;
            for (int q = gx[v]; q < gx[v + 1]; ++q) { const int t = p[ga[q]]; if (t == 0) ++c0; else if (t == 1) ++c1; }
            if (s0 + 1 <= cap && c1 - 1 < bdelta) { bdelta = c1 - 1; bv = v; ba = 0; }
            if (s1 + 1 <= cap && c0 - 1 < bdelta) { bdelta = c0 - 1; bv = v; ba = 1; }
        }
        work.resize(w);
        if (bv < 0) break;
        p[bv] = ba; locked[bv] = 1;
        if (ba == 0) ++s0; else ++s1;
        const int opp = 1 - ba;
        for (int q = gx[bv]; q < gx[bv + 1]; ++q) {
            const int u = ga[q];
            if (p[u] == opp) { p[u] = 2; if (opp == 0) --s0; else --s1; work.push_back(u); }
        }
    }
}

// Both-starts (vertex 0 + pseudo-peripheral) cut on a graph -> part {0,1,2}. true on valid bisection.
bool separator_cpu(int gn, const int* gx, const int* ga, std::vector<int>& part)
{
    if (gn <= 1) return false;
    std::vector<int> lvl(gn);
    cpu_bfs(gn, gx, ga, 0, lvl);
    int peri = 0, dmax = -1;
    for (int v = 0; v < gn; ++v) if (lvl[v] > dmax) { dmax = lvl[v]; peri = v; }
    std::vector<int> lvl0(lvl);
    cpu_bfs(gn, gx, ga, peri, lvl);
    std::vector<int> candA, candP;
    const int sepA = build_part_cpu(gn, gx, ga, lvl0, candA);
    const int sepP = build_part_cpu(gn, gx, ga, lvl, candP);
    if (sepA < 0 && sepP < 0) return false;
    const bool useP = (sepP >= 0) && (sepA < 0 || sepP <= sepA);
    part = useP ? candP : candA;
    fm_refine_cpu(gn, gx, ga, part);
    return true;
}

// Greedy-matching coarsening: cmap[v] = coarse id (matched pairs + singletons merge), builds the
// coarse CSR (cx,ca) with deduped adjacency (no self-loops). Returns the coarse vertex count.
// GPU matching: fills matched[v] = partner (or -1). Returns false on any CUDA failure (caller
// falls back to CPU matching). Allocates/uploads/runs/downloads per call (resident reuse comes
// when contraction is also GPU).
bool gpu_match(int gn, const int* gx, const int* ga, int gnnz, std::vector<int>& matched)
{
    int *d_x = nullptr, *d_a = nullptr, *d_m = nullptr, *d_p = nullptr, *d_ch = nullptr;
    auto cleanup = [&] { cudaFree(d_x); cudaFree(d_a); cudaFree(d_m); cudaFree(d_p); cudaFree(d_ch); };
    if (cudaMalloc(&d_x, (long)(gn + 1) * sizeof(int)) != cudaSuccess) { cleanup(); return false; }
    if (cudaMalloc(&d_a, (long)gnnz * sizeof(int)) != cudaSuccess) { cleanup(); return false; }
    if (cudaMalloc(&d_m, (long)gn * sizeof(int)) != cudaSuccess) { cleanup(); return false; }
    if (cudaMalloc(&d_p, (long)gn * sizeof(int)) != cudaSuccess) { cleanup(); return false; }
    if (cudaMalloc(&d_ch, sizeof(int)) != cudaSuccess) { cleanup(); return false; }
    cudaMemcpy(d_x, gx, (long)(gn + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, ga, (long)gnnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_m, 0xff, (long)gn * sizeof(int));  // matched[] = -1
    const int T = 256, B = (gn + T - 1) / T;
    int maxdeg = 1;
    for (int v = 0; v < gn; ++v) maxdeg = std::max(maxdeg, gx[v + 1] - gx[v]);
    // cy218: geometric degree buckets (1,2,4,...,>=maxdeg) -- low-degree vertices match first.
    for (int dt = 1; dt < maxdeg * 2; dt *= 2) {
        for (int round = 0; round < gn; ++round) {
            int changed = 0;
            cudaMemcpy(d_ch, &changed, sizeof(int), cudaMemcpyHostToDevice);
            match_propose<<<B, T>>>(gn, d_x, d_a, d_m, d_p, dt);
            match_commit<<<B, T>>>(gn, d_m, d_p, d_ch);
            cudaMemcpy(&changed, d_ch, sizeof(int), cudaMemcpyDeviceToHost);
            if (!changed) break;
        }
    }
    matched.resize(gn);
    cudaMemcpy(matched.data(), d_m, (long)gn * sizeof(int), cudaMemcpyDeviceToHost);
    cleanup();
    return true;
}

int coarsen(int gn, const int* gx, const int* ga, std::vector<int>& cmap,
            std::vector<int>& cx, std::vector<int>& ca)
{
    cmap.assign(gn, -1);
    int cn = 0;
    // cy212: opt-in GPU matching (env GPU_ND_GMATCH) for large levels -> assign coarse ids from
    // the GPU pairing. Falls back to the CPU heuristic on small levels or CUDA failure.
    bool cpu_match = true;
    const char* gm = std::getenv("GPU_ND_GMATCH");
    if (gm != nullptr && gn >= 50000) {
        std::vector<int> matched;
        if (gpu_match(gn, gx, ga, (int)gx[gn], matched)) {
            for (int v = 0; v < gn; ++v) {
                if (cmap[v] >= 0) continue;
                cmap[v] = cn;
                if (matched[v] >= 0) cmap[matched[v]] = cn;
                ++cn;
            }
            cpu_match = false;
        }
    }
    if (cpu_match) {
        // cy209: process LOW-DEGREE vertices first, match each to its lowest-degree unmatched
        // neighbor (HEM-style for unweighted graphs -> matches more, keeps the coarse graph sparse).
        // cy210: bucket-sort the degree ordering in O(gn) (was std::sort) -- ordering dominates A.
        int maxdeg = 0;
        for (int v = 0; v < gn; ++v) maxdeg = std::max(maxdeg, gx[v + 1] - gx[v]);
        std::vector<int> w(maxdeg + 2, 0);
        for (int v = 0; v < gn; ++v) w[gx[v + 1] - gx[v]]++;
        for (int d = 1; d <= maxdeg + 1; ++d) w[d] += w[d - 1];  // w[d] = end offset of degree<=d
        std::vector<int> order(gn);
        for (int v = gn - 1; v >= 0; --v) order[--w[gx[v + 1] - gx[v]]] = v;  // stable bucket sort
        for (int oi = 0; oi < gn; ++oi) {
            const int v = order[oi];
            if (cmap[v] >= 0) continue;
            int partner = -1, bestdeg = gn + 1;
            for (int p = gx[v]; p < gx[v + 1]; ++p) {
                const int u = ga[p];
                if (u != v && cmap[u] < 0) {
                    const int d = gx[u + 1] - gx[u];
                    if (d < bestdeg) { bestdeg = d; partner = u; }
                }
            }
            cmap[v] = cn;
            if (partner >= 0) cmap[partner] = cn;
            ++cn;
        }
    }
    // cy210: contraction in O(gn+nnz) via a monotonic marker dedup (was per-coarse std::sort+unique).
    // Group fine vertices by coarse id (linked list), then for each coarse c collect its members'
    // coarse-neighbors, deduping with seen[cu]==c. The marker c increases monotonically, so stale
    // marks from earlier coarse vertices are auto-ignored -> no per-vertex reset, no sorting.
    std::vector<int> head(cn, -1), nxt(gn);
    for (int v = gn - 1; v >= 0; --v) { nxt[v] = head[cmap[v]]; head[cmap[v]] = v; }
    std::vector<std::vector<int>> cadj(cn);
    std::vector<int> seen(cn, -1);
    for (int c = 0; c < cn; ++c)
        for (int v = head[c]; v != -1; v = nxt[v])
            for (int p = gx[v]; p < gx[v + 1]; ++p) {
                const int cu = cmap[ga[p]];
                if (cu != c && seen[cu] != c) { seen[cu] = c; cadj[c].push_back(cu); }
            }
    cx.assign(cn + 1, 0);
    for (int c = 0; c < cn; ++c) cx[c + 1] = cx[c] + (int)cadj[c].size();
    ca.resize(cx[cn]);
    for (int c = 0; c < cn; ++c) std::copy(cadj[c].begin(), cadj[c].end(), ca.begin() + cx[c]);
    return cn;
}

// cy213: DEVICE-RESIDENT GPU coarsening. Builds the whole coarsening hierarchy on the GPU --
// matching + id-assignment + contraction per level -- keeping the graph resident (the coarse CSR
// of each level feeds the next level's matching with NO host round-trip; only the small cmap is
// downloaded per level for the CPU projection). This is the configuration that avoids cy212's
// transfer overhead. Fills host cmaps[] (per level) + ns[] (sizes) + the coarsest host CSR.
// Returns false on CUDA/thrust failure (caller falls back to CPU coarsening).
bool gpu_coarsen_resident(int n, const int* xadj, const int* adjncy, int nnz, int coarse_thr,
                          std::vector<std::vector<int>>& gxs, std::vector<std::vector<int>>& gas,
                          std::vector<std::vector<int>>& cmaps, std::vector<int>& ns)
{
    try {
        thrust::device_vector<int> dx(xadj, xadj + n + 1), da(adjncy, adjncy + nnz);
        int gn = n;
        long gnnz = nnz;
        gxs.clear(); gas.clear(); cmaps.clear(); ns.clear();
        gxs.emplace_back(xadj, xadj + n + 1);
        gas.emplace_back(adjncy, adjncy + nnz);
        ns.push_back(n);
        thrust::device_vector<int> chg(1);
        while (gn > coarse_thr) {
            const int T = 256, B = (gn + T - 1) / T;
            int* px = thrust::raw_pointer_cast(dx.data());
            int* pa = thrust::raw_pointer_cast(da.data());
            // --- matching ---
            thrust::device_vector<int> dm(gn, -1), dp(gn);
            int* pm = thrust::raw_pointer_cast(dm.data());
            int* pp = thrust::raw_pointer_cast(dp.data());
            int* pchg = thrust::raw_pointer_cast(chg.data());
            // cy218: per-level max degree (on device) for the geometric degree-bucket schedule.
            thrust::device_vector<int> degb(gn);
            k_degbuf<<<B, T>>>(gn, px, thrust::raw_pointer_cast(degb.data()));
            const int maxdeg = thrust::reduce(degb.begin(), degb.end(), 1, thrust::maximum<int>());
            // cy218: geometric degree buckets (low-degree vertices match first). cy220 found the
            // coarsening (18ms) is matching-ROUNDS-bound, not sync-bound: fewer rounds (fixed-3)
            // is faster but undershoots convergence -> fill regresses; batched-convergence does the
            // same total rounds -> no speedup. So keep adaptive per-round convergence (best fill).
            // cy240: cap matching rounds/bucket (env GPU_ND_MROUNDS) -- re-testing fewer rounds now
            // that the METIS coarsest cut (cy237) compensates for slightly-worse coarsening. Default
            // large = adaptive convergence (cy218).
            // cy240: default 3 rounds/bucket. With the METIS coarsest cut (cy237) compensating for
            // slightly-incomplete coarsening, capping at 3 gives IDENTICAL fill (SyntheticUSA fill_nnz
            // 1.5601M vs adaptive 1.5607M) but faster coarsening (15.6->13.8ms). Was cy218 adaptive.
            const char* mr = std::getenv("GPU_ND_MROUNDS");
            const int max_rounds = (mr && std::atoi(mr) > 0) ? std::atoi(mr) : 3;
            // cy248: run the (capped) rounds SYNC-FREE -- no per-round chg[0] host round-trip. The
            // early-break rarely fires at MROUNDS=3, so the ~180 host syncs cost more than the few
            // extra (async, cheap) kernel launches they'd save. (env GPU_ND_MSYNC restores the check.)
            const bool msync = std::getenv("GPU_ND_MSYNC") != nullptr;
            for (int dt = 1; dt < maxdeg * 2; dt *= 2) {
                for (int round = 0; round < max_rounds; ++round) {
                    if (msync) chg[0] = 0;
                    match_propose<<<B, T>>>(gn, px, pa, pm, pp, dt);
                    match_commit<<<B, T>>>(gn, pm, pp, pchg);
                    if (msync && chg[0] == 0) break;
                }
            }
            // --- coarse-id assignment ---
            thrust::device_vector<int> isrep(gn), cid(gn);
            k_isrep<<<B, T>>>(gn, pm, thrust::raw_pointer_cast(isrep.data()));
            thrust::exclusive_scan(isrep.begin(), isrep.end(), cid.begin());
            const int cn = cid[gn - 1] + isrep[gn - 1];
            if (cn < 2 || cn > (gn * 95) / 100) break;  // no progress -> current is coarsest
            thrust::device_vector<int> dcmap(gn);
            k_cmap<<<B, T>>>(gn, pm, thrust::raw_pointer_cast(cid.data()),
                             thrust::raw_pointer_cast(dcmap.data()));
            std::vector<int> hcmap(gn);
            thrust::copy(dcmap.begin(), dcmap.end(), hcmap.begin());
            cmaps.push_back(std::move(hcmap));  // pushed only on progress -> cmaps.size()==ns.size()-1
            // --- contraction (coarse CSR via sort + unique) ---
            thrust::device_vector<long> keys(gnnz);
            k_emit<<<B, T>>>(gn, px, pa, thrust::raw_pointer_cast(dcmap.data()), (long)cn,
                             thrust::raw_pointer_cast(keys.data()));
            thrust::sort(keys.begin(), keys.end());
            auto uend = thrust::unique(keys.begin(), keys.end());
            const long num_real =
                thrust::lower_bound(keys.begin(), uend, (long)cn * cn) - keys.begin();
            thrust::device_vector<int> ndeg(cn, 0), ncx(cn + 1), nca(num_real > 0 ? num_real : 1);
            const long Bk = (num_real + 255) / 256;
            if (num_real > 0) {
                k_deg<<<(unsigned)Bk, 256>>>(num_real, thrust::raw_pointer_cast(keys.data()),
                                             (long)cn, thrust::raw_pointer_cast(ndeg.data()));
                k_dst<<<(unsigned)Bk, 256>>>(num_real, thrust::raw_pointer_cast(keys.data()),
                                             (long)cn, thrust::raw_pointer_cast(nca.data()));
            }
            thrust::exclusive_scan(ndeg.begin(), ndeg.end(), ncx.begin());
            ncx[cn] = (int)num_real;
            dx.swap(ncx);
            da.swap(nca);
            gn = cn;
            gnnz = num_real;
            // download this coarse level's CSR for the CPU projection + FM (graph stays resident
            // on device for the NEXT level's matching -- no re-upload).
            std::vector<int> hx(gn + 1), ha(gnnz);
            thrust::copy(dx.begin(), dx.begin() + gn + 1, hx.begin());
            thrust::copy(da.begin(), da.begin() + gnnz, ha.begin());
            gxs.push_back(std::move(hx));
            gas.push_back(std::move(ha));
            ns.push_back(cn);
        }
        return cudaGetLastError() == cudaSuccess;
    } catch (...) {
        return false;  // thrust bad_alloc / CUDA error -> CPU fallback
    }
}

// cy237: cut a graph with METIS_ComputeVertexSeparator (high-quality), filling part[] in {0,1,2}.
// Used (opt-in) for the tiny COARSEST graph so the projection gets a METIS-quality seed -- the
// coarsen (GPU) + project/FM (the bulk) stay ours; the coarsest cut is ~free on <=200 vtx.
bool metis_cut(int gn, const int* gx, const int* ga, std::vector<int>& part)
{
    if (gn <= 1) return false;
    idx_t nv = gn, sepsize = 0;
    std::vector<idx_t> xx(gx, gx + gn + 1), aa(ga, ga + gx[gn]), pt(gn);
    idx_t opt[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opt);
    opt[METIS_OPTION_NUMBERING] = 0;
    opt[METIS_OPTION_SEED] = 42;
    if (METIS_ComputeVertexSeparator(&nv, xx.data(), aa.data(), nullptr, opt, &sepsize, pt.data()) != METIS_OK)
        return false;
    part.assign(gn, 0);
    int c0 = 0, c1 = 0;
    for (int v = 0; v < gn; ++v) { part[v] = (int)pt[v]; if (part[v] == 0) ++c0; else if (part[v] == 1) ++c1; }
    return c0 > 0 && c1 > 0;
}

// Multilevel ND separator: coarsen to a hierarchy, cut the coarsest graph, project up with FM.
bool multilevel_separator(int n, const int* xadj, const int* adjncy, int nnz, int* part)
{
    // cy211: SIZE GUARD. Multilevel pays off (METIS-parity fill + parallel-recursion A win) only on
    // LARGE graphs; on small subgraphs recursive multilevel sub-cuts are slightly worse than METIS's
    // tuned NodeND (case8387 F 0.71->1.03 at PAR_ND=4). Return false for small n so the caller falls
    // back to METIS -> best of both (multilevel for big cuts, METIS for small). Env GPU_ND_ML_MIN.
    // cy219: default raised 30000->100000. The GPU-ND root cut matches METIS fill only for VERY
    // large roots (SyntheticUSA 156k: parity + A win); at medium size it regresses fill (ACTIVSg25k
    // 47k: F 1.58->1.76). So GPU-accelerate only the very-large root separator; METIS handles the rest.
    const char* mm = std::getenv("GPU_ND_ML_MIN");
    const int min_n = (mm && std::atoi(mm) > 0) ? std::atoi(mm) : 100000;
    if (n < min_n) return false;
    std::vector<std::vector<int>> gxs, gas, cmaps;
    std::vector<int> ns;
    // cy222: coarsest-graph size threshold (env GPU_ND_CTHR for A/B).
    // cy239: default 200->1000. With the METIS coarsest cut (cy237), a LARGER coarsest is high-quality
    // (cy223's BFS-coarsest catastrophe at CTHR>200 is gone), so coarsen LESS -> fewer levels -> faster
    // coarsening (18.3->15.8ms) AND better fill (SyntheticUSA F ~4.41->~4.21). METIS scales fine to 1000.
    const char* ct = std::getenv("GPU_ND_CTHR");
    const int COARSE_THR = (ct && std::atoi(ct) > 0) ? std::atoi(ct) : 1000;
    // cy220: opt-in phase profiling (env GPU_ND_PROF) to target the root-cut bottleneck.
    const bool prof = std::getenv("GPU_ND_PROF") != nullptr;
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto lap = [&](const char* nm, std::chrono::steady_clock::time_point& a) {
        if (prof) { auto b = clk::now();
            std::fprintf(stderr, "    [ml-prof] %-12s %.1f ms\n", nm,
                std::chrono::duration<double, std::milli>(b - a).count()); a = b; }
    };
    // cy225: device-RESIDENT GPU coarsening is the DEFAULT (matching + contraction on the GPU,
    // graph resident across levels). GPU_ND_NORESIDENT selects CPU coarsening (A/B). Falls back to
    // CPU automatically on any CUDA/thrust failure.
    bool coarsened = false;
    if (std::getenv("GPU_ND_NORESIDENT") == nullptr)
        coarsened = gpu_coarsen_resident(n, xadj, adjncy, nnz, COARSE_THR, gxs, gas, cmaps, ns);
    if (!coarsened) {
        gxs.clear(); gas.clear(); cmaps.clear(); ns.clear();
        gxs.emplace_back(xadj, xadj + n + 1);
        gas.emplace_back(adjncy, adjncy + nnz);
        ns.push_back(n);
        while (ns.back() > COARSE_THR) {
            std::vector<int> cmap, cx, ca;
            const int cn = coarsen(ns.back(), gxs.back().data(), gas.back().data(), cmap, cx, ca);
            if (cn < 2 || cn > (ns.back() * 95) / 100) break;  // no further progress
            cmaps.push_back(std::move(cmap));
            gxs.push_back(std::move(cx));
            gas.push_back(std::move(ca));
            ns.push_back(cn);
        }
    }
    lap("coarsen", t0);
    std::vector<int> p;
    // cy237: METIS coarsest cut is the DEFAULT -- a high-quality seed on the tiny (<=COARSE_THR)
    // coarsest graph (~free) fixed the medium-size fill regression (ACTIVSg25k 1.07x->1.00x) and
    // improved large fill (SyntheticUSA -> 0.99x, beats METIS). The coarsen (GPU) + project/FM stay
    // ours. GPU_ND_NO_METIS_COARSE selects the BFS-bisection separator_cpu (A/B); fallback on failure.
    bool ok = false;
    if (std::getenv("GPU_ND_NO_METIS_COARSE") == nullptr)
        ok = metis_cut(ns.back(), gxs.back().data(), gas.back().data(), p);
    if (!ok) ok = separator_cpu(ns.back(), gxs.back().data(), gas.back().data(), p);
    if (!ok) return false;
    lap("cut", t0);
    for (int l = (int)cmaps.size() - 1; l >= 0; --l) {
        std::vector<int> fp(ns[l]);
        const std::vector<int>& cm = cmaps[l];
        for (int v = 0; v < ns[l]; ++v) fp[v] = p[cm[v]];  // project the coarse cut down
        fm_refine_cpu(ns[l], gxs[l].data(), gas[l].data(), fp);
        p.swap(fp);
    }
    lap("project+FM", t0);
    int c0 = 0, c1 = 0, c2 = 0;
    for (int v = 0; v < n; ++v) { part[v] = p[v]; if (p[v] == 0) ++c0; else if (p[v] == 1) ++c1; else ++c2; }
#ifdef GPU_ND_DEBUG
    std::fprintf(stderr, "  [gpu_nd_ml] n=%d levels=%d coarsest=%d | c0=%d c1=%d sep=%d\n",
                 n, (int)ns.size(), ns.back(), c0, c1, c2);
#endif
    return c0 > 0 && c1 > 0;
}

}  // namespace

bool gpu_nd_separator(int n, const int* xadj, const int* adjncy, int nnz, int* part)
{
    if (n <= 0 || nnz <= 0) return false;
    // cy225: the MULTILEVEL path is the DEFAULT (the validated win-config: fill parity + the GPU-accel
    // A win, root-only via par_nd_rec). A single GPU_ND=1 now gives the win. GPU_ND_SINGLE selects the
    // old single-level BFS-bisection cut (1.0-2.6x fill, kept for A/B); GPU_ND_ML kept as a synonym.
    if (std::getenv("GPU_ND_SINGLE") == nullptr || std::getenv("GPU_ND_ML") != nullptr)
        return multilevel_separator(n, xadj, adjncy, nnz, part);
    int *d_xadj = nullptr, *d_adj = nullptr, *d_level = nullptr, *d_changed = nullptr;
    if (cudaMalloc(&d_xadj, (long)(n + 1) * sizeof(int)) != cudaSuccess) return false;
    if (cudaMalloc(&d_adj, (long)nnz * sizeof(int)) != cudaSuccess) { cudaFree(d_xadj); return false; }
    if (cudaMalloc(&d_level, (long)n * sizeof(int)) != cudaSuccess) {
        cudaFree(d_xadj); cudaFree(d_adj); return false;
    }
    if (cudaMalloc(&d_changed, sizeof(int)) != cudaSuccess) {
        cudaFree(d_xadj); cudaFree(d_adj); cudaFree(d_level); return false;
    }
    cudaMemcpy(d_xadj, xadj, (long)(n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj, adjncy, (long)nnz * sizeof(int), cudaMemcpyHostToDevice);

    const int T = 256, B = (n + T - 1) / T;
    // cy207: BATCH the per-level termination check. The level-sync BFS round-trips the `changed`
    // flag to the host every level (3 memcpy + sync). Instead launch K level-kernels back-to-back
    // on the ordered (default) stream -- each correctly consumes the frontier the previous one
    // produced -- then read `changed` once per batch (K-fold fewer host syncs). A batch with no
    // advance means BFS is done; trailing no-op kernels are cheap. Env BFS_BATCH for A/B.
    const char* bb = std::getenv("BFS_BATCH");
    const int Kbatch = (bb && std::atoi(bb) > 0) ? std::atoi(bb) : 16;
    std::vector<int> level(n, -1);
    // GPU level-synchronous BFS from `start`, filling host `level` (unreached stay -1). Returns
    // the max level reached (eccentricity of `start` within its component).
    auto run_bfs = [&](int start) -> int {
        std::fill(level.begin(), level.end(), -1);
        level[start] = 0;
        cudaMemcpy(d_level, level.data(), (long)n * sizeof(int), cudaMemcpyHostToDevice);
        for (int d = 0; d < n;) {
            int changed = 0;
            cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);
            for (int k = 0; k < Kbatch && d < n; ++k, ++d)
                bfs_expand<<<B, T>>>(n, d_xadj, d_adj, d_level, d, d_changed);
            cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
            if (!changed) break;
        }
        cudaMemcpy(level.data(), d_level, (long)n * sizeof(int), cudaMemcpyDeviceToHost);
        int mx = 0;
        for (int v = 0; v < n; ++v) if (level[v] > mx) mx = level[v];
        return mx;
    };
    // Given a BFS level structure `lvl`, build the best balanced bisection + vertex separator and
    // write it into `cand`. cy203 cut-position search: scan cuts between levels L-1/L that leave
    // both sides >= 35%, score min(width(L-1),width(L)) (upper bound on the smaller boundary),
    // take the smaller-boundary separator at the best cut. Returns the separator size, or -1 if no
    // balanced cut exists. (`bestL` reported via out-param for debug.)
    auto build_part = [&](const std::vector<int>& lvl, std::vector<int>& cand, int& bestL_out) -> int {
        int maxl = 0;
        for (int v = 0; v < n; ++v) if (lvl[v] > maxl) maxl = lvl[v];
        std::vector<int> L(lvl);
        for (int v = 0; v < n; ++v) if (L[v] < 0) L[v] = maxl + 1;  // unreached -> last level
        std::vector<int> cnt(static_cast<std::size_t>(maxl) + 2, 0);
        for (int v = 0; v < n; ++v) cnt[L[v]]++;
        const int side_min = (n * 35) / 100;
        int cum = 0, bestL = -1, bestscore = n + 1;
        for (int l = 1; l <= maxl; ++l) {
            cum += cnt[l - 1];                 // vertices at levels < l
            const int p0 = cum, p1 = n - cum;  // the two halves if cut between l-1 and l
            const int proxy = std::min(cnt[l - 1], cnt[l]);
            if (p0 >= side_min && p1 >= side_min && proxy < bestscore) { bestscore = proxy; bestL = l; }
        }
        bestL_out = bestL;
        if (bestL < 0) return -1;  // no balanced cut
        cand.assign(n, 0);
        for (int v = 0; v < n; ++v) cand[v] = (L[v] < bestL) ? 0 : 1;

        // cy205: MINIMUM vertex separator from the cut = minimum vertex cover of the bipartite
        // boundary graph (left = part-0 boundary, right = part-1 boundary, edges = cut edges).
        // By Koenig's theorem min-vertex-cover = max bipartite matching; the cover covers every
        // cut edge so removing it disconnects the halves -> a valid, provably minimal separator.
        // cy204 took the smaller boundary SIDE (a 2-approximation); the exact cover is <= that.
        std::vector<int> ridx(n, -1), rightv, leftv;
        for (int v = 0; v < n; ++v)
            if (cand[v] == 1)
                for (int p = xadj[v]; p < xadj[v + 1]; ++p)
                    if (cand[adjncy[p]] == 0) { ridx[v] = (int)rightv.size(); rightv.push_back(v); break; }
        std::vector<std::vector<int>> adjL;
        for (int v = 0; v < n; ++v) {
            if (cand[v] != 0) continue;
            std::vector<int> nbr;
            for (int p = xadj[v]; p < xadj[v + 1]; ++p)
                if (cand[adjncy[p]] == 1) nbr.push_back(ridx[adjncy[p]]);
            if (!nbr.empty()) { leftv.push_back(v); adjL.push_back(std::move(nbr)); }
        }
        const int nl = (int)leftv.size(), nr = (int)rightv.size();
        if (nl == 0 || nr == 0) return 0;  // no cut edges -> halves already disconnected, empty sep
        // Kuhn's augmenting-path max matching (boundaries are small: O(V*E) is fine).
        std::vector<int> matchR(nr, -1), matchL(nl, -1);
        std::function<bool(int, std::vector<char>&)> aug = [&](int u, std::vector<char>& vis) -> bool {
            for (int j : adjL[u])
                if (!vis[j]) {
                    vis[j] = 1;
                    if (matchR[j] < 0 || aug(matchR[j], vis)) { matchR[j] = u; matchL[u] = j; return true; }
                }
            return false;
        };
        for (int i = 0; i < nl; ++i) { std::vector<char> vis(nr, 0); aug(i, vis); }
        // Koenig: Z = vertices on alternating paths from unmatched-left. Cover = (L\Z) U (R intersect Z).
        std::vector<char> visL(nl, 0), visR(nr, 0);
        std::function<void(int)> dfs = [&](int u) {
            visL[u] = 1;
            for (int j : adjL[u])
                if (!visR[j]) { visR[j] = 1; if (matchR[j] >= 0 && !visL[matchR[j]]) dfs(matchR[j]); }
        };
        for (int i = 0; i < nl; ++i) if (matchL[i] < 0 && !visL[i]) dfs(i);
        int sep = 0;
        for (int i = 0; i < nl; ++i) if (!visL[i]) { cand[leftv[i]] = 2; ++sep; }
        for (int j = 0; j < nr; ++j) if (visR[j]) { cand[rightv[j]] = 2; ++sep; }
        return sep;
    };

    // cy206: greedy FM-style separator refinement. min-cover is exact only for a GIVEN cut; relocate
    // a separator vertex v entirely into side `a`, pulling its opposite-side neighbors into the
    // separator. The net separator change is (#opposite-side neighbors of v) - 1, so accept only
    // delta <= 0 moves (relocations that don't grow it). After min-cover many separator vertices
    // have ALL their opposite neighbors already in the separator (delta = -1, pure removal) -> this
    // shrinks the separator below the per-cut minimum. Balance-capped, vertices locked once moved.
    auto fm_refine = [&](std::vector<int>& p) {
        int s0 = 0, s1 = 0;
        for (int v = 0; v < n; ++v) { if (p[v] == 0) ++s0; else if (p[v] == 1) ++s1; }
        const int cap = (n * 65) / 100;  // max size of either side
        std::vector<char> locked(n, 0);
        std::vector<int> work;
        for (int v = 0; v < n; ++v) if (p[v] == 2) work.push_back(v);
        for (int guard = 0; guard < n + 64; ++guard) {
            int bv = -1, ba = -1, bdelta = 1;  // accept only delta <= 0
            int w = 0;
            for (int idx = 0; idx < (int)work.size(); ++idx) {
                const int v = work[idx];
                if (p[v] != 2 || locked[v]) continue;
                work[w++] = v;
                int c0 = 0, c1 = 0;
                for (int q = xadj[v]; q < xadj[v + 1]; ++q) {
                    const int t = p[adjncy[q]];
                    if (t == 0) ++c0; else if (t == 1) ++c1;
                }
                if (s0 + 1 <= cap && c1 - 1 < bdelta) { bdelta = c1 - 1; bv = v; ba = 0; }
                if (s1 + 1 <= cap && c0 - 1 < bdelta) { bdelta = c0 - 1; bv = v; ba = 1; }
            }
            work.resize(w);
            if (bv < 0) break;  // no delta<=0 move left
            p[bv] = ba; locked[bv] = 1;
            if (ba == 0) ++s0; else ++s1;
            const int opp = 1 - ba;
            for (int q = xadj[bv]; q < xadj[bv + 1]; ++q) {
                const int u = adjncy[q];
                if (p[u] == opp) { p[u] = 2; if (opp == 0) --s0; else --s1; work.push_back(u); }
            }
        }
    };

    // cy202 pseudo-peripheral start + cy204: evaluate BOTH starts, keep the lower-separator one.
    // A single start vertex is a heuristic -- the peripheral start helps most graphs (more, thinner
    // levels) but HURT case_ACTIVSg25k (cy201 vertex-0 gave 1.54x, cy203 peripheral 2.0x). BFS from
    // 0 -> jump to deepest vertex (peripheral) -> BFS again; score the cut from each level structure
    // and take whichever yields the smaller vertex separator (a good lower-fill proxy under the
    // balance constraint).
    run_bfs(0);
    std::vector<int> lvl0(level);
    int peri = 0, dmax = -1;
    for (int v = 0; v < n; ++v) if (lvl0[v] > dmax) { dmax = lvl0[v]; peri = v; }
    run_bfs(peri);  // leaves the peripheral level structure in `level`
    cudaFree(d_xadj); cudaFree(d_adj); cudaFree(d_level); cudaFree(d_changed);

    std::vector<int> candA, candP;
    int bestLA = -1, bestLP = -1;
    const int sepA = build_part(lvl0, candA, bestLA);
    const int sepP = build_part(level, candP, bestLP);
    if (sepA < 0 && sepP < 0) return false;  // neither start gave a balanced cut -> METIS fallback
    const bool useP = (sepP >= 0) && (sepA < 0 || sepP <= sepA);  // smaller separator wins
    std::vector<int>& win = useP ? candP : candA;
    fm_refine(win);  // cy206: shrink the separator below the per-cut minimum
    for (int v = 0; v < n; ++v) part[v] = win[v];
    int c0 = 0, c1 = 0, c2 = 0;
    for (int v = 0; v < n; ++v) { if (part[v] == 0) ++c0; else if (part[v] == 1) ++c1; else ++c2; }
#ifdef GPU_ND_DEBUG
    std::fprintf(stderr, "  [gpu_nd] n=%d start=%s bestL=%d sepA=%d sepP=%d | c0=%d c1=%d sep=%d\n",
                 n, useP ? "peri" : "v0", useP ? bestLP : bestLA, sepA, sepP, c0, c1, c2);
#endif
    return c0 > 0 && c1 > 0;  // non-degenerate bisection
}

}  // namespace mysolver::reordering
