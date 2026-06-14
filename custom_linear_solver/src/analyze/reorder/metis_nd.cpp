#include "analyze/reorder/metis_nd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <vector>

#include <metis.h>

namespace custom_linear_solver::reordering {

namespace {

#ifndef CLS_PAR_ND_DEPTH
#define CLS_PAR_ND_DEPTH 4
#endif
#ifndef CLS_PAR_ND_SMALL_BASE_THR
#define CLS_PAR_ND_SMALL_BASE_THR 4000
#endif
#ifndef CLS_PAR_ND_LARGE_BASE_THR
#define CLS_PAR_ND_LARGE_BASE_THR 20000
#endif

constexpr int kParNdDepth = CLS_PAR_ND_DEPTH;
constexpr int kParNdTopInduceThreshold = 49152;

int parallel_nd_base_threshold(int n)
{
    return n < 20000 ? CLS_PAR_ND_SMALL_BASE_THR : CLS_PAR_ND_LARGE_BASE_THR;
}

// Parallel nested dissection. METIS_NodeND is single-threaded; ND is recursively parallel
// (after a vertex separator, the two halves are independent). Find a separator with METIS,
// recurse on each half on separate threads, order the separator last. Same ND algorithm
// → fill ~= serial METIS, but uses all cores. Returns perm in METIS convention:
// perm[old_local] = new_local_position.
void induce(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
            const std::vector<idx_t>& part, idx_t which, std::vector<idx_t>& sx,
            std::vector<idx_t>& sa, std::vector<int>& map)
{
    const int n = static_cast<int>(xadj.size()) - 1;
    std::vector<int> g2l(n, -1);
    map.clear();
    for (int v = 0; v < n; ++v)
        if (part[v] == which) { g2l[v] = static_cast<int>(map.size()); map.push_back(v); }
    const int sn = static_cast<int>(map.size());
    sx.assign(sn + 1, 0);
    for (int li = 0; li < sn; ++li) {
        const int v = map[li];
        for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
            if (part[adj[p]] == which) sx[li + 1]++;
    }
    for (int i = 0; i < sn; ++i) sx[i + 1] += sx[i];
    sa.resize(sx[sn]);
    std::vector<idx_t> pos(sx.begin(), sx.end());
    for (int li = 0; li < sn; ++li) {
        const int v = map[li];
        for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
            if (part[adj[p]] == which) sa[pos[li]++] = g2l[adj[p]];
    }
}

// Parallel induced-subgraph extraction (same output as induce(), multi-core). Used at the
// root recursion level where only one thread is otherwise active; the inner par_for self-
// limits to serial for small subgraphs, so deeper (already thread-saturated) levels keep the
// serial induce().
void induce_par(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                const std::vector<idx_t>& part, idx_t which, std::vector<idx_t>& sx,
                std::vector<idx_t>& sa, std::vector<int>& map);

void base_nodend(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                 std::vector<int>& perm, int seed)
{
    const int n = static_cast<int>(xadj.size()) - 1;
    perm.assign(n, 0);
    if (n <= 1) { if (n == 1) perm[0] = 0; return; }
    if (adj.empty()) { for (int i = 0; i < n; ++i) perm[i] = i; return; }
    idx_t nv = n;
    // METIS_NodeND is read-only on the graph; base_nodend is the recursion terminal so
    // xadj/adj aren't reused — pass them directly without copying.
    std::vector<idx_t> p(n), ip(n);
    idx_t opt[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opt);
    opt[METIS_OPTION_NUMBERING] = 0;
    opt[METIS_OPTION_SEED] = seed;  // fixed seed -> deterministic across threads
    std::srand(seed);  // reseed per call -> each subgraph starts from a fixed RNG state
                       // (deterministic with the LD_PRELOAD thread-safe rand; no-op otherwise)
    if (METIS_NodeND(&nv, const_cast<idx_t*>(xadj.data()), const_cast<idx_t*>(adj.data()),
                     nullptr, opt, p.data(), ip.data()) != METIS_OK) {
        for (int i = 0; i < n; ++i) perm[i] = i;
        return;
    }
    for (int i = 0; i < n; ++i) perm[i] = static_cast<int>(p[i]);
}

void par_nd_rec(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                std::vector<int>& perm, int depth, int base_thr, int seed, bool is_root = true)
{
    const int n = static_cast<int>(xadj.size()) - 1;
    if (depth <= 0 || n < base_thr || adj.empty()) { base_nodend(xadj, adj, perm, seed); return; }
    idx_t nv = n, sepsize = 0;
    // No graph copy. METIS_ComputeVertexSeparator treats the input graph as read-only (it
    // copies into its own internal graph_t), and induce() below reads the ORIGINAL xadj/adj
    // — so pass them directly (const_cast only for the C API signature). Removes a full-graph
    // copy at every recursion level (the root copy is on the sequential parND critical path).
    std::vector<idx_t> part(n);
    idx_t* mx = const_cast<idx_t*>(xadj.data());
    idx_t* ma = const_cast<idx_t*>(adj.data());
    idx_t opt[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opt);
    opt[METIS_OPTION_NUMBERING] = 0;
    opt[METIS_OPTION_SEED] = seed;  // fixed seed -> deterministic across threads
    std::srand(seed);  // reseed per call (see base_nodend)
    bool got_sep = false;
    if (!got_sep &&
        METIS_ComputeVertexSeparator(&nv, mx, ma, nullptr, opt, &sepsize,
                                     part.data()) != METIS_OK) {
        base_nodend(xadj, adj, perm, seed);
        return;
    }
    std::vector<idx_t> x0, a0, x1, a1;
    std::vector<int> m0, m1;
    // Use the multi-core induce on the few TOP levels (large subgraphs, few active recursion
    // threads), where the single-threaded induce sits on the critical path. The inner par_for
    // self-limits to serial below 32768, so deeper/smaller subgraphs keep serial induce and the
    // oversubscription from concurrent branches stays bounded.
    if (is_root || n >= kParNdTopInduceThreshold) {
        std::thread ti([&] { induce_par(xadj, adj, part, 0, x0, a0, m0); });
        induce_par(xadj, adj, part, 1, x1, a1, m1);
        ti.join();
    } else {
        induce(xadj, adj, part, 0, x0, a0, m0);
        induce(xadj, adj, part, 1, x1, a1, m1);
    }
    const int n0 = static_cast<int>(m0.size()), n1 = static_cast<int>(m1.size());
    if (n0 == 0 || n1 == 0) { base_nodend(xadj, adj, perm, seed); return; }  // degenerate split
    std::vector<int> p0, p1;
    std::thread th([&] { par_nd_rec(x0, a0, p0, depth - 1, base_thr, seed, false); });
    par_nd_rec(x1, a1, p1, depth - 1, base_thr, seed, false);  // recursive calls go through METIS
    th.join();
    // perm convention is perm[new_position] = old_vertex (matches the caller's permute_sym
    // and METIS_NodeND's first output). p0[np]=old_local in part0 at sub-newpos np.
    perm.assign(n, 0);
    for (int np = 0; np < n0; ++np) perm[np] = m0[p0[np]];          // part 0 -> [0, n0)
    for (int np = 0; np < n1; ++np) perm[n0 + np] = m1[p1[np]];     // part 1 -> [n0, n0+n1)
    int j = 0;
    for (int v = 0; v < n; ++v)
        if (part[v] == 2) perm[n0 + n1 + (j++)] = v;               // separator last
}

// =======================================================================================
//  GPU/TC-OBJECTIVE NESTED DISSECTION   (exp_260612 — own the recursion + objective)
// =======================================================================================
//
// We own the recursion, the per-split objective, and the stopping granularity; METIS_Compute-
// VertexSeparator stays the (well-tuned) bisection primitive. At each subgraph we generate C
// candidate separators spanning METIS's (separator-size ↔ balance) tradeoff curve (varied
// UFACTOR), then keep the GPU-best by a cost that — unlike METIS's fill objective — discounts
// large separators on the TF32/Ozaki path (TC accelerates the big top fronts at B=1, doc 11) and
// penalizes imbalance (children fill the GPU in parallel). The recursion stops at a GPU-tuned leaf
// size (front-tier lever METIS doesn't expose). perm convention matches par_nd_rec
// (perm[new_pos]=old_vertex, separator last) — drop-in for the pipeline.
struct GpuNdParams {
    int cand = 4;          // candidate separators per node (CLS_GPU_ND_CAND)
    int leaf = 2000;       // stop dissecting below this n -> base_nodend (CLS_GPU_ND_LEAF)
    double lambda = 0.5;   // imbalance penalty, as a fraction of separator cost (CLS_GPU_ND_LAMBDA)
    double tc_g = 1.0;     // TC trailing speedup for the separator-cost discount (CLS_GPU_ND_TC_G)
    int fm = 1;            // FM separator refinement toward the GPU objective (CLS_GPU_ND_FM)
    int fm_passes = 4;     // FM passes (CLS_GPU_ND_FM_PASS)
    int seed = 42;
};

// Critical-path cost of a separator front of size s, TC-discounted (doc 11: trailing share s_share
// ≈ (s−nc)/s runs on tensor cores → divided by tc_g; the panel-LU part stays scalar). nc unknown at
// graph level → approximate by the panel-width cap (~16). tc_g≤1 → plain s³ (no discount).
double sep_front_cost(double s, double tc_g)
{
    double c = s * s * s;
    if (tc_g > 1.0 && s > 0.0) {
        const double nc = 16.0;
        const double share = s > nc ? (s - nc) / s : 0.0;
        c *= (1.0 - share) + share / tc_g;
    }
    return c;
}

// Full GPU separator-front cost = sep_front_cost(|S|) scaled by an imbalance penalty (balanced
// children → parallelism/short critical path). This is the objective the FM refinement minimizes —
// unlike METIS, which only minimizes |S| (fill).
double gpu_cost(double ns, double na, double nb, double lambda, double tc_g)
{
    const double tot = na + nb;
    const double imb = tot > 0.0 ? std::abs(na - nb) / tot : 0.0;
    return sep_front_cost(ns, tc_g) * (1.0 + lambda * imb);
}

// FM vertex-separator refinement toward the GPU objective (exp_260612 — owns the SEPARATOR
// objective, the part METIS won't give). Starts from a valid separator (part[v] ∈ {0=A,1=B,2=S})
// and greedily applies the node-move "v∈S → side d": v leaves S to d, and every neighbor of v on
// the opposite side (1−d) is forced into S (preserves validity — no A-B edge). |S| changes by
// (oppcnt−1) and balance shifts; gpu_cost weighs both, so the objective can ACCEPT a larger
// separator when it buys balance — the divergence from METIS's pure-fill local minimum. Greedy
// hill-climb over the live separator boundary; power-grid separators are small (≈√n) so it's cheap.
void gpu_sep_refine(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                    std::vector<idx_t>& part, int& na, int& nb, int& ns,
                    double lambda, double tc_g, int max_passes)
{
    if (lambda <= 0.0 && tc_g <= 1.0) return;   // pure |S| objective: METIS is already at its min
    const int n = static_cast<int>(xadj.size()) - 1;
    auto opp_cnt = [&](int v, int d) {
        const idx_t other = static_cast<idx_t>(1 - d); int c = 0;
        for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p) if (part[adj[p]] == other) ++c;
        return c;
    };
    for (int pass = 0; pass < max_passes; ++pass) {
        std::vector<int> sep;
        sep.reserve(static_cast<std::size_t>(ns) + 16);
        for (int v = 0; v < n; ++v) if (part[v] == 2) sep.push_back(v);
        bool improved = false;
        const long move_cap = 4L * static_cast<long>(sep.size()) + 16;
        long moves = 0;
        while (moves < move_cap) {
            const double cur = gpu_cost(ns, na, nb, lambda, tc_g);
            double best_gain = cur * 1e-9 + 1e-12;
            int best_v = -1, best_d = 0;
            for (int v : sep) {
                if (part[v] != 2) continue;             // stale (already moved)
                for (int d = 0; d < 2; ++d) {
                    const int cnt = opp_cnt(v, d);
                    const double new_ns = (double)ns - 1.0 + cnt;
                    const double new_na = (d == 0) ? na + 1 : na - cnt;
                    const double new_nb = (d == 1) ? nb + 1 : nb - cnt;
                    if (new_na < 1.0 || new_nb < 1.0) continue;
                    const double gain = cur - gpu_cost(new_ns, new_na, new_nb, lambda, tc_g);
                    if (gain > best_gain) { best_gain = gain; best_v = v; best_d = d; }
                }
            }
            if (best_v < 0) break;
            const int v = best_v, d = best_d, other = 1 - d;
            part[v] = static_cast<idx_t>(d); --ns; if (d == 0) ++na; else ++nb;
            for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p) {
                const int u = adj[p];
                if (part[u] == other) { part[u] = 2; ++ns; if (other == 0) --na; else --nb; sep.push_back(u); }
            }
            improved = true; ++moves;
        }
        if (!improved) break;
    }
}

void gpu_nd_rec(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                std::vector<int>& perm, const GpuNdParams& prm)
{
    const int n = static_cast<int>(xadj.size()) - 1;
    if (n < prm.leaf || adj.empty()) { base_nodend(xadj, adj, perm, prm.seed); return; }

    // Generate C candidate separators (varied UFACTOR spans balanced↔imbalanced cuts), keep GPU-best.
    std::vector<idx_t> best_part;
    double best_score = -1.0;
    int best_n0 = 0, best_n1 = 0;
    idx_t* mx = const_cast<idx_t*>(xadj.data());
    idx_t* ma = const_cast<idx_t*>(adj.data());
    for (int c = 0; c < std::max(1, prm.cand); ++c) {
        idx_t nv = n, sepsize = 0;
        std::vector<idx_t> part(n);
        idx_t opt[METIS_NOPTIONS];
        METIS_SetDefaultOptions(opt);
        opt[METIS_OPTION_NUMBERING] = 0;
        opt[METIS_OPTION_SEED] = prm.seed + 1 + c;   // seed-varied candidates (robust)
        // Cap imbalance: a small, FIXED tolerance for ALL candidates. The earlier escalating
        // UFACTOR=1+c*120 produced extreme-imbalance traps at high CAND that the lambda penalty
        // couldn't always reject (CAND≥6 regressed 20-130%). Vary the SEED, not the balance.
        opt[METIS_OPTION_UFACTOR] = 30;              // ~3% imbalance tolerance, all candidates
        std::srand(prm.seed + 1 + c);
        if (METIS_ComputeVertexSeparator(&nv, mx, ma, nullptr, opt, &sepsize, part.data()) != METIS_OK)
            continue;
        int n0 = 0, n1 = 0, ns = 0;
        for (int v = 0; v < n; ++v) { const idx_t pv = part[v]; if (pv == 0) ++n0; else if (pv == 1) ++n1; else ++ns; }
        if (n0 == 0 || n1 == 0) continue;   // degenerate
        const double imb = (double)std::abs(n0 - n1) / (double)(n0 + n1);  // 0=balanced
        const double score = sep_front_cost((double)ns, prm.tc_g) * (1.0 + prm.lambda * imb);
        if (best_score < 0.0 || score < best_score) {
            best_score = score; best_part = part; best_n0 = n0; best_n1 = n1;
        }
    }
    if (best_score < 0.0) { base_nodend(xadj, adj, perm, prm.seed); return; }

    // FM-refine the chosen separator toward the GPU objective (move vertices off METIS's fill
    // local-minimum toward better balance — the one thing METIS's fill objective cannot do).
    if (prm.fm) {
        int ns = n - best_n0 - best_n1;
        gpu_sep_refine(xadj, adj, best_part, best_n0, best_n1, ns, prm.lambda, prm.tc_g, prm.fm_passes);
    }

    std::vector<idx_t> x0, a0, x1, a1;
    std::vector<int> m0, m1;
    induce(xadj, adj, best_part, 0, x0, a0, m0);
    induce(xadj, adj, best_part, 1, x1, a1, m1);
    const int n0 = static_cast<int>(m0.size()), n1 = static_cast<int>(m1.size());
    if (n0 == 0 || n1 == 0) { base_nodend(xadj, adj, perm, prm.seed); return; }
    std::vector<int> p0, p1;
    gpu_nd_rec(x0, a0, p0, prm);
    gpu_nd_rec(x1, a1, p1, prm);
    perm.assign(n, 0);
    for (int np = 0; np < n0; ++np) perm[np] = m0[p0[np]];
    for (int np = 0; np < n1; ++np) perm[n0 + np] = m1[p1[np]];
    int j = 0;
    for (int v = 0; v < n; ++v)
        if (best_part[v] == 2) perm[n0 + n1 + (j++)] = v;
}

template <typename Fn>
void par_for(int lo, int hi, Fn&& fn)
{
    unsigned hw = std::thread::hardware_concurrency();
    const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
    if (hi - lo < 32768 || nth <= 1) {
        fn(lo, hi);
        return;
    }
    std::vector<std::thread> th;
    const int chunk = (hi - lo + nth - 1) / nth;
    for (int t = 0; t < nth; ++t) {
        const int a = lo + t * chunk;
        const int b = std::min(hi, a + chunk);
        if (a < b) th.emplace_back([&fn, a, b] { fn(a, b); });
    }
    for (auto& x : th) x.join();
}

void induce_par(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                const std::vector<idx_t>& part, idx_t which, std::vector<idx_t>& sx,
                std::vector<idx_t>& sa, std::vector<int>& map)
{
    const int n = static_cast<int>(xadj.size()) - 1;
    // Local relabel: keep vertices with part[v]==which in ascending vertex order. The scan
    // that assigns local indices is serial (O(n), ~0.1ms) but the count/fill passes are
    // multi-core. Output is byte-identical to serial induce().
    std::vector<int> g2l(n, -1);
    map.clear();
    for (int v = 0; v < n; ++v)
        if (part[v] == which) { g2l[v] = static_cast<int>(map.size()); map.push_back(v); }
    const int sn = static_cast<int>(map.size());
    sx.assign(sn + 1, 0);
    par_for(0, sn, [&](int lo, int hi) {
        for (int li = lo; li < hi; ++li) {
            const int v = map[li];
            int cnt = 0;
            for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
                if (part[adj[p]] == which) ++cnt;
            sx[li + 1] = cnt;
        }
    });
    for (int i = 0; i < sn; ++i) sx[i + 1] += sx[i];
    sa.resize(static_cast<std::size_t>(sx[sn]));
    par_for(0, sn, [&](int lo, int hi) {
        for (int li = lo; li < hi; ++li) {
            const int v = map[li];
            idx_t w = sx[li];
            for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
                if (part[adj[p]] == which) sa[w++] = g2l[adj[p]];
        }
    });
}

// Build an EDGE-WEIGHTED symmetric adjacency from the CSR pattern+values of J. Edge (i,j) weight =
// quantized |J[i,j]|+|J[j,i]| — the electrical coupling strength. METIS edge-weighted partitioning
// then minimizes the *weighted* cut, i.e. cuts the electrically WEAK tie-lines (high-impedance
// interfaces) — the natural power-grid separators that fill-objective METIS (no edge weights) cannot
// target. Weights quantized to [1, kWmax] (METIS needs positive int weights). (exp_260612 Stage 2)
void build_weighted_symmetric_adjacency(int n, const int* rowptr, const int* colidx,
                                        const double* vals, std::vector<idx_t>& xadj,
                                        std::vector<idx_t>& adjncy, std::vector<idx_t>& adjwgt)
{
    struct E { int a, b; double w; };
    std::vector<E> e;
    e.reserve(static_cast<std::size_t>(rowptr[n]));
    for (int i = 0; i < n; ++i)
        for (int p = rowptr[i]; p < rowptr[i + 1]; ++p) {
            const int j = colidx[p];
            if (j == i || j < 0 || j >= n) continue;
            const double w = std::fabs(vals[p]);
            e.push_back({std::min(i, j), std::max(i, j), w});   // undirected pair
        }
    std::sort(e.begin(), e.end(), [](const E& x, const E& y) {
        return x.a != y.a ? x.a < y.a : x.b < y.b;
    });
    // Accumulate duplicate (a,b) pairs (J[i,j] and J[j,i]) into one undirected weight.
    std::vector<E> uniq;
    uniq.reserve(e.size());
    double wmax = 0.0;
    for (std::size_t k = 0; k < e.size();) {
        std::size_t m = k;
        double s = 0.0;
        while (m < e.size() && e[m].a == e[k].a && e[m].b == e[k].b) { s += e[m].w; ++m; }
        uniq.push_back({e[k].a, e[k].b, s});
        if (s > wmax) wmax = s;
        k = m;
    }
    constexpr idx_t kWmax = 1000;
    const double scale = wmax > 0.0 ? wmax / static_cast<double>(kWmax) : 1.0;
    // Directed CSR: each undirected (a,b) emits a→b and b→a with the same quantized weight.
    std::vector<int> deg(n, 0);
    for (const E& u : uniq) { ++deg[u.a]; ++deg[u.b]; }
    xadj.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int v = 0; v < n; ++v) xadj[v + 1] = xadj[v] + deg[v];
    adjncy.resize(static_cast<std::size_t>(xadj[n]));
    adjwgt.resize(static_cast<std::size_t>(xadj[n]));
    std::vector<idx_t> pos(xadj.begin(), xadj.end());
    for (const E& u : uniq) {
        idx_t qw = scale > 0.0 ? static_cast<idx_t>(u.w / scale) : 1;
        if (qw < 1) qw = 1;
        adjncy[pos[u.a]] = u.b; adjwgt[pos[u.a]] = qw; ++pos[u.a];
        adjncy[pos[u.b]] = u.a; adjwgt[pos[u.b]] = qw; ++pos[u.b];
    }
}

// induce() that also carries edge weights into the subgraph (for weighted recursion).
void induce_weighted(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                     const std::vector<idx_t>& wgt, const std::vector<idx_t>& part, idx_t which,
                     std::vector<idx_t>& sx, std::vector<idx_t>& sa, std::vector<idx_t>& sw,
                     std::vector<int>& map)
{
    const int n = static_cast<int>(xadj.size()) - 1;
    std::vector<int> g2l(n, -1);
    map.clear();
    for (int v = 0; v < n; ++v)
        if (part[v] == which) { g2l[v] = static_cast<int>(map.size()); map.push_back(v); }
    const int sn = static_cast<int>(map.size());
    sx.assign(sn + 1, 0);
    for (int li = 0; li < sn; ++li) {
        const int v = map[li];
        for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p) if (part[adj[p]] == which) sx[li + 1]++;
    }
    for (int i = 0; i < sn; ++i) sx[i + 1] += sx[i];
    sa.resize(sx[sn]); sw.resize(sx[sn]);
    std::vector<idx_t> pos(sx.begin(), sx.end());
    for (int li = 0; li < sn; ++li) {
        const int v = map[li];
        for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
            if (part[adj[p]] == which) { sa[pos[li]] = g2l[adj[p]]; sw[pos[li]] = wgt[p]; ++pos[li]; }
    }
}

// Top-of-tree electrical recursion: for the first `depth` levels, bisect by EDGE-WEIGHTED METIS
// partitioning (cut weak ties) and derive a vertex separator from the cut boundary; FM-refine it to
// the GPU balance objective; recurse. Below `depth`, hand to the (unweighted) gpu_nd_rec. The
// near-planar power grid makes the top separator dominate the B=1 span, so the electrical structure
// matters most exactly here. (exp_260612 Stage 2)
void gpu_nd_weighted_rec(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                         const std::vector<idx_t>& wgt, std::vector<int>& perm,
                         const GpuNdParams& prm, int depth)
{
    const int n = static_cast<int>(xadj.size()) - 1;
    if (depth <= 0 || n < prm.leaf || adj.empty()) { gpu_nd_rec(xadj, adj, perm, prm); return; }

    idx_t nv = n, ncon = 1, nparts = 2, edgecut = 0;
    std::vector<idx_t> part(n, 0);
    idx_t opt[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opt);
    opt[METIS_OPTION_NUMBERING] = 0;
    opt[METIS_OPTION_SEED] = prm.seed;
    opt[METIS_OPTION_UFACTOR] = 30;   // ~3% balance tolerance
    if (METIS_PartGraphRecursive(&nv, &ncon, const_cast<idx_t*>(xadj.data()),
                                 const_cast<idx_t*>(adj.data()), nullptr, nullptr,
                                 const_cast<idx_t*>(wgt.data()), &nparts, nullptr, nullptr,
                                 opt, &edgecut, part.data()) != METIS_OK) {
        gpu_nd_rec(xadj, adj, perm, prm); return;
    }
    // Derive a vertex separator: move part-1 vertices that border part-0 into S (part==2).
    for (int v = 0; v < n; ++v) {
        if (part[v] != 1) continue;
        for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
            if (part[adj[p]] == 0) { part[v] = 2; break; }
    }
    int na = 0, nb = 0, ns = 0;
    for (int v = 0; v < n; ++v) { const idx_t pv = part[v]; if (pv == 0) ++na; else if (pv == 1) ++nb; else ++ns; }
    if (na == 0 || nb == 0) { gpu_nd_rec(xadj, adj, perm, prm); return; }
    if (prm.fm) gpu_sep_refine(xadj, adj, part, na, nb, ns, prm.lambda, prm.tc_g, prm.fm_passes);

    std::vector<idx_t> x0, a0, w0, x1, a1, w1;
    std::vector<int> m0, m1;
    induce_weighted(xadj, adj, wgt, part, 0, x0, a0, w0, m0);
    induce_weighted(xadj, adj, wgt, part, 1, x1, a1, w1, m1);
    const int n0 = static_cast<int>(m0.size()), n1 = static_cast<int>(m1.size());
    if (n0 == 0 || n1 == 0) { gpu_nd_rec(xadj, adj, perm, prm); return; }
    std::vector<int> p0, p1;
    gpu_nd_weighted_rec(x0, a0, w0, p0, prm, depth - 1);
    gpu_nd_weighted_rec(x1, a1, w1, p1, prm, depth - 1);
    perm.assign(n, 0);
    for (int np = 0; np < n0; ++np) perm[np] = m0[p0[np]];
    for (int np = 0; np < n1; ++np) perm[n0 + np] = m1[p1[np]];
    int j = 0;
    for (int v = 0; v < n; ++v) if (part[v] == 2) perm[n0 + n1 + (j++)] = v;
}

void build_symmetric_adjacency(int n, const int* col_ptr, const int* row_idx,
                               std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy)
{
    std::vector<int> off(static_cast<std::size_t>(n) + 1, 0);
    auto valid = [&](int row, int col) { return row >= 0 && row < n && row != col; };
    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            const int row = row_idx[p];
            if (!valid(row, col)) continue;
            ++off[row + 1];
            ++off[col + 1];
        }
    }
    for (int v = 0; v < n; ++v) off[v + 1] += off[v];

    std::vector<int> adj(static_cast<std::size_t>(off[n]));
    {
        std::vector<int> next(off.begin(), off.end());
        for (int col = 0; col < n; ++col) {
            for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
                const int row = row_idx[p];
                if (!valid(row, col)) continue;
                adj[next[row]++] = col;
                adj[next[col]++] = row;
            }
        }
    }

    std::vector<int> unique_counts(n, 0);
    par_for(0, n, [&](int lo, int hi) {
        for (int v = lo; v < hi; ++v) {
            const int b = off[v];
            const int e = off[v + 1];
            std::sort(adj.begin() + b, adj.begin() + e);
            int count = 0;
            int last = -1;
            for (int p = b; p < e; ++p) {
                if (adj[p] != last) {
                    ++count;
                    last = adj[p];
                }
            }
            unique_counts[v] = count;
        }
    });

    xadj.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int v = 0; v < n; ++v) {
        xadj[v + 1] = xadj[v] + static_cast<idx_t>(unique_counts[v]);
    }
    adjncy.resize(static_cast<std::size_t>(xadj[n]));

    par_for(0, n, [&](int lo, int hi) {
        for (int v = lo; v < hi; ++v) {
            idx_t w = xadj[v];
            int last = -1;
            for (int p = off[v]; p < off[v + 1]; ++p) {
                if (adj[p] != last) {
                    adjncy[static_cast<std::size_t>(w++)] = static_cast<idx_t>(adj[p]);
                    last = adj[p];
                }
            }
        }
    });
}

// Run the ND ordering (parallel nested dissection or serial METIS NodeND) on an already-built
// symmetric idx_t graph. Fills perm in METIS convention (perm[new_pos] = old_vertex). Shared by
// metis_nd (CPU graph build) and metis_nd_from_graph (GPU graph build).
bool run_nd_on_graph(int n, std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy,
                     std::vector<int>& perm, bool parallel, int seed, double t_build_ms)
{
    (void)t_build_ms;

    if (adjncy.empty()) {  // no edges: natural order is optimal
        for (int i = 0; i < n; ++i) perm[i] = i;
        return true;
    }

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED] = seed;

    if (parallel) {
        const int base_thr = parallel_nd_base_threshold(n);
        std::vector<int> pp;
        par_nd_rec(xadj, adjncy, pp, kParNdDepth, base_thr, seed);
        for (int i = 0; i < n; ++i) perm[i] = pp[i];
        return true;
    }
    idx_t nvtxs = n;
    std::vector<idx_t> mperm(n);
    std::vector<idx_t> miperm(n);
    const int rc = METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), nullptr, options,
                                mperm.data(), miperm.data());
    if (rc != METIS_OK) {
        for (int i = 0; i < n; ++i) perm[i] = i;
        return true;
    }
    for (int i = 0; i < n; ++i) perm[i] = static_cast<int>(mperm[i]);
    return true;
}

}  // namespace

bool metis_nd_from_graph(int n, std::vector<int>& xadj_in, std::vector<int>& adjncy_in,
                         std::vector<int>& perm, bool parallel, int seed)
{
    if (n < 0) return false;
    perm.assign(static_cast<std::size_t>(n), 0);
    if (n <= 1) { if (n == 1) perm[0] = 0; return true; }

    // Adopt the prebuilt (GPU-produced) symmetric graph as idx_t. When idx_t == int the
    // host arrays are moved in with no copy; otherwise widen element-wise.
    std::vector<idx_t> xadj, adjncy;
    if constexpr (std::is_same_v<idx_t, int>) {
        xadj = std::move(xadj_in);
        adjncy = std::move(adjncy_in);
    } else {
        xadj.assign(xadj_in.begin(), xadj_in.end());
        adjncy.assign(adjncy_in.begin(), adjncy_in.end());
    }
    return run_nd_on_graph(n, xadj, adjncy, perm, parallel, seed, 0.0);
}

bool gpu_nd_from_graph(int n, std::vector<int>& xadj_in, std::vector<int>& adjncy_in,
                       std::vector<int>& perm, int seed)
{
    if (n < 0) return false;
    perm.assign(static_cast<std::size_t>(n), 0);
    if (n <= 1) { if (n == 1) perm[0] = 0; return true; }

    std::vector<idx_t> xadj, adjncy;
    if constexpr (std::is_same_v<idx_t, int>) {
        xadj = std::move(xadj_in);
        adjncy = std::move(adjncy_in);
    } else {
        xadj.assign(xadj_in.begin(), xadj_in.end());
        adjncy.assign(adjncy_in.begin(), adjncy_in.end());
    }
    if (adjncy.empty()) { for (int i = 0; i < n; ++i) perm[i] = i; return true; }

    auto envi = [](const char* k, int d) { const char* s = std::getenv(k); return s ? std::atoi(s) : d; };
    auto envd = [](const char* k, double d) { const char* s = std::getenv(k); return s ? std::atof(s) : d; };
    GpuNdParams prm;
    prm.cand = std::max(1, envi("CLS_GPU_ND_CAND", 4));
    prm.leaf = std::max(1, envi("CLS_GPU_ND_LEAF", 2000));
    prm.lambda = std::max(0.0, envd("CLS_GPU_ND_LAMBDA", 0.5));
    prm.tc_g = std::max(0.0, envd("CLS_GPU_ND_TC_G", 1.0));
    prm.fm = envi("CLS_GPU_ND_FM", 1);
    prm.fm_passes = std::max(0, envi("CLS_GPU_ND_FM_PASS", 4));
    prm.seed = seed;

    std::vector<int> p;
    gpu_nd_rec(xadj, adjncy, p, prm);
    perm = std::move(p);
    return true;
}

bool gpu_nd_weighted_from_graph(int n, const int* rowptr, const int* colidx, const double* vals,
                                std::vector<int>& perm, int seed)
{
    if (n < 0) return false;
    perm.assign(static_cast<std::size_t>(n), 0);
    if (n <= 1) { if (n == 1) perm[0] = 0; return true; }

    std::vector<idx_t> xadj, adjncy, adjwgt;
    build_weighted_symmetric_adjacency(n, rowptr, colidx, vals, xadj, adjncy, adjwgt);
    if (adjncy.empty()) { for (int i = 0; i < n; ++i) perm[i] = i; return true; }

    auto envi = [](const char* k, int d) { const char* s = std::getenv(k); return s ? std::atoi(s) : d; };
    auto envd = [](const char* k, double d) { const char* s = std::getenv(k); return s ? std::atof(s) : d; };
    GpuNdParams prm;
    prm.cand = std::max(1, envi("CLS_GPU_ND_CAND", 4));
    prm.leaf = std::max(1, envi("CLS_GPU_ND_LEAF", 2000));
    prm.lambda = std::max(0.0, envd("CLS_GPU_ND_LAMBDA", 0.5));
    prm.tc_g = std::max(0.0, envd("CLS_GPU_ND_TC_G", 1.0));
    prm.fm = envi("CLS_GPU_ND_FM", 1);
    prm.fm_passes = std::max(0, envi("CLS_GPU_ND_FM_PASS", 4));
    prm.seed = seed;
    const int ew_depth = std::max(1, envi("CLS_GPU_ND_EW_DEPTH", 3));   // top levels cut electrically

    std::vector<int> pp;
    gpu_nd_weighted_rec(xadj, adjncy, adjwgt, pp, prm, ew_depth);
    perm = std::move(pp);
    return true;
}

bool metis_nd(int n, const int* col_ptr, const int* row_idx, std::vector<int>& perm,
              bool parallel, std::vector<int>* sym_col_ptr, std::vector<int>* sym_row_idx,
              int seed)
{
    if (n < 0 || (n > 0 && (col_ptr == nullptr || (col_ptr[n] > 0 && row_idx == nullptr)))) {
        return false;
    }
    perm.assign(static_cast<std::size_t>(n), 0);
    if (n <= 1) {
        if (n == 1) {
            perm[0] = 0;
        }
        return true;
    }

    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    build_symmetric_adjacency(n, col_ptr, row_idx, xadj, adjncy);
    auto export_sym_graph = [&] {
        if (sym_col_ptr != nullptr) {
            if constexpr (std::is_same_v<idx_t, int>) {
                *sym_col_ptr = std::move(xadj);
            } else {
                sym_col_ptr->resize(static_cast<std::size_t>(n) + 1);
                for (int i = 0; i <= n; ++i) (*sym_col_ptr)[i] = static_cast<int>(xadj[i]);
            }
        }
        if (sym_row_idx != nullptr) {
            if constexpr (std::is_same_v<idx_t, int>) {
                *sym_row_idx = std::move(adjncy);
            } else {
                sym_row_idx->resize(adjncy.size());
                for (std::size_t i = 0; i < adjncy.size(); ++i) {
                    (*sym_row_idx)[i] = static_cast<int>(adjncy[i]);
                }
            }
        }
    };

    // No edges at all (e.g. a diagonal block): natural order is optimal.
    if (adjncy.empty()) {
        for (int i = 0; i < n; ++i) {
            perm[i] = i;
        }
        export_sym_graph();
        return true;
    }

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED] = seed;
    // Faster METIS opts (NITER=1/2, RM coarsening) were tested: ~-2..-16% on analyze wall but
    // produced orderings whose pivot structure fails the no-pivot GPU factor on some matrices,
    // and changed the fill basis (downstream factor/solve regressed). Default METIS_NodeND
    // stays — analyze is METIS_NodeND-bound (~92-95%).

    // PARALLEL nested dissection: same ND quality, multi-core. Enabled by the `parallel` arg.
    if (parallel) {
        // Base-case threshold for the parallel recursion. The two regimes:
        //   - Small Jacobians (n < 20000): recurse deeper (base 4000) — faster AND lower fill.
        //   - Large Jacobians (n >= 20000): base 20000. Going deeper over-recurses on the long
        //     elimination chains typical of large power-grid Jacobians, raising fill and
        //     regressing the downstream factor/solve.
        const int base_thr = parallel_nd_base_threshold(n);
        std::vector<int> pp;
        par_nd_rec(xadj, adjncy, pp, kParNdDepth, base_thr, seed);
        for (int i = 0; i < n; ++i) perm[i] = pp[i];
        export_sym_graph();
        return true;
    }
    idx_t nvtxs = n;
    std::vector<idx_t> mperm(n);
    std::vector<idx_t> miperm(n);
    const int rc = METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), nullptr,
                                options, mperm.data(), miperm.data());
    if (rc != METIS_OK) {
        // Degrade gracefully: natural order keeps the solver usable, just
        // without the fill reduction for this block.
        for (int i = 0; i < n; ++i) {
            perm[i] = i;
        }
        export_sym_graph();
        return true;
    }

    for (int i = 0; i < n; ++i) {
        perm[i] = static_cast<int>(mperm[i]);
    }
    export_sym_graph();
    return true;
}

}  // namespace custom_linear_solver::reordering
