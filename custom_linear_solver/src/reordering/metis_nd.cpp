#include "reordering/metis_nd.hpp"

#include <algorithm>
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
