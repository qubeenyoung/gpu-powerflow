#include "reordering/metis_nd.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <vector>

#include <metis.h>

namespace custom_linear_solver::reordering {

namespace {

// cy155: PARALLEL nested dissection (research-grade A rewrite). METIS_NodeND is
// single-threaded; ND is recursively parallel (after a vertex separator, the two halves
// are independent). Find a separator with METIS, recurse on each half on separate threads,
// order the separator last. Same ND algorithm -> fill ~= serial METIS (F-safe), but uses
// the 12 cores. Returns perm in METIS convention: perm[old_local] = new_local_position.
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

void base_nodend(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                 std::vector<int>& perm)
{
    const int n = static_cast<int>(xadj.size()) - 1;
    perm.assign(n, 0);
    if (n <= 1) { if (n == 1) perm[0] = 0; return; }
    if (adj.empty()) { for (int i = 0; i < n; ++i) perm[i] = i; return; }
    idx_t nv = n;
    std::vector<idx_t> p(n), ip(n);  // cy263: no xx/aa copy (METIS_NodeND is read-only on the
                                     // graph; base_nodend is terminal so xadj/adj aren't reused).
    idx_t opt[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opt);
    opt[METIS_OPTION_NUMBERING] = 0;
    opt[METIS_OPTION_SEED] = 42;  // fixed seed -> deterministic across threads
    std::srand(42);  // cy168: reseed per call -> each subgraph from a fixed RNG state
                     // (deterministic with the LD_PRELOAD thread-safe rand; no-op otherwise)
    if (METIS_NodeND(&nv, const_cast<idx_t*>(xadj.data()), const_cast<idx_t*>(adj.data()),
                     nullptr, opt, p.data(), ip.data()) != METIS_OK) {
        for (int i = 0; i < n; ++i) perm[i] = i;
        return;
    }
    for (int i = 0; i < n; ++i) perm[i] = static_cast<int>(p[i]);
}

void par_nd_rec(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                std::vector<int>& perm, int depth, int base_thr, bool is_root = true)
{
    const int n = static_cast<int>(xadj.size()) - 1;
    if (depth <= 0 || n < base_thr || adj.empty()) { base_nodend(xadj, adj, perm); return; }
    idx_t nv = n, sepsize = 0;
    // cy263: no xx/aa graph copy. METIS_ComputeVertexSeparator treats the input graph as
    // read-only (it copies into its own internal graph_t), and induce() below reads the
    // ORIGINAL xadj/adj -- so pass them directly (const_cast only for the C API signature).
    // Removes a full-graph copy at every recursion level (the root copy is on the sequential
    // parND critical path). gpu_test + fill identity verify METIS leaves the input untouched.
    std::vector<idx_t> part(n);
    idx_t* mx = const_cast<idx_t*>(xadj.data());
    idx_t* ma = const_cast<idx_t*>(adj.data());
    idx_t opt[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opt);
    opt[METIS_OPTION_NUMBERING] = 0;
    opt[METIS_OPTION_SEED] = 42;  // fixed seed -> deterministic across threads
    std::srand(42);  // cy168: reseed per call (see base_nodend)
    bool got_sep = false;
    if (!got_sep &&
        METIS_ComputeVertexSeparator(&nv, mx, ma, nullptr, opt, &sepsize,
                                     part.data()) != METIS_OK) {
        base_nodend(xadj, adj, perm);
        return;
    }
    std::vector<idx_t> x0, a0, x1, a1;
    std::vector<int> m0, m1;
    induce(xadj, adj, part, 0, x0, a0, m0);
    induce(xadj, adj, part, 1, x1, a1, m1);
    const int n0 = static_cast<int>(m0.size()), n1 = static_cast<int>(m1.size());
    if (n0 == 0 || n1 == 0) { base_nodend(xadj, adj, perm); return; }  // degenerate split
    std::vector<int> p0, p1;
    std::thread th([&] { par_nd_rec(x0, a0, p0, depth - 1, base_thr, false); });
    par_nd_rec(x1, a1, p1, depth - 1, base_thr, false);  // cy219: recursion uses METIS (is_root=false)
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
                     std::vector<int>& perm, bool parallel, double t_build_ms)
{
    const bool tm = std::getenv("METIS_TIME") != nullptr;
    auto t1 = std::chrono::steady_clock::now();

    if (adjncy.empty()) {  // no edges: natural order is optimal
        for (int i = 0; i < n; ++i) perm[i] = i;
        return true;
    }

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;

    const char* pd = std::getenv("PAR_ND");
    if (parallel || pd) {
        const int depth = (pd && std::atoi(pd) > 0) ? std::atoi(pd) : 4;
        const char* bs = std::getenv("PAR_ND_BASE");
        const int base_thr = bs ? std::atoi(bs) : (n < 20000 ? 4000 : 20000);
        std::vector<int> pp;
        par_nd_rec(xadj, adjncy, pp, depth, base_thr);
        for (int i = 0; i < n; ++i) perm[i] = pp[i];
        if (tm) {
            auto t2 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "  [metis-PAR] n=%d adj_build=%.1fms parND(d=%d)=%.1fms\n", n,
                         t_build_ms, depth,
                         std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        return true;
    }
    idx_t nvtxs = n;
    std::vector<idx_t> mperm(n);
    std::vector<idx_t> miperm(n);
    const int rc = METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), nullptr, options,
                                mperm.data(), miperm.data());
    if (tm) {
        auto t2 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "  [metis] n=%d adj_build=%.1fms NodeND=%.1fms\n", n, t_build_ms,
                     std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    if (rc != METIS_OK) {
        for (int i = 0; i < n; ++i) perm[i] = i;
        return true;
    }
    for (int i = 0; i < n; ++i) perm[i] = static_cast<int>(mperm[i]);
    return true;
}

}  // namespace

bool metis_nd_from_graph(int n, std::vector<int>& xadj_in, std::vector<int>& adjncy_in,
                         std::vector<int>& perm, bool parallel)
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
    return run_nd_on_graph(n, xadj, adjncy, perm, parallel, 0.0);
}

bool metis_nd(int n, const int* col_ptr, const int* row_idx, std::vector<int>& perm,
              bool parallel, std::vector<int>* sym_col_ptr, std::vector<int>* sym_row_idx)
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

    const bool tm = std::getenv("METIS_TIME") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
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
    // cy152: tested faster METIS opts (NITER=1/2, RM coarsening) -> only -2..-16% NodeND
    // BUT broke onetone2 (ok=0: the different ordering's pivot structure fails the no-pivot
    // GPU factor) + changes fill basis. Not viable; A stays METIS_NodeND-bound (92-95% of A).

    auto t1 = std::chrono::steady_clock::now();
    // cy155/164: PARALLEL nested dissection (research-grade A). Same ND quality, multi-core.
    // Enabled by the `parallel` arg (production A win) or the PAR_ND env (A/B sweeps).
    const char* pd = std::getenv("PAR_ND");
    if (parallel || pd) {
        const int depth = (pd && std::atoi(pd) > 0) ? std::atoi(pd) : 4;  // cy156: 4 best
        // cy178: base-case threshold. Small matrices (n<20000) used to fall back to SERIAL
        // METIS NodeND -> cuDSS edged them on A. Recursing them with parND (base 4000) is
        // faster AND lower-fill (A/F/S all improve: case8387 beats cuDSS A, case6468 matches).
        // BIG matrices keep 20000 -> their tuned fill is preserved (base 4000 over-recurses
        // SyntheticUSA/onetone2 -> +fill -> F/S regress). PAR_ND_BASE env overrides.
        const char* bs = std::getenv("PAR_ND_BASE");
        const int base_thr = bs ? std::atoi(bs) : (n < 20000 ? 4000 : 20000);
        std::vector<int> pp;
        par_nd_rec(xadj, adjncy, pp, depth, base_thr);
        for (int i = 0; i < n; ++i) perm[i] = pp[i];
        if (tm) {
            auto t2 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "  [metis-PAR] n=%d adj_build=%.1fms parND(d=%d)=%.1fms\n", n,
                         std::chrono::duration<double, std::milli>(t1 - t0).count(), depth,
                         std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        export_sym_graph();
        return true;
    }
    idx_t nvtxs = n;
    std::vector<idx_t> mperm(n);
    std::vector<idx_t> miperm(n);
    const int rc = METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), nullptr,
                                options, mperm.data(), miperm.data());
    if (tm) {
        auto t2 = std::chrono::steady_clock::now();
        const double bld = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double nd = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::fprintf(stderr, "  [metis] n=%d adj_build=%.1fms NodeND=%.1fms\n", n, bld, nd);
    }
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
