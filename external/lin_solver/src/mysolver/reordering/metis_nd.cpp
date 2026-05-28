#include "mysolver/reordering/metis_nd.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <metis.h>

#ifdef MYSOLVER_HAVE_GPU_ND
#include "mysolver/reordering/gpu_nd.hpp"
#endif

namespace mysolver::reordering {

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
#ifdef MYSOLVER_HAVE_GPU_ND
    // cy201: opt-in GPU nested-dissection separator (env GPU_ND). Replaces only the separator
    // computation; induce/recurse/assemble below are reused. Falls back to METIS on failure
    // (degenerate bisection or CUDA error) -> production parND is unaffected (default off).
    // cy219: ROOT-ONLY -- GPU-ND only for the root (largest) cut; METIS parND for the recursion.
    // Recursive GPU-ND multilevel sub-cuts drift on fill (PAR_ND=4 SyntheticUSA F 4.55->5.56,
    // cy219) whereas the GPU top cut + METIS parND recursion keeps fill PARITY and wins A
    // (SyntheticUSA A ~316->~297, reproducible). So GPU-accelerate only the expensive root separator.
    // cy229: GPU_ND_RECURSE lets non-root nodes also try GPU-ND (size-gated by GPU_ND_ML_MIN) --
    // tests extending the win past root-only to the top few large levels (A/B). Default = root-only.
    if ((is_root || std::getenv("GPU_ND_RECURSE") != nullptr) && std::getenv("GPU_ND") != nullptr) {
        std::vector<int> xi(xadj.begin(), xadj.end()), ai(adj.begin(), adj.end()), pi(n);
        if (gpu_nd_separator(n, xi.data(), ai.data(), static_cast<int>(ai.size()), pi.data())) {
            for (int v = 0; v < n; ++v) part[v] = static_cast<idx_t>(pi[v]);
            got_sep = true;
        }
    }
#endif
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

}  // namespace

bool metis_nd(int n, const int* col_ptr, const int* row_idx, std::vector<int>& perm,
              bool parallel)
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
    // Build the symmetric adjacency (A + Aᵀ pattern), excluding the diagonal.
    std::vector<std::vector<int>> adjacency(n);
    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            const int row = row_idx[p];
            if (row == col || row < 0 || row >= n) {
                continue;
            }
            adjacency[row].push_back(col);
            adjacency[col].push_back(row);
        }
    }

    // cy177: parallelize the per-vertex sort+dedup + flatten. adj_build is ~10-20% of A on
    // the large matrices (SyntheticUSA 37.7ms) and was single-threaded. Each vertex's list
    // is independent -> output is byte-identical to the serial flatten (deterministic sort,
    // concatenated in vertex order via prefix-sum offsets) -> same ordering, fill, F.
    auto par_for = [](int lo, int hi, auto&& fn) {
        unsigned hw = std::thread::hardware_concurrency();
        const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
        if (hi - lo < 32768 || nth <= 1) { fn(lo, hi); return; }  // small: thread overhead > gain
        std::vector<std::thread> th;
        const int chunk = (hi - lo + nth - 1) / nth;
        for (int t = 0; t < nth; ++t) {
            const int a = lo + t * chunk, b = std::min(hi, a + chunk);
            if (a < b) th.emplace_back([&fn, a, b] { fn(a, b); });
        }
        for (auto& x : th) x.join();
    };
    par_for(0, n, [&](int lo, int hi) {
        for (int v = lo; v < hi; ++v) {
            std::vector<int>& nb = adjacency[v];
            std::sort(nb.begin(), nb.end());
            nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        }
    });
    std::vector<idx_t> xadj(n + 1);
    xadj[0] = 0;
    for (int v = 0; v < n; ++v)
        xadj[v + 1] = xadj[v] + static_cast<idx_t>(adjacency[v].size());
    std::vector<idx_t> adjncy(xadj[n]);
    par_for(0, n, [&](int lo, int hi) {
        for (int v = lo; v < hi; ++v) {
            idx_t off = xadj[v];
            for (int u : adjacency[v]) adjncy[off++] = static_cast<idx_t>(u);
        }
    });

    // No edges at all (e.g. a diagonal block): natural order is optimal.
    if (adjncy.empty()) {
        for (int i = 0; i < n; ++i) {
            perm[i] = i;
        }
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
        // Degrade gracefully: natural order keeps mysolver correct (KLU still
        // factors), just without the fill reduction for this block.
        for (int i = 0; i < n; ++i) {
            perm[i] = i;
        }
        return true;
    }

    for (int i = 0; i < n; ++i) {
        perm[i] = static_cast<int>(mperm[i]);
    }
    return true;
}

}  // namespace mysolver::reordering

int klu_metis_user_order(int n, int* Ap, int* Ai, int* Perm, struct klu_common_struct*)
{
    std::vector<int> perm;
    if (!mysolver::reordering::metis_nd(n, Ap, Ai, perm)) {
        return 0;  // KLU treats 0 as failure.
    }
    for (int i = 0; i < n; ++i) {
        Perm[i] = perm[i];
    }
    // Return a positive lnz estimate (KLU only checks != 0 for success).
    return (n > 0 ? Ap[n] + n : 1);
}
