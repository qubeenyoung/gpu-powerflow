#include "analyze/reorder/metis_nd.hpp"

#include <metis.h>

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <vector>

namespace custom_linear_solver::reordering {

namespace {

// Subgraph size at/above which the top-level multi-core Induce() is used.
constexpr int kParNdTopInduceThreshold = 49152;

// Problem size at which the parallel-ND base-case threshold switches from the
// "small" to the "large" tuning value: graphs >= this recurse deeper before
// dropping to serial METIS_NodeND.
constexpr int kParNdSizeSplit = 20000;

int ParallelNdBaseThreshold(int n, int small_thr, int large_thr) {
  return n < kParNdSizeSplit ? small_thr : large_thr;
}

// Serial induced-subgraph extraction: build the sub-CSR (sx/sa) over the
// vertices with part[v]==which, in ascending vertex order, and the local->global
// vertex map. Edges to other partitions are dropped.
void Induce(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
            const std::vector<idx_t>& part, idx_t which, std::vector<idx_t>& sx,
            std::vector<idx_t>& sa, std::vector<int>& map) {
  // 1. Assign local indices to kept vertices.
  const int n = static_cast<int>(xadj.size()) - 1;
  std::vector<int> g2l(n, -1);
  map.clear();
  for (int v = 0; v < n; ++v)
    if (part[v] == which) {
      g2l[v] = static_cast<int>(map.size());
      map.push_back(v);
    }

  // 2. Count intra-partition edges per local vertex and prefix-sum into sx.
  const int sn = static_cast<int>(map.size());
  sx.assign(sn + 1, 0);
  for (int li = 0; li < sn; ++li) {
    const int v = map[li];
    for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
      if (part[adj[p]] == which) sx[li + 1]++;
  }
  for (int i = 0; i < sn; ++i) sx[i + 1] += sx[i];

  // 3. Scatter relabeled intra-partition neighbors into sa.
  sa.resize(sx[sn]);
  std::vector<idx_t> pos(sx.begin(), sx.end());
  for (int li = 0; li < sn; ++li) {
    const int v = map[li];
    for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
      if (part[adj[p]] == which) sa[pos[li]++] = g2l[adj[p]];
  }
}

// Parallel induced-subgraph extraction: same output as Induce(), multi-core.
void InducePar(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
               const std::vector<idx_t>& part, idx_t which,
               std::vector<idx_t>& sx, std::vector<idx_t>& sa,
               std::vector<int>& map);

// Recursion terminal: serial METIS_NodeND on the (sub)graph. Falls back to
// natural order for trivial / edgeless graphs or a METIS failure.
void BaseNodend(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
                std::vector<int>& perm, int seed) {
  // 1. Handle trivial and edgeless graphs.
  const int n = static_cast<int>(xadj.size()) - 1;
  perm.assign(n, 0);
  if (n <= 1) {
    if (n == 1) perm[0] = 0;
    return;
  }
  if (adj.empty()) {
    for (int i = 0; i < n; ++i) perm[i] = i;
    return;
  }

  // 2. Run METIS_NodeND. The graph is read-only and not reused, so pass it
  // directly (const_cast only for the C API signature).
  idx_t nv = n;
  std::vector<idx_t> p(n), ip(n);
  idx_t opt[METIS_NOPTIONS];
  METIS_SetDefaultOptions(opt);
  opt[METIS_OPTION_NUMBERING] = 0;
  opt[METIS_OPTION_SEED] = seed;  // fixed seed -> deterministic across threads
  // Reseed per call so each subgraph starts from a fixed RNG state
  // (deterministic with the LD_PRELOAD thread-safe rand; no-op otherwise).
  std::srand(seed);
  if (METIS_NodeND(&nv, const_cast<idx_t*>(xadj.data()),
                   const_cast<idx_t*>(adj.data()), nullptr, opt, p.data(),
                   ip.data()) != METIS_OK) {
    for (int i = 0; i < n; ++i) perm[i] = i;
    return;
  }
  for (int i = 0; i < n; ++i) perm[i] = static_cast<int>(p[i]);
}

// Parallel nested dissection. ND is recursively parallel: a vertex separator
// splits the graph into two independent halves that recurse on separate threads;
// the separator is ordered last. Same ND algorithm as serial METIS, so fill is
// comparable, but it uses all cores. Drops to serial BaseNodend at the base
// case. perm convention: perm[new_pos] = old_vertex.
void ParNdRec(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
              std::vector<int>& perm, int depth, int base_thr, int seed,
              bool is_root = true) {
  // 1. Base case: depth/size limit or edgeless graph -> serial METIS.
  const int n = static_cast<int>(xadj.size()) - 1;
  if (depth <= 0 || n < base_thr || adj.empty()) {
    BaseNodend(xadj, adj, perm, seed);
    return;
  }

  // 2. Compute a vertex separator. The graph is read-only (METIS copies it
  // internally and Induce() reads the original), so pass it without a copy.
  idx_t nv = n, sepsize = 0;
  std::vector<idx_t> part(n);
  idx_t* mx = const_cast<idx_t*>(xadj.data());
  idx_t* ma = const_cast<idx_t*>(adj.data());
  idx_t opt[METIS_NOPTIONS];
  METIS_SetDefaultOptions(opt);
  opt[METIS_OPTION_NUMBERING] = 0;
  opt[METIS_OPTION_SEED] = seed;  // fixed seed -> deterministic across threads
  std::srand(seed);               // reseed per call (see BaseNodend)
  if (METIS_ComputeVertexSeparator(&nv, mx, ma, nullptr, opt, &sepsize,
                                   part.data()) != METIS_OK) {
    BaseNodend(xadj, adj, perm, seed);
    return;
  }

  // 3. Induce the two halves. Use multi-core Induce on top levels (large
  // subgraphs, few active threads) where serial Induce is on the critical path;
  // deeper/smaller subgraphs keep serial Induce to bound oversubscription.
  std::vector<idx_t> x0, a0, x1, a1;
  std::vector<int> m0, m1;
  if (is_root || n >= kParNdTopInduceThreshold) {
    std::thread ti([&] { InducePar(xadj, adj, part, 0, x0, a0, m0); });
    InducePar(xadj, adj, part, 1, x1, a1, m1);
    ti.join();
  } else {
    Induce(xadj, adj, part, 0, x0, a0, m0);
    Induce(xadj, adj, part, 1, x1, a1, m1);
  }
  // 4. Degenerate split (one side empty) -> fall back to serial METIS.
  const int n0 = static_cast<int>(m0.size()), n1 = static_cast<int>(m1.size());
  if (n0 == 0 || n1 == 0) {
    BaseNodend(xadj, adj, perm, seed);
    return;
  }

  // 5. Recurse on the two halves concurrently.
  std::vector<int> p0, p1;
  std::thread th(
      [&] { ParNdRec(x0, a0, p0, depth - 1, base_thr, seed, false); });
  ParNdRec(x1, a1, p1, depth - 1, base_thr, seed, false);
  th.join();

  // 6. Stitch: part 0 in [0,n0), part 1 in [n0,n0+n1), separator last. Each
  // sub-perm maps sub-newpos -> sub-local, then m* maps local -> global vertex.
  perm.assign(n, 0);
  for (int np = 0; np < n0; ++np) perm[np] = m0[p0[np]];
  for (int np = 0; np < n1; ++np) perm[n0 + np] = m1[p1[np]];
  int j = 0;
  for (int v = 0; v < n; ++v)
    if (part[v] == 2) perm[n0 + n1 + (j++)] = v;
}

// Run a parallel-for over [lo, hi) chunked across up to 12 host threads. Falls
// back to a single-thread call for small ranges or unknown concurrency.
template <typename Fn>
void ParFor(int lo, int hi, Fn&& fn) {
  // 1. Pick thread count; small ranges run serially.
  unsigned hw = std::thread::hardware_concurrency();
  const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
  if (hi - lo < 32768 || nth <= 1) {
    fn(lo, hi);
    return;
  }

  // 2. Split the range into chunks and join.
  std::vector<std::thread> th;
  const int chunk = (hi - lo + nth - 1) / nth;
  for (int t = 0; t < nth; ++t) {
    const int a = lo + t * chunk;
    const int b = std::min(hi, a + chunk);
    if (a < b) th.emplace_back([&fn, a, b] { fn(a, b); });
  }
  for (auto& x : th) x.join();
}

void InducePar(const std::vector<idx_t>& xadj, const std::vector<idx_t>& adj,
               const std::vector<idx_t>& part, idx_t which,
               std::vector<idx_t>& sx, std::vector<idx_t>& sa,
               std::vector<int>& map) {
  // 1. Serial local relabel (ascending vertex order); output is byte-identical
  // to Induce().
  const int n = static_cast<int>(xadj.size()) - 1;
  std::vector<int> g2l(n, -1);
  map.clear();
  for (int v = 0; v < n; ++v)
    if (part[v] == which) {
      g2l[v] = static_cast<int>(map.size());
      map.push_back(v);
    }

  // 2. Count intra-partition edges per local vertex (parallel) and prefix-sum.
  const int sn = static_cast<int>(map.size());
  sx.assign(sn + 1, 0);
  ParFor(0, sn, [&](int lo, int hi) {
    for (int li = lo; li < hi; ++li) {
      const int v = map[li];
      int cnt = 0;
      for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
        if (part[adj[p]] == which) ++cnt;
      sx[li + 1] = cnt;
    }
  });
  for (int i = 0; i < sn; ++i) sx[i + 1] += sx[i];

  // 3. Scatter relabeled intra-partition neighbors into sa (parallel).
  sa.resize(static_cast<std::size_t>(sx[sn]));
  ParFor(0, sn, [&](int lo, int hi) {
    for (int li = lo; li < hi; ++li) {
      const int v = map[li];
      idx_t w = sx[li];
      for (idx_t p = xadj[v]; p < xadj[v + 1]; ++p)
        if (part[adj[p]] == which) sa[w++] = g2l[adj[p]];
    }
  });
}

// Run the ND ordering (parallel nested dissection or serial METIS NodeND) on an
// already-built symmetric idx_t graph. Fills perm in METIS convention
// (perm[new_pos] = old_vertex).
bool RunNdOnGraph(int n, std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy,
                  std::vector<int>& perm, bool parallel, int seed,
                  int par_nd_depth, int par_nd_base_small,
                  int par_nd_base_large) {
  // 1. No edges: natural order is optimal.
  if (adjncy.empty()) {
    for (int i = 0; i < n; ++i) perm[i] = i;
    return true;
  }

  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_SEED] = seed;

  // 2. Parallel path: pick the base-case threshold by size and recurse.
  if (parallel) {
    const int base_thr =
        ParallelNdBaseThreshold(n, par_nd_base_small, par_nd_base_large);
    std::vector<int> pp;
    ParNdRec(xadj, adjncy, pp, par_nd_depth, base_thr, seed);
    for (int i = 0; i < n; ++i) perm[i] = pp[i];
    return true;
  }

  // 3. Serial path: single-threaded METIS_NodeND; natural order on failure.
  idx_t nvtxs = n;
  std::vector<idx_t> mperm(n);
  std::vector<idx_t> miperm(n);
  const int rc = METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), nullptr,
                              options, mperm.data(), miperm.data());
  if (rc != METIS_OK) {
    for (int i = 0; i < n; ++i) perm[i] = i;
    return true;
  }
  for (int i = 0; i < n; ++i) perm[i] = static_cast<int>(mperm[i]);
  return true;
}

}  // namespace

bool MetisNdFromGraph(int n, std::vector<int>& xadj_in,
                      std::vector<int>& adjncy_in, std::vector<int>& perm,
                      bool parallel, int seed, int par_nd_depth,
                      int par_nd_base_small, int par_nd_base_large) {
  // 1. Validate and handle trivial sizes.
  if (n < 0) return false;
  perm.assign(static_cast<std::size_t>(n), 0);
  if (n <= 1) {
    if (n == 1) perm[0] = 0;
    return true;
  }

  // 2. Adopt the prebuilt symmetric graph as idx_t: move (no copy) when
  // idx_t == int, otherwise widen element-wise.
  std::vector<idx_t> xadj, adjncy;
  if constexpr (std::is_same_v<idx_t, int>) {
    xadj = std::move(xadj_in);
    adjncy = std::move(adjncy_in);
  } else {
    xadj.assign(xadj_in.begin(), xadj_in.end());
    adjncy.assign(adjncy_in.begin(), adjncy_in.end());
  }

  // 3. Run the ordering.
  return RunNdOnGraph(n, xadj, adjncy, perm, parallel, seed, par_nd_depth,
                      par_nd_base_small, par_nd_base_large);
}

}  // namespace custom_linear_solver::reordering
