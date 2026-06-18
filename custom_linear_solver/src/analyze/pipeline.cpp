#include <algorithm>
#include <cstdio>
#include <deque>
#include <exception>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "analyze/analyze.hpp"
#include "analyze/plan/lower.hpp"
#include "analyze/reorder/metis_nd.hpp"
#include "analyze/symbolic/elimination_tree.hpp"
#include "analyze/symbolic/supernode.hpp"

namespace custom_linear_solver::plan {

namespace {

// Run a parallel-for over [lo, hi) chunked across up to 12 host threads. Falls
// back to a single-thread call for small ranges or when the system reports zero
// hardware concurrency.
template <typename Fn>
void ParFor(int lo, int hi, Fn&& fn) {
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

// Relabel a symmetric CSC pattern (col_ptr / row_idx) under (perm, iperm) into
// a new CSC. new_col = iperm[old_col]; new_row = iperm[old_row]. Used to
// prepare Etree / FillPattern in the post-METIS ordering.
void PermuteSymmetricPattern(int n, const std::vector<int>& col_ptr,
                             const std::vector<int>& row_idx,
                             const std::vector<int>& perm,
                             const std::vector<int>& iperm,
                             std::vector<int>& out_col_ptr,
                             std::vector<int>& out_row_idx) {
  out_col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (int new_col = 0; new_col < n; ++new_col) {
    const int old_col = perm[new_col];
    out_col_ptr[new_col + 1] =
        out_col_ptr[new_col] + (col_ptr[old_col + 1] - col_ptr[old_col]);
  }
  out_row_idx.resize(static_cast<std::size_t>(out_col_ptr[n]));
  ParFor(0, n, [&](int lo, int hi) {
    for (int new_col = lo; new_col < hi; ++new_col) {
      const int old_col = perm[new_col];
      int w = out_col_ptr[new_col];
      for (int p = col_ptr[old_col]; p < col_ptr[old_col + 1]; ++p) {
        out_row_idx[w++] = iperm[row_idx[p]];
      }
    }
  });
}

bool StructuralMatchingFromCsc(int n, const std::vector<int>& col_ptr,
                               const std::vector<int>& row_idx,
                               std::vector<int>& row_to_col) {
  std::vector<int> degree(static_cast<std::size_t>(n), 0);
  for (int col = 0; col < n; ++col) {
    for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
      const int row = row_idx[p];
      if (0 <= row && row < n) ++degree[row];
    }
  }
  std::vector<int> row_ptr(static_cast<std::size_t>(n) + 1, 0);
  for (int r = 0; r < n; ++r) row_ptr[r + 1] = row_ptr[r] + degree[r];
  std::vector<int> adj(static_cast<std::size_t>(row_ptr[n]));
  std::vector<int> cursor = row_ptr;
  for (int col = 0; col < n; ++col) {
    for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
      const int row = row_idx[p];
      if (0 <= row && row < n) adj[cursor[row]++] = col;
    }
  }
  for (int r = 0; r < n; ++r)
    if (row_ptr[r + 1] == row_ptr[r]) return false;

  row_to_col.assign(static_cast<std::size_t>(n), -1);
  std::vector<int> col_to_row(static_cast<std::size_t>(n), -1);
  std::vector<int> dist(static_cast<std::size_t>(n), -1);

  auto bfs = [&]() {
    std::deque<int> q;
    bool found = false;
    for (int r = 0; r < n; ++r) {
      if (row_to_col[r] < 0) {
        dist[r] = 0;
        q.push_back(r);
      } else {
        dist[r] = -1;
      }
    }
    while (!q.empty()) {
      const int r = q.front();
      q.pop_front();
      for (int p = row_ptr[r]; p < row_ptr[r + 1]; ++p) {
        const int c = adj[p];
        const int rr = col_to_row[c];
        if (rr < 0) {
          found = true;
        } else if (dist[rr] < 0) {
          dist[rr] = dist[r] + 1;
          q.push_back(rr);
        }
      }
    }
    return found;
  };

  auto dfs = [&](auto&& self, int r) -> bool {
    for (int p = row_ptr[r]; p < row_ptr[r + 1]; ++p) {
      const int c = adj[p];
      const int rr = col_to_row[c];
      if (rr < 0 || (dist[rr] == dist[r] + 1 && self(self, rr))) {
        row_to_col[r] = c;
        col_to_row[c] = r;
        return true;
      }
    }
    dist[r] = -1;
    return false;
  };

  int matched = 0;
  while (bfs()) {
    for (int r = 0; r < n; ++r) {
      if (row_to_col[r] < 0 && dfs(dfs, r)) ++matched;
    }
  }
  return matched == n;
}

// Dump per-front structure to a CSV when SolverConfig.analyze_dump_fronts_path
// is non-empty. Used by offline front-distribution and parent-update analysis
// scripts.
void MaybeDumpFronts(const MultifrontalPlan& plan, const std::string& path) {
  if (path.empty()) return;
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "[Analyze] dump-fronts: failed to open %s\n",
                 path.c_str());
    return;
  }
  std::fprintf(f, "q,p,fsz,nc,uc,level,parent,asm_len,extend_elems\n");
  for (int L = 0; L < plan.num_plevels; ++L) {
    for (int q = plan.panel_level_ptr[L]; q < plan.panel_level_ptr[L + 1];
         ++q) {
      const int p = plan.h_plcols[q];
      const int fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
      const int nc = plan.h_ncols[p];
      const int uc = fsz - nc;
      const int parent =
          plan.h_panel_parent.empty() ? -1 : plan.h_panel_parent[p];
      const int asm_len = plan.h_asm_ptr.empty()
                              ? 0
                              : plan.h_asm_ptr[p + 1] - plan.h_asm_ptr[p];
      const long extend_elems = (parent >= 0) ? (long)uc * uc : 0;
      std::fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%d,%ld\n", q, p, fsz, nc, uc, L,
                   parent, asm_len, extend_elems);
    }
  }
  std::fclose(f);
  std::fprintf(stderr, "[Analyze] wrote %d front entries to %s\n",
               plan.num_panels, path.c_str());
}

}  // namespace

// Build the plan for one fixed ND seed (the original single-ordering pipeline).
static bool BuildPlanSeed(const CsrMatrixView& matrix,
                          const PlanBuildOptions& options, int metis_seed,
                          bool parallel_nd, PlanBuildResult& out) {
  try {
    const int n = static_cast<int>(matrix.nrows);
    const int nnz = static_cast<int>(matrix.nnz);
    const auto* d_csr_row_ptr = static_cast<const int*>(matrix.row_offsets);
    const auto* d_csr_col_idx = static_cast<const int*>(matrix.col_indices);

    // 1. CSR → CSC on device.
    matrix::DeviceCscPattern csc_device;
    if (matrix::BuildCscFromCsrDevice(n, nnz, d_csr_row_ptr, d_csr_col_idx,
                                      csc_device) != Status::kSuccess)
      return false;

    // 2. Symmetric adjacency graph (A + A^T) on device. Reused below for
    // permute_metis_graph.
    std::vector<int> metis_sym_col_ptr, metis_sym_row_idx;
    if (matrix::BuildSymmetricGraphDevice(csc_device, metis_sym_col_ptr,
                                          metis_sym_row_idx) !=
        Status::kSuccess)
      return false;

    std::vector<int> csc_col_ptr, csc_row_idx;
    std::vector<int> row_to_col;
    if (options.use_matching) {
      if (matrix::DownloadCscStructure(csc_device, csc_col_ptr, csc_row_idx) !=
          Status::kSuccess)
        return false;
      if (!StructuralMatchingFromCsc(n, csc_col_ptr, csc_row_idx, row_to_col)) {
        std::fprintf(stderr,
                     "[Analyze] structural matching requested but no perfect "
                     "matching exists\n");
        return false;
      }
    }

    // 3. METIS nested-dissection.
    std::vector<int> nd_perm(static_cast<std::size_t>(n), 0);
    {
      std::vector<int> nd_xadj =
          metis_sym_col_ptr;  // consumed (moved-from) by ND call
      std::vector<int> nd_adjncy = metis_sym_row_idx;
      if (!reordering::MetisNdFromGraph(n, nd_xadj, nd_adjncy, nd_perm,
                                        parallel_nd, metis_seed))
        return false;
    }
    out.perm = nd_perm;  // row order: new row -> original row
    out.iperm.assign(static_cast<std::size_t>(n),
                     0);  // column inverse: original col -> new col
    if (options.use_matching) {
      std::vector<int> col_perm(static_cast<std::size_t>(n), 0);
      for (int k = 0; k < n; ++k) col_perm[k] = row_to_col[out.perm[k]];
      for (int k = 0; k < n; ++k) out.iperm[col_perm[k]] = k;
    } else {
      for (int k = 0; k < n; ++k) out.iperm[out.perm[k]] = k;
    }
    std::vector<int> row_iperm(static_cast<std::size_t>(n), 0);
    for (int k = 0; k < n; ++k) row_iperm[out.perm[k]] = k;
    if (out.d_perm.upload(out.perm) != Status::kSuccess) return false;
    if (out.d_iperm.upload(out.iperm) != Status::kSuccess) return false;

    // 4. Apply permutation to CSC; capture ordered_value_to_csr mapping.
    matrix::IntDeviceBuffer d_row_iperm;
    if (options.use_matching &&
        d_row_iperm.upload(row_iperm) != Status::kSuccess)
      return false;
    matrix::DeviceCscPattern ordered_device;
    if (options.use_matching) {
      if (matrix::PermuteCscDeviceRc(csc_device, d_row_iperm.ptr,
                                     out.d_iperm.ptr,
                                     ordered_device) != Status::kSuccess)
        return false;
    } else {
      if (matrix::PermuteCscDevice(csc_device, out.d_iperm.ptr,
                                   ordered_device) != Status::kSuccess)
        return false;
    }
    out.d_ordered_value_to_csr = std::move(ordered_device.source_pos);

    // 5. Relabel the symmetric adjacency under the permutation for Etree /
    // FillPattern.
    std::vector<int> sym_col_ptr, sym_row_idx;
    if (options.use_matching) {
      if (matrix::BuildSymmetricGraphDevice(ordered_device, sym_col_ptr,
                                            sym_row_idx) != Status::kSuccess)
        return false;
    } else {
      PermuteSymmetricPattern(n, metis_sym_col_ptr, metis_sym_row_idx, out.perm,
                              out.iperm, sym_col_ptr, sym_row_idx);
    }

    // 6. Elimination tree.
    std::vector<int> parent =
        symbolic::Etree(n, sym_col_ptr.data(), sym_row_idx.data());

    // 7. Fill pattern. The METIS ordering is a Postorder → fill-neutral, so we
    // can compute fill in METIS order and relabel below without a second
    // FillPattern pass.
    std::vector<int> Lp, Li;
    symbolic::FillPattern(n, sym_col_ptr.data(), sym_row_idx.data(), parent, Lp,
                          Li);
    {
      long h = 0;
      std::vector<int> dep(n, 0);
      for (int v = 0; v < n; ++v) {
        int p = parent[v], d = 0;
        while (p >= 0 && d < n) {
          d++;
          p = parent[p];
        }
        if (d > h) h = d;
      }
      std::fprintf(stderr, "SYMB fill_nnz=%zu etree_height=%ld\n", Li.size(),
                   h);
    }

    // 8. Build the multifrontal plan.
    out.plan = AnalyzeMultifrontal(
        n, nnz, ordered_device.col_ptr.ptr, ordered_device.row_idx.ptr, Lp, Li,
        parent, options.max_panel_width, /*forced_panels=*/nullptr,
        options.float_front, options.emit_analyze_info);
    if (out.plan.num_panels == 0) return false;

    MaybeDumpFronts(out.plan, options.dump_fronts_csv_path);
    return true;
  } catch (const std::bad_alloc&) {
    return false;
  } catch (const std::exception&) {
    return false;
  }
}

bool BuildPlanFromCsr(const CsrMatrixView& matrix,
                      const PlanBuildOptions& options, PlanBuildResult& out) {
  return BuildPlanSeed(matrix, options, options.metis_seed,
                       options.use_parallel_nested_dissection, out);
}

}  // namespace custom_linear_solver::plan
