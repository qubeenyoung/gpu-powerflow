#include "plan/analyze.hpp"

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "symbolic/multifrontal.hpp"
#include "symbolic/supernode.hpp"

namespace custom_linear_solver::plan {

namespace {

__device__ int front_lower_bound(const int* front_rows, int begin, int end, int value)
{
    int lo = begin;
    int hi = end;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (front_rows[mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return lo - begin;
}
__global__ void build_a_pos_device(int n, const int* __restrict__ Ap,
                                   const int* __restrict__ Ai,
                                   const int* __restrict__ panel_of,
                                   const int* __restrict__ panel_first,
                                   const int* __restrict__ panel_ncols,
                                   const int* __restrict__ front_off,
                                   const int* __restrict__ front_ptr,
                                   const int* __restrict__ front_rows,
                                   int* __restrict__ a_pos)
{
    const int j = blockIdx.x;
    if (j >= n) return;
    for (int q = Ap[j] + threadIdx.x; q < Ap[j + 1]; q += blockDim.x) {
        const int i = Ai[q];
        const int mn = i < j ? i : j;
        const int mx = i < j ? j : i;
        const int owner = panel_of[mn];
        const int first = panel_first[owner];
        const int ncols = panel_ncols[owner];
        const int fbegin = front_ptr[owner];
        const int fend = front_ptr[owner + 1];
        const int fsz = fend - fbegin;
        const int mn_local = mn - first;
        const int mx_local = (mx < first + ncols)
                                 ? (mx - first)
                                 : front_lower_bound(front_rows, fbegin, fend, mx);
        const int ri = (i < j) ? mn_local : mx_local;
        const int ci = (i < j) ? mx_local : mn_local;
        a_pos[q] = front_off[owner] + ri * fsz + ci;
    }
}

// Determine whether every numerical scatter destination in `a_pos` is unique. This is kept as
// analyze-time metadata/diagnostics; the active scatter path still uses atomicAdd. Optionally emits
// front fill / coverage statistics. Returns false on cudaMemcpy failure.
static bool determine_a_pos_unique(const int* d_a_pos, int nnz_a, long front_total,
                                   bool emit_info)
{
    std::vector<int> h_a_pos(static_cast<std::size_t>(nnz_a));
    if (cudaMemcpy(h_a_pos.data(), d_a_pos, h_a_pos.size() * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    std::sort(h_a_pos.begin(), h_a_pos.end());

    bool unique = true;
    for (std::size_t i = 1; i < h_a_pos.size(); ++i) {
        if (h_a_pos[i] >= 0 && h_a_pos[i] == h_a_pos[i - 1]) {
            unique = false;
            break;
        }
    }

    if (emit_info) {
        long present = 0, fill_runs = 0, fill_slots = 0, last = -1;
        for (int pos : h_a_pos) {
            if (pos < 0 || pos == last) continue;
            if (pos > last + 1) {
                ++fill_runs;
                fill_slots += pos - last - 1;
            }
            ++present;
            last = pos;
        }
        if (last + 1 < front_total) {
            ++fill_runs;
            fill_slots += front_total - last - 1;
        }
        std::fprintf(stderr,
                     "[analyze] front fill slots=%ld (%.1f%%) runs=%ld avg_run=%.1f\n",
                     fill_slots,
                     100.0 * (double)fill_slots / std::max(1.0, (double)front_total),
                     fill_runs,
                     fill_runs ? (double)fill_slots / (double)fill_runs : 0.0);
        std::fprintf(stderr,
                     "[analyze] front present slots=%ld (%.1f%% of front_total)\n",
                     present,
                     100.0 * (double)present / std::max(1.0, (double)front_total));
        std::fprintf(stderr, "[analyze] a_pos_unique=%s\n", unique ? "true" : "false");
    }

    return unique;
}

#if defined(CLS_TC_CLOSURE_PANEL_AMALGAMATE)
static bool valid_multifrontal_symbolic(const symbolic::MultifrontalSymbolic& mf)
{
    for (int p = 0; p < mf.num_panels; ++p) {
        const int par = mf.panel_parent[p];
        if (par >= 0 && par <= p) {
            return false;
        }
    }
    for (int idx : mf.asm_idx) {
        if (idx < 0) {
            return false;
        }
    }
    return true;
}
#endif

#if defined(CLS_TC_CLOSURE_PANEL_AMALGAMATE)
static std::vector<int> merged_panel_front_rows(const std::vector<int>& Lp,
                                                const std::vector<int>& Li, int first_col,
                                                int ncols)
{
    std::vector<int> rows;
    for (int j = first_col; j < first_col + ncols; ++j) {
        for (int q = Lp[j]; q < Lp[j + 1]; ++q) {
            rows.push_back(Li[q]);
        }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

static bool parent_front_contains_row(const symbolic::PanelPartition& base,
                                      const symbolic::MultifrontalSymbolic& base_mf,
                                      int parent_panel, int row)
{
    const int pfirst = base.first[parent_panel];
    const int plast = pfirst + base.ncols[parent_panel];
    if (row >= pfirst && row < plast) {
        return true;
    }
    const auto begin = base_mf.front_rows.begin() + base_mf.front_ptr[parent_panel];
    const auto end = base_mf.front_rows.begin() + base_mf.front_ptr[parent_panel + 1];
    return std::binary_search(begin, end, row);
}

static bool closure_group_valid(const symbolic::PanelPartition& base,
                                const symbolic::MultifrontalSymbolic& base_mf,
                                const std::vector<int>& Lp, const std::vector<int>& Li,
                                int start_panel, int end_panel)
{
    const int first_col = base.first[start_panel];
    const int ncols = base.first[end_panel] + base.ncols[end_panel] - first_col;
    const std::vector<int> rows = merged_panel_front_rows(Lp, Li, first_col, ncols);
    if (static_cast<int>(rows.size()) <= ncols) {
        return true;
    }

    const int parent_col = rows[ncols];
    const int parent_panel = base.panel_of[parent_col];
    if (parent_panel <= end_panel) {
        return false;
    }

    for (int k = ncols; k < static_cast<int>(rows.size()); ++k) {
        if (!parent_front_contains_row(base, base_mf, parent_panel, rows[k])) {
            return false;
        }
    }
    return true;
}
#endif





#ifdef CLS_TC_CLOSURE_PANEL_AMALGAMATE
static symbolic::PanelPartition tc_closure_amalgamate_panels(
    int n, const std::vector<int>& Lp, const std::vector<int>& Li,
    const symbolic::PanelPartition& base, int effective_panel_cap, bool emit_info)
{
    const int P = base.num_panels;
    if (P <= 1) {
        return base;
    }
    const symbolic::MultifrontalSymbolic base_mf =
        symbolic::multifrontal_symbolic(n, Lp, Li, base);
    if (!valid_multifrontal_symbolic(base_mf)) {
        if (emit_info) {
            std::fprintf(stderr, "[analyze] tc-closure-amalgamate: base symbolic invalid, skip\n");
        }
        return base;
    }

    int cap = CLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP;
    if (cap < 1) cap = 1;
    if (effective_panel_cap > 0 && cap > effective_panel_cap) cap = effective_panel_cap;

    int accepted_groups = 0;
    int accepted_extra_panels = 0;
    int closure_valid_candidates = 0;
    int tc_routable_candidates = 0;
    symbolic::PanelPartition out;
    out.panel_of.assign(n < 0 ? 0 : n, -1);
    for (int p = 0; p < P;) {
        int best_end = p;
        int best_ncols = base.ncols[p];
        int best_width = base.width[p];
        int candidate_ncols = base.ncols[p];
        int candidate_width = base.width[p];
        for (int end_panel = p + 1; end_panel < P; ++end_panel) {
            candidate_ncols += base.ncols[end_panel];
            if (candidate_ncols > cap) {
                break;
            }
            candidate_width = std::max(candidate_width, base.width[end_panel]);
            if (!closure_group_valid(base, base_mf, Lp, Li, p, end_panel)) {
                continue;
            }
            ++closure_valid_candidates;
            const int first_col = base.first[p];
            const int fsz = static_cast<int>(
                merged_panel_front_rows(Lp, Li, first_col, candidate_ncols).size());
            const int uc = fsz - candidate_ncols;
            if (fsz > 32 && uc >= 16 && candidate_ncols >= 4 && candidate_ncols <= 32) {
                ++tc_routable_candidates;
                best_end = end_panel;
                best_ncols = candidate_ncols;
                best_width = candidate_width;
            }
        }

        if (best_end > p) {
            ++accepted_groups;
            accepted_extra_panels += best_end - p;
        }
        const int id = out.num_panels++;
        const int first = base.first[p];
        out.first.push_back(first);
        out.ncols.push_back(best_ncols);
        out.width.push_back(best_width);
        out.padded_fill += static_cast<long>(best_ncols) * best_width;
        for (int c = first; c < first + best_ncols; ++c) {
            out.panel_of[c] = id;
        }
        p = best_end + 1;
    }

    const symbolic::MultifrontalSymbolic out_mf =
        symbolic::multifrontal_symbolic(n, Lp, Li, out);
    if (!valid_multifrontal_symbolic(out_mf)) {
        if (emit_info) {
            std::fprintf(stderr,
                         "[analyze] tc-closure-amalgamate: candidate invalid, fallback base\n");
        }
        return base;
    }
    if (emit_info) {
        std::fprintf(stderr,
                     "[analyze] tc-closure-amalgamate: panels %d -> %d, groups=%d extra=%d closure_candidates=%d tc_candidates=%d, padded_fill %.3fx\n",
                     P, out.num_panels, accepted_groups, accepted_extra_panels,
                     closure_valid_candidates, tc_routable_candidates,
                     base.padded_fill > 0
                         ? static_cast<double>(out.padded_fill) / base.padded_fill
                         : 1.0);
    }
    return out;
}
#endif  // CLS_TC_CLOSURE_PANEL_AMALGAMATE

// Adaptive panel cap by problem size. A bigger cap merges longer etree chains into one panel,
// cutting the front / solve-kernel count at the price of padded fill; larger matrices have the
// margin to amalgamate more aggressively. Upper bound 64 from the shared pivot buffer (nc <= cap).
static int compute_effective_panel_cap(int n, int panel_cap, bool float_front)
{
#ifdef CLS_RESPECT_PANEL_CAP
    int effective_panel_cap = panel_cap;
#else
    int effective_panel_cap = n >= 80000 ? 20 : (n >= 16000 ? 12 : panel_cap);
#endif
    // B=1 float-front factorization benefits from a slightly wider panel around the 3K-bus /
    // 6K-unknown class: it removes tiny-front overhead without the large-front padding that hurts
    // the 6K-8K bus cases.
    if (float_front && n >= 5000 && n < 8000 && effective_panel_cap < 18) effective_panel_cap = 18;
    if (effective_panel_cap < 1) effective_panel_cap = 1;
    if (effective_panel_cap > 64) effective_panel_cap = 64;
    return effective_panel_cap;
}

// Per-panel front offset (in doubles) into the front arena = prefix sum of fsz², in panel-id
// (postorder) order, which keeps each parent front adjacent to its children for extend-add.
// Returns false if the arena would exceed 1G doubles (8 GB).
static bool layout_front_arena(MultifrontalPlan& plan, const symbolic::MultifrontalSymbolic& symbolic_factor,
                               int P, std::vector<int>& front_off, long& total)
{
    front_off.assign(P + 1, 0);
    total = 0;
    for (int p = 0; p < P; ++p) {
        const long fsz = symbolic_factor.front_ptr[p + 1] - symbolic_factor.front_ptr[p];
        front_off[p] = static_cast<int>(total);
        total += fsz * fsz;
        if (total > (1L << 30)) return false;
    }
    front_off[P] = static_cast<int>(total);
    plan.front_total = total;
    plan.h_front_off = front_off;  // host mirror for per-panel cuBLAS dispatch
    return true;
}

// Per-panel pivot offsets (one slot per pivot column); total_pivots == n.
static void build_pivot_offsets(MultifrontalPlan& plan, const symbolic::PanelPartition& panels,
                                int P)
{
    plan.h_pivot_offset.assign(P + 1, 0);
    for (int p = 0; p < P; ++p)
        plan.h_pivot_offset[p + 1] = plan.h_pivot_offset[p] + panels.ncols[p];
    plan.total_pivots = plan.h_pivot_offset[P];
    if (cudaMalloc(&plan.d_pivot_offset, (size_t)(P + 1) * sizeof(int)) == cudaSuccess) {
        cudaMemcpy(plan.d_pivot_offset, plan.h_pivot_offset.data(),
                   (size_t)(P + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }
}

// Panel-etree levels (parent id > child in postorder, so one forward pass suffices). Fills plevel
// and plan.panel_level_ptr (the level CSR); returns the level count.
static int build_panel_levels(MultifrontalPlan& plan, const symbolic::MultifrontalSymbolic& symbolic_factor,
                              int P, std::vector<int>& plevel)
{
    plevel.assign(P, 0);
    for (int p = 0; p < P; ++p) {
        const int par = symbolic_factor.panel_parent[p];
        if (par != -1) plevel[par] = std::max(plevel[par], plevel[p] + 1);
    }
    int num_plevels = 0;
    for (int p = 0; p < P; ++p) num_plevels = std::max(num_plevels, plevel[p] + 1);
    plan.num_plevels = num_plevels;
    plan.panel_level_ptr.assign(num_plevels + 1, 0);
    for (int p = 0; p < P; ++p) ++plan.panel_level_ptr[plevel[p] + 1];
    for (int L = 0; L < num_plevels; ++L)
        plan.panel_level_ptr[L + 1] += plan.panel_level_ptr[L];
    return num_plevels;
}

// Front-size distribution + per-level structure (analyze-info dump, stderr).
static void dump_front_distribution(const symbolic::MultifrontalSymbolic& symbolic_factor,
                                    const std::vector<int>& plevel, int n, int P, int num_plevels,
                                    int effective_panel_cap, long total)
{
    const int NB = 7;
    const int edges[NB] = {16, 32, 48, 64, 96, 160, 1 << 30};
    long cnt[NB] = {0}, sf2[NB] = {0};
    double sf3[NB] = {0};
    long tot2 = 0; double tot3 = 0;
    for (int p = 0; p < P; ++p) {
        const long fsz = symbolic_factor.front_ptr[p + 1] - symbolic_factor.front_ptr[p];
        int bck = 0; while (fsz > edges[bck]) ++bck;
        cnt[bck]++; sf2[bck] += fsz * fsz; sf3[bck] += (double)fsz * fsz * fsz;
        tot2 += fsz * fsz; tot3 += (double)fsz * fsz * fsz;
    }
    fprintf(stderr, "[analyze] n=%d P=%d levels=%d cap=%d front_total(MB f32)=%.1f\n",
            n, P, num_plevels, effective_panel_cap, total * 4.0 / 1e6);
    const int lo[NB] = {1, 17, 33, 49, 65, 97, 161};
    for (int bk = 0; bk < NB; ++bk)
        fprintf(stderr, "  fsz[%4d..%-9d] cnt=%-8ld f2%%=%5.1f f3%%=%5.1f\n", lo[bk],
                edges[bk] == (1 << 30) ? 999999 : edges[bk], cnt[bk],
                100.0 * sf2[bk] / std::max(1L, tot2), 100.0 * sf3[bk] / std::max(1e-9, tot3));
    for (int L = 0; L < num_plevels && L < 40; ++L) {
        long c = 0, m = 0, s2 = 0;
        for (int p = 0; p < P; ++p)
            if (plevel[p] == L) {
                const long fsz = symbolic_factor.front_ptr[p + 1] - symbolic_factor.front_ptr[p];
                ++c; m = std::max(m, fsz); s2 += fsz * fsz;
            }
        fprintf(stderr, "  L%-2d cnt=%-7ld maxfsz=%-4ld f2=%ld\n", L, c, m, s2);
    }
}

// Spine + K-subtree partition. Spine = contiguous "cnt=1 chain" at the top of the panel etree.
// Below the spine, each panel's subtree root is the highest ancestor whose parent is in the spine
// (or -1); panels sharing a root form one subtree. Roots are capped at kMaxSubtreeStreams (largest
// kept, the rest merged into one spillover subtree), then plcols is re-sorted so each level groups
// panels subtree-major and tier-major within each subtree (spine last).
static void partition_subtrees(MultifrontalPlan& plan, const symbolic::MultifrontalSymbolic& symbolic_factor,
                               std::vector<int>& plcols, int P, int num_plevels, bool emit_info)
{
    auto level_cnt = [&](int L) { return plan.panel_level_ptr[L + 1] - plan.panel_level_ptr[L]; };

    // Spine: walk down from the top level while cnt == 1.
    int spine_top = num_plevels - 1;
    int spine_bot = spine_top;
    while (spine_bot >= 0 && level_cnt(spine_bot) == 1) --spine_bot;
    ++spine_bot;  // now spine_bot = lowest L with cnt=1 in the chain (start of spine)
    if (spine_bot <= spine_top) {
        plan.spine_start_level = spine_bot;
        // Collect spine panels in factor order (bottom -> top).
        plan.h_spine_panels.clear();
        for (int L = spine_bot; L <= spine_top; ++L) {
            for (int q = plan.panel_level_ptr[L]; q < plan.panel_level_ptr[L + 1]; ++q) {
                plan.h_spine_panels.push_back(plan.h_plcols[q]);
            }
        }
    } else {
        // No spine at all (root level has cnt > 1).
        plan.spine_start_level = -1;
        plan.h_spine_panels.clear();
    }

    // A panel p's subtree root is the highest ancestor whose panel_parent is in the spine (or -1).
    // All panels with the same root form one subtree. Parent id > child id holds for postorder
    // panel ids, so a reverse-id pass lets each child inherit its already-resolved parent's root.
    plan.h_subtree_of_panel.assign(P, -1);
    std::vector<char> is_spine(P, 0);
    if (plan.spine_start_level >= 0) {
        for (int sp : plan.h_spine_panels) is_spine[sp] = 1;
    }
    std::vector<int> subtree_root_of(P, -1);
    for (int p = P - 1; p >= 0; --p) {
        if (is_spine[p]) continue;
        const int par = symbolic_factor.panel_parent[p];
        if (par < 0 || is_spine[par]) {
            subtree_root_of[p] = p;
        } else {
            subtree_root_of[p] = subtree_root_of[par];
        }
    }
    // Collect distinct subtree roots with their member counts.
    std::vector<int> root_member_count(P, 0);
    for (int p = 0; p < P; ++p) {
        if (subtree_root_of[p] >= 0) ++root_member_count[subtree_root_of[p]];
    }
    std::vector<int> distinct_roots;
    for (int p = 0; p < P; ++p) {
        if (root_member_count[p] > 0) distinct_roots.push_back(p);
    }
    // Sort by member count desc.
    std::sort(distinct_roots.begin(), distinct_roots.end(),
              [&](int a, int b) { return root_member_count[a] > root_member_count[b]; });

    // Cap to kMaxSubtreeStreams. When more distinct roots exist (common on large power-grid
    // Jacobians), keep the top (K-1) largest as separate subtrees and merge ALL remaining roots'
    // members into one "spillover" subtree (id K-1), processed on its own stream level-by-level.
    constexpr int MAX_SUBTREES = kMaxSubtreeStreams;
    plan.h_subtree_roots.clear();
    std::vector<int> root_to_id(P, -1);
    const int kept = std::min((int)distinct_roots.size(), MAX_SUBTREES - 1);
    for (int i = 0; i < kept; ++i) {
        root_to_id[distinct_roots[i]] = i;
        plan.h_subtree_roots.push_back(distinct_roots[i]);
    }
    // If there are more roots, all "extra" roots map to id = kept (spillover bin).
    if ((int)distinct_roots.size() > kept) {
        for (int i = kept; i < (int)distinct_roots.size(); ++i) {
            root_to_id[distinct_roots[i]] = kept;
        }
        // The spillover bin needs a representative root for h_subtree_roots[kept].
        plan.h_subtree_roots.push_back(distinct_roots[kept]);
    }
    plan.num_subtrees = (int)plan.h_subtree_roots.size();
    // Assign subtree id per panel via its subtree_root_of.
    for (int p = 0; p < P; ++p) {
        const int r = subtree_root_of[p];
        if (r < 0) continue;
        plan.h_subtree_of_panel[p] = root_to_id[r];
    }

    // Re-sort plcols within each level so panels of the same subtree are contiguous (subtree 0
    // first, ..., then -1=spine). Same-level panels are independent.
    if (plan.num_subtrees > 0) {
        constexpr int NT = MultifrontalPlan::kNumTiers;
        auto tier_of = [&](int p) {
            const int fsz = symbolic_factor.front_ptr[p + 1] - symbolic_factor.front_ptr[p];
            return front_bucket(fsz);
        };
        plan.h_subtree_level_off.assign((long)plan.num_subtrees * num_plevels, 0);
        plan.h_subtree_level_cnt.assign((long)plan.num_subtrees * num_plevels, 0);
        plan.h_subtree_level_tier_off.assign(
            (long)plan.num_subtrees * num_plevels * (NT + 1), 0);
        // For each level: bucket panels by subtree (spine=-1 last); each subtree cell is then
        // ordered tier-major so multistream dispatch can tier-split it (see h_subtree_level_*).
        for (int L = 0; L < num_plevels; ++L) {
            const int lo = plan.panel_level_ptr[L], hi = plan.panel_level_ptr[L + 1];
            std::vector<int> bucketed;
            bucketed.reserve(hi - lo);
            for (int k = 0; k < plan.num_subtrees; ++k) {
                const int cell_start = (int)bucketed.size() + lo;
                plan.h_subtree_level_off[(long)k * num_plevels + L] = cell_start;
                const long tier_base = ((long)k * num_plevels + L) * (NT + 1);
                int cnt = 0;
                for (int t = 0; t < NT; ++t) {  // tier-major within the cell
                    plan.h_subtree_level_tier_off[tier_base + t] = (int)bucketed.size() + lo;
                    for (int q = lo; q < hi; ++q) {
                        const int p = plcols[q];
                        if (plan.h_subtree_of_panel[p] == k && tier_of(p) == t) {
                            bucketed.push_back(p);
                            ++cnt;
                        }
                    }
                }
                plan.h_subtree_level_tier_off[tier_base + NT] = (int)bucketed.size() + lo;
                plan.h_subtree_level_cnt[(long)k * num_plevels + L] = cnt;
            }
            // Then spine panels (subtree_of_panel == -1) at the end (cnt=1 chain; no split).
            for (int q = lo; q < hi; ++q) {
                const int p = plcols[q];
                if (plan.h_subtree_of_panel[p] == -1) bucketed.push_back(p);
            }
            for (size_t i = 0; i < bucketed.size(); ++i) plcols[lo + i] = bucketed[i];
        }
        plan.h_plcols = plcols;  // refresh with the subtree-grouped order
    }

    if (emit_info) {
        fprintf(stderr, "[analyze] P=%d levels=%d  spine=[L%d..L%d] (%zu panels)  K=%d\n",
                P, num_plevels, plan.spine_start_level, num_plevels - 1,
                plan.h_spine_panels.size(), plan.num_subtrees);
        for (int k = 0; k < plan.num_subtrees; ++k) {
            int subtree_size = 0;
            for (int p = 0; p < P; ++p) if (plan.h_subtree_of_panel[p] == k) ++subtree_size;
            fprintf(stderr, "  subtree %d: root panel %d, %d panels\n",
                    k, plan.h_subtree_roots[k], subtree_size);
            for (int L = 0; L < num_plevels && L < 12; ++L) {
                const int c = plan.h_subtree_level_cnt[(long)k * num_plevels + L];
                if (c > 0) fprintf(stderr, "    sub%d L%d cnt=%d\n", k, L, c);
            }
        }
    }
}

// Tier-homogeneous dispatch order. Within each level, group panels by kernel tier
// (classify_front_tier) so the single-stream factor walk launches one right-sized kernel per
// (level, tier) instead of promoting a whole mixed level to its largest front's tier.
static void build_tier_order(MultifrontalPlan& plan, const symbolic::MultifrontalSymbolic& symbolic_factor,
                             const std::vector<int>& plcols, int P, int num_plevels)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    auto tier_of = [&](int p) {
        const int fsz = symbolic_factor.front_ptr[p + 1] - symbolic_factor.front_ptr[p];
        return front_bucket(fsz);
    };
    plan.h_plcols_tier.assign(P, 0);
    plan.h_level_tier_off.assign((long)num_plevels * (NT + 1), 0);
    int w = 0;
    for (int L = 0; L < num_plevels; ++L) {
        const int lo = plan.panel_level_ptr[L], hi = plan.panel_level_ptr[L + 1];
        for (int t = 0; t < NT; ++t) {
            plan.h_level_tier_off[(long)L * (NT + 1) + t] = w;
            for (int q = lo; q < hi; ++q) {
                const int p = plcols[q];
                if (tier_of(p) == t) plan.h_plcols_tier[w++] = p;
            }
        }
        plan.h_level_tier_off[(long)L * (NT + 1) + NT] = w;  // == hi
    }
    if (cudaMalloc(&plan.d_plcols_tier, (size_t)std::max(1, P) * sizeof(int)) == cudaSuccess) {
        cudaMemcpy(plan.d_plcols_tier, plan.h_plcols_tier.data(),
                   (size_t)P * sizeof(int), cudaMemcpyHostToDevice);
    }
}

// Parent-local extend-add map: asm_idx is a global front_rows index; subtract the parent front's
// start so the kernel indexes into the parent's fsz x fsz. Returns the map; sets plan.asm_total.
static std::vector<int> build_assembly_map(MultifrontalPlan& plan,
                                           const symbolic::MultifrontalSymbolic& symbolic_factor, int P)
{
    std::vector<int> asm_local(symbolic_factor.asm_idx.size());
    for (int p = 0; p < P; ++p) {
        const int par = symbolic_factor.panel_parent[p];
        const int base = (par >= 0) ? symbolic_factor.front_ptr[par] : 0;
        for (int a = symbolic_factor.asm_ptr[p]; a < symbolic_factor.asm_ptr[p + 1]; ++a)
            asm_local[a] = symbolic_factor.asm_idx[a] - base;
    }
    plan.asm_total = static_cast<int>(asm_local.size());
    return asm_local;
}

// Lay out the single device arena (kArenaAlignmentBytes-aligned sub-arrays), allocate it, upload
// the host plan arrays, and build the assembly position map (a_pos) on device. Returns false if
// the a_pos kernel fails.
static bool allocate_and_upload_plan(MultifrontalPlan& plan,
                                     const symbolic::MultifrontalSymbolic& symbolic_factor,
                                     const symbolic::PanelPartition& panels,
                                     const std::vector<int>& front_off,
                                     const std::vector<int>& plcols,
                                     const std::vector<int>& asm_local, const int* d_Ap,
                                     const int* d_Ai, int n, int P, long total, bool float_front,
                                     bool emit_info)
{
    // One arena with kArenaAlignmentBytes-aligned sub-arrays (avoids an L2 cache-line straddle).
    auto align_up = [](long b) {
        return (b + kArenaAlignmentBytes - 1) & ~static_cast<long>(kArenaAlignmentBytes - 1);
    };
    long off = 0;
    const long o_front = off; off = align_up(off + total * sizeof(double));
    const long o_foff = off;  off = align_up(off + (long)(P + 1) * sizeof(int));
    const long o_fptr = off;  off = align_up(off + (long)(P + 1) * sizeof(int));
    const long o_nc = off;    off = align_up(off + (long)P * sizeof(int));
    const long o_plc = off;   off = align_up(off + (long)P * sizeof(int));
    const long o_par = off;   off = align_up(off + (long)P * sizeof(int));
    const long o_aptr = off;  off = align_up(off + (long)(P + 1) * sizeof(int));
    const long o_aloc = off;  off = align_up(off + (long)std::max(1, plan.asm_total) * sizeof(int));
    const long o_apos = off;  off = align_up(off + (long)std::max(1, plan.nnz) * sizeof(int));
    const int front_store = symbolic_factor.front_ptr[P];
    plan.front_store = front_store;
    const long o_fr = off;    off = align_up(off + (long)std::max(1, front_store) * sizeof(int));
    const long o_pf = off;    off = align_up(off + (long)std::max(1, P) * sizeof(int));
    const long o_pof = off;   off = align_up(off + (long)std::max(1, n) * sizeof(int));
    const long o_y = off;     off = align_up(off + (long)n * sizeof(double));
    const long o_sing = off;  off = align_up(off + sizeof(int));
    cudaMalloc(&plan.arena, off);
    char* base = static_cast<char*>(plan.arena);
    plan.d_front = reinterpret_cast<double*>(base + o_front);
    // When the factor/solve front is float (FP32 / FP16 / TF32), allocate the float front arena.
    if (float_front) cudaMalloc(&plan.d_front_f, (long)total * sizeof(float));
    plan.d_front_off = reinterpret_cast<int*>(base + o_foff);
    plan.d_front_ptr = reinterpret_cast<int*>(base + o_fptr);
    plan.d_ncols = reinterpret_cast<int*>(base + o_nc);
    plan.d_plcols = reinterpret_cast<int*>(base + o_plc);
    plan.d_panel_parent = reinterpret_cast<int*>(base + o_par);
    plan.d_asm_ptr = reinterpret_cast<int*>(base + o_aptr);
    plan.d_asm_local = reinterpret_cast<int*>(base + o_aloc);
    plan.d_a_pos = reinterpret_cast<int*>(base + o_apos);
    plan.d_front_rows = reinterpret_cast<int*>(base + o_fr);
    plan.d_y = reinterpret_cast<double*>(base + o_y);
    plan.d_sing = reinterpret_cast<int*>(base + o_sing);
    if (float_front) cudaMalloc(&plan.d_y_f, (long)n * sizeof(float));
    int* d_panel_first = reinterpret_cast<int*>(base + o_pf);
    int* d_panel_of = reinterpret_cast<int*>(base + o_pof);

    auto copy_to_device = [](int* d, const std::vector<int>& v) {
        if (!v.empty()) cudaMemcpy(d, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice);
    };
    copy_to_device(plan.d_front_off, front_off);
    copy_to_device(plan.d_front_ptr, symbolic_factor.front_ptr);
    copy_to_device(plan.d_ncols, panels.ncols);
    copy_to_device(plan.d_plcols, plcols);
    copy_to_device(plan.d_panel_parent, symbolic_factor.panel_parent);
    copy_to_device(plan.d_asm_ptr, symbolic_factor.asm_ptr);
    copy_to_device(plan.d_asm_local, asm_local);
    copy_to_device(plan.d_front_rows, symbolic_factor.front_rows);
    // Upload spine panel list. Separate allocation so it survives plan moves.
    if (!plan.h_spine_panels.empty()) {
        cudaMalloc(&plan.d_spine_panels, plan.h_spine_panels.size() * sizeof(int));
        cudaMemcpy(plan.d_spine_panels, plan.h_spine_panels.data(),
                   plan.h_spine_panels.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    copy_to_device(d_panel_first, panels.first);
    copy_to_device(d_panel_of, panels.panel_of);
    build_a_pos_device<<<n, 128>>>(n, d_Ap, d_Ai, d_panel_of, d_panel_first, plan.d_ncols,
                                   plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                                   plan.d_a_pos);
    const cudaError_t apos_err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (apos_err != cudaSuccess) return false;
    plan.a_pos_unique = determine_a_pos_unique(plan.d_a_pos, plan.nnz, total, emit_info);
    return true;
}

}  // namespace

MultifrontalPlan analyze_multifrontal(int n, int nnz_a, const int* d_Ap, const int* d_Ai,
                                      const std::vector<int>& Lp,
                                      const std::vector<int>& Li,
                                      const std::vector<int>& parent, int panel_cap,
                                      const custom_linear_solver::symbolic::PanelPartition* forced_panels,
                                      bool float_front, bool emit_info)
{
    namespace sym = custom_linear_solver::symbolic;
    MultifrontalPlan plan;
    plan.num_rows = n;
    plan.nnz = nnz_a;
    if (n <= 0) return plan;

    std::vector<int> colcount(n);
    for (int j = 0; j < n; ++j) colcount[j] = Lp[j + 1] - Lp[j];
    const int effective_panel_cap = compute_effective_panel_cap(n, panel_cap, float_front);
    sym::PanelPartition panels =
        forced_panels ? *forced_panels : sym::relaxed_panels(n, parent, colcount, effective_panel_cap);
#ifdef CLS_TC_CLOSURE_PANEL_AMALGAMATE
    if (!forced_panels) {
        panels = tc_closure_amalgamate_panels(n, Lp, Li, panels, effective_panel_cap, emit_info);
    }
#endif
    const sym::MultifrontalSymbolic symbolic_factor = sym::multifrontal_symbolic(n, Lp, Li, panels);
    const int P = panels.num_panels;
    plan.num_panels = P;
    plan.h_front_ptr = symbolic_factor.front_ptr;   // host copies for batched dispatch / TC shared sizing
    plan.h_ncols = panels.ncols;
    plan.h_panel_parent = symbolic_factor.panel_parent;
    plan.h_asm_ptr = symbolic_factor.asm_ptr;

    std::vector<int> front_off;
    long total = 0;
    if (!layout_front_arena(plan, symbolic_factor, P, front_off, total)) return MultifrontalPlan{};

    build_pivot_offsets(plan, panels, P);

    std::vector<int> plevel;
    const int num_plevels = build_panel_levels(plan, symbolic_factor, P, plevel);
    std::vector<int> plcols(P);

    if (emit_info) dump_front_distribution(symbolic_factor, plevel, n, P, num_plevels, effective_panel_cap, total);

    {
        std::vector<int> next(plan.panel_level_ptr.begin(), plan.panel_level_ptr.end());
        for (int p = 0; p < P; ++p) plcols[next[plevel[p]]++] = p;
    }
    plan.h_plcols = plcols;

    partition_subtrees(plan, symbolic_factor, plcols, P, num_plevels, emit_info);
    build_tier_order(plan, symbolic_factor, plcols, P, num_plevels);
    const std::vector<int> asm_local = build_assembly_map(plan, symbolic_factor, P);
    if (!allocate_and_upload_plan(plan, symbolic_factor, panels, front_off, plcols, asm_local, d_Ap, d_Ai,
                                  n, P, total, float_front, emit_info))
        return MultifrontalPlan{};


    return plan;
}

}  // namespace custom_linear_solver::plan
