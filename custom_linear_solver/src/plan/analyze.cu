#include "plan/analyze.hpp"

#include <algorithm>
#include <cstdio>
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

}  // namespace

MultifrontalPlan analyze_multifrontal(int n, int nnz_a, const int* d_Ap, const int* d_Ai,
                                      const std::vector<int>& Lp,
                                      const std::vector<int>& Li,
                                      const std::vector<int>& parent, int panel_cap,
                                      const custom_linear_solver::symbolic::PanelPartition* forced_panels,
                                      bool float_front, bool emit_info)
{
    namespace sym = custom_linear_solver::symbolic;
    auto lap = [](const char*) {};
    MultifrontalPlan plan;
    plan.n = n;
    plan.nnz_a = nnz_a;
    if (n <= 0) return plan;

    std::vector<int> colcount(n);
    for (int j = 0; j < n; ++j) colcount[j] = Lp[j + 1] - Lp[j];
    // Adaptive panel cap by problem size. A bigger cap merges longer etree chains into a single
    // panel, reducing the number of fronts and solve-level kernels (the dominant solve cost) at
    // the price of padded fill. Larger matrices have the margin to amalgamate more aggressively.
    //   n >= 80000 → cap 20
    //   n >= 16000 → cap 12
    //   else       → caller-supplied panel_cap (SolverConfig.panel_cap, default 8)
    // Upper bound 64 from the shared pivot buffer (nc <= cap).
    int eff_cap = n >= 80000 ? 20 : (n >= 16000 ? 12 : panel_cap);
    if (eff_cap < 1) eff_cap = 1;
    if (eff_cap > 64) eff_cap = 64;
    const sym::PanelPartition panels =
        forced_panels ? *forced_panels : sym::relaxed_panels(n, parent, colcount, eff_cap);
    lap("relaxed_panels");
    const sym::MultifrontalSymbolic mf = sym::multifrontal_symbolic(n, Lp, Li, panels);
    lap("multifrontal_symbolic");
    const int P = panels.num_panels;
    plan.num_panels = P;
    plan.h_front_ptr = mf.front_ptr;   // host copies for batched dispatch / TC shared sizing
    plan.h_ncols = panels.ncols;
    // h_plcols filled after plcols is built below.

    // Per-panel front offset (in doubles) into the front arena = prefix sum of fsz².
    std::vector<int> front_off(P + 1, 0);
    long total = 0;
    for (int p = 0; p < P; ++p) {
        const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
        front_off[p] = static_cast<int>(total);
        total += fsz * fsz;
        if (total > (1L << 30)) {  // > 1G doubles (8GB) -> bail out; analyze fails for huge fronts
            return MultifrontalPlan{};
        }
    }
    front_off[P] = static_cast<int>(total);
    plan.front_total = total;
    plan.h_front_off = front_off;  // host mirror for per-panel cuBLAS dispatch (host mirror of front offsets)

    // Per-panel pivot offsets (one slot per pivot column). The within-pivot pivoting code path is
    plan.h_pivot_offset.assign(P + 1, 0);
    for (int p = 0; p < P; ++p) plan.h_pivot_offset[p + 1] = plan.h_pivot_offset[p] + panels.ncols[p];
    plan.total_pivots = plan.h_pivot_offset[P];  // = n
    if (cudaMalloc(&plan.d_pivot_offset, (size_t)(P + 1) * sizeof(int)) == cudaSuccess) {
        cudaMemcpy(plan.d_pivot_offset, plan.h_pivot_offset.data(),
                   (size_t)(P + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Panel-etree levels (parent id > child in postorder -> single forward pass).
    std::vector<int> plevel(P, 0);
    int num_plevels = 0;
    for (int p = 0; p < P; ++p) {
        const int par = mf.panel_parent[p];
        if (par != -1) plevel[par] = std::max(plevel[par], plevel[p] + 1);
    }
    for (int p = 0; p < P; ++p) num_plevels = std::max(num_plevels, plevel[p] + 1);
    plan.num_plevels = num_plevels;
    plan.plptr.assign(num_plevels + 1, 0);
    for (int p = 0; p < P; ++p) ++plan.plptr[plevel[p] + 1];
    for (int L = 0; L < num_plevels; ++L) plan.plptr[L + 1] += plan.plptr[L];
    std::vector<int> plcols(P);

    if (emit_info) {  // front-size distribution + per-level structure (analyze-info dump)
        const int NB = 7;
        const int edges[NB] = {16, 32, 48, 64, 96, 160, 1 << 30};
        long cnt[NB] = {0}, sf2[NB] = {0};
        double sf3[NB] = {0};
        long tot2 = 0; double tot3 = 0;
        for (int p = 0; p < P; ++p) {
            const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
            int bck = 0; while (fsz > edges[bck]) ++bck;
            cnt[bck]++; sf2[bck] += fsz * fsz; sf3[bck] += (double)fsz * fsz * fsz;
            tot2 += fsz * fsz; tot3 += (double)fsz * fsz * fsz;
        }
        fprintf(stderr, "[analyze] n=%d P=%d levels=%d cap=%d front_total(MB f32)=%.1f\n",
                n, P, num_plevels, eff_cap, total * 4.0 / 1e6);
        const int lo[NB] = {1, 17, 33, 49, 65, 97, 161};
        for (int bk = 0; bk < NB; ++bk)
            fprintf(stderr, "  fsz[%4d..%-9d] cnt=%-8ld f2%%=%5.1f f3%%=%5.1f\n", lo[bk],
                    edges[bk] == (1 << 30) ? 999999 : edges[bk], cnt[bk],
                    100.0 * sf2[bk] / std::max(1L, tot2), 100.0 * sf3[bk] / std::max(1e-9, tot3));
        for (int L = 0; L < num_plevels && L < 40; ++L) {
            long c = 0, m = 0, s2 = 0;
            for (int p = 0; p < P; ++p)
                if (plevel[p] == L) {
                    const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
                    ++c; m = std::max(m, fsz); s2 += fsz * fsz;
                }
            fprintf(stderr, "  L%-2d cnt=%-7ld maxfsz=%-4ld f2=%ld\n", L, c, m, s2);
        }
    }
    {
        std::vector<int> next(plan.plptr.begin(), plan.plptr.end());
        for (int p = 0; p < P; ++p) plcols[next[plevel[p]]++] = p;
    }
    plan.h_plcols = plcols;

    // --- Spine + K-subtree partition (partition computation) ---
    // Spine = contiguous "cnt=1 chain" at the top of the panel etree.
    // Below the spine: K subtree roots = panels at the level just below the spine bottom.
    // Each subtree is the closure of one of those roots (all descendants by panel_parent).
    {
        // Compute cnt-per-level (we already have plptr which gives cnt = plptr[L+1] - plptr[L]).
        auto level_cnt = [&](int L) { return plan.plptr[L + 1] - plan.plptr[L]; };

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
                for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
                    plan.h_spine_panels.push_back(plan.h_plcols[q]);
                }
            }
        } else {
            // No spine at all (root level has cnt > 1).
            plan.spine_start_level = -1;
            plan.h_spine_panels.clear();
        }

        // Subtree partition. The previous version assumed every below-
        // spine panel is reachable from some panel at level (spine_start_level - 1). But the
        // panel etree can have a panel at L<spine_start_level whose parent is in the spine
        // *directly* (skipping the "expected root level"). Those panels were stranded,
        // causing multi-stream dispatch to lose them and producing relres ~ 1e+28 garbage.
        //
        // Correct definition: a panel p's subtree root is the highest ancestor whose
        // panel_parent is in the spine (or is -1, i.e. p is its own root). All panels with
        // the same subtree-root form one subtree. Subtree roots = the set of such anchors.
        plan.h_subtree_of_panel.assign(P, -1);
        // Mark spine panels (those with subtree_of remaining -1 == spine).
        std::vector<char> is_spine(P, 0);
        if (plan.spine_start_level >= 0) {
            for (int sp : plan.h_spine_panels) is_spine[sp] = 1;
        }
        // For each non-spine panel, find its subtree root: walk up panel_parent until we
        // hit a panel whose parent is in spine (or -1). Topological order (parent id > child id)
        // means we can dynamic-program in increasing id with one pass.
        std::vector<int> subtree_root_of(P, -1);
        for (int p = 0; p < P; ++p) {
            if (is_spine[p]) continue;
            const int par = mf.panel_parent[p];
            if (par < 0 || is_spine[par]) {
                subtree_root_of[p] = p;  // p itself is the subtree root
            } else {
                subtree_root_of[p] = subtree_root_of[par];  // inherit (parent already processed)
            }
        }
        // (Note: parent id > child id holds for postorder panel ids, so subtree_root_of[par]
        // is set before we read it... wait, parent id > child id means par > p, so when we
        // process p we haven't processed par yet! Fix: iterate in REVERSE id order. Parent
        // comes before child in REVERSE, so subtree_root_of[par] is set first.)
        std::fill(subtree_root_of.begin(), subtree_root_of.end(), -1);
        for (int p = P - 1; p >= 0; --p) {
            if (is_spine[p]) continue;
            const int par = mf.panel_parent[p];
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

        // Cap to MAX_SUBTREES (matches TCState.subtree_streams[8]). When more distinct
        // roots exist (common on large power-grid Jacobians, where the etree fans out into
        // many small subtrees + a few large ones), keep the top (K_cap-1) largest as separate
        // subtrees and merge ALL remaining roots' members into one "spillover" subtree
        // (id K_cap-1). The spillover subtree is processed on its own stream level-by-level;
        // correctness holds because within one stream the level dispatch order respects
        // panel-etree dependencies.
        constexpr int MAX_SUBTREES = 8;
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
            // The spillover bin needs a representative root for h_subtree_roots[kept]; use
            // the first "extra" root (member count smaller but valid panel id).
            plan.h_subtree_roots.push_back(distinct_roots[kept]);
        }
        plan.num_subtrees = (int)plan.h_subtree_roots.size();
        // Assign subtree id per panel via its subtree_root_of.
        for (int p = 0; p < P; ++p) {
            const int r = subtree_root_of[p];
            if (r < 0) continue;
            plan.h_subtree_of_panel[p] = root_to_id[r];
        }

        // Re-sort plcols within each level so panels of the same subtree are
        // contiguous (subtree 0 first, then subtree 1, ..., then -1=spine). After this,
        // for subtree k at level L the range is [off, off+cnt) inside plcols.
        if (plan.num_subtrees > 0) {
            plan.h_subtree_level_off.assign((long)plan.num_subtrees * num_plevels, 0);
            plan.h_subtree_level_cnt.assign((long)plan.num_subtrees * num_plevels, 0);
            // For each level: bucket-sort panels by subtree_of_panel (spine=-1 goes last)
            for (int L = 0; L < num_plevels; ++L) {
                const int lo = plan.plptr[L], hi = plan.plptr[L + 1];
                std::vector<int> bucketed;
                bucketed.reserve(hi - lo);
                // First, subtree 0..K-1
                for (int k = 0; k < plan.num_subtrees; ++k) {
                    plan.h_subtree_level_off[(long)k * num_plevels + L] = (int)bucketed.size() + lo;
                    int cnt = 0;
                    for (int q = lo; q < hi; ++q) {
                        const int p = plcols[q];
                        if (plan.h_subtree_of_panel[p] == k) {
                            bucketed.push_back(p);
                            ++cnt;
                        }
                    }
                    plan.h_subtree_level_cnt[(long)k * num_plevels + L] = cnt;
                }
                // Then spine panels (subtree_of_panel == -1) at the end
                for (int q = lo; q < hi; ++q) {
                    const int p = plcols[q];
                    if (plan.h_subtree_of_panel[p] == -1) bucketed.push_back(p);
                }
                // Write back to plcols
                for (size_t i = 0; i < bucketed.size(); ++i) plcols[lo + i] = bucketed[i];
            }
            // h_plcols was already assigned from plcols above; refresh.
            plan.h_plcols = plcols;
        }

        if (emit_info) {
            fprintf(stderr, "[analyze] P=%d levels=%d  spine=[L%d..L%d] (%zu panels)  K=%d\n",
                    P, num_plevels, plan.spine_start_level, num_plevels - 1,
                    plan.h_spine_panels.size(), plan.num_subtrees);
            for (int k = 0; k < plan.num_subtrees; ++k) {
                int sz = 0;
                for (int p = 0; p < P; ++p) if (plan.h_subtree_of_panel[p] == k) ++sz;
                fprintf(stderr, "  subtree %d: root panel %d, %d panels\n",
                        k, plan.h_subtree_roots[k], sz);
                for (int L = 0; L < num_plevels && L < 12; ++L) {
                    const int c = plan.h_subtree_level_cnt[(long)k * num_plevels + L];
                    if (c > 0) fprintf(stderr, "    sub%d L%d cnt=%d\n", k, L, c);
                }
            }
        }
    }

    // Parent-local extend-add map: asm_idx is a global front_rows index; subtract
    // the parent front's start so the kernel indexes into the parent's fsz x fsz.
    std::vector<int> asm_local(mf.asm_idx.size());
    for (int p = 0; p < P; ++p) {
        const int par = mf.panel_parent[p];
        const int base = (par >= 0) ? mf.front_ptr[par] : 0;
        for (int a = mf.asm_ptr[p]; a < mf.asm_ptr[p + 1]; ++a)
            asm_local[a] = mf.asm_idx[a] - base;
    }
    plan.asm_total = static_cast<int>(asm_local.size());

    // One arena, 256-byte-aligned sub-arrays (avoids the L2 straddle from cycle 39).
    auto al = [](long b) { return (b + 255) & ~static_cast<long>(255); };
    long off = 0;
    const long o_front = off; off = al(off + total * sizeof(double));
    const long o_foff = off;  off = al(off + (long)(P + 1) * sizeof(int));
    const long o_fptr = off;  off = al(off + (long)(P + 1) * sizeof(int));
    const long o_nc = off;    off = al(off + (long)P * sizeof(int));
    const long o_plc = off;   off = al(off + (long)P * sizeof(int));
    const long o_par = off;   off = al(off + (long)P * sizeof(int));
    const long o_aptr = off;  off = al(off + (long)(P + 1) * sizeof(int));
    const long o_aloc = off;  off = al(off + (long)std::max(1, plan.asm_total) * sizeof(int));
    const long o_apos = off;  off = al(off + (long)std::max(1, plan.nnz_a) * sizeof(int));
    const int front_store = mf.front_ptr[P];
    plan.front_store = front_store;
    const long o_fr = off;    off = al(off + (long)std::max(1, front_store) * sizeof(int));
    const long o_pf = off;    off = al(off + (long)std::max(1, P) * sizeof(int));
    const long o_pof = off;   off = al(off + (long)std::max(1, n) * sizeof(int));
    const long o_y = off;     off = al(off + (long)n * sizeof(double));
    const long o_sing = off;  off = al(off + sizeof(int));
    cudaMalloc(&plan.arena, off);
    char* base = static_cast<char*>(plan.arena);
    plan.d_front = reinterpret_cast<double*>(base + o_front);
    // When the factor/solve front is float (FP32 / FP16 / TF32), allocate the float front arena.
    if (float_front) cudaMalloc(&plan.d_frontf, (long)total * sizeof(float));
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
    if (float_front) cudaMalloc(&plan.d_yf, (long)n * sizeof(float));
    int* d_panel_first = reinterpret_cast<int*>(base + o_pf);
    int* d_panel_of = reinterpret_cast<int*>(base + o_pof);

    auto H2D = [](int* d, const std::vector<int>& v) {
        if (!v.empty()) cudaMemcpy(d, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice);
    };
    H2D(plan.d_front_off, front_off);
    H2D(plan.d_front_ptr, mf.front_ptr);
    H2D(plan.d_ncols, panels.ncols);
    H2D(plan.d_plcols, plcols);
    H2D(plan.d_panel_parent, mf.panel_parent);
    H2D(plan.d_asm_ptr, mf.asm_ptr);
    H2D(plan.d_asm_local, asm_local);
    H2D(plan.d_front_rows, mf.front_rows);
    // Upload spine panel list (Phase 4). Separate allocation so it survives plan moves.
    if (!plan.h_spine_panels.empty()) {
        cudaMalloc(&plan.d_spine_panels, plan.h_spine_panels.size() * sizeof(int));
        cudaMemcpy(plan.d_spine_panels, plan.h_spine_panels.data(),
                   plan.h_spine_panels.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    lap("arena_malloc+H2D");
    H2D(d_panel_first, panels.first);
    H2D(d_panel_of, panels.panel_of);
    build_a_pos_device<<<n, 128>>>(n, d_Ap, d_Ai, d_panel_of, d_panel_first, plan.d_ncols,
                                   plan.d_front_off, plan.d_front_ptr, plan.d_front_rows,
                                   plan.d_a_pos);
    const cudaError_t apos_err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (apos_err != cudaSuccess) return MultifrontalPlan{};
    lap("a_pos_device");


    return plan;
}

}  // namespace custom_linear_solver::plan
