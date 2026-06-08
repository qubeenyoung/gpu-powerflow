#pragma once

#include <vector>

namespace custom_linear_solver::plan {

struct MultifrontalPlan {
    int n = 0, num_panels = 0, num_plevels = 0;
    int nnz_a = 0, asm_total = 0;
    long front_total = 0;
    void* arena = nullptr;
    double* d_front = nullptr;
    float* d_frontf = nullptr;
    int *d_front_off = nullptr, *d_front_ptr = nullptr, *d_ncols = nullptr;
    int *d_plcols = nullptr, *d_panel_parent = nullptr;
    int *d_asm_ptr = nullptr, *d_asm_local = nullptr;
    int* d_a_pos = nullptr;
    int* d_sing = nullptr;
    int* d_front_rows = nullptr;
    double* d_y = nullptr;
    float* d_yf = nullptr;
    int front_store = 0;
    bool a_pos_unique = false;  // true when numeric scatter can store instead of atomic-add
    std::vector<int> plptr;
    std::vector<int> h_front_ptr;  // host copy of front_ptr (per-panel front size), for kernel dispatch
    std::vector<int> h_ncols;      // host copy of panel ncols, for tensor-core shared sizing
    std::vector<int> h_plcols;     // host copy of panels-by-level order (indexes into h_front_ptr)
    std::vector<int> h_front_off;  // host copy of front_off (per-panel arena offset), for per-panel cuBLAS dispatch

    // Phase Σ.8 — within-panel partial pivoting infrastructure (CLS_USE_PIVOTING=1).
    // pivot_offset[p] = prefix sum of ncols[0..p-1], giving the start of panel p's pivots in
    // the per-system pivot array (length = sum_p ncols[p] = n). Total per-batch pivot storage
    // is therefore B * n ints. Allocated/owned by per-state (per State allocation).
    std::vector<int> h_pivot_offset;  // P+1 entries
    int* d_pivot_offset = nullptr;    // device mirror
    int total_pivots = 0;             // = h_pivot_offset[P] = n

    // Tree-restructuring metadata (computed at analyze time, used by tree-restructured dispatch).
    // The spine is the contiguous "cnt=1 chain" at the top of the panel etree -- a sequential chain
    // of single-panel levels. The K subtree roots are the panels immediately below the spine; each
    // rooted subtree is independent and can be factored in parallel (Phase 3). The spine itself
    // gets fused into a single persistent kernel (Phase 4).
    std::vector<int> h_spine_panels;        // panel IDs in the spine, in factorization order (bottom -> top)
    int spine_start_level = -1;             // lowest L of the spine (top of the non-spine region)
    int num_subtrees = 0;                   // K (= cnt at the level just below the spine)
    std::vector<int> h_subtree_roots;       // K panel IDs of subtree roots (children of the spine bottom)
    std::vector<int> h_subtree_of_panel;    // per-panel subtree id (0..K-1) or -1 if panel is in the spine

    // Device copy of h_spine_panels for Phase 4 spine kernel.
    int* d_spine_panels = nullptr;

    // Per-(subtree, level) ranges in h_plcols / d_plcols. After analyze re-sorts plcols within
    // each level so all subtree-0 panels come first, then subtree-1, etc. Then for subtree k at
    // level L: dispatch range = [subtree_level_off[k * num_plevels + L], + subtree_level_cnt[..]).
    // Spine panels are NOT included in any subtree's range — they are handled separately by the
    // spine kernel (Phase 4). Sized K * num_plevels.
    std::vector<int> h_subtree_level_off;
    std::vector<int> h_subtree_level_cnt;

    void* stream = nullptr;
    bool owns_stream = false;
    void* graph_exec = nullptr;
    void* solve_graph_exec = nullptr;
    void* solve_graph = nullptr;

    MultifrontalPlan() = default;
    ~MultifrontalPlan();
    MultifrontalPlan(const MultifrontalPlan&) = delete;
    MultifrontalPlan& operator=(const MultifrontalPlan&) = delete;
    MultifrontalPlan(MultifrontalPlan&&) noexcept;
    MultifrontalPlan& operator=(MultifrontalPlan&&) noexcept;
};

}  // namespace custom_linear_solver::plan
