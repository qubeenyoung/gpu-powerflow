#include "internal/plan/multifrontal_plan.hpp"

#include <utility>

#include <cuda_runtime.h>

namespace custom_linear_solver::plan {

MultifrontalPlan::~MultifrontalPlan()
{
    if (d_spine_panels) cudaFree(d_spine_panels);
    if (d_plcols_tier) cudaFree(d_plcols_tier);
    if (d_y_f) cudaFree(d_y_f);
    if (d_front_f) cudaFree(d_front_f);
    if (d_pivot_offset) cudaFree(d_pivot_offset);
    if (arena) cudaFree(arena);
}

MultifrontalPlan::MultifrontalPlan(MultifrontalPlan&& o) noexcept { *this = std::move(o); }

MultifrontalPlan& MultifrontalPlan::operator=(MultifrontalPlan&& o) noexcept
{
    if (this != &o) {
        if (d_spine_panels) cudaFree(d_spine_panels);
        if (d_plcols_tier) cudaFree(d_plcols_tier);
        if (d_y_f) cudaFree(d_y_f);
        if (d_front_f) cudaFree(d_front_f);
        if (d_pivot_offset) cudaFree(d_pivot_offset);
        if (arena) cudaFree(arena);

        num_rows = o.num_rows;
        num_panels = o.num_panels;
        num_plevels = o.num_plevels;
        nnz = o.nnz;
        asm_total = o.asm_total;
        front_total = o.front_total;
        arena = o.arena;
        d_front = o.d_front;
        d_front_f = o.d_front_f;
        d_front_off = o.d_front_off;
        d_front_ptr = o.d_front_ptr;
        d_ncols = o.d_ncols;
        d_plcols = o.d_plcols;
        d_panel_parent = o.d_panel_parent;
        d_asm_ptr = o.d_asm_ptr;
        d_asm_local = o.d_asm_local;
        d_a_pos = o.d_a_pos;
        d_sing = o.d_sing;
        d_front_rows = o.d_front_rows;
        d_y = o.d_y;
        d_y_f = o.d_y_f;
        front_store = o.front_store;
        a_pos_unique = o.a_pos_unique;
        panel_level_ptr = std::move(o.panel_level_ptr);
        h_front_ptr = std::move(o.h_front_ptr);
        h_ncols = std::move(o.h_ncols);
        h_plcols = std::move(o.h_plcols);
        h_front_off = std::move(o.h_front_off);
        h_panel_parent = std::move(o.h_panel_parent);
        h_asm_ptr = std::move(o.h_asm_ptr);
        h_pivot_offset = std::move(o.h_pivot_offset);
        d_pivot_offset = o.d_pivot_offset;
        total_pivots = o.total_pivots;
        o.d_pivot_offset = nullptr;
        h_spine_panels = std::move(o.h_spine_panels);
        spine_start_level = o.spine_start_level;
        num_subtrees = o.num_subtrees;
        h_subtree_roots = std::move(o.h_subtree_roots);
        h_subtree_of_panel = std::move(o.h_subtree_of_panel);
        h_subtree_level_off = std::move(o.h_subtree_level_off);
        h_subtree_level_cnt = std::move(o.h_subtree_level_cnt);
        h_subtree_level_tier_off = std::move(o.h_subtree_level_tier_off);
        h_plcols_tier = std::move(o.h_plcols_tier);
        h_level_tier_off = std::move(o.h_level_tier_off);
        d_plcols_tier = o.d_plcols_tier;
        o.d_plcols_tier = nullptr;
        d_spine_panels = o.d_spine_panels;
        o.d_spine_panels = nullptr;

        o.arena = nullptr;
        o.d_front_f = nullptr;
        o.d_y_f = nullptr;
        o.d_pivot_offset = nullptr;
    }
    return *this;
}

}  // namespace custom_linear_solver::plan
