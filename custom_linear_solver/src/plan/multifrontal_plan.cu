#include "plan/multifrontal_plan.hpp"

#include <utility>

#include <cuda_runtime.h>

namespace custom_linear_solver::plan {

MultifrontalPlan::~MultifrontalPlan()
{
    if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
    if (solve_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(solve_graph_exec));
    if (solve_graph) cudaGraphDestroy(static_cast<cudaGraph_t>(solve_graph));
    if (stream && owns_stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    if (d_spine_panels) cudaFree(d_spine_panels);
    if (d_yf) cudaFree(d_yf);
    if (d_frontf) cudaFree(d_frontf);
    if (d_pivot_offset) cudaFree(d_pivot_offset);
    if (arena) cudaFree(arena);
}

MultifrontalPlan::MultifrontalPlan(MultifrontalPlan&& o) noexcept { *this = std::move(o); }

MultifrontalPlan& MultifrontalPlan::operator=(MultifrontalPlan&& o) noexcept
{
    if (this != &o) {
        if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
        if (solve_graph_exec)
            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(solve_graph_exec));
        if (solve_graph) cudaGraphDestroy(static_cast<cudaGraph_t>(solve_graph));
        if (stream && owns_stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        if (d_spine_panels) cudaFree(d_spine_panels);
        if (d_yf) cudaFree(d_yf);
        if (d_frontf) cudaFree(d_frontf);
        if (arena) cudaFree(arena);

        n = o.n;
        num_panels = o.num_panels;
        num_plevels = o.num_plevels;
        nnz_a = o.nnz_a;
        asm_total = o.asm_total;
        front_total = o.front_total;
        arena = o.arena;
        d_front = o.d_front;
        d_frontf = o.d_frontf;
        fp32 = o.fp32;
        pure_fp32 = o.pure_fp32;
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
        d_yf = o.d_yf;
        front_store = o.front_store;
        plptr = std::move(o.plptr);
        h_front_ptr = std::move(o.h_front_ptr);
        h_ncols = std::move(o.h_ncols);
        h_plcols = std::move(o.h_plcols);
        h_front_off = std::move(o.h_front_off);
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
        d_spine_panels = o.d_spine_panels;
        o.d_spine_panels = nullptr;
        stream = o.stream;
        owns_stream = o.owns_stream;
        graph_exec = o.graph_exec;
        solve_graph_exec = o.solve_graph_exec;
        solve_graph = o.solve_graph;

        o.arena = nullptr;
        o.stream = nullptr;
        o.owns_stream = false;
        o.graph_exec = nullptr;
        o.solve_graph_exec = nullptr;
        o.d_frontf = nullptr;
        o.d_yf = nullptr;
        o.solve_graph = nullptr;
    }
    return *this;
}

}  // namespace custom_linear_solver::plan
