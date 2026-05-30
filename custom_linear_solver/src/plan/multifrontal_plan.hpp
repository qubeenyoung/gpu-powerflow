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
    bool fp32 = false;
    int *d_front_off = nullptr, *d_front_ptr = nullptr, *d_ncols = nullptr;
    int *d_plcols = nullptr, *d_panel_parent = nullptr;
    int *d_asm_ptr = nullptr, *d_asm_local = nullptr;
    int* d_a_pos = nullptr;
    int* d_sing = nullptr;
    int* d_front_rows = nullptr;
    double* d_y = nullptr;
    int front_store = 0;
    std::vector<int> plptr;
    void* stream = nullptr;
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
