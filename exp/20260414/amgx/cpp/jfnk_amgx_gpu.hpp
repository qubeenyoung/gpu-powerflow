#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/ops/op_interfaces.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace exp_20260414::newton_krylov {

struct JfnkOptions {
    std::string solver = "fgmres";
    std::string preconditioner = "amg_fd";
    double linear_tolerance = 1e-3;
    int32_t max_inner_iterations = 1000;
    int32_t gmres_restart = 30;
    bool auto_epsilon = true;
    double fixed_epsilon = 1e-6;
    double ilut_drop_tol = 1e-4;
    int32_t ilut_fill_factor = 10;
    double ilu_pivot_tol = 1e-12;
    std::string permutation = "none";
    std::string preconditioner_combine = "single";
    std::string residual_trace_path;
    std::string residual_trace_case;
    std::string jacobian_error_path;
};

struct JfnkStats {
    bool last_success = false;
    int32_t last_iterations = 0;
    int64_t last_jv_calls = 0;
    double last_estimated_error = 0.0;
    double last_epsilon = 0.0;
    std::string last_failure_reason;

    int64_t total_inner_iterations = 0;
    int64_t total_jv_calls = 0;
    int32_t max_inner_iterations = 0;
    int32_t linear_failures = 0;

    double total_solve_sec = 0.0;
    double total_preconditioner_setup_sec = 0.0;
    double total_jv_sec = 0.0;
    double total_jv_mismatch_sec = 0.0;
    double total_jv_update_sec = 0.0;
};

class JfnkLinearSolveAmgx final : public ILinearSolveOp {
public:
    JfnkLinearSolveAmgx(IStorage& storage, JfnkOptions options);
    ~JfnkLinearSolveAmgx();

    JfnkLinearSolveAmgx(const JfnkLinearSolveAmgx&) = delete;
    JfnkLinearSolveAmgx& operator=(const JfnkLinearSolveAmgx&) = delete;

    void analyze(const AnalyzeContext& ctx) override;
    void run(IterationContext& ctx) override;

    const JfnkStats& stats() const { return stats_; }

private:
    CudaFp64Storage& storage_;
    JfnkOptions options_;
    JfnkStats stats_;

    std::vector<int32_t> ybus_indptr_;
    std::vector<int32_t> ybus_indices_;
    std::vector<int32_t> pv_;
    std::vector<int32_t> pq_;
    int32_t n_bus_ = 0;
    bool analyzed_ = false;
};

}  // namespace exp_20260414::newton_krylov

#endif  // CUPF_WITH_CUDA
