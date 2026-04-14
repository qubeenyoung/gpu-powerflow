#pragma once

#include "newton_solver/ops/op_interfaces.hpp"
#include "newton_solver/ops/mismatch/cpu_f64.hpp"
#include "newton_solver/ops/voltage_update/cpu_f64.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <cstdint>
#include <string>

namespace exp_20260414::newton_krylov {

struct JfnkOptions {
    std::string solver = "bicgstab_none";
    std::string preconditioner = "none";
    double linear_tolerance = 1e-3;
    int32_t max_inner_iterations = 1000;
    int32_t gmres_restart = 30;
    bool auto_epsilon = true;
    double fixed_epsilon = 1e-6;
    double ilut_drop_tol = 1e-4;
    int32_t ilut_fill_factor = 10;
    double ilu_pivot_tol = 1e-12;
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

class JfnkLinearSolveBiCGSTAB final : public ILinearSolveOp {
public:
    JfnkLinearSolveBiCGSTAB(IStorage& storage, JfnkOptions options);

    void analyze(const AnalyzeContext& ctx) override;
    void run(IterationContext& ctx) override;

    const JfnkStats& stats() const { return stats_; }

private:
    CpuFp64Storage& storage_;
    CpuFp64Storage scratch_;
    CpuMismatchOpF64 scratch_mismatch_;
    CpuVoltageUpdateF64 scratch_voltage_update_;
    JfnkOptions options_;
    JfnkStats stats_;
    bool scratch_prepared_ = false;
};

}  // namespace exp_20260414::newton_krylov
