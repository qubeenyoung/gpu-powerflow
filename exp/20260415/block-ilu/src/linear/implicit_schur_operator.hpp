#pragma once

#include "linear/csr_spmv.hpp"
#include "linear/partitioned_dense_lu_j11.hpp"
#include "linear/permuted_ilu0_block.hpp"

#include <cstdint>
#include <string>

namespace exp_20260415::block_ilu {

enum class SchurPreconditionerKind {
    None,
    J22Ilu0,
    J22BlockDenseLu,
};

SchurPreconditionerKind parse_schur_preconditioner_kind(const std::string& name);
const char* schur_preconditioner_kind_name(SchurPreconditionerKind kind);

struct SchurOperatorStats {
    int32_t schur_matvec_calls = 0;
    int32_t schur_preconditioner_calls = 0;
    int32_t spmv_calls = 0;
    int32_t j11_solve_calls = 0;
    double matvec_sec = 0.0;
    double rhs_sec = 0.0;
    double recover_sec = 0.0;
    double spmv_sec = 0.0;
    double j11_solve_sec = 0.0;
    double schur_preconditioner_sec = 0.0;
    double vector_update_sec = 0.0;
};

class ImplicitSchurOperator {
public:
    ImplicitSchurOperator();
    ~ImplicitSchurOperator();

    ImplicitSchurOperator(const ImplicitSchurOperator&) = delete;
    ImplicitSchurOperator& operator=(const ImplicitSchurOperator&) = delete;

    void analyze(DeviceCsrMatrixView j11,
                 const HostCsrPattern& host_j11,
                 DeviceCsrMatrixView j12,
                 DeviceCsrMatrixView j21,
                 DeviceCsrMatrixView j22,
                 int32_t n_pvpq,
                 int32_t n_pq,
                 J11ReorderMode j11_reorder_mode,
                 J11SolverKind j11_solver_kind,
                 int32_t j11_dense_block_size,
                 J11PartitionMode partition_mode);
    void factorize_j11();

    void build_rhs(const double* rhs_full_device,
                   double* rhs_schur_device,
                   SchurOperatorStats& stats,
                   bool collect_timing_breakdown);
    void apply(const double* x_vm_device,
               double* y_q_device,
               SchurOperatorStats& stats,
               bool collect_timing_breakdown);
    void recover_solution(const double* rhs_full_device,
                          const double* dvm_device,
                          double* dx_full_device,
                          SchurOperatorStats& stats,
                          bool collect_timing_breakdown);

    int32_t n_pvpq() const { return n_pvpq_; }
    int32_t n_pq() const { return n_pq_; }
    int32_t j11_zero_pivot() const;

private:
    void ensure_ready(const char* method) const;
    void ensure_workspace();
    void record_default_input();
    void wait_default_on_chain();
    void factorize_active_j11();
    void solve_active_j11(const double* rhs_device, double* out_device);

    int32_t n_pvpq_ = 0;
    int32_t n_pq_ = 0;
    bool analyzed_ = false;
    J11SolverKind j11_solver_kind_ = J11SolverKind::Ilu0;

    CsrSpmv j12_spmv_;
    CsrSpmv j21_spmv_;
    CsrSpmv j22_spmv_;
    PermutedIlu0Block j11_ilu_;
    PartitionedDenseLuJ11Block j11_dense_;

    cudaStream_t j22_stream_ = nullptr;
    cudaStream_t chain_stream_ = nullptr;
    cudaEvent_t input_ready_ = nullptr;
    cudaEvent_t j22_done_ = nullptr;
    cudaEvent_t chain_done_ = nullptr;

    DeviceBuffer<double> j22_x_;
    DeviceBuffer<double> j12_x_;
    DeviceBuffer<double> j11_solve_;
    DeviceBuffer<double> j21_tmp_;
    DeviceBuffer<double> theta_rhs_;
};

}  // namespace exp_20260415::block_ilu
