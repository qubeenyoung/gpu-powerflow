#pragma once

#include "linear/cusparse_ilu0_block_f32.hpp"
#include "linear/csr_spmv.hpp"
#include "linear/exact_klu_j11_f32.hpp"
#include "linear/implicit_schur_operator.hpp"
#include "linear/partitioned_dense_lu_j11_f32.hpp"

namespace exp_20260415::block_ilu {

class ImplicitSchurOperatorF32 {
public:
    ImplicitSchurOperatorF32();
    ~ImplicitSchurOperatorF32();

    ImplicitSchurOperatorF32(const ImplicitSchurOperatorF32&) = delete;
    ImplicitSchurOperatorF32& operator=(const ImplicitSchurOperatorF32&) = delete;

    void analyze(DeviceCsrMatrixView j11,
                 const HostCsrPattern& host_j11,
                 const HostCsrPattern& host_j22,
                 DeviceCsrMatrixView j12,
                 DeviceCsrMatrixView j21,
                 DeviceCsrMatrixView j22,
                 int32_t n_pvpq,
                 int32_t n_pq,
                 J11SolverKind j11_solver_kind,
                 J11ReorderMode j11_reorder_mode,
                 int32_t j11_dense_block_size,
                 J11DenseBackend dense_backend,
                 J11PartitionMode partition_mode,
                 SchurPreconditionerKind schur_preconditioner_kind);
    void factorize_j11();

    void build_rhs(const double* rhs_full_device,
                   float* rhs_schur_device,
                   SchurOperatorStats& stats,
                   bool collect_timing_breakdown);
    void apply(const float* x_vm_device,
               float* y_q_device,
               SchurOperatorStats& stats,
               bool collect_timing_breakdown);
    void apply_schur_preconditioner(const float* rhs_device,
                                    float* out_device,
                                    SchurOperatorStats& stats,
                                    bool collect_timing_breakdown);
    void recover_solution(const double* rhs_full_device,
                          const float* dvm_device,
                          double* dx_full_device,
                          SchurOperatorStats& stats,
                          bool collect_timing_breakdown);

    int32_t n_pq() const { return n_pq_; }
    int32_t j11_zero_pivot() const
    {
        return (j11_solver_kind_ == J11SolverKind::ExactKlu)
                   ? j11_exact_.last_zero_pivot()
                   : j11_dense_.last_zero_pivot();
    }
    int32_t j22_zero_pivot() const
    {
        if (schur_preconditioner_kind_ == SchurPreconditionerKind::J22Ilu0) {
            return j22_ilu_.last_zero_pivot();
        }
        if (schur_preconditioner_kind_ == SchurPreconditionerKind::J22BlockDenseLu) {
            return j22_dense_.last_zero_pivot();
        }
        return -1;
    }
    bool schur_preconditioner_enabled() const
    {
        return schur_preconditioner_kind_ != SchurPreconditionerKind::None;
    }

private:
    void ensure_ready(const char* method) const;
    void ensure_workspace();
    void refresh_values();
    void record_default_input();
    void wait_default_on_chain();
    void factorize_active_j11();
    void solve_active_j11(const float* rhs_device, float* out_device);

    int32_t n_pvpq_ = 0;
    int32_t n_pq_ = 0;
    bool analyzed_ = false;
    SchurPreconditionerKind schur_preconditioner_kind_ =
        SchurPreconditionerKind::None;
    J11SolverKind j11_solver_kind_ = J11SolverKind::PartitionDenseLu;

    DeviceCsrMatrixView j11_source_;
    DeviceCsrMatrixView j12_source_;
    DeviceCsrMatrixView j21_source_;
    DeviceCsrMatrixView j22_source_;

    DeviceBuffer<float> d_j11_values_;
    DeviceBuffer<float> d_j12_values_;
    DeviceBuffer<float> d_j21_values_;
    DeviceBuffer<float> d_j22_values_;
    DeviceBuffer<float> d_j22_ilu_values_;

    CsrSpmvF32 j12_spmv_;
    CsrSpmvF32 j21_spmv_;
    CsrSpmvF32 j22_spmv_;
    CusparseIlu0BlockF32 j22_ilu_;
    PartitionedDenseLuJ11BlockF32 j22_dense_;
    PartitionedDenseLuJ11BlockF32 j11_dense_;
    ExactKluJ11F32 j11_exact_;

    cudaStream_t j22_stream_ = nullptr;
    cudaStream_t chain_stream_ = nullptr;
    cudaEvent_t input_ready_ = nullptr;
    cudaEvent_t j22_done_ = nullptr;
    cudaEvent_t chain_done_ = nullptr;

    DeviceBuffer<float> rhs_p_;
    DeviceBuffer<float> rhs_q_;
    DeviceBuffer<float> j22_x_;
    DeviceBuffer<float> j12_x_;
    DeviceBuffer<float> j11_solve_;
    DeviceBuffer<float> j21_tmp_;
    DeviceBuffer<float> theta_rhs_;
};

}  // namespace exp_20260415::block_ilu
