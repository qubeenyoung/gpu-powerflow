#include "linear/implicit_schur_operator_f32.hpp"

#include <chrono>
#include <stdexcept>

namespace exp_20260415::block_ilu {
namespace {

constexpr int32_t kBlockSize = 256;

int32_t grid_for(int32_t n)
{
    return (n + kBlockSize - 1) / kBlockSize;
}

double elapsed_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template <typename Work>
void time_stage(double& bucket, bool collect, Work&& work)
{
    if (collect) {
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto start = std::chrono::steady_clock::now();
        work();
        CUDA_CHECK(cudaDeviceSynchronize());
        bucket += elapsed_since(start);
    } else {
        work();
    }
}

__global__ void double_to_float_kernel(int32_t n,
                                       const double* __restrict__ input,
                                       float* __restrict__ output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = static_cast<float>(input[i]);
    }
}

__global__ void float_to_double_kernel(int32_t n,
                                       const float* __restrict__ input,
                                       double* __restrict__ output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = static_cast<double>(input[i]);
    }
}

__global__ void subtract_kernel_f32(int32_t n,
                                    const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] - b[i];
    }
}

__global__ void copy_kernel_f32(int32_t n,
                                const float* __restrict__ src,
                                float* __restrict__ dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

}  // namespace

ImplicitSchurOperatorF32::ImplicitSchurOperatorF32()
    : j22_ilu_("J22_schur_ilu0_f32")
    , j22_dense_("J22_schur_dense_f32")
    , j11_dense_("J11_dense_f32")
    , j11_exact_("J11_exact_klu_f32")
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&j22_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&chain_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&input_ready_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&j22_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&chain_done_, cudaEventDisableTiming));
    j11_dense_.set_stream(chain_stream_);
    j22_ilu_.set_stream(chain_stream_);
    j22_dense_.set_stream(chain_stream_);
}

ImplicitSchurOperatorF32::~ImplicitSchurOperatorF32()
{
    if (j22_stream_ != nullptr) cudaStreamSynchronize(j22_stream_);
    if (chain_stream_ != nullptr) cudaStreamSynchronize(chain_stream_);
    j22_dense_.clear_stream_noexcept();
    j22_ilu_.clear_stream_noexcept();
    j11_dense_.clear_stream_noexcept();
    if (j22_done_ != nullptr) cudaEventDestroy(j22_done_);
    if (chain_done_ != nullptr) cudaEventDestroy(chain_done_);
    if (input_ready_ != nullptr) cudaEventDestroy(input_ready_);
    if (j22_stream_ != nullptr) cudaStreamDestroy(j22_stream_);
    if (chain_stream_ != nullptr) cudaStreamDestroy(chain_stream_);
}

void ImplicitSchurOperatorF32::analyze(DeviceCsrMatrixView j11,
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
                                       SchurPreconditionerKind schur_preconditioner_kind)
{
    if (n_pvpq <= 0 || n_pq <= 0 ||
        j11.rows != n_pvpq || j11.cols != n_pvpq ||
        j12.rows != n_pvpq || j12.cols != n_pq ||
        j21.rows != n_pq || j21.cols != n_pvpq ||
        j22.rows != n_pq || j22.cols != n_pq) {
        throw std::runtime_error("ImplicitSchurOperatorF32::analyze received invalid dimensions");
    }
    if (host_j22.rows != j22.rows || host_j22.cols != j22.cols ||
        host_j22.nnz() != j22.nnz) {
        throw std::runtime_error("ImplicitSchurOperatorF32::analyze received invalid J22 host pattern");
    }

    n_pvpq_ = n_pvpq;
    n_pq_ = n_pq;
    schur_preconditioner_kind_ = schur_preconditioner_kind;
    j11_solver_kind_ = j11_solver_kind;
    j11_source_ = j11;
    j12_source_ = j12;
    j21_source_ = j21;
    j22_source_ = j22;
    d_j11_values_.resize(static_cast<std::size_t>(j11.nnz));
    d_j12_values_.resize(static_cast<std::size_t>(j12.nnz));
    d_j21_values_.resize(static_cast<std::size_t>(j21.nnz));
    d_j22_values_.resize(static_cast<std::size_t>(j22.nnz));
    if (schur_preconditioner_kind_ == SchurPreconditionerKind::J22Ilu0) {
        d_j22_ilu_values_.resize(static_cast<std::size_t>(j22.nnz));
    }
    ensure_workspace();

    DeviceCsrMatrixViewF32 j11_f32{j11.rows, j11.cols, j11.nnz,
                                   j11.row_ptr, j11.col_idx, d_j11_values_.data()};
    DeviceCsrMatrixViewF32 j12_f32{j12.rows, j12.cols, j12.nnz,
                                   j12.row_ptr, j12.col_idx, d_j12_values_.data()};
    DeviceCsrMatrixViewF32 j21_f32{j21.rows, j21.cols, j21.nnz,
                                   j21.row_ptr, j21.col_idx, d_j21_values_.data()};
    DeviceCsrMatrixViewF32 j22_f32{j22.rows, j22.cols, j22.nnz,
                                   j22.row_ptr, j22.col_idx, d_j22_values_.data()};
    DeviceCsrMatrixViewF32 j22_ilu_f32{j22.rows, j22.cols, j22.nnz,
                                       j22.row_ptr, j22.col_idx,
                                       d_j22_ilu_values_.data()};
    refresh_values();
    CUDA_CHECK(cudaDeviceSynchronize());
    if (j11_solver_kind_ == J11SolverKind::ExactKlu) {
        j11_exact_.analyze(j11, host_j11);
    } else if (j11_solver_kind_ == J11SolverKind::PartitionDenseLu) {
        j11_dense_.analyze(j11_f32,
                           host_j11,
                           j11_reorder_mode,
                           j11_dense_block_size,
                           dense_backend,
                           partition_mode);
    } else {
        throw std::runtime_error("ImplicitSchurOperatorF32 supports partition-dense-lu or exact-klu J11");
    }
    j12_spmv_.bind(j12_f32);
    j21_spmv_.bind(j21_f32);
    j22_spmv_.bind(j22_f32);
    if (schur_preconditioner_kind_ == SchurPreconditionerKind::J22Ilu0) {
        j22_ilu_.analyze(j22_ilu_f32);
    } else if (schur_preconditioner_kind_ ==
               SchurPreconditionerKind::J22BlockDenseLu) {
        j22_dense_.analyze(j22_f32,
                           host_j22,
                           j11_reorder_mode,
                           j11_dense_block_size,
                           J11DenseBackend::CublasGetrf,
                           partition_mode);
    }
    analyzed_ = true;
}

void ImplicitSchurOperatorF32::ensure_workspace()
{
    rhs_p_.resize(static_cast<std::size_t>(n_pvpq_));
    rhs_q_.resize(static_cast<std::size_t>(n_pq_));
    j22_x_.resize(static_cast<std::size_t>(n_pq_));
    j12_x_.resize(static_cast<std::size_t>(n_pvpq_));
    j11_solve_.resize(static_cast<std::size_t>(n_pvpq_));
    j21_tmp_.resize(static_cast<std::size_t>(n_pq_));
    theta_rhs_.resize(static_cast<std::size_t>(n_pvpq_));
}

void ImplicitSchurOperatorF32::ensure_ready(const char* method) const
{
    if (!analyzed_) {
        throw std::runtime_error(std::string("ImplicitSchurOperatorF32::") +
                                 method + " called before analyze");
    }
}

void ImplicitSchurOperatorF32::refresh_values()
{
    double_to_float_kernel<<<grid_for(j11_source_.nnz), kBlockSize>>>(
        j11_source_.nnz, j11_source_.values, d_j11_values_.data());
    double_to_float_kernel<<<grid_for(j12_source_.nnz), kBlockSize>>>(
        j12_source_.nnz, j12_source_.values, d_j12_values_.data());
    double_to_float_kernel<<<grid_for(j21_source_.nnz), kBlockSize>>>(
        j21_source_.nnz, j21_source_.values, d_j21_values_.data());
    double_to_float_kernel<<<grid_for(j22_source_.nnz), kBlockSize>>>(
        j22_source_.nnz, j22_source_.values, d_j22_values_.data());
    if (schur_preconditioner_kind_ == SchurPreconditionerKind::J22Ilu0) {
        double_to_float_kernel<<<grid_for(j22_source_.nnz), kBlockSize>>>(
            j22_source_.nnz, j22_source_.values, d_j22_ilu_values_.data());
    }
    CUDA_CHECK(cudaGetLastError());
}

void ImplicitSchurOperatorF32::record_default_input()
{
    CUDA_CHECK(cudaEventRecord(input_ready_, nullptr));
    CUDA_CHECK(cudaStreamWaitEvent(chain_stream_, input_ready_, 0));
}

void ImplicitSchurOperatorF32::wait_default_on_chain()
{
    CUDA_CHECK(cudaEventRecord(chain_done_, chain_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, chain_done_, 0));
}

void ImplicitSchurOperatorF32::factorize_j11()
{
    ensure_ready("factorize_j11");
    record_default_input();
    refresh_values();
    CUDA_CHECK(cudaDeviceSynchronize());
    factorize_active_j11();
    if (schur_preconditioner_kind_ == SchurPreconditionerKind::J22Ilu0) {
        j22_ilu_.factorize();
        wait_default_on_chain();
    } else if (schur_preconditioner_kind_ ==
               SchurPreconditionerKind::J22BlockDenseLu) {
        j22_dense_.factorize();
        wait_default_on_chain();
    }
}

void ImplicitSchurOperatorF32::factorize_active_j11()
{
    if (j11_solver_kind_ == J11SolverKind::ExactKlu) {
        CUDA_CHECK(cudaDeviceSynchronize());
        j11_exact_.factorize();
    } else {
        j11_dense_.factorize();
        wait_default_on_chain();
    }
}

void ImplicitSchurOperatorF32::solve_active_j11(const float* rhs_device,
                                                float* out_device)
{
    if (j11_solver_kind_ == J11SolverKind::ExactKlu) {
        CUDA_CHECK(cudaStreamSynchronize(chain_stream_));
        j11_exact_.solve(rhs_device, out_device);
        CUDA_CHECK(cudaEventRecord(input_ready_, nullptr));
        CUDA_CHECK(cudaStreamWaitEvent(chain_stream_, input_ready_, 0));
    } else {
        j11_dense_.solve(rhs_device, out_device);
    }
}

void ImplicitSchurOperatorF32::build_rhs(const double* rhs_full_device,
                                         float* rhs_schur_device,
                                         SchurOperatorStats& stats,
                                         bool collect_timing_breakdown)
{
    ensure_ready("build_rhs");
    time_stage(stats.rhs_sec, collect_timing_breakdown, [&]() {
        double_to_float_kernel<<<grid_for(n_pvpq_), kBlockSize>>>(
            n_pvpq_, rhs_full_device, rhs_p_.data());
        double_to_float_kernel<<<grid_for(n_pq_), kBlockSize>>>(
            n_pq_, rhs_full_device + n_pvpq_, rhs_q_.data());
        CUDA_CHECK(cudaGetLastError());
        record_default_input();
        solve_active_j11(rhs_p_.data(), j11_solve_.data());
        j21_spmv_.apply_async(j11_solve_.data(), j21_tmp_.data(), chain_stream_);
        wait_default_on_chain();
        subtract_kernel_f32<<<grid_for(n_pq_), kBlockSize>>>(
            n_pq_, rhs_q_.data(), j21_tmp_.data(), rhs_schur_device);
        CUDA_CHECK(cudaGetLastError());
    });
    ++stats.j11_solve_calls;
    ++stats.spmv_calls;
}

void ImplicitSchurOperatorF32::apply(const float* x_vm_device,
                                     float* y_q_device,
                                     SchurOperatorStats& stats,
                                     bool collect_timing_breakdown)
{
    ensure_ready("apply");
    ++stats.schur_matvec_calls;
    stats.spmv_calls += 3;
    ++stats.j11_solve_calls;

    if (collect_timing_breakdown) {
        const auto matvec_start = std::chrono::steady_clock::now();
        time_stage(stats.spmv_sec, true, [&]() {
            j22_spmv_.apply_async(x_vm_device, j22_x_.data(), j22_stream_);
            CUDA_CHECK(cudaStreamSynchronize(j22_stream_));
        });
        time_stage(stats.spmv_sec, true, [&]() {
            j12_spmv_.apply_async(x_vm_device, j12_x_.data(), chain_stream_);
            CUDA_CHECK(cudaStreamSynchronize(chain_stream_));
        });
        time_stage(stats.j11_solve_sec, true, [&]() {
            solve_active_j11(j12_x_.data(), j11_solve_.data());
            CUDA_CHECK(cudaStreamSynchronize(chain_stream_));
        });
        time_stage(stats.spmv_sec, true, [&]() {
            j21_spmv_.apply_async(j11_solve_.data(), j21_tmp_.data(), chain_stream_);
            CUDA_CHECK(cudaStreamSynchronize(chain_stream_));
        });
        time_stage(stats.vector_update_sec, true, [&]() {
            subtract_kernel_f32<<<grid_for(n_pq_), kBlockSize>>>(
                n_pq_, j22_x_.data(), j21_tmp_.data(), y_q_device);
            CUDA_CHECK(cudaGetLastError());
        });
        stats.matvec_sec += elapsed_since(matvec_start);
        return;
    }

    CUDA_CHECK(cudaEventRecord(input_ready_, nullptr));
    CUDA_CHECK(cudaStreamWaitEvent(j22_stream_, input_ready_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(chain_stream_, input_ready_, 0));
    j22_spmv_.apply_async(x_vm_device, j22_x_.data(), j22_stream_);
    j12_spmv_.apply_async(x_vm_device, j12_x_.data(), chain_stream_);
    solve_active_j11(j12_x_.data(), j11_solve_.data());
    j21_spmv_.apply_async(j11_solve_.data(), j21_tmp_.data(), chain_stream_);
    CUDA_CHECK(cudaEventRecord(j22_done_, j22_stream_));
    CUDA_CHECK(cudaEventRecord(chain_done_, chain_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, j22_done_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, chain_done_, 0));
    subtract_kernel_f32<<<grid_for(n_pq_), kBlockSize>>>(
        n_pq_, j22_x_.data(), j21_tmp_.data(), y_q_device);
    CUDA_CHECK(cudaGetLastError());
}

void ImplicitSchurOperatorF32::apply_schur_preconditioner(
    const float* rhs_device,
    float* out_device,
    SchurOperatorStats& stats,
    bool collect_timing_breakdown)
{
    ensure_ready("apply_schur_preconditioner");
    if (rhs_device == nullptr || out_device == nullptr) {
        throw std::runtime_error("ImplicitSchurOperatorF32::apply_schur_preconditioner received null input");
    }

    if (schur_preconditioner_kind_ == SchurPreconditionerKind::None) {
        copy_kernel_f32<<<grid_for(n_pq_), kBlockSize>>>(n_pq_, rhs_device, out_device);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    time_stage(stats.schur_preconditioner_sec, collect_timing_breakdown, [&]() {
        record_default_input();
        if (schur_preconditioner_kind_ == SchurPreconditionerKind::J22Ilu0) {
            j22_ilu_.solve(rhs_device, out_device);
        } else {
            j22_dense_.solve(rhs_device, out_device);
        }
        wait_default_on_chain();
    });
    ++stats.schur_preconditioner_calls;
}

void ImplicitSchurOperatorF32::recover_solution(const double* rhs_full_device,
                                                const float* dvm_device,
                                                double* dx_full_device,
                                                SchurOperatorStats& stats,
                                                bool collect_timing_breakdown)
{
    ensure_ready("recover_solution");
    time_stage(stats.recover_sec, collect_timing_breakdown, [&]() {
        double_to_float_kernel<<<grid_for(n_pvpq_), kBlockSize>>>(
            n_pvpq_, rhs_full_device, rhs_p_.data());
        CUDA_CHECK(cudaGetLastError());
        record_default_input();
        j12_spmv_.apply_async(dvm_device, j12_x_.data(), chain_stream_);
        wait_default_on_chain();
        subtract_kernel_f32<<<grid_for(n_pvpq_), kBlockSize>>>(
            n_pvpq_, rhs_p_.data(), j12_x_.data(), theta_rhs_.data());
        CUDA_CHECK(cudaGetLastError());
        record_default_input();
        solve_active_j11(theta_rhs_.data(), j11_solve_.data());
        wait_default_on_chain();
        float_to_double_kernel<<<grid_for(n_pvpq_), kBlockSize>>>(
            n_pvpq_, j11_solve_.data(), dx_full_device);
        float_to_double_kernel<<<grid_for(n_pq_), kBlockSize>>>(
            n_pq_, dvm_device, dx_full_device + n_pvpq_);
        CUDA_CHECK(cudaGetLastError());
    });
    ++stats.j11_solve_calls;
    ++stats.spmv_calls;
}

}  // namespace exp_20260415::block_ilu
