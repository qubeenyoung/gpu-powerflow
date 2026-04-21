#include "linear/implicit_schur_operator.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
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
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start)
        .count();
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

__global__ void subtract_kernel(int32_t n,
                                const double* __restrict__ a,
                                const double* __restrict__ b,
                                double* __restrict__ out)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] - b[i];
    }
}

__global__ void copy_kernel(int32_t n,
                            const double* __restrict__ src,
                            double* __restrict__ dst)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

}  // namespace

SchurPreconditionerKind parse_schur_preconditioner_kind(const std::string& name)
{
    std::string lowered = name;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered == "none") {
        return SchurPreconditionerKind::None;
    }
    if (lowered == "j22-ilu0" || lowered == "j22_ilu0") {
        return SchurPreconditionerKind::J22Ilu0;
    }
    if (lowered == "j22-block-dense-lu" ||
        lowered == "j22_block_dense_lu" ||
        lowered == "j22-dense-lu" ||
        lowered == "j22_dense_lu") {
        return SchurPreconditionerKind::J22BlockDenseLu;
    }
    throw std::runtime_error("unknown Schur preconditioner: " + name);
}

const char* schur_preconditioner_kind_name(SchurPreconditionerKind kind)
{
    switch (kind) {
    case SchurPreconditionerKind::None:
        return "none";
    case SchurPreconditionerKind::J22Ilu0:
        return "j22_ilu0";
    case SchurPreconditionerKind::J22BlockDenseLu:
        return "j22_block_dense_lu";
    default:
        return "unknown";
    }
}

ImplicitSchurOperator::ImplicitSchurOperator()
    : j11_ilu_("J11")
    , j11_dense_("J11_dense")
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&j22_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&chain_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&input_ready_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&j22_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&chain_done_, cudaEventDisableTiming));
    j11_ilu_.set_stream(chain_stream_);
    j11_dense_.set_stream(chain_stream_);
}

ImplicitSchurOperator::~ImplicitSchurOperator()
{
    if (j22_stream_ != nullptr) {
        cudaStreamSynchronize(j22_stream_);
    }
    if (chain_stream_ != nullptr) {
        cudaStreamSynchronize(chain_stream_);
    }
    j11_ilu_.clear_stream_noexcept();
    j11_dense_.clear_stream_noexcept();
    if (j22_done_ != nullptr) {
        cudaEventDestroy(j22_done_);
        j22_done_ = nullptr;
    }
    if (chain_done_ != nullptr) {
        cudaEventDestroy(chain_done_);
        chain_done_ = nullptr;
    }
    if (input_ready_ != nullptr) {
        cudaEventDestroy(input_ready_);
        input_ready_ = nullptr;
    }
    if (j22_stream_ != nullptr) {
        cudaStreamDestroy(j22_stream_);
        j22_stream_ = nullptr;
    }
    if (chain_stream_ != nullptr) {
        cudaStreamDestroy(chain_stream_);
        chain_stream_ = nullptr;
    }
}

void ImplicitSchurOperator::analyze(DeviceCsrMatrixView j11,
                                    const HostCsrPattern& host_j11,
                                    DeviceCsrMatrixView j12,
                                    DeviceCsrMatrixView j21,
                                    DeviceCsrMatrixView j22,
                                    int32_t n_pvpq,
                                    int32_t n_pq,
                                    J11ReorderMode j11_reorder_mode,
                                    J11SolverKind j11_solver_kind,
                                    int32_t j11_dense_block_size,
                                    J11PartitionMode partition_mode)
{
    if (n_pvpq <= 0 || n_pq <= 0 ||
        j11.rows != n_pvpq || j11.cols != n_pvpq ||
        host_j11.rows != n_pvpq || host_j11.cols != n_pvpq ||
        j12.rows != n_pvpq || j12.cols != n_pq ||
        j21.rows != n_pq || j21.cols != n_pvpq ||
        j22.rows != n_pq || j22.cols != n_pq) {
        throw std::runtime_error("ImplicitSchurOperator::analyze received invalid dimensions");
    }

    n_pvpq_ = n_pvpq;
    n_pq_ = n_pq;
    j11_solver_kind_ = j11_solver_kind;
    if (j11_solver_kind_ == J11SolverKind::Ilu0) {
        j11_ilu_.analyze(j11, host_j11, j11_reorder_mode);
    } else if (j11_solver_kind_ == J11SolverKind::PartitionDenseLu) {
        j11_dense_.analyze(j11,
                           host_j11,
                           j11_reorder_mode,
                           j11_dense_block_size,
                           partition_mode);
    } else {
        throw std::runtime_error("ImplicitSchurOperator::analyze received unknown J11 solver");
    }
    j12_spmv_.bind(j12);
    j21_spmv_.bind(j21);
    j22_spmv_.bind(j22);
    ensure_workspace();
    analyzed_ = true;
}

void ImplicitSchurOperator::ensure_workspace()
{
    j22_x_.resize(static_cast<std::size_t>(n_pq_));
    j12_x_.resize(static_cast<std::size_t>(n_pvpq_));
    j11_solve_.resize(static_cast<std::size_t>(n_pvpq_));
    j21_tmp_.resize(static_cast<std::size_t>(n_pq_));
    theta_rhs_.resize(static_cast<std::size_t>(n_pvpq_));
}

void ImplicitSchurOperator::ensure_ready(const char* method) const
{
    if (!analyzed_ || n_pvpq_ <= 0 || n_pq_ <= 0) {
        throw std::runtime_error(std::string("ImplicitSchurOperator::") +
                                 method + " called before analyze");
    }
}

void ImplicitSchurOperator::record_default_input()
{
    CUDA_CHECK(cudaEventRecord(input_ready_, nullptr));
    CUDA_CHECK(cudaStreamWaitEvent(chain_stream_, input_ready_, 0));
}

void ImplicitSchurOperator::wait_default_on_chain()
{
    CUDA_CHECK(cudaEventRecord(chain_done_, chain_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, chain_done_, 0));
}

void ImplicitSchurOperator::factorize_j11()
{
    ensure_ready("factorize_j11");
    record_default_input();
    factorize_active_j11();
    wait_default_on_chain();
}

int32_t ImplicitSchurOperator::j11_zero_pivot() const
{
    if (j11_solver_kind_ == J11SolverKind::PartitionDenseLu) {
        return j11_dense_.last_zero_pivot();
    }
    return j11_ilu_.last_zero_pivot();
}

void ImplicitSchurOperator::factorize_active_j11()
{
    if (j11_solver_kind_ == J11SolverKind::PartitionDenseLu) {
        j11_dense_.factorize();
    } else {
        j11_ilu_.factorize();
    }
}

void ImplicitSchurOperator::solve_active_j11(const double* rhs_device, double* out_device)
{
    if (j11_solver_kind_ == J11SolverKind::PartitionDenseLu) {
        j11_dense_.solve(rhs_device, out_device);
    } else {
        j11_ilu_.solve(rhs_device, out_device);
    }
}

void ImplicitSchurOperator::build_rhs(const double* rhs_full_device,
                                      double* rhs_schur_device,
                                      SchurOperatorStats& stats,
                                      bool collect_timing_breakdown)
{
    ensure_ready("build_rhs");
    if (rhs_full_device == nullptr || rhs_schur_device == nullptr) {
        throw std::runtime_error("ImplicitSchurOperator::build_rhs received null input");
    }

    time_stage(stats.rhs_sec, collect_timing_breakdown, [&]() {
        record_default_input();
        solve_active_j11(rhs_full_device, j11_solve_.data());
        j21_spmv_.apply_async(j11_solve_.data(), j21_tmp_.data(), chain_stream_);
        wait_default_on_chain();
        subtract_kernel<<<grid_for(n_pq_), kBlockSize>>>(
            n_pq_, rhs_full_device + n_pvpq_, j21_tmp_.data(), rhs_schur_device);
        CUDA_CHECK(cudaGetLastError());
    });

    ++stats.j11_solve_calls;
    ++stats.spmv_calls;
}

void ImplicitSchurOperator::apply(const double* x_vm_device,
                                  double* y_q_device,
                                  SchurOperatorStats& stats,
                                  bool collect_timing_breakdown)
{
    ensure_ready("apply");
    if (x_vm_device == nullptr || y_q_device == nullptr) {
        throw std::runtime_error("ImplicitSchurOperator::apply received null input");
    }

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
            subtract_kernel<<<grid_for(n_pq_), kBlockSize>>>(
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
    subtract_kernel<<<grid_for(n_pq_), kBlockSize>>>(
        n_pq_, j22_x_.data(), j21_tmp_.data(), y_q_device);
    CUDA_CHECK(cudaGetLastError());
}

void ImplicitSchurOperator::recover_solution(const double* rhs_full_device,
                                             const double* dvm_device,
                                             double* dx_full_device,
                                             SchurOperatorStats& stats,
                                             bool collect_timing_breakdown)
{
    ensure_ready("recover_solution");
    if (rhs_full_device == nullptr || dvm_device == nullptr || dx_full_device == nullptr) {
        throw std::runtime_error("ImplicitSchurOperator::recover_solution received null input");
    }

    time_stage(stats.recover_sec, collect_timing_breakdown, [&]() {
        record_default_input();
        j12_spmv_.apply_async(dvm_device, j12_x_.data(), chain_stream_);
        wait_default_on_chain();
        subtract_kernel<<<grid_for(n_pvpq_), kBlockSize>>>(
            n_pvpq_, rhs_full_device, j12_x_.data(), theta_rhs_.data());
        CUDA_CHECK(cudaGetLastError());

        record_default_input();
        solve_active_j11(theta_rhs_.data(), dx_full_device);
        wait_default_on_chain();

        copy_kernel<<<grid_for(n_pq_), kBlockSize>>>(
            n_pq_, dvm_device, dx_full_device + n_pvpq_);
        CUDA_CHECK(cudaGetLastError());
    });

    ++stats.j11_solve_calls;
    ++stats.spmv_calls;
}

}  // namespace exp_20260415::block_ilu
