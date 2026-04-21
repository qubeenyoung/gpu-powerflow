#include "linear/block_ilu_preconditioner.hpp"

#include <stdexcept>

namespace exp_20260415::block_ilu {

BlockIluPreconditioner::BlockIluPreconditioner()
    : j11_("J11")
    , j22_("J22")
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&j11_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&j22_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&input_ready_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&j11_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&j22_done_, cudaEventDisableTiming));
    j11_.set_stream(j11_stream_);
    j22_.set_stream(j22_stream_);
}

BlockIluPreconditioner::~BlockIluPreconditioner()
{
    if (j11_stream_ != nullptr) {
        cudaStreamSynchronize(j11_stream_);
    }
    if (j22_stream_ != nullptr) {
        cudaStreamSynchronize(j22_stream_);
    }
    j11_.clear_stream_noexcept();
    j22_.clear_stream_noexcept();
    if (j11_done_ != nullptr) {
        cudaEventDestroy(j11_done_);
        j11_done_ = nullptr;
    }
    if (j22_done_ != nullptr) {
        cudaEventDestroy(j22_done_);
        j22_done_ = nullptr;
    }
    if (input_ready_ != nullptr) {
        cudaEventDestroy(input_ready_);
        input_ready_ = nullptr;
    }
    if (j11_stream_ != nullptr) {
        cudaStreamDestroy(j11_stream_);
        j11_stream_ = nullptr;
    }
    if (j22_stream_ != nullptr) {
        cudaStreamDestroy(j22_stream_);
        j22_stream_ = nullptr;
    }
}

void BlockIluPreconditioner::analyze(DeviceCsrMatrixView j11,
                                     DeviceCsrMatrixView j22,
                                     int32_t n_pvpq,
                                     int32_t n_pq)
{
    if (n_pvpq <= 0 || n_pq <= 0 || j11.rows != n_pvpq || j22.rows != n_pq) {
        throw std::runtime_error("BlockIluPreconditioner::analyze received invalid dimensions");
    }
    n_pvpq_ = n_pvpq;
    n_pq_ = n_pq;
    j11_.analyze(j11);
    j22_.analyze(j22);
}

void BlockIluPreconditioner::factorize()
{
    CUDA_CHECK(cudaEventRecord(input_ready_, nullptr));
    CUDA_CHECK(cudaStreamWaitEvent(j11_stream_, input_ready_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(j22_stream_, input_ready_, 0));
    j11_.factorize();
    j22_.factorize();
    CUDA_CHECK(cudaEventRecord(j11_done_, j11_stream_));
    CUDA_CHECK(cudaEventRecord(j22_done_, j22_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, j11_done_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, j22_done_, 0));
}

void BlockIluPreconditioner::apply(const double* rhs_device, double* out_device)
{
    if (n_pvpq_ <= 0 || n_pq_ <= 0) {
        throw std::runtime_error("BlockIluPreconditioner::apply called before analyze");
    }
    if (rhs_device == nullptr || out_device == nullptr) {
        throw std::runtime_error("BlockIluPreconditioner::apply received null input");
    }

    CUDA_CHECK(cudaEventRecord(input_ready_, nullptr));
    CUDA_CHECK(cudaStreamWaitEvent(j11_stream_, input_ready_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(j22_stream_, input_ready_, 0));

    j11_.solve(rhs_device, out_device);
    j22_.solve(rhs_device + n_pvpq_, out_device + n_pvpq_);

    CUDA_CHECK(cudaEventRecord(j11_done_, j11_stream_));
    CUDA_CHECK(cudaEventRecord(j22_done_, j22_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, j11_done_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, j22_done_, 0));
}

}  // namespace exp_20260415::block_ilu
