#pragma once

#include "linear/cusparse_ilu0_block.hpp"

#include <cstdint>

namespace exp_20260415::block_ilu {

class BlockIluPreconditioner {
public:
    BlockIluPreconditioner();
    ~BlockIluPreconditioner();

    BlockIluPreconditioner(const BlockIluPreconditioner&) = delete;
    BlockIluPreconditioner& operator=(const BlockIluPreconditioner&) = delete;

    void analyze(DeviceCsrMatrixView j11, DeviceCsrMatrixView j22, int32_t n_pvpq, int32_t n_pq);
    void factorize();
    void apply(const double* rhs_device, double* out_device);

    int32_t n_pvpq() const { return n_pvpq_; }
    int32_t n_pq() const { return n_pq_; }
    int32_t dim() const { return n_pvpq_ + n_pq_; }
    int32_t j11_zero_pivot() const { return j11_.last_zero_pivot(); }
    int32_t j22_zero_pivot() const { return j22_.last_zero_pivot(); }

private:
    int32_t n_pvpq_ = 0;
    int32_t n_pq_ = 0;
    CusparseIlu0Block j11_;
    CusparseIlu0Block j22_;
    cudaStream_t j11_stream_ = nullptr;
    cudaStream_t j22_stream_ = nullptr;
    cudaEvent_t input_ready_ = nullptr;
    cudaEvent_t j11_done_ = nullptr;
    cudaEvent_t j22_done_ = nullptr;
};

}  // namespace exp_20260415::block_ilu
