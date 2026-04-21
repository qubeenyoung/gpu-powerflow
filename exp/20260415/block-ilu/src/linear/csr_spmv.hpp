#pragma once

#include "assembly/reduced_jacobian_assembler.hpp"

namespace exp_20260415::block_ilu {

class CsrSpmv {
public:
    CsrSpmv();
    ~CsrSpmv();

    CsrSpmv(const CsrSpmv&) = delete;
    CsrSpmv& operator=(const CsrSpmv&) = delete;

    void bind(DeviceCsrMatrixView matrix);
    void enable_parallel_row_split(int32_t row_split);
    void disable_parallel_row_split();
    void apply(const double* x_device, double* y_device) const;
    void apply_async(const double* x_device, double* y_device, cudaStream_t stream) const;
    int32_t rows() const { return matrix_.rows; }
    int32_t cols() const { return matrix_.cols; }
    bool parallel_row_split_enabled() const { return row_split_ > 0; }

private:
    DeviceCsrMatrixView matrix_;
    int32_t row_split_ = 0;
    mutable cudaStream_t top_stream_ = nullptr;
    mutable cudaStream_t bottom_stream_ = nullptr;
    mutable cudaEvent_t input_ready_ = nullptr;
    mutable cudaEvent_t top_done_ = nullptr;
    mutable cudaEvent_t bottom_done_ = nullptr;
};

class CsrSpmvF32 {
public:
    CsrSpmvF32();
    ~CsrSpmvF32();

    CsrSpmvF32(const CsrSpmvF32&) = delete;
    CsrSpmvF32& operator=(const CsrSpmvF32&) = delete;

    void bind(DeviceCsrMatrixViewF32 matrix);
    void apply_async(const float* x_device, float* y_device, cudaStream_t stream) const;
    int32_t rows() const { return matrix_.rows; }
    int32_t cols() const { return matrix_.cols; }

private:
    DeviceCsrMatrixViewF32 matrix_;
};

}  // namespace exp_20260415::block_ilu
