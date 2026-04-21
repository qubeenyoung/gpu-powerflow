#pragma once

#include "assembly/reduced_jacobian_assembler.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <string>

namespace exp_20260415::block_ilu {

class CusparseIlu0Block {
public:
    explicit CusparseIlu0Block(std::string name);
    ~CusparseIlu0Block();

    CusparseIlu0Block(const CusparseIlu0Block&) = delete;
    CusparseIlu0Block& operator=(const CusparseIlu0Block&) = delete;

    void analyze(DeviceCsrMatrixView matrix);
    void factorize();
    void solve(const double* rhs_device, double* out_device);
    void set_stream(cudaStream_t stream);
    void clear_stream_noexcept();

    int32_t rows() const { return rows_; }
    int32_t nnz() const { return nnz_; }
    int32_t last_zero_pivot() const { return last_zero_pivot_; }
    bool ready() const { return ready_; }

private:
    void destroy();
    void check_analyzed(const char* method) const;
    void check_zero_pivot(const char* stage);
    void analyze_triangular_solves();
    void ensure_spsv_buffer_size(std::size_t required);

    std::string name_;
    int32_t rows_ = 0;
    int32_t nnz_ = 0;
    const int32_t* row_ptr_ = nullptr;
    const int32_t* col_idx_ = nullptr;
    double* values_ = nullptr;
    int32_t last_zero_pivot_ = -1;
    bool ready_ = false;
    bool factorized_ = false;

    cusparseHandle_t handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cusparseMatDescr_t legacy_descr_ = nullptr;
    csrilu02Info_t ilu_info_ = nullptr;
    DeviceBuffer<char> d_ilu_buffer_;

    cusparseSpMatDescr_t lower_mat_ = nullptr;
    cusparseSpMatDescr_t upper_mat_ = nullptr;
    cusparseSpSVDescr_t lower_spsv_ = nullptr;
    cusparseSpSVDescr_t upper_spsv_ = nullptr;
    cusparseDnVecDescr_t vec_rhs_ = nullptr;
    cusparseDnVecDescr_t vec_tmp_ = nullptr;
    cusparseDnVecDescr_t vec_out_ = nullptr;
    DeviceBuffer<double> d_tmp_;
    DeviceBuffer<double> d_dummy_rhs_;
    DeviceBuffer<double> d_dummy_out_;
    DeviceBuffer<char> d_spsv_buffer_;
    double alpha_ = 1.0;
};

}  // namespace exp_20260415::block_ilu
