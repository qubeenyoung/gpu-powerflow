#include "linear/cusparse_ilu0_block.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace exp_20260415::block_ilu {
namespace {

const char* status_name(cusparseStatus_t status)
{
    switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
        return "success";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "not_initialized";
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "alloc_failed";
    case CUSPARSE_STATUS_INVALID_VALUE:
        return "invalid_value";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "arch_mismatch";
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "mapping_error";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "execution_failed";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "internal_error";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "matrix_type_not_supported";
    case CUSPARSE_STATUS_ZERO_PIVOT:
        return "zero_pivot";
    case CUSPARSE_STATUS_NOT_SUPPORTED:
        return "not_supported";
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        return "insufficient_resources";
    default:
        return "unknown";
    }
}

void check_status(cusparseStatus_t status, const std::string& message)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(message + ": " + status_name(status));
    }
}

void set_spmat_attributes(cusparseSpMatDescr_t mat,
                          cusparseFillMode_t fill_mode,
                          cusparseDiagType_t diag_type)
{
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(mat,
                                             CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_mode,
                                             sizeof(fill_mode)));
    CUSPARSE_CHECK(cusparseSpMatSetAttribute(mat,
                                             CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_type,
                                             sizeof(diag_type)));
}

}  // namespace

CusparseIlu0Block::CusparseIlu0Block(std::string name)
    : name_(std::move(name))
{
    CUSPARSE_CHECK(cusparseCreate(&handle_));
}

CusparseIlu0Block::~CusparseIlu0Block()
{
    destroy();
}

void CusparseIlu0Block::set_stream(cudaStream_t stream)
{
    stream_ = stream;
    CUSPARSE_CHECK(cusparseSetStream(handle_, stream_));
}

void CusparseIlu0Block::clear_stream_noexcept()
{
    stream_ = nullptr;
    if (handle_ != nullptr) {
        cusparseSetStream(handle_, nullptr);
    }
}

void CusparseIlu0Block::destroy()
{
    if (vec_out_ != nullptr) {
        cusparseDestroyDnVec(vec_out_);
        vec_out_ = nullptr;
    }
    if (vec_tmp_ != nullptr) {
        cusparseDestroyDnVec(vec_tmp_);
        vec_tmp_ = nullptr;
    }
    if (vec_rhs_ != nullptr) {
        cusparseDestroyDnVec(vec_rhs_);
        vec_rhs_ = nullptr;
    }
    if (upper_spsv_ != nullptr) {
        cusparseSpSV_destroyDescr(upper_spsv_);
        upper_spsv_ = nullptr;
    }
    if (lower_spsv_ != nullptr) {
        cusparseSpSV_destroyDescr(lower_spsv_);
        lower_spsv_ = nullptr;
    }
    if (upper_mat_ != nullptr) {
        cusparseDestroySpMat(upper_mat_);
        upper_mat_ = nullptr;
    }
    if (lower_mat_ != nullptr) {
        cusparseDestroySpMat(lower_mat_);
        lower_mat_ = nullptr;
    }
    if (ilu_info_ != nullptr) {
        cusparseDestroyCsrilu02Info(ilu_info_);
        ilu_info_ = nullptr;
    }
    if (legacy_descr_ != nullptr) {
        cusparseDestroyMatDescr(legacy_descr_);
        legacy_descr_ = nullptr;
    }
    if (handle_ != nullptr) {
        cusparseDestroy(handle_);
        handle_ = nullptr;
    }
    ready_ = false;
    factorized_ = false;
}

void CusparseIlu0Block::check_analyzed(const char* method) const
{
    if (!ready_ || rows_ <= 0 || nnz_ <= 0 || row_ptr_ == nullptr ||
        col_idx_ == nullptr || values_ == nullptr) {
        throw std::runtime_error(name_ + "::" + method + " called before analyze");
    }
}

void CusparseIlu0Block::analyze(DeviceCsrMatrixView matrix)
{
    if (matrix.rows <= 0 || matrix.rows != matrix.cols || matrix.nnz <= 0 ||
        matrix.row_ptr == nullptr || matrix.col_idx == nullptr || matrix.values == nullptr) {
        throw std::runtime_error(name_ + "::analyze received an invalid matrix");
    }

    rows_ = matrix.rows;
    nnz_ = matrix.nnz;
    row_ptr_ = matrix.row_ptr;
    col_idx_ = matrix.col_idx;
    values_ = matrix.values;
    last_zero_pivot_ = -1;
    factorized_ = false;

    CUSPARSE_CHECK(cusparseCreateMatDescr(&legacy_descr_));
    CUSPARSE_CHECK(cusparseSetMatType(legacy_descr_, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(legacy_descr_, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseCreateCsrilu02Info(&ilu_info_));

    int ilu_buffer_bytes = 0;
    CUSPARSE_CHECK(cusparseDcsrilu02_bufferSize(handle_,
                                                rows_,
                                                nnz_,
                                                legacy_descr_,
                                                values_,
                                                row_ptr_,
                                                col_idx_,
                                                ilu_info_,
                                                &ilu_buffer_bytes));
    d_ilu_buffer_.resize(static_cast<std::size_t>(ilu_buffer_bytes));

    CUSPARSE_CHECK(cusparseDcsrilu02_analysis(handle_,
                                              rows_,
                                              nnz_,
                                              legacy_descr_,
                                              values_,
                                              row_ptr_,
                                              col_idx_,
                                              ilu_info_,
                                              CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                              d_ilu_buffer_.data()));
    check_zero_pivot("analysis");

    CUSPARSE_CHECK(cusparseCreateCsr(&lower_mat_,
                                     rows_,
                                     rows_,
                                     nnz_,
                                     const_cast<int32_t*>(row_ptr_),
                                     const_cast<int32_t*>(col_idx_),
                                     values_,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&upper_mat_,
                                     rows_,
                                     rows_,
                                     nnz_,
                                     const_cast<int32_t*>(row_ptr_),
                                     const_cast<int32_t*>(col_idx_),
                                     values_,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));

    set_spmat_attributes(lower_mat_,
                         CUSPARSE_FILL_MODE_LOWER,
                         CUSPARSE_DIAG_TYPE_UNIT);
    set_spmat_attributes(upper_mat_,
                         CUSPARSE_FILL_MODE_UPPER,
                         CUSPARSE_DIAG_TYPE_NON_UNIT);

    CUSPARSE_CHECK(cusparseSpSV_createDescr(&lower_spsv_));
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&upper_spsv_));

    d_tmp_.resize(static_cast<std::size_t>(rows_));
    d_dummy_rhs_.resize(static_cast<std::size_t>(rows_));
    d_dummy_out_.resize(static_cast<std::size_t>(rows_));

    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_rhs_, rows_, d_dummy_rhs_.data(), CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_tmp_, rows_, d_tmp_.data(), CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_out_, rows_, d_dummy_out_.data(), CUDA_R_64F));

    ready_ = true;
}

void CusparseIlu0Block::check_zero_pivot(const char* stage)
{
    last_zero_pivot_ = -1;
    const cusparseStatus_t status =
        cusparseXcsrilu02_zeroPivot(handle_, ilu_info_, &last_zero_pivot_);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        throw std::runtime_error(name_ + " ILU0 " + stage +
                                 " zero pivot at row " + std::to_string(last_zero_pivot_));
    }
    check_status(status, name_ + " ILU0 " + stage + " zero-pivot query failed");
}

void CusparseIlu0Block::ensure_spsv_buffer_size(std::size_t required)
{
    if (d_spsv_buffer_.size() < required) {
        d_spsv_buffer_.resize(required);
    }
}

void CusparseIlu0Block::analyze_triangular_solves()
{
    std::size_t lower_bytes = 0;
    std::size_t upper_bytes = 0;
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(handle_,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_,
                                           lower_mat_,
                                           vec_rhs_,
                                           vec_tmp_,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           lower_spsv_,
                                           &lower_bytes));
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(handle_,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_,
                                           upper_mat_,
                                           vec_tmp_,
                                           vec_out_,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           upper_spsv_,
                                           &upper_bytes));
    ensure_spsv_buffer_size(std::max(lower_bytes, upper_bytes));

    CUSPARSE_CHECK(cusparseSpSV_analysis(handle_,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha_,
                                         lower_mat_,
                                         vec_rhs_,
                                         vec_tmp_,
                                         CUDA_R_64F,
                                         CUSPARSE_SPSV_ALG_DEFAULT,
                                         lower_spsv_,
                                         d_spsv_buffer_.data()));
    CUSPARSE_CHECK(cusparseSpSV_analysis(handle_,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha_,
                                         upper_mat_,
                                         vec_tmp_,
                                         vec_out_,
                                         CUDA_R_64F,
                                         CUSPARSE_SPSV_ALG_DEFAULT,
                                         upper_spsv_,
                                         d_spsv_buffer_.data()));
}

void CusparseIlu0Block::factorize()
{
    check_analyzed("factorize");

    CUSPARSE_CHECK(cusparseDcsrilu02(handle_,
                                     rows_,
                                     nnz_,
                                     legacy_descr_,
                                     values_,
                                     row_ptr_,
                                     col_idx_,
                                     ilu_info_,
                                     CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                     d_ilu_buffer_.data()));
    check_zero_pivot("factorization");
    analyze_triangular_solves();
    factorized_ = true;
}

void CusparseIlu0Block::solve(const double* rhs_device, double* out_device)
{
    check_analyzed("solve");
    if (!factorized_) {
        throw std::runtime_error(name_ + "::solve called before factorize");
    }
    if (rhs_device == nullptr || out_device == nullptr) {
        throw std::runtime_error(name_ + "::solve received null device pointer");
    }

    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_rhs_, const_cast<double*>(rhs_device)));
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_tmp_, d_tmp_.data()));
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_out_, out_device));

    CUSPARSE_CHECK(cusparseSpSV_solve(handle_,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha_,
                                      lower_mat_,
                                      vec_rhs_,
                                      vec_tmp_,
                                      CUDA_R_64F,
                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                      lower_spsv_));
    CUSPARSE_CHECK(cusparseSpSV_solve(handle_,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha_,
                                      upper_mat_,
                                      vec_tmp_,
                                      vec_out_,
                                      CUDA_R_64F,
                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                      upper_spsv_));
}

}  // namespace exp_20260415::block_ilu
