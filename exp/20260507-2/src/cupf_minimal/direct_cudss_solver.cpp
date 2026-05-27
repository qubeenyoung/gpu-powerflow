#include "cupf_minimal/direct_cudss_solver.hpp"

// experimental minimal cuPF NR port

#include "cuiter/common/cuda_utils.hpp"

#include <chrono>
#include <stdexcept>

namespace cupf_minimal {
namespace {

#define CUPF_MINIMAL_CUDSS_CHECK(call) \
    ::cupf_minimal::DirectCudssSolver::check((call), #call, __FILE__, __LINE__)

template <typename Fn>
double timed_phase(Fn&& fn)
{
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    fn();
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

const char* cudss_status_name(cudssStatus_t status)
{
    switch (status) {
    case CUDSS_STATUS_SUCCESS:
        return "CUDSS_STATUS_SUCCESS";
    case CUDSS_STATUS_NOT_INITIALIZED:
        return "CUDSS_STATUS_NOT_INITIALIZED";
    case CUDSS_STATUS_ALLOC_FAILED:
        return "CUDSS_STATUS_ALLOC_FAILED";
    case CUDSS_STATUS_INVALID_VALUE:
        return "CUDSS_STATUS_INVALID_VALUE";
    case CUDSS_STATUS_NOT_SUPPORTED:
        return "CUDSS_STATUS_NOT_SUPPORTED";
    case CUDSS_STATUS_EXECUTION_FAILED:
        return "CUDSS_STATUS_EXECUTION_FAILED";
    case CUDSS_STATUS_INTERNAL_ERROR:
        return "CUDSS_STATUS_INTERNAL_ERROR";
    }
    return "CUDSS_STATUS_UNKNOWN";
}

}  // namespace

DirectCudssSolver::DirectCudssSolver()
{
    CUPF_MINIMAL_CUDSS_CHECK(cudssCreate(&handle_));
    CUPF_MINIMAL_CUDSS_CHECK(cudssConfigCreate(&config_));
    CUPF_MINIMAL_CUDSS_CHECK(cudssDataCreate(handle_, &data_));
}

DirectCudssSolver::~DirectCudssSolver()
{
    destroy_matrices();
    if (data_ != nullptr) {
        cudssDataDestroy(handle_, data_);
        data_ = nullptr;
    }
    if (config_ != nullptr) {
        cudssConfigDestroy(config_);
        config_ = nullptr;
    }
    if (handle_ != nullptr) {
        cudssDestroy(handle_);
        handle_ = nullptr;
    }
}

void DirectCudssSolver::check(cudssStatus_t status, const char* call, const char* file, int line)
{
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuDSS error at ") + file + ":" +
                                 std::to_string(line) + " in " + call + " - " +
                                 cudss_status_name(status));
    }
}

void DirectCudssSolver::destroy_matrices()
{
    if (matrix_ != nullptr) {
        cudssMatrixDestroy(matrix_);
        matrix_ = nullptr;
    }
    if (rhs_ != nullptr) {
        cudssMatrixDestroy(rhs_);
        rhs_ = nullptr;
    }
    if (solution_ != nullptr) {
        cudssMatrixDestroy(solution_);
        solution_ = nullptr;
    }
    initialized_ = false;
}

void DirectCudssSolver::initialize(const cuiter::CsrMatrix& pattern,
                                   const int32_t* d_row_ptr,
                                   const int32_t* d_col_idx,
                                   const double* d_values,
                                   const double* d_rhs,
                                   double* d_x)
{
    if (pattern.rows <= 0 || pattern.rows != pattern.cols || pattern.nnz() <= 0 ||
        d_row_ptr == nullptr || d_col_idx == nullptr || d_values == nullptr ||
        d_rhs == nullptr || d_x == nullptr) {
        throw std::runtime_error("DirectCudssSolver::initialize: invalid input");
    }
    destroy_matrices();
    CUPF_MINIMAL_CUDSS_CHECK(cudssMatrixCreateCsr(&matrix_,
                                                  pattern.rows,
                                                  pattern.cols,
                                                  static_cast<int64_t>(pattern.nnz()),
                                                  const_cast<int32_t*>(d_row_ptr),
                                                  nullptr,
                                                  const_cast<int32_t*>(d_col_idx),
                                                  const_cast<double*>(d_values),
                                                  CUDA_R_32I,
                                                  CUDA_R_64F,
                                                  CUDSS_MTYPE_GENERAL,
                                                  CUDSS_MVIEW_FULL,
                                                  CUDSS_BASE_ZERO));
    CUPF_MINIMAL_CUDSS_CHECK(cudssMatrixCreateDn(&rhs_,
                                                 pattern.rows,
                                                 1,
                                                 pattern.rows,
                                                 const_cast<double*>(d_rhs),
                                                 CUDA_R_64F,
                                                 CUDSS_LAYOUT_COL_MAJOR));
    CUPF_MINIMAL_CUDSS_CHECK(cudssMatrixCreateDn(&solution_,
                                                 pattern.rows,
                                                 1,
                                                 pattern.rows,
                                                 d_x,
                                                 CUDA_R_64F,
                                                 CUDSS_LAYOUT_COL_MAJOR));
    initialized_ = true;
}

double DirectCudssSolver::analyze()
{
    if (!initialized_) {
        throw std::runtime_error("DirectCudssSolver::analyze before initialize");
    }
    const double seconds = timed_phase([&] {
        CUPF_MINIMAL_CUDSS_CHECK(cudssExecute(handle_,
                                              CUDSS_PHASE_ANALYSIS,
                                              config_,
                                              data_,
                                              matrix_,
                                              solution_,
                                              rhs_));
    });
    timings_.analyze_seconds += seconds;
    return seconds;
}

double DirectCudssSolver::factorize()
{
    if (!initialized_) {
        throw std::runtime_error("DirectCudssSolver::factorize before initialize");
    }
    const double seconds = timed_phase([&] {
        CUPF_MINIMAL_CUDSS_CHECK(cudssExecute(handle_,
                                              CUDSS_PHASE_FACTORIZATION,
                                              config_,
                                              data_,
                                              matrix_,
                                              solution_,
                                              rhs_));
    });
    timings_.factorize_seconds += seconds;
    return seconds;
}

double DirectCudssSolver::solve()
{
    if (!initialized_) {
        throw std::runtime_error("DirectCudssSolver::solve before initialize");
    }
    const double seconds = timed_phase([&] {
        CUPF_MINIMAL_CUDSS_CHECK(cudssExecute(handle_,
                                              CUDSS_PHASE_SOLVE,
                                              config_,
                                              data_,
                                              matrix_,
                                              solution_,
                                              rhs_));
    });
    timings_.solve_seconds += seconds;
    return seconds;
}

void DirectCudssSolver::set_stream(cudaStream_t stream)
{
    CUPF_MINIMAL_CUDSS_CHECK(cudssSetStream(handle_, stream));
}

void DirectCudssSolver::factorize_async()
{
    if (!initialized_) {
        throw std::runtime_error("DirectCudssSolver::factorize_async before initialize");
    }
    CUPF_MINIMAL_CUDSS_CHECK(cudssExecute(handle_,
                                          CUDSS_PHASE_FACTORIZATION,
                                          config_,
                                          data_,
                                          matrix_,
                                          solution_,
                                          rhs_));
}

void DirectCudssSolver::solve_async()
{
    if (!initialized_) {
        throw std::runtime_error("DirectCudssSolver::solve_async before initialize");
    }
    CUPF_MINIMAL_CUDSS_CHECK(cudssExecute(handle_,
                                          CUDSS_PHASE_SOLVE,
                                          config_,
                                          data_,
                                          matrix_,
                                          solution_,
                                          rhs_));
}

}  // namespace cupf_minimal
