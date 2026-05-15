#pragma once

// experimental minimal cuPF NR port

#include "cuiter/core/csr_matrix.hpp"

#include <cudss.h>

#include <cstdint>
#include <string>

namespace cupf_minimal {

struct DirectCudssTimings {
    double analyze_seconds = 0.0;
    double factorize_seconds = 0.0;
    double solve_seconds = 0.0;
};

class DirectCudssSolver {
public:
    DirectCudssSolver();
    ~DirectCudssSolver();

    DirectCudssSolver(const DirectCudssSolver&) = delete;
    DirectCudssSolver& operator=(const DirectCudssSolver&) = delete;

    void initialize(const cuiter::CsrMatrix& pattern,
                    const int32_t* d_row_ptr,
                    const int32_t* d_col_idx,
                    const double* d_values,
                    const double* d_rhs,
                    double* d_x);
    double analyze();
    double factorize();
    double solve();
    void set_stream(cudaStream_t stream);
    void factorize_async();
    void solve_async();

    const DirectCudssTimings& timings() const { return timings_; }
    static void check(cudssStatus_t status, const char* call, const char* file, int line);

private:
    void destroy_matrices();

    cudssHandle_t handle_ = nullptr;
    cudssConfig_t config_ = nullptr;
    cudssData_t data_ = nullptr;
    cudssMatrix_t matrix_ = nullptr;
    cudssMatrix_t rhs_ = nullptr;
    cudssMatrix_t solution_ = nullptr;
    DirectCudssTimings timings_;
    bool initialized_ = false;
};

}  // namespace cupf_minimal
