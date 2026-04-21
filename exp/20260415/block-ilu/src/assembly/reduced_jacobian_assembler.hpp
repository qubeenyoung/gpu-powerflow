#pragma once

#include "model/reduced_jacobian.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <vector>

namespace exp_20260415::block_ilu {

struct DeviceCsrMatrixView {
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t nnz = 0;
    const int32_t* row_ptr = nullptr;
    const int32_t* col_idx = nullptr;
    double* values = nullptr;
};

struct DeviceCsrMatrixViewF32 {
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t nnz = 0;
    const int32_t* row_ptr = nullptr;
    const int32_t* col_idx = nullptr;
    float* values = nullptr;
};

class ReducedJacobianAssembler {
public:
    void analyze(const ReducedJacobianPatterns& patterns);

    void assemble(const double* ybus_re_device,
                  const double* ybus_im_device,
                  const double* voltage_re_device,
                  const double* voltage_im_device);

    DeviceCsrMatrixView full_view() const;
    DeviceCsrMatrixView j11_view() const;
    DeviceCsrMatrixView j12_view() const;
    DeviceCsrMatrixView j21_view() const;
    DeviceCsrMatrixView j22_view() const;

    const ReducedJacobianPatterns& host_patterns() const;

    void download_full_values(std::vector<double>& values) const;
    void download_j11_values(std::vector<double>& values) const;
    void download_j12_values(std::vector<double>& values) const;
    void download_j21_values(std::vector<double>& values) const;
    void download_j22_values(std::vector<double>& values) const;

private:
    ReducedJacobianPatterns patterns_;
    int32_t ybus_nnz_ = 0;

    DeviceBuffer<int32_t> d_ybus_row_;
    DeviceBuffer<int32_t> d_ybus_col_;

    DeviceBuffer<int32_t> d_full_row_ptr_;
    DeviceBuffer<int32_t> d_full_col_idx_;
    DeviceBuffer<double> d_full_values_;

    DeviceBuffer<int32_t> d_j11_row_ptr_;
    DeviceBuffer<int32_t> d_j11_col_idx_;
    DeviceBuffer<double> d_j11_values_;

    DeviceBuffer<int32_t> d_j12_row_ptr_;
    DeviceBuffer<int32_t> d_j12_col_idx_;
    DeviceBuffer<double> d_j12_values_;

    DeviceBuffer<int32_t> d_j21_row_ptr_;
    DeviceBuffer<int32_t> d_j21_col_idx_;
    DeviceBuffer<double> d_j21_values_;

    DeviceBuffer<int32_t> d_j22_row_ptr_;
    DeviceBuffer<int32_t> d_j22_col_idx_;
    DeviceBuffer<double> d_j22_values_;

    DeviceBuffer<int32_t> d_map11_;
    DeviceBuffer<int32_t> d_map12_;
    DeviceBuffer<int32_t> d_map21_;
    DeviceBuffer<int32_t> d_map22_;
    DeviceBuffer<int32_t> d_diag11_;
    DeviceBuffer<int32_t> d_diag12_;
    DeviceBuffer<int32_t> d_diag21_;
    DeviceBuffer<int32_t> d_diag22_;

    DeviceBuffer<int32_t> d_map_b11_;
    DeviceBuffer<int32_t> d_diag_b11_;
    DeviceBuffer<int32_t> d_map_b12_;
    DeviceBuffer<int32_t> d_diag_b12_;
    DeviceBuffer<int32_t> d_map_b21_;
    DeviceBuffer<int32_t> d_diag_b21_;
    DeviceBuffer<int32_t> d_map_b22_;
    DeviceBuffer<int32_t> d_diag_b22_;
};

}  // namespace exp_20260415::block_ilu
