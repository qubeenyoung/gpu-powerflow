#pragma once

#include "linear/amgx_preconditioner.hpp"
#include "model/bus_local_index.hpp"
#include "model/bus_local_jacobian_pattern.hpp"

#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <vector>

namespace exp_20260414::amgx_v2 {

// Owns the device CSR matrix used by AMGX.
//
// Host work is limited to analyze-time sparsity bookkeeping. Once analyze()
// returns, row/column maps and values live on device. Each assemble() call only
// launches kernels and does not copy Jacobian values back to the host.
class BusLocalJacobianAssembler {
public:
    void analyze(const BusLocalIndex& index,
                 const BusLocalJacobianPattern& pattern,
                 const std::vector<int32_t>& ybus_row_ptr,
                 const std::vector<int32_t>& ybus_col_idx);

    void assemble(const double* ybus_re_device,
                  const double* ybus_im_device,
                  const double* voltage_re_device,
                  const double* voltage_im_device);

    CsrMatrixView device_matrix_view() const;
    void download_values(std::vector<double>& values) const;

    int32_t dim() const { return dim_; }
    int32_t nnz() const { return nnz_; }

private:
    int32_t dim_ = 0;
    int32_t nnz_ = 0;
    int32_t ybus_nnz_ = 0;

    DeviceBuffer<int32_t> d_row_ptr_;
    DeviceBuffer<int32_t> d_col_idx_;
    DeviceBuffer<double> d_values_;

    DeviceBuffer<int32_t> d_y_row_;
    DeviceBuffer<int32_t> d_y_col_;
    DeviceBuffer<int32_t> d_map11_;
    DeviceBuffer<int32_t> d_map21_;
    DeviceBuffer<int32_t> d_map12_;
    DeviceBuffer<int32_t> d_map22_;
    DeviceBuffer<int32_t> d_diag11_;
    DeviceBuffer<int32_t> d_diag21_;
    DeviceBuffer<int32_t> d_diag12_;
    DeviceBuffer<int32_t> d_diag22_;
    DeviceBuffer<int32_t> d_fixed_identity_pos_;
};

}  // namespace exp_20260414::amgx_v2
