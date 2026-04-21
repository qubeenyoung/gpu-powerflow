#pragma once

#include "linear/amgx_preconditioner.hpp"
#include "linear/bus_block_jacobi_preconditioner.hpp"
#include "model/bus_local_index.hpp"

#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <vector>

namespace exp_20260414::amgx_v2 {

// Directly assembles the AMGX-facing fixed 2x2 bus-block Jacobian.
//
// The scalar Jacobian remains the linear operator for FGMRES. This matrix is a
// separate preconditioner representation with one block row per bus-local
// position and row-major dense blocks:
//
//   [ dP/dtheta  dP/dVm ]
//   [ dQ/dtheta  dQ/dVm ]
//
// Fixed slack/PV slots are represented as identity rows inside the diagonal
// 2x2 block.
class BusBlockJacobianAssembler {
public:
    void analyze(const BusLocalIndex& index,
                 const std::vector<int32_t>& ybus_row_ptr,
                 const std::vector<int32_t>& ybus_col_idx);

    void assemble(const double* ybus_re_device,
                  const double* ybus_im_device,
                  const double* voltage_re_device,
                  const double* voltage_im_device);

    BlockCsrMatrixView device_matrix_view() const;
    BusBlockJacobiView jacobi_view() const;

    int32_t rows() const { return block_rows_; }
    int32_t nnz() const { return block_nnz_; }

private:
    int32_t n_bus_ = 0;
    int32_t ybus_nnz_ = 0;
    int32_t block_rows_ = 0;
    int32_t block_nnz_ = 0;

    DeviceBuffer<int32_t> d_row_ptr_;
    DeviceBuffer<int32_t> d_col_idx_;
    DeviceBuffer<double> d_values_;

    DeviceBuffer<int32_t> d_y_row_;
    DeviceBuffer<int32_t> d_y_col_;
    DeviceBuffer<int32_t> d_edge_block_value_base_;
    DeviceBuffer<int32_t> d_diag_block_value_base_by_bus_;
    DeviceBuffer<int32_t> d_diag_block_value_base_by_position_;
    DeviceBuffer<int32_t> d_p_active_;
    DeviceBuffer<int32_t> d_q_active_;
    DeviceBuffer<int32_t> d_fixed_identity_value_pos_;
};

}  // namespace exp_20260414::amgx_v2
