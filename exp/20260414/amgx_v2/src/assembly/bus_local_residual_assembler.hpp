#pragma once

#include "model/bus_local_index.hpp"

#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <vector>

namespace exp_20260414::amgx_v2 {

// Builds the augmented bus-local residual vector on device.
//
// Active slots receive the physical mismatch. Fixed slots receive zero, so the
// paired identity Jacobian rows enforce a zero update.
class BusLocalResidualAssembler {
public:
    void analyze(const BusLocalIndex& index);

    void assemble(const int32_t* ybus_row_ptr_device,
                  const int32_t* ybus_col_idx_device,
                  const double* ybus_re_device,
                  const double* ybus_im_device,
                  const double* voltage_re_device,
                  const double* voltage_im_device,
                  const double* sbus_re_device,
                  const double* sbus_im_device);

    const double* values_device() const { return d_values_.data(); }
    double* values_device() { return d_values_.data(); }
    int32_t dim() const { return dim_; }

    void download_values(std::vector<double>& values) const;

private:
    int32_t n_bus_ = 0;
    int32_t dim_ = 0;

    DeviceBuffer<int32_t> d_theta_slot_;
    DeviceBuffer<int32_t> d_vm_slot_;
    DeviceBuffer<int32_t> d_p_active_;
    DeviceBuffer<int32_t> d_q_active_;
    DeviceBuffer<double> d_values_;
};

}  // namespace exp_20260414::amgx_v2
