#pragma once

#include "model/bus_local_index.hpp"

#include "utils/cuda_utils.hpp"

#include <cstdint>

namespace exp_20260414::amgx_v2 {

// Applies an augmented bus-local Newton step to the physical voltage state.
//
// The dx vector has two formal slots per bus. Only active slots are applied:
// theta is active for PV/PQ buses, Vm is active only for PQ buses. Fixed slots
// are ignored even though they exist in the augmented linear system.
class BusLocalVoltageUpdate {
public:
    void analyze(const BusLocalIndex& index);

    void apply(const double* dx_device,
               double* voltage_re_device,
               double* voltage_im_device);

private:
    int32_t n_bus_ = 0;

    DeviceBuffer<int32_t> d_theta_slot_;
    DeviceBuffer<int32_t> d_vm_slot_;
    DeviceBuffer<int32_t> d_theta_active_;
    DeviceBuffer<int32_t> d_vm_active_;
    DeviceBuffer<double> d_va_;
    DeviceBuffer<double> d_vm_;
};

}  // namespace exp_20260414::amgx_v2
