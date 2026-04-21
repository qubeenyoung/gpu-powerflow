#pragma once

#include "utils/cuda_utils.hpp"

#include <cstdint>

namespace exp_20260414::amgx_v2 {

class DeviceVectorOps {
public:
    void negate(int32_t n, const double* input_device, double* output_device);
    double norm_inf(int32_t n, const double* input_device);

private:
    DeviceBuffer<double> d_partial_;
    DeviceBuffer<double> d_scratch_;
};

}  // namespace exp_20260414::amgx_v2
