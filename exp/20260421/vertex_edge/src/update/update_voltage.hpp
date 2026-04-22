#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace exp20260421::vertex_edge::voltage_update {

// Voltage state is split rectangular FP64 and batch-major device-resident.
// The Newton update dx remains FP32 because cuDSS solves the FP32 Jacobian.

void init_voltage(const double* v0_re,
                  const double* v0_im,
                  double* v_re,
                  double* v_im,
                  int32_t n_bus,
                  int32_t batch_size,
                  cudaStream_t stream = nullptr);

void update_voltage(double* v_re,
                    double* v_im,
                    const float* dx,
                    const int32_t* pvpq,
                    int32_t n_pvpq,
                    const int32_t* pq,
                    int32_t n_pq,
                    int32_t n_bus,
                    int32_t batch_size,
                    cudaStream_t stream = nullptr);

}  // namespace exp20260421::vertex_edge::voltage_update
