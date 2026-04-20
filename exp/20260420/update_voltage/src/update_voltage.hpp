#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace exp20260420::voltage_update {

// All voltage state is FP32 and device-resident.
//
// v_re/v_im are the split-complex voltage used by mismatch and Jacobian.
// va/vm are the polar state updated by Newton dx.
// v_norm_re/v_norm_im store V / |V| for Jacobian dS/dVm terms.

void init_voltage(const float* v0_re,
                  const float* v0_im,
                  float* v_re,
                  float* v_im,
                  float* va,
                  float* vm,
                  float* v_norm_re,
                  float* v_norm_im,
                  int32_t n_bus,
                  cudaStream_t stream = nullptr);

void update_voltage(float* v_re,
                    float* v_im,
                    float* va,
                    float* vm,
                    float* v_norm_re,
                    float* v_norm_im,
                    const float* dx,
                    const int32_t* pvpq,
                    int32_t n_pvpq,
                    const int32_t* pq,
                    int32_t n_pq,
                    int32_t n_bus,
                    cudaStream_t stream = nullptr);

}  // namespace exp20260420::voltage_update
