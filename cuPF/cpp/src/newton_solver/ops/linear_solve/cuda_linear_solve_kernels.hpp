#pragma once

// ---------------------------------------------------------------------------
// cuda_linear_solve_kernels.hpp
//
// Declarations for the small CUDA support kernels used by the linear-solve and
// torch-bridge paths. Definitions are split across two TUs:
//   linear_solve_kernels.cu  - launch_prepare_rhs, launch_transpose_csr_values
//   torch_bridge_kernels.cu  - launch_gather_adjoint_rhs,
//                              launch_project_load_gradients,
//                              launch_set_pf_inputs_from_load,
//                              launch_copy_voltage_outputs
// Callers (cuda_cudss.cpp, torch_bridge.cpp) include only this header.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include <cstdint>

void launch_prepare_rhs(const double* src, float* dst, int32_t count);
void launch_transpose_csr_values(const double* src,
                                 double* dst,
                                 const int32_t* src_to_transpose_pos,
                                 int32_t nnz,
                                 int32_t batch_size);
void launch_transpose_csr_values(const float* src,
                                 float* dst,
                                 const int32_t* src_to_transpose_pos,
                                 int32_t nnz,
                                 int32_t batch_size);

template <typename T>
void launch_gather_adjoint_rhs(const T* grad_va,
                               const T* grad_vm,
                               T* grad_state,
                               const int32_t* pv,
                               int32_t n_pv,
                               const int32_t* pq,
                               int32_t n_pq,
                               int32_t n_bus,
                               int32_t batch_size);

template <typename T>
void launch_project_load_gradients(const T* lambda,
                                   T* grad_load_p,
                                   T* grad_load_q,
                                   const int32_t* pv,
                                   int32_t n_pv,
                                   const int32_t* pq,
                                   int32_t n_pq,
                                   int32_t n_bus,
                                   int32_t batch_size);

template <typename InputT, typename StorageT>
void launch_set_pf_inputs_from_load(const InputT* sbus_base_re,
                                    const InputT* sbus_base_im,
                                    const InputT* load_p,
                                    const InputT* load_q,
                                    const InputT* v0_va,
                                    const InputT* v0_vm,
                                    StorageT* sbus_re,
                                    StorageT* sbus_im,
                                    StorageT* va,
                                    StorageT* vm,
                                    StorageT* v_re,
                                    StorageT* v_im,
                                    int32_t n_bus,
                                    int32_t batch_size);

template <typename StorageT, typename OutputT>
void launch_copy_voltage_outputs(const StorageT* va,
                                 const StorageT* vm,
                                 OutputT* va_out,
                                 OutputT* vm_out,
                                 int32_t total_bus);

#endif  // CUPF_WITH_CUDA
