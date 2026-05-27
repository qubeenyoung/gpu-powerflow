#pragma once

// experimental minimal cuPF NR port

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cupf_minimal {

void launch_compute_ibus(int32_t n_bus,
                         const int32_t* y_row_ptr,
                         const int32_t* y_col,
                         const double* y_re,
                         const double* y_im,
                         const double* v_re,
                         const double* v_im,
                         double* ibus_re,
                         double* ibus_im);

void launch_compute_mismatch_from_ibus(int32_t dimF,
                                       int32_t n_bus,
                                       int32_t n_pv,
                                       int32_t n_pq,
                                       const double* v_re,
                                       const double* v_im,
                                       const double* ibus_re,
                                       const double* ibus_im,
                                       const double* sbus_re,
                                       const double* sbus_im,
                                       const int32_t* pv,
                                       const int32_t* pq,
                                       double* F);

void launch_reduce_abs_max(int32_t n, const double* values, double* out);

void launch_fill_jacobian(int32_t nnz_ybus,
                          int32_t nnz_J,
                          int32_t n_bus,
                          const double* y_re,
                          const double* y_im,
                          const int32_t* y_row,
                          const int32_t* y_col,
                          const int32_t* y_row_ptr,
                          const double* v_re,
                          const double* v_im,
                          const double* vm,
                          const double* ibus_re,
                          const double* ibus_im,
                          const int32_t* map11,
                          const int32_t* map21,
                          const int32_t* map12,
                          const int32_t* map22,
                          const int32_t* diag11,
                          const int32_t* diag21,
                          const int32_t* diag12,
                          const int32_t* diag22,
                          double* J_values);

void launch_voltage_update(int32_t n_bus,
                           int32_t dimF,
                           int32_t n_pv,
                           int32_t n_pq,
                           double* va,
                           double* vm,
                           double* v_re,
                           double* v_im,
                           const double* dx,
                           const int32_t* pv,
                           const int32_t* pq,
                           double damping_factor);

void launch_count_nonfinite(int32_t n, const double* values, int32_t* count);

void launch_scatter_field_values(int32_t nnz,
                                 const int32_t* full_positions,
                                 const double* full_values,
                                 double* field_values,
                                 cudaStream_t stream = nullptr);

void launch_copy_field_rhs(int32_t n_p,
                           int32_t n_q,
                           const double* residual,
                           double* rhs_p,
                           double* rhs_q,
                           cudaStream_t stream = nullptr);

void launch_accumulate_field_dx(int32_t n_p,
                                int32_t n_q,
                                const double* dtheta0,
                                const double* dvm0,
                                const double* dtheta1,
                                const double* dvm1,
                                bool include_round1,
                                double* dx,
                                cudaStream_t stream = nullptr);

void launch_build_fdlf_rhs(int32_t n_pv,
                           int32_t n_pq,
                           const double* residual_p,
                           const double* residual_q,
                           const double* vm,
                           const int32_t* pv,
                           const int32_t* pq,
                           double p_sign,
                           double q_sign,
                           bool p_scale_by_v,
                           bool q_scale_by_v,
                           double* rhs_p,
                           double* rhs_q,
                           cudaStream_t stream = nullptr);

void launch_build_fdlf_rhs_p(int32_t n_pv,
                             int32_t n_pq,
                             const double* residual_p,
                             const double* vm,
                             const int32_t* pv,
                             const int32_t* pq,
                             double sign,
                             bool scale_by_v,
                             double* rhs_p,
                             cudaStream_t stream = nullptr);

void launch_build_fdlf_rhs_q(int32_t n_pq,
                             const double* residual_q,
                             const double* vm,
                             const int32_t* pq,
                             double sign,
                             bool scale_by_v,
                             double* rhs_q,
                             cudaStream_t stream = nullptr);

}  // namespace cupf_minimal
