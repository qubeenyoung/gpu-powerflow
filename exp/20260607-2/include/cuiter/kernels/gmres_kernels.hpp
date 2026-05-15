#pragma once

#include <cstdint>

namespace cuiter::kernels {

void launch_set_zero(int32_t n, double* x);
void launch_set_constant(int32_t n, double value, double* x);
void launch_copy(int32_t n, const double* src, double* dst);
void launch_scale_copy(int32_t n, double alpha, const double* src, double* dst);
void launch_multiply_by_scale(int32_t n, const double* scale, const double* src, double* dst);
void launch_sub_scaled_device_scalar(int32_t n, const double* basis, const double* alpha, double* target);
void launch_combine_solution(int32_t n, int32_t basis_count, const double* z_basis, const double* y, double* x);
void launch_residual(int32_t n, const double* rhs, const double* ax, double* r);
void launch_residual_scaled(int32_t n, const double* rhs, const double* ax, double alpha, double* r);
void launch_residual_scaled_device_scalar(int32_t n,
                                          const double* rhs,
                                          const double* ax,
                                          const double* alpha,
                                          double* r);
void launch_scale_copy_device_scalar(int32_t n, const double* alpha, const double* src, double* dst);
void launch_mr1_two_dot_reduction(int32_t n,
                                  const double* w,
                                  const double* r,
                                  double* dot_wr_dot_ww);
void launch_mr2_five_dot_reduction(int32_t n,
                                   const double* w0,
                                   const double* w1,
                                   const double* r,
                                   double* dots);
void launch_linear_combination2(int32_t n,
                                double alpha0,
                                const double* x0,
                                double alpha1,
                                const double* x1,
                                double* out);
void launch_residual_two_scaled(int32_t n,
                                const double* rhs,
                                const double* w0,
                                double alpha0,
                                const double* w1,
                                double alpha1,
                                double* r);
void launch_bicgstab_update_p(int32_t n,
                              const double* r,
                              double beta,
                              double omega,
                              const double* v,
                              double* p);
void launch_bicgstab_update_p_device_scalar(int32_t n,
                                            const double* r,
                                            const double* beta,
                                            const double* omega,
                                            const double* v,
                                            double* p);
void launch_bicgstab_update_x_r(int32_t n,
                                double alpha,
                                const double* p_hat,
                                double omega,
                                const double* s_hat,
                                const double* s,
                                const double* t,
                                double* x,
                                double* r);
void launch_bicgstab_update_x_r_device_scalar(int32_t n,
                                              const double* alpha,
                                              const double* p_hat,
                                              const double* omega,
                                              const double* s_hat,
                                              const double* s,
                                              const double* t,
                                              double* x,
                                              double* r);
void launch_bicgstab_compute_beta(double* scalars);
void launch_bicgstab_compute_alpha(double* scalars);
void launch_bicgstab_compute_omega_and_advance(double* scalars);
void launch_csr_spmv(int32_t rows,
                     const int32_t* row_ptr,
                     const int32_t* col_idx,
                     const double* values,
                     const double* x,
                     double* y);
void launch_permute_vector(int32_t n, const int32_t* new_to_old, const double* x_old, double* x_new);
void launch_unpermute_vector(int32_t n, const int32_t* new_to_old, const double* x_new, double* x_old);
void launch_compute_scaled_row_l2_norms(int32_t rows,
                                        const int32_t* row_ptr,
                                        const int32_t* col_idx,
                                        const double* values,
                                        const double* row_scale,
                                        const double* col_scale,
                                        double* row_norms);
void launch_compute_scaled_col_l2_norms(int32_t rows,
                                        int32_t cols,
                                        const int32_t* row_ptr,
                                        const int32_t* col_idx,
                                        const double* values,
                                        const double* row_scale,
                                        const double* col_scale,
                                        double* col_norms);
void launch_update_ruiz_scale(int32_t n,
                              const double* norms,
                              double eps,
                              double clamp,
                              double* scale);
void launch_apply_scaled_csr_values(int32_t rows,
                                    const int32_t* row_ptr,
                                    const int32_t* col_idx,
                                    const double* values,
                                    const double* row_scale,
                                    const double* col_scale,
                                    double* scaled_values);

}  // namespace cuiter::kernels
