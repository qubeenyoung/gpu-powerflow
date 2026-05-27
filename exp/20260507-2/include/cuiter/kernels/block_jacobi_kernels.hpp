#pragma once

#include <cstdint>

namespace cuiter::kernels {

void launch_scatter_csr_values(int32_t nnz,
                               const int32_t* perm_value_source,
                               const double* original_values,
                               double* permuted_values);
void launch_extract_dense_blocks_f32(int32_t nnz_perm,
                                     const int32_t* dense_block_offsets,
                                     const int32_t* dense_local_rows,
                                     const int32_t* dense_local_cols,
                                     const double* permuted_values,
                                     float* dense_blocks);
void launch_extract_dense_blocks_f64(int32_t nnz_perm,
                                     const int32_t* dense_block_offsets,
                                     const int32_t* dense_local_rows,
                                     const int32_t* dense_local_cols,
                                     const double* permuted_values,
                                     double* dense_blocks);
void launch_extract_ras_dense_blocks_f32(int32_t map_nnz,
                                         const int32_t* source_positions,
                                         const int32_t* dense_block_offsets,
                                         const int32_t* dense_local_rows,
                                         const int32_t* dense_local_cols,
                                         const double* permuted_values,
                                         float* dense_blocks);
void launch_extract_ras_dense_blocks_f64(int32_t map_nnz,
                                         const int32_t* source_positions,
                                         const int32_t* dense_block_offsets,
                                         const int32_t* dense_local_rows,
                                         const int32_t* dense_local_cols,
                                         const double* permuted_values,
                                         double* dense_blocks);
void launch_add_block_diagonal_shift_f32(int32_t num_blocks,
                                         int32_t leading_dim,
                                         const int32_t* block_sizes,
                                         float shift,
                                         float* dense_blocks);
void launch_add_block_diagonal_shift_f64(int32_t num_blocks,
                                         int32_t leading_dim,
                                         const int32_t* block_sizes,
                                         double shift,
                                         double* dense_blocks);
void launch_block_inverse_apply_f32(int32_t num_blocks,
                                    int32_t leading_dim,
                                    const int32_t* block_starts,
                                    const int32_t* block_sizes,
                                    const float* inverse_blocks,
                                    const double* rhs,
                                    double* out);
void launch_block_inverse_apply_f64(int32_t num_blocks,
                                    int32_t leading_dim,
                                    const int32_t* block_starts,
                                    const int32_t* block_sizes,
                                    const double* inverse_blocks,
                                    const double* rhs,
                                    double* out);
void launch_ras_inverse_apply_f32(int32_t num_blocks,
                                  int32_t leading_dim,
                                  const int32_t* local_offsets,
                                  const int32_t* local_to_global,
                                  const int32_t* owned_sizes,
                                  const int32_t* local_sizes,
                                  const float* inverse_blocks,
                                  const double* rhs,
                                  double* out);
void launch_ras_inverse_apply_f64(int32_t num_blocks,
                                  int32_t leading_dim,
                                  const int32_t* local_offsets,
                                  const int32_t* local_to_global,
                                  const int32_t* owned_sizes,
                                  const int32_t* local_sizes,
                                  const double* inverse_blocks,
                                  const double* rhs,
                                  double* out);
void launch_block_lu_solve_apply_f32(int32_t num_blocks,
                                     int32_t leading_dim,
                                     const int32_t* block_starts,
                                     const int32_t* block_sizes,
                                     const float* lu_blocks,
                                     const int32_t* pivots,
                                     const double* rhs,
                                     double* out);
void launch_block_lu_solve_apply_f64(int32_t num_blocks,
                                     int32_t leading_dim,
                                     const int32_t* block_starts,
                                     const int32_t* block_sizes,
                                     const double* lu_blocks,
                                     const int32_t* pivots,
                                     const double* rhs,
                                     double* out);
void launch_assemble_coarse_matrix_f32(int32_t rows,
                                       int32_t coarse_dim,
                                       const int32_t* row_ptr,
                                       const int32_t* col_idx,
                                       const double* values,
                                       const int32_t* block_ids,
                                       const float* weights,
                                       float* coarse_matrix);
void launch_assemble_coarse_matrix_f64(int32_t rows,
                                       int32_t coarse_dim,
                                       const int32_t* row_ptr,
                                       const int32_t* col_idx,
                                       const double* values,
                                       const int32_t* block_ids,
                                       const double* weights,
                                       double* coarse_matrix);
void launch_compress_coarse_rhs_f32(int32_t n,
                                    const int32_t* block_ids,
                                    const float* weights,
                                    const double* residual,
                                    float* coarse_rhs);
void launch_compress_coarse_rhs_f64(int32_t n,
                                    const int32_t* block_ids,
                                    const double* weights,
                                    const double* residual,
                                    double* coarse_rhs);
void launch_expand_add_coarse_solution_f32(int32_t n,
                                           const int32_t* block_ids,
                                           const float* weights,
                                           const float* coarse_solution,
                                           double* out);
void launch_expand_add_coarse_solution_f64(int32_t n,
                                           const int32_t* block_ids,
                                           const double* weights,
                                           const double* coarse_solution,
                                           double* out);
void launch_check_finite(int32_t n, const double* x, int32_t* flag);

}  // namespace cuiter::kernels
