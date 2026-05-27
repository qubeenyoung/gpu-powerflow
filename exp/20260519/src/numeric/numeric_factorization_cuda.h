#ifndef EXP_20260519_NUMERIC_FACTORIZATION_CUDA_H
#define EXP_20260519_NUMERIC_FACTORIZATION_CUDA_H

#include "matrix/csc_matrix.h"
#include "symbolic/symbolic_factorization.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double pivot_tol;
    int enable_diagonal_perturbation;
    double perturb_value;
    int enable_debug_print;
    int enable_timing;
    int use_cublas;
} NumericFactorizationOptions;

typedef struct {
    int num_fronts;
    int num_a_values;

    double *d_values;
    double *h_values;
    double *d_a_values;
    double *d_lu_work;

    int *d_front_status;
    int *d_pivot_index;
    int *d_global_status;
    int *d_global_pivot_front;
    int *d_global_pivot_index;
    int *d_lu_info;
    int *d_ipiv;

    int *d_entry_ptr;
    int *d_entry_source_index;
    size_t *d_entry_target_offset;
    int *h_entry_ptr;

    int *d_update_to_parent;
    int *h_update_to_parent_ptr;

    size_t *h_L11_offset;
    size_t *h_U11_offset;
    size_t *h_L21_offset;
    size_t *h_U12_offset;
    size_t *h_C_offset;

    void *cublas_handle;
    void *cusolver_handle;

    int max_npiv;
    int lu_work_size;

    size_t total_dense_entries;
    size_t total_dense_bytes;

    int kernel_launches;
    int host_to_device_copies;
    int device_to_host_copies;
    int cuda_synchronizes;

    int zero_pivot_count;
    int first_failed_front;
    int first_failed_pivot;

    double factorization_ms;
} NumericFactorization;

int numeric_factorization_create_cuda(const SymbolicFactorization *symbolic,
                                      NumericFactorization *numeric);

int numeric_factorization_destroy_cuda(NumericFactorization *numeric);

int numeric_factorization_factorize_cuda(const CSCMatrix *a_perm,
                                         const SymbolicFactorization *symbolic,
                                         const NumericFactorizationOptions *options,
                                         NumericFactorization *numeric);

int numeric_factorization_print_summary(const SymbolicFactorization *symbolic,
                                        const NumericFactorization *numeric);

int launch_zero_front_blocks_kernel(double *d_U11,
                                    double *d_U12,
                                    double *d_L21,
                                    double *d_C,
                                    int npiv,
                                    int nupd);

int launch_assemble_entries_to_blocks_kernel(const double *d_a_values,
                                             const int *d_entry_source_index,
                                             const size_t *d_entry_target_offset,
                                             int entry_begin,
                                             int entry_count,
                                             double *d_numeric_values);

int launch_assemble_child_contribution_kernel(const double *d_child_C,
                                              const int *d_update_to_parent,
                                              int child_nupd,
                                              double *d_U11,
                                              double *d_U12,
                                              double *d_L21,
                                              double *d_C,
                                              int parent_npiv,
                                              int parent_nupd);

int launch_apply_row_pivots_kernel(double *d_U12,
                                   int npiv,
                                   int nupd,
                                   const int *d_ipiv);

int launch_extract_lu_factors_kernel(double *d_U11,
                                     double *d_L11,
                                     int npiv);

int launch_perturb_diagonal_kernel(double *d_U11,
                                   int npiv,
                                   double pivot_tol,
                                   double perturb_value);

int launch_record_getrf_status_kernel(const int *d_lu_info,
                                      int *d_front_status,
                                      int *d_pivot_index,
                                      int *d_global_status,
                                      int *d_global_pivot_front,
                                      int *d_global_pivot_index,
                                      int front_id);

int launch_check_diagonal_pivots_kernel(const double *d_U11,
                                        int npiv,
                                        double pivot_tol,
                                        int *d_front_status,
                                        int *d_pivot_index,
                                        int *d_global_status,
                                        int *d_global_pivot_front,
                                        int *d_global_pivot_index,
                                        int front_id);

#ifdef __cplusplus
}
#endif

#endif
