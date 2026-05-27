#ifndef EXP_20260519_NUMERIC_CUDA_INTERNAL_H
#define EXP_20260519_NUMERIC_CUDA_INTERNAL_H

#include "numeric/numeric_factorization_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

typedef struct {
    int num_fronts;
    int *entry_counts;
    int *entry_ptr;
    int *entry_cursor;
    int *entry_source;
    size_t *entry_target_offset;
    int *update_to_parent_ptr;
    int *update_to_parent;
} HostNumericPlans;

int numeric_cuda_status(cudaError_t err);
int numeric_cublas_status(cublasStatus_t status);
int numeric_cusolver_status(cusolverStatus_t status);

int numeric_cuda_copy_to_device(void **d_ptr,
                                const void *h_ptr,
                                size_t count,
                                size_t elem_size,
                                NumericFactorization *numeric);

int numeric_build_compact_block_layout(const SymbolicFactorization *symbolic,
                                       NumericFactorization *numeric);

int numeric_build_host_plans(const SymbolicFactorization *symbolic,
                             const NumericFactorization *numeric,
                             HostNumericPlans *plans);

void numeric_destroy_host_plans(HostNumericPlans *plans);

int numeric_upload_plans(const HostNumericPlans *plans,
                         NumericFactorization *numeric);

int numeric_initialize_device_status(NumericFactorization *numeric);

int numeric_fetch_final_status(NumericFactorization *numeric);

int numeric_prepare_dense_lu_cuda(const SymbolicFactorization *symbolic,
                                  NumericFactorization *numeric);

int numeric_front_factor_phase1_cuda(const SymbolicFactorization *symbolic,
                                     NumericFactorization *numeric,
                                     int front_id,
                                     double *d_U11,
                                     double *d_U12,
                                     double *d_L21,
                                     double *d_C);

int numeric_front_factor_phase2_cuda(const SymbolicFactorization *symbolic,
                                     const NumericFactorizationOptions *options,
                                     NumericFactorization *numeric,
                                     int front_id,
                                     double *d_L11,
                                     double *d_U11,
                                     double *d_U12,
                                     double *d_L21,
                                     double *d_C);

#endif
