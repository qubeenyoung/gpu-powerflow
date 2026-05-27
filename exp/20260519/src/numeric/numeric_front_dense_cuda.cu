#include "numeric/numeric_cuda_internal.h"

/*
 * STRUMPACK FrontDense-inspired block flow:
 *   phase1: create F11/F12/F21/F22 blocks directly from original entries and
 *           child contribution blocks.
 *   phase2: factor F11, pivot F12, triangular solves, and GEMM update F22.
 */

int numeric_prepare_dense_lu_cuda(const SymbolicFactorization *symbolic,
                                  NumericFactorization *numeric)
{
    int rc;

    if (!symbolic || !numeric) {
        return SDS_ERR_BAD_INPUT;
    }

    numeric->max_npiv = 0;
    for (int front = 0; front < symbolic->num_fronts; ++front) {
        if (symbolic->fronts[front].num_pivots > numeric->max_npiv) {
            numeric->max_npiv = symbolic->fronts[front].num_pivots;
        }
    }
    if (numeric->max_npiv <= 0) {
        return SDS_ERR_BAD_INPUT;
    }

    rc = numeric_cublas_status(cublasCreate((cublasHandle_t *)&numeric->cublas_handle));
    if (rc == SDS_OK) {
        rc = numeric_cusolver_status(
            cusolverDnCreate((cusolverDnHandle_t *)&numeric->cusolver_handle));
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_ipiv,
                                            (size_t)numeric->max_npiv * sizeof(int)));
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_lu_info,
                                            sizeof(int)));
    }
    if (rc == SDS_OK) {
        rc = numeric_cusolver_status(
            cusolverDnDgetrf_bufferSize((cusolverDnHandle_t)numeric->cusolver_handle,
                                        numeric->max_npiv, numeric->max_npiv,
                                        numeric->d_values,
                                        numeric->max_npiv,
                                        &numeric->lu_work_size));
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_lu_work,
                                            (size_t)numeric->lu_work_size * sizeof(double)));
    }
    return rc;
}

int numeric_front_factor_phase1_cuda(const SymbolicFactorization *symbolic,
                                     NumericFactorization *numeric,
                                     int front_id,
                                     double *d_U11,
                                     double *d_U12,
                                     double *d_L21,
                                     double *d_C)
{
    const int npiv = symbolic->fronts[front_id].num_pivots;
    const int nupd = symbolic->fronts[front_id].num_updates;
    const int entry_begin = numeric->h_entry_ptr[front_id];
    const int entry_count = numeric->h_entry_ptr[front_id + 1] - entry_begin;
    int rc;

    rc = launch_zero_front_blocks_kernel(d_U11, d_U12, d_L21, d_C, npiv, nupd);
    if (rc != SDS_OK) {
        return rc;
    }
    ++numeric->kernel_launches;

    rc = launch_assemble_entries_to_blocks_kernel(
        numeric->d_a_values,
        numeric->d_entry_source_index,
        numeric->d_entry_target_offset,
        entry_begin,
        entry_count,
        numeric->d_values);
    if (rc != SDS_OK) {
        return rc;
    }
    ++numeric->kernel_launches;

    for (int child_slot = 0; child_slot < 2; ++child_slot) {
        const int child_id = child_slot == 0
            ? symbolic->fronts[front_id].left_child
            : symbolic->fronts[front_id].right_child;
        const int child_nupd = child_id >= 0
            ? symbolic->fronts[child_id].num_updates
            : 0;

        if (child_id < 0 || child_nupd <= 0) {
            continue;
        }
        rc = launch_assemble_child_contribution_kernel(
            numeric->d_values + numeric->h_C_offset[child_id],
            numeric->d_update_to_parent +
                numeric->h_update_to_parent_ptr[child_id],
            child_nupd,
            d_U11, d_U12, d_L21, d_C,
            npiv, nupd);
        if (rc != SDS_OK) {
            return rc;
        }
        ++numeric->kernel_launches;
    }

    return SDS_OK;
}

int numeric_front_factor_phase2_cuda(const SymbolicFactorization *symbolic,
                                     const NumericFactorizationOptions *options,
                                     NumericFactorization *numeric,
                                     int front_id,
                                     double *d_L11,
                                     double *d_U11,
                                     double *d_U12,
                                     double *d_L21,
                                     double *d_C)
{
    const int npiv = symbolic->fronts[front_id].num_pivots;
    const int nupd = symbolic->fronts[front_id].num_updates;
    const double one = 1.0;
    const double minus_one = -1.0;
    int rc;

    if (npiv < 0) {
        return SDS_ERR_BAD_INPUT;
    }
    if (npiv == 0) {
        return SDS_OK;
    }

    if (options->enable_diagonal_perturbation) {
        rc = launch_perturb_diagonal_kernel(d_U11, npiv,
                                            options->pivot_tol,
                                            options->perturb_value);
        if (rc != SDS_OK) {
            return rc;
        }
        ++numeric->kernel_launches;
    }

    rc = numeric_cusolver_status(
        cusolverDnDgetrf((cusolverDnHandle_t)numeric->cusolver_handle,
                         npiv, npiv, d_U11, npiv,
                         numeric->d_lu_work,
                         numeric->d_ipiv,
                         numeric->d_lu_info));
    if (rc != SDS_OK) {
        return rc;
    }

    rc = launch_record_getrf_status_kernel(numeric->d_lu_info,
                                           numeric->d_front_status,
                                           numeric->d_pivot_index,
                                           numeric->d_global_status,
                                           numeric->d_global_pivot_front,
                                           numeric->d_global_pivot_index,
                                           front_id);
    if (rc != SDS_OK) {
        return rc;
    }
    ++numeric->kernel_launches;

    rc = launch_check_diagonal_pivots_kernel(d_U11, npiv,
                                             options->pivot_tol,
                                             numeric->d_front_status,
                                             numeric->d_pivot_index,
                                             numeric->d_global_status,
                                             numeric->d_global_pivot_front,
                                             numeric->d_global_pivot_index,
                                             front_id);
    if (rc != SDS_OK) {
        return rc;
    }
    ++numeric->kernel_launches;

    if (nupd > 0) {
        rc = launch_apply_row_pivots_kernel(d_U12, npiv, nupd, numeric->d_ipiv);
        if (rc != SDS_OK) {
            return rc;
        }
        ++numeric->kernel_launches;
    }

    rc = launch_extract_lu_factors_kernel(d_U11, d_L11, npiv);
    if (rc != SDS_OK) {
        return rc;
    }
    ++numeric->kernel_launches;

    if (nupd > 0) {
        rc = numeric_cublas_status(
            cublasDtrsm((cublasHandle_t)numeric->cublas_handle,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_UNIT,
                        npiv, nupd,
                        &one,
                        d_L11, npiv,
                        d_U12, npiv));
        if (rc != SDS_OK) {
            return rc;
        }

        rc = numeric_cublas_status(
            cublasDtrsm((cublasHandle_t)numeric->cublas_handle,
                        CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        nupd, npiv,
                        &one,
                        d_U11, npiv,
                        d_L21, nupd));
        if (rc != SDS_OK) {
            return rc;
        }

        rc = numeric_cublas_status(
            cublasDgemm((cublasHandle_t)numeric->cublas_handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        nupd, nupd, npiv,
                        &minus_one,
                        d_L21, nupd,
                        d_U12, npiv,
                        &one,
                        d_C, nupd));
        if (rc != SDS_OK) {
            return rc;
        }
    }

    return SDS_OK;
}
