#include "numeric/numeric_cuda_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Algorithmic reference:
 *   third_party/lin_sol/strumpack/src/sparse/EliminationTree.cpp
 *   third_party/lin_sol/strumpack/src/sparse/fronts/Front.hpp
 *   third_party/lin_sol/strumpack/src/sparse/fronts/FrontDense.cpp
 *   third_party/lin_sol/strumpack/src/sparse/fronts/Front.cpp
 *
 * This file owns the public numeric lifecycle and factor-order replay only.
 * CUDA kernels live in numeric_kernels_cuda.cu. Host/device plan upload and
 * status handling live in numeric_plan_cuda.cu and numeric_cuda_runtime.cu.
 */

int numeric_factorization_create_cuda(const SymbolicFactorization *symbolic,
                                      NumericFactorization *numeric)
{
    HostNumericPlans plans;
    int rc;
    const int progress = getenv("SDS_NUMERIC_CREATE_PROGRESS") != NULL;

    if (!symbolic || !numeric || symbolic->storage.total_dense_entries == 0) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(numeric, 0, sizeof(*numeric));
    memset(&plans, 0, sizeof(plans));

    numeric->num_fronts = symbolic->num_fronts;
    numeric->first_failed_front = -1;
    numeric->first_failed_pivot = -1;

    if (progress) {
        fprintf(stderr, "[numeric-create] compact block layout begin\n");
    }
    rc = numeric_build_compact_block_layout(symbolic, numeric);
    if (rc != SDS_OK) {
        numeric_factorization_destroy_cuda(numeric);
        return rc;
    }
    if (progress) {
        fprintf(stderr, "[numeric-create] compact block layout done bytes=%zu\n",
                numeric->total_dense_bytes);
        fprintf(stderr, "[numeric-create] cudaMalloc d_values begin\n");
    }

    rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_values,
                                        numeric->total_dense_bytes));
    if (rc != SDS_OK) {
        numeric_factorization_destroy_cuda(numeric);
        return rc;
    }
    if (progress) {
        fprintf(stderr, "[numeric-create] cudaMalloc d_values done\n");
    }
    rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_front_status,
                                        (size_t)numeric->num_fronts * sizeof(int)));
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_pivot_index,
                                            (size_t)numeric->num_fronts * sizeof(int)));
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_global_status,
                                            sizeof(int)));
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_global_pivot_front,
                                            sizeof(int)));
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_global_pivot_index,
                                            sizeof(int)));
    }
    if (rc != SDS_OK) {
        numeric_factorization_destroy_cuda(numeric);
        return rc;
    }

    if (progress) {
        fprintf(stderr, "[numeric-create] dense lu prepare begin\n");
    }
    rc = numeric_prepare_dense_lu_cuda(symbolic, numeric);
    if (rc != SDS_OK) {
        numeric_factorization_destroy_cuda(numeric);
        return rc;
    }
    if (progress) {
        fprintf(stderr, "[numeric-create] dense lu prepare done\n");
        fprintf(stderr, "[numeric-create] host plans begin entry=%d contrib=%d\n",
                symbolic->entry_assembly.num_entries,
                symbolic->contribution_assembly.total_update_indices);
    }

    rc = numeric_build_host_plans(symbolic, numeric, &plans);
    if (progress) {
        fprintf(stderr, "[numeric-create] host plans done rc=%d\n", rc);
    }
    if (rc == SDS_OK) {
        if (progress) {
            fprintf(stderr, "[numeric-create] upload plans begin\n");
        }
        rc = numeric_upload_plans(&plans, numeric);
        if (progress) {
            fprintf(stderr, "[numeric-create] upload plans done rc=%d\n", rc);
        }
    }
    numeric_destroy_host_plans(&plans);
    if (rc != SDS_OK) {
        numeric_factorization_destroy_cuda(numeric);
        return rc;
    }

    return SDS_OK;
}

int numeric_factorization_destroy_cuda(NumericFactorization *numeric)
{
    if (!numeric) {
        return SDS_OK;
    }
    cudaFree(numeric->d_values);
    cudaFree(numeric->d_a_values);
    cudaFree(numeric->d_lu_work);
    cudaFree(numeric->d_front_status);
    cudaFree(numeric->d_pivot_index);
    cudaFree(numeric->d_global_status);
    cudaFree(numeric->d_global_pivot_front);
    cudaFree(numeric->d_global_pivot_index);
    cudaFree(numeric->d_lu_info);
    cudaFree(numeric->d_ipiv);
    cudaFree(numeric->d_entry_ptr);
    cudaFree(numeric->d_entry_source_index);
    cudaFree(numeric->d_entry_target_offset);
    cudaFree(numeric->d_update_to_parent);
    if (numeric->cublas_handle) {
        cublasDestroy((cublasHandle_t)numeric->cublas_handle);
    }
    if (numeric->cusolver_handle) {
        cusolverDnDestroy((cusolverDnHandle_t)numeric->cusolver_handle);
    }
    free(numeric->h_values);
    free(numeric->h_entry_ptr);
    free(numeric->h_update_to_parent_ptr);
    free(numeric->h_L11_offset);
    free(numeric->h_U11_offset);
    free(numeric->h_L21_offset);
    free(numeric->h_U12_offset);
    free(numeric->h_C_offset);
    memset(numeric, 0, sizeof(*numeric));
    numeric->first_failed_front = -1;
    numeric->first_failed_pivot = -1;
    return SDS_OK;
}

int numeric_factorization_factorize_cuda(const CSCMatrix *a_perm,
                                         const SymbolicFactorization *symbolic,
                                         const NumericFactorizationOptions *options,
                                         NumericFactorization *numeric)
{
    NumericFactorizationOptions local_options;
    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    int rc;

    if (!a_perm || !a_perm->values || !symbolic || !numeric ||
        !numeric->d_values || a_perm->nnz < 0) {
        return SDS_ERR_BAD_INPUT;
    }

    local_options.pivot_tol = 1e-12;
    local_options.enable_diagonal_perturbation = 0;
    local_options.perturb_value = 1e-12;
    local_options.enable_debug_print = 0;
    local_options.enable_timing = 1;
    local_options.use_cublas = 0;
    if (options) {
        local_options = *options;
    }
    if (local_options.pivot_tol <= 0.0) {
        local_options.pivot_tol = 1e-12;
    }
    if (local_options.perturb_value <= 0.0) {
        local_options.perturb_value = local_options.pivot_tol;
    }

    numeric->num_a_values = a_perm->nnz;
    if (!numeric->d_a_values) {
        rc = numeric_cuda_status(cudaMalloc((void **)&numeric->d_a_values,
                                            (size_t)a_perm->nnz * sizeof(double)));
        if (rc != SDS_OK) {
            return rc;
        }
    }
    rc = numeric_cuda_status(cudaMemcpy(numeric->d_a_values, a_perm->values,
                                        (size_t)a_perm->nnz * sizeof(double),
                                        cudaMemcpyHostToDevice));
    if (rc != SDS_OK) {
        return rc;
    }
    ++numeric->host_to_device_copies;

    rc = numeric_initialize_device_status(numeric);
    if (rc != SDS_OK) {
        return rc;
    }
    rc = numeric_cuda_status(cudaMemset(numeric->d_values, 0,
                                        numeric->total_dense_bytes));
    if (rc != SDS_OK) {
        return rc;
    }

    if (local_options.enable_timing) {
        rc = numeric_cuda_status(cudaEventCreate(&start));
        if (rc == SDS_OK) {
            rc = numeric_cuda_status(cudaEventCreate(&stop));
        }
        if (rc == SDS_OK) {
            rc = numeric_cuda_status(cudaEventRecord(start, 0));
        }
        if (rc != SDS_OK) {
            return rc;
        }
    }

    for (int order_idx = 0; order_idx < symbolic->schedule.num_fronts; ++order_idx) {
        const int front_id = symbolic->schedule.factor_order[order_idx];
        const FrontSymbolic *front = &symbolic->fronts[front_id];
        const int npiv = front->num_pivots;
        const int nupd = front->num_updates;
        double *d_L11 = numeric->d_values + numeric->h_L11_offset[front_id];
        double *d_U11 = numeric->d_values + numeric->h_U11_offset[front_id];
        double *d_L21 = numeric->d_values + numeric->h_L21_offset[front_id];
        double *d_U12 = numeric->d_values + numeric->h_U12_offset[front_id];
        double *d_C = numeric->d_values + numeric->h_C_offset[front_id];

        if (local_options.enable_debug_print) {
            printf("numeric front=%d npiv=%d nupd=%d nfront=%d parent=%d L11=%zu U11=%zu L21=%zu U12=%zu C=%zu\n",
                   front_id, npiv, nupd, front->num_front_vars, front->parent,
                   numeric->h_L11_offset[front_id], numeric->h_U11_offset[front_id],
                   numeric->h_L21_offset[front_id], numeric->h_U12_offset[front_id],
                   numeric->h_C_offset[front_id]);
        }

        rc = numeric_front_factor_phase1_cuda(symbolic, numeric, front_id,
                                              d_U11, d_U12, d_L21, d_C);
        if (rc != SDS_OK) {
            return rc;
        }
        rc = numeric_front_factor_phase2_cuda(symbolic, &local_options, numeric,
                                              front_id, d_L11, d_U11, d_U12,
                                              d_L21, d_C);
        if (rc != SDS_OK) {
            return rc;
        }
    }

    if (local_options.enable_timing) {
        float elapsed_ms = 0.0f;
        rc = numeric_cuda_status(cudaEventRecord(stop, 0));
        if (rc == SDS_OK) {
            rc = numeric_cuda_status(cudaEventSynchronize(stop));
        }
        if (rc == SDS_OK) {
            ++numeric->cuda_synchronizes;
            rc = numeric_cuda_status(cudaEventElapsedTime(&elapsed_ms, start, stop));
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (rc != SDS_OK) {
            return rc;
        }
        numeric->factorization_ms = (double)elapsed_ms;
    }

    rc = numeric_fetch_final_status(numeric);
    if (rc != SDS_OK) {
        return rc;
    }
    if (numeric->first_failed_front >= 0) {
        return SDS_ERR_ZERO_PIVOT;
    }
    return SDS_OK;
}

int numeric_factorization_print_summary(const SymbolicFactorization *symbolic,
                                        const NumericFactorization *numeric)
{
    int largest_front = -1;
    int largest_npiv = 0;
    int largest_nupd = 0;
    int largest_nfront = 0;

    if (!symbolic || !numeric) {
        return SDS_ERR_BAD_INPUT;
    }

    for (int front = 0; front < symbolic->num_fronts; ++front) {
        const int npiv = symbolic->fronts[front].num_pivots;
        const int nupd = symbolic->fronts[front].num_updates;
        const int nfront = symbolic->fronts[front].num_front_vars;
        if (nfront > largest_nfront) {
            largest_front = front;
            largest_npiv = npiv;
            largest_nupd = nupd;
            largest_nfront = nfront;
        }
    }

    printf("NumericFactorization CUDA summary:\n");
    printf("  num_fronts=%d\n", numeric->num_fronts);
    printf("  total_numeric_storage_bytes=%zu\n", numeric->total_dense_bytes);
    printf("  largest_front=%d npiv=%d nupd=%d nfront=%d\n",
           largest_front, largest_npiv, largest_nupd, largest_nfront);
    printf("  zero_pivot_count=%d first_failed_front=%d first_failed_pivot=%d\n",
           numeric->zero_pivot_count,
           numeric->first_failed_front,
           numeric->first_failed_pivot);
    printf("  factorization_ms=%.6f\n", numeric->factorization_ms);
    printf("  kernel_launches=%d h2d_copies=%d d2h_copies=%d cuda_syncs=%d\n",
           numeric->kernel_launches,
           numeric->host_to_device_copies,
           numeric->device_to_host_copies,
           numeric->cuda_synchronizes);
    return SDS_OK;
}
