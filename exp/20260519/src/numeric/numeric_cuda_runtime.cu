#include "numeric/numeric_cuda_internal.h"

#include <stdio.h>
#include <stdlib.h>

int numeric_cuda_status(cudaError_t err)
{
    if (err == cudaSuccess) {
        return SDS_OK;
    }
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    return SDS_ERR_CUDA;
}

int numeric_cublas_status(cublasStatus_t status)
{
    if (status == CUBLAS_STATUS_SUCCESS) {
        return SDS_OK;
    }
    fprintf(stderr, "cuBLAS error: %d\n", (int)status);
    return SDS_ERR_CUDA;
}

int numeric_cusolver_status(cusolverStatus_t status)
{
    if (status == CUSOLVER_STATUS_SUCCESS) {
        return SDS_OK;
    }
    fprintf(stderr, "cuSOLVER error: %d\n", (int)status);
    return SDS_ERR_CUDA;
}

int numeric_cuda_copy_to_device(void **d_ptr,
                                const void *h_ptr,
                                size_t count,
                                size_t elem_size,
                                NumericFactorization *numeric)
{
    int rc;
    const size_t bytes = count * elem_size;

    if (count == 0) {
        *d_ptr = NULL;
        return SDS_OK;
    }

    rc = numeric_cuda_status(cudaMalloc(d_ptr, bytes));
    if (rc != SDS_OK) {
        return rc;
    }
    rc = numeric_cuda_status(cudaMemcpy(*d_ptr, h_ptr, bytes,
                                        cudaMemcpyHostToDevice));
    if (rc == SDS_OK && numeric) {
        ++numeric->host_to_device_copies;
    }
    return rc;
}

int numeric_initialize_device_status(NumericFactorization *numeric)
{
    int rc;

    rc = numeric_cuda_status(cudaMemset(numeric->d_front_status, 0,
                                        (size_t)numeric->num_fronts * sizeof(int)));
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMemset(numeric->d_pivot_index, 0xff,
                                            (size_t)numeric->num_fronts * sizeof(int)));
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMemset(numeric->d_global_status, 0,
                                            sizeof(int)));
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMemset(numeric->d_global_pivot_front, 0xff,
                                            sizeof(int)));
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_status(cudaMemset(numeric->d_global_pivot_index, 0xff,
                                            sizeof(int)));
    }
    return rc;
}

int numeric_fetch_final_status(NumericFactorization *numeric)
{
    int global_status = SDS_OK;
    int *front_status = NULL;
    int rc;

    rc = numeric_cuda_status(cudaMemcpy(&global_status, numeric->d_global_status,
                                        sizeof(int), cudaMemcpyDeviceToHost));
    if (rc != SDS_OK) {
        return rc;
    }
    ++numeric->device_to_host_copies;

    rc = numeric_cuda_status(cudaMemcpy(&numeric->first_failed_front,
                                        numeric->d_global_pivot_front,
                                        sizeof(int), cudaMemcpyDeviceToHost));
    if (rc != SDS_OK) {
        return rc;
    }
    ++numeric->device_to_host_copies;

    rc = numeric_cuda_status(cudaMemcpy(&numeric->first_failed_pivot,
                                        numeric->d_global_pivot_index,
                                        sizeof(int), cudaMemcpyDeviceToHost));
    if (rc != SDS_OK) {
        return rc;
    }
    ++numeric->device_to_host_copies;

    front_status = (int *)malloc((size_t)numeric->num_fronts * sizeof(int));
    if (!front_status) {
        return SDS_ERR_ALLOC;
    }
    rc = numeric_cuda_status(cudaMemcpy(front_status, numeric->d_front_status,
                                        (size_t)numeric->num_fronts * sizeof(int),
                                        cudaMemcpyDeviceToHost));
    if (rc != SDS_OK) {
        free(front_status);
        return rc;
    }
    ++numeric->device_to_host_copies;

    numeric->zero_pivot_count = 0;
    for (int i = 0; i < numeric->num_fronts; ++i) {
        if (front_status[i] == SDS_ERR_ZERO_PIVOT) {
            ++numeric->zero_pivot_count;
        }
    }
    free(front_status);

    if (global_status == SDS_ERR_ZERO_PIVOT) {
        return SDS_OK;
    }
    return global_status == SDS_OK ? SDS_OK : SDS_ERR_NUMERIC;
}
