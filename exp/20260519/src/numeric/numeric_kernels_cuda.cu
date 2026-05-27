#include "numeric/numeric_cuda_internal.h"

#include <math.h>

__device__ static double atomic_add_double(double *address, double value)
{
    unsigned long long int *address_as_ull =
        (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(value +
                            __longlong_as_double((long long int)assumed)));
    } while (assumed != old);

    return __longlong_as_double((long long int)old);
}

__global__ static void zero_front_blocks_kernel(double *U11,
                                                double *U12,
                                                double *L21,
                                                double *C,
                                                int npiv,
                                                int nupd)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int U11_size = npiv * npiv;
    const int U12_size = npiv * nupd;
    const int L21_size = nupd * npiv;
    const int C_size = nupd * nupd;
    const int total = U11_size + U12_size + L21_size + C_size;

    if (idx >= total) {
        return;
    }
    if (idx < U11_size) {
        U11[idx] = 0.0;
    } else if (idx < U11_size + U12_size) {
        U12[idx - U11_size] = 0.0;
    } else if (idx < U11_size + U12_size + L21_size) {
        L21[idx - U11_size - U12_size] = 0.0;
    } else {
        C[idx - U11_size - U12_size - L21_size] = 0.0;
    }
}

__global__ static void assemble_to_blocks_kernel(const double *source_values,
                                                 const int *source_index,
                                                 const size_t *target_offset,
                                                 int begin,
                                                 int count,
                                                 double *numeric_values)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const int plan_idx = begin + idx;
        atomic_add_double(&numeric_values[target_offset[plan_idx]],
                          source_values[source_index[plan_idx]]);
    }
}

__global__ static void assemble_child_contribution_kernel(const double *child_C,
                                                          const int *update_to_parent,
                                                          int child_nupd,
                                                          double *U11,
                                                          double *U12,
                                                          double *L21,
                                                          double *C,
                                                          int parent_npiv,
                                                          int parent_nupd)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = child_nupd * child_nupd;
    if (idx < count) {
        const int child_row = idx % child_nupd;
        const int child_col = idx / child_nupd;
        const int parent_row = update_to_parent[child_row];
        const int parent_col = update_to_parent[child_col];
        const double value = child_C[child_row + child_col * child_nupd];

        if (parent_row < parent_npiv && parent_col < parent_npiv) {
            atomic_add_double(&U11[parent_row + parent_col * parent_npiv],
                              value);
        } else if (parent_row < parent_npiv) {
            atomic_add_double(
                &U12[parent_row + (parent_col - parent_npiv) * parent_npiv],
                value);
        } else if (parent_col < parent_npiv) {
            atomic_add_double(
                &L21[(parent_row - parent_npiv) + parent_col * parent_nupd],
                value);
        } else {
            atomic_add_double(
                &C[(parent_row - parent_npiv) +
                   (parent_col - parent_npiv) * parent_nupd],
                value);
        }
    }
}

__global__ static void apply_row_pivots_kernel(double *U12,
                                               int npiv,
                                               int nupd,
                                               const int *ipiv)
{
    for (int k = 0; k < npiv; ++k) {
        const int pivot_row = ipiv[k] - 1;
        if (pivot_row != k) {
            for (int col = threadIdx.x; col < nupd; col += blockDim.x) {
                const double tmp = U12[k + col * npiv];
                U12[k + col * npiv] = U12[pivot_row + col * npiv];
                U12[pivot_row + col * npiv] = tmp;
            }
        }
        __syncthreads();
    }
}

__global__ static void extract_lu_factors_kernel(double *U11,
                                                 double *L11,
                                                 int npiv)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = npiv * npiv;
    if (idx >= total) {
        return;
    }

    const int row = idx % npiv;
    const int col = idx / npiv;
    const double value = U11[idx];
    if (row > col) {
        L11[idx] = value;
        U11[idx] = 0.0;
    } else if (row == col) {
        L11[idx] = 1.0;
    } else {
        L11[idx] = 0.0;
    }
}

__global__ static void perturb_diagonal_kernel(double *U11,
                                               int npiv,
                                               double pivot_tol,
                                               double perturb_value)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npiv) {
        const int offset = idx + idx * npiv;
        const double pivot = U11[offset];
        if (fabs(pivot) < pivot_tol) {
            U11[offset] = pivot < 0.0 ? -perturb_value : perturb_value;
        }
    }
}

__global__ static void record_getrf_status_kernel(const int *lu_info,
                                                  int *front_status,
                                                  int *pivot_index,
                                                  int *global_status,
                                                  int *global_pivot_front,
                                                  int *global_pivot_index,
                                                  int front_id)
{
    if (threadIdx.x == 0 && blockIdx.x == 0 && *lu_info != 0) {
        const int failed_pivot = *lu_info > 0 ? *lu_info - 1 : 0;
        front_status[front_id] = SDS_ERR_ZERO_PIVOT;
        pivot_index[front_id] = failed_pivot;
        atomicCAS(global_status, SDS_OK, SDS_ERR_ZERO_PIVOT);
        atomicCAS(global_pivot_front, -1, front_id);
        atomicCAS(global_pivot_index, -1, failed_pivot);
    }
}

__global__ static void check_diagonal_pivots_kernel(const double *U11,
                                                    int npiv,
                                                    double pivot_tol,
                                                    int *front_status,
                                                    int *pivot_index,
                                                    int *global_status,
                                                    int *global_pivot_front,
                                                    int *global_pivot_index,
                                                    int front_id)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npiv && fabs(U11[idx + idx * npiv]) < pivot_tol) {
        front_status[front_id] = SDS_ERR_ZERO_PIVOT;
        pivot_index[front_id] = idx;
        atomicCAS(global_status, SDS_OK, SDS_ERR_ZERO_PIVOT);
        atomicCAS(global_pivot_front, -1, front_id);
        atomicCAS(global_pivot_index, -1, idx);
    }
}

int launch_zero_front_blocks_kernel(double *d_U11,
                                    double *d_U12,
                                    double *d_L21,
                                    double *d_C,
                                    int npiv,
                                    int nupd)
{
    const int count = npiv * npiv + npiv * nupd + nupd * npiv + nupd * nupd;
    const int block = 256;
    const int grid = (count + block - 1) / block;
    if (count <= 0) {
        return SDS_OK;
    }
    zero_front_blocks_kernel<<<grid, block>>>(d_U11, d_U12, d_L21, d_C,
                                              npiv, nupd);
    return numeric_cuda_status(cudaGetLastError());
}

int launch_assemble_entries_to_blocks_kernel(const double *d_a_values,
                                             const int *d_entry_source_index,
                                             const size_t *d_entry_target_offset,
                                             int entry_begin,
                                             int entry_count,
                                             double *d_numeric_values)
{
    const int block = 256;
    const int grid = (entry_count + block - 1) / block;
    if (entry_count <= 0) {
        return SDS_OK;
    }
    assemble_to_blocks_kernel<<<grid, block>>>(d_a_values, d_entry_source_index,
                                               d_entry_target_offset,
                                               entry_begin, entry_count,
                                               d_numeric_values);
    return numeric_cuda_status(cudaGetLastError());
}

int launch_assemble_child_contribution_kernel(const double *d_child_C,
                                              const int *d_update_to_parent,
                                              int child_nupd,
                                              double *d_U11,
                                              double *d_U12,
                                              double *d_L21,
                                              double *d_C,
                                              int parent_npiv,
                                              int parent_nupd)
{
    const int block = 256;
    const int count = child_nupd * child_nupd;
    const int grid = (count + block - 1) / block;
    if (count <= 0) {
        return SDS_OK;
    }
    assemble_child_contribution_kernel<<<grid, block>>>(
        d_child_C, d_update_to_parent, child_nupd,
        d_U11, d_U12, d_L21, d_C, parent_npiv, parent_nupd);
    return numeric_cuda_status(cudaGetLastError());
}

int launch_apply_row_pivots_kernel(double *d_U12,
                                   int npiv,
                                   int nupd,
                                   const int *d_ipiv)
{
    if (npiv <= 0 || nupd <= 0) {
        return SDS_OK;
    }
    apply_row_pivots_kernel<<<1, 256>>>(d_U12, npiv, nupd, d_ipiv);
    return numeric_cuda_status(cudaGetLastError());
}

int launch_extract_lu_factors_kernel(double *d_U11,
                                     double *d_L11,
                                     int npiv)
{
    const int count = npiv * npiv;
    const int block = 256;
    const int grid = (count + block - 1) / block;
    if (count <= 0) {
        return SDS_OK;
    }
    extract_lu_factors_kernel<<<grid, block>>>(d_U11, d_L11, npiv);
    return numeric_cuda_status(cudaGetLastError());
}

int launch_perturb_diagonal_kernel(double *d_U11,
                                   int npiv,
                                   double pivot_tol,
                                   double perturb_value)
{
    const int block = 256;
    const int grid = (npiv + block - 1) / block;
    if (npiv <= 0) {
        return SDS_OK;
    }
    perturb_diagonal_kernel<<<grid, block>>>(d_U11, npiv, pivot_tol,
                                             perturb_value);
    return numeric_cuda_status(cudaGetLastError());
}

int launch_record_getrf_status_kernel(const int *d_lu_info,
                                      int *d_front_status,
                                      int *d_pivot_index,
                                      int *d_global_status,
                                      int *d_global_pivot_front,
                                      int *d_global_pivot_index,
                                      int front_id)
{
    record_getrf_status_kernel<<<1, 1>>>(d_lu_info, d_front_status,
                                         d_pivot_index, d_global_status,
                                         d_global_pivot_front,
                                         d_global_pivot_index, front_id);
    return numeric_cuda_status(cudaGetLastError());
}

int launch_check_diagonal_pivots_kernel(const double *d_U11,
                                        int npiv,
                                        double pivot_tol,
                                        int *d_front_status,
                                        int *d_pivot_index,
                                        int *d_global_status,
                                        int *d_global_pivot_front,
                                        int *d_global_pivot_index,
                                        int front_id)
{
    const int block = 256;
    const int grid = (npiv + block - 1) / block;
    if (npiv <= 0) {
        return SDS_OK;
    }
    check_diagonal_pivots_kernel<<<grid, block>>>(d_U11, npiv, pivot_tol,
                                                  d_front_status,
                                                  d_pivot_index,
                                                  d_global_status,
                                                  d_global_pivot_front,
                                                  d_global_pivot_index,
                                                  front_id);
    return numeric_cuda_status(cudaGetLastError());
}
