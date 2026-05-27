#include "matrix/csc_matrix.h"
#include "matrix/csr_matrix.h"
#include "numeric/numeric_factorization_cuda.h"
#include "reordering/metis_nd.h"
#include "symbolic/symbolic_factorization.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int create_safe_matrix(CSRMatrix *A);
static int create_tiny_pivot_matrix(CSRMatrix *A);
static int run_numeric_case(CSRMatrix *A,
                            const NumericFactorizationOptions *options,
                            int expected_status,
                            int print_summary);

int main(void)
{
    CSRMatrix A;
    NumericFactorizationOptions options;

    memset(&A, 0, sizeof(A));
    memset(&options, 0, sizeof(options));
    options.pivot_tol = 1e-12;
    options.perturb_value = 1e-10;
    options.enable_timing = 1;

    if (create_safe_matrix(&A) != SDS_OK) {
        return 1;
    }
    if (run_numeric_case(&A, &options, SDS_OK, 1) != 0) {
        csr_destroy(&A);
        return 1;
    }
    csr_destroy(&A);

    if (create_tiny_pivot_matrix(&A) != SDS_OK) {
        return 1;
    }
    options.enable_diagonal_perturbation = 0;
    if (run_numeric_case(&A, &options, SDS_ERR_ZERO_PIVOT, 1) != 0) {
        csr_destroy(&A);
        return 1;
    }

    options.enable_diagonal_perturbation = 1;
    if (run_numeric_case(&A, &options, SDS_OK, 1) != 0) {
        csr_destroy(&A);
        return 1;
    }
    csr_destroy(&A);

    printf("numeric CUDA tests passed\n");
    return 0;
}

static int create_safe_matrix(CSRMatrix *A)
{
    int rc = csr_create(A, 6, 6, 20);
    if (rc != SDS_OK) {
        return rc;
    }

    A->rowptr[0] = 0;
    A->rowptr[1] = 3;
    A->rowptr[2] = 7;
    A->rowptr[3] = 11;
    A->rowptr[4] = 15;
    A->rowptr[5] = 18;
    A->rowptr[6] = 20;

    {
        int cols[20] = {
            0, 1, 2,
            0, 1, 2, 3,
            0, 1, 2, 4,
            1, 3, 4, 5,
            2, 3, 4,
            3, 5
        };
        double vals[20] = {
            10.0, 0.2, -0.1,
            0.7, 11.0, 0.3, 0.4,
            0.5, -0.2, 12.0, 0.6,
            0.8, 13.0, -0.4, 0.9,
            0.1, 1.0, 14.0,
            0.2, 15.0
        };
        memcpy(A->colind, cols, sizeof(cols));
        memcpy(A->values, vals, sizeof(vals));
    }
    return SDS_OK;
}

static int create_tiny_pivot_matrix(CSRMatrix *A)
{
    int rc = csr_create(A, 3, 3, 7);
    if (rc != SDS_OK) {
        return rc;
    }

    A->rowptr[0] = 0;
    A->rowptr[1] = 2;
    A->rowptr[2] = 5;
    A->rowptr[3] = 7;

    A->colind[0] = 0;
    A->colind[1] = 1;
    A->colind[2] = 0;
    A->colind[3] = 1;
    A->colind[4] = 2;
    A->colind[5] = 1;
    A->colind[6] = 2;

    A->values[0] = 1.0;
    A->values[1] = 0.0;
    A->values[2] = 0.0;
    A->values[3] = 1e-20;
    A->values[4] = 0.0;
    A->values[5] = 0.0;
    A->values[6] = 2.0;
    return SDS_OK;
}

static int run_numeric_case(CSRMatrix *A,
                            const NumericFactorizationOptions *options,
                            int expected_status,
                            int print_summary)
{
    CSRMatrix graph;
    CSRMatrix A_perm;
    CSCMatrix A_perm_csc;
    SymbolicFactorization symbolic;
    NumericFactorization numeric;
    int *perm = NULL;
    int *inv_perm = NULL;
    int rc;

    memset(&graph, 0, sizeof(graph));
    memset(&A_perm, 0, sizeof(A_perm));
    memset(&A_perm_csc, 0, sizeof(A_perm_csc));
    memset(&symbolic, 0, sizeof(symbolic));
    memset(&numeric, 0, sizeof(numeric));

    perm = (int *)malloc((size_t)A->nrows * sizeof(int));
    inv_perm = (int *)malloc((size_t)A->nrows * sizeof(int));
    if (!perm || !inv_perm) {
        free(perm);
        free(inv_perm);
        return -1;
    }

    rc = build_metis_graph(A, &graph, 1);
    if (rc == SDS_OK) {
        rc = compute_metis_nd_ordering(&graph, perm, inv_perm);
    }
    if (rc == SDS_OK) {
        rc = apply_symmetric_permutation(A, perm, &A_perm);
    }
    if (rc == SDS_OK) {
        rc = csr_to_csc(&A_perm, &A_perm_csc);
    }
    if (rc == SDS_OK) {
        rc = symbolic_factorization_analyze(&A_perm_csc, perm, inv_perm, &symbolic);
    }
    if (rc == SDS_OK) {
        rc = numeric_factorization_create_cuda(&symbolic, &numeric);
    }
    if (rc == SDS_OK) {
        rc = numeric_factorization_factorize_cuda(&A_perm_csc, &symbolic,
                                                  options, &numeric);
    }

    if (print_summary && numeric.d_values) {
        numeric_factorization_print_summary(&symbolic, &numeric);
    }

    numeric_factorization_destroy_cuda(&numeric);
    symbolic_factorization_destroy(&symbolic);
    csc_destroy(&A_perm_csc);
    csr_destroy(&A_perm);
    csr_destroy(&graph);
    free(perm);
    free(inv_perm);

    if (expected_status == SDS_OK) {
        return rc == SDS_OK ? 0 : -1;
    }
    return rc == expected_status ? 0 : -1;
}
