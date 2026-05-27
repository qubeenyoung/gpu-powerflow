#include "matrix/csc_matrix.h"
#include "matrix/csr_matrix.h"
#include "reordering/metis_nd.h"
#include "symbolic/symbolic_factorization.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int create_test_matrix(CSRMatrix *A);
static int create_invalid_pattern(CSRMatrix *A);
static int check_unique_array(const int *values, int n);
static int check_inverse_arrays(const int *perm, const int *inv_perm, int n);
static int run_symbolic_demo(void);
static int run_invalid_pattern_test(void);

int main(void)
{
    if (run_symbolic_demo() != 0) {
        return 1;
    }
    if (run_invalid_pattern_test() != 0) {
        return 1;
    }
    printf("symbolic tests passed\n");
    return 0;
}

static int create_test_matrix(CSRMatrix *A)
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
            10.0, 2.0, -1.0,
            7.0, 11.0, 3.0, 4.0,
            5.0, -2.0, 12.0, 6.0,
            8.0, 13.0, -4.0, 9.0,
            1.0, 10.0, 14.0,
            2.0, 15.0
        };
        memcpy(A->colind, cols, sizeof(cols));
        memcpy(A->values, vals, sizeof(vals));
    }

    return SDS_OK;
}

static int create_invalid_pattern(CSRMatrix *A)
{
    int rc = csr_create(A, 3, 3, 4);
    if (rc != SDS_OK) {
        return rc;
    }
    A->rowptr[0] = 0;
    A->rowptr[1] = 2;
    A->rowptr[2] = 3;
    A->rowptr[3] = 4;
    A->colind[0] = 0;
    A->colind[1] = 1;
    A->colind[2] = 1;
    A->colind[3] = 2;
    A->values[0] = 1.0;
    A->values[1] = 2.0;
    A->values[2] = 3.0;
    A->values[3] = 4.0;
    return SDS_OK;
}

static int check_unique_array(const int *values, int n)
{
    int *seen = (int *)calloc((size_t)n, sizeof(int));
    if (!seen) {
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        if (values[i] < 0 || values[i] >= n || seen[values[i]]) {
            free(seen);
            return -1;
        }
        seen[values[i]] = 1;
    }
    free(seen);
    return 0;
}

static int run_symbolic_demo(void)
{
    CSRMatrix A;
    CSRMatrix graph;
    CSRMatrix A_perm;
    CSCMatrix A_perm_csc;
    SymbolicFactorization symbolic;
    int *perm = NULL;
    int *inv_perm = NULL;
    int rc;

    memset(&A, 0, sizeof(A));
    memset(&graph, 0, sizeof(graph));
    memset(&A_perm, 0, sizeof(A_perm));
    memset(&A_perm_csc, 0, sizeof(A_perm_csc));
    memset(&symbolic, 0, sizeof(symbolic));

    rc = create_test_matrix(&A);
    if (rc != SDS_OK) {
        return -1;
    }
    rc = build_metis_graph(&A, &graph, 1);
    if (rc != SDS_OK) {
        csr_destroy(&A);
        return -1;
    }

    perm = (int *)malloc((size_t)A.nrows * sizeof(int));
    inv_perm = (int *)malloc((size_t)A.nrows * sizeof(int));
    if (!perm || !inv_perm) {
        csr_destroy(&graph);
        csr_destroy(&A);
        free(perm);
        free(inv_perm);
        return -1;
    }
    rc = compute_metis_nd_ordering(&graph, perm, inv_perm);
    if (rc != SDS_OK || check_unique_array(perm, A.nrows) != 0 ||
        check_unique_array(inv_perm, A.nrows) != 0) {
        csr_destroy(&graph);
        csr_destroy(&A);
        free(perm);
        free(inv_perm);
        return -1;
    }

    rc = apply_symmetric_permutation(&A, perm, &A_perm);
    if (rc == SDS_OK) {
        rc = csr_to_csc(&A_perm, &A_perm_csc);
    }
    if (rc == SDS_OK) {
        rc = symbolic_factorization_analyze(&A_perm_csc, perm, inv_perm, &symbolic);
    }
    if (rc != SDS_OK || symbolic_factorization_validate(&symbolic) != SDS_OK ||
        check_unique_array(symbolic.etree_postorder, symbolic.n) != 0 ||
        check_unique_array(symbolic.input_perm, symbolic.n) != 0 ||
        check_unique_array(symbolic.input_inv_perm, symbolic.n) != 0 ||
        check_unique_array(symbolic.separator_perm, symbolic.n) != 0 ||
        check_unique_array(symbolic.separator_inv_perm, symbolic.n) != 0 ||
        check_unique_array(symbolic.final_perm, symbolic.n) != 0 ||
        check_unique_array(symbolic.final_inv_perm, symbolic.n) != 0 ||
        check_inverse_arrays(symbolic.input_perm, symbolic.input_inv_perm, symbolic.n) != 0 ||
        check_inverse_arrays(symbolic.separator_inv_perm, symbolic.separator_perm, symbolic.n) != 0 ||
        check_inverse_arrays(symbolic.final_perm, symbolic.final_inv_perm, symbolic.n) != 0 ||
        symbolic.schedule.num_fronts != symbolic.num_fronts ||
        symbolic.schedule.num_levels <= 0 ||
        symbolic.storage.num_fronts != symbolic.num_fronts ||
        symbolic.storage.total_dense_entries == 0 ||
        symbolic.entry_assembly.num_entries != A_perm_csc.nnz ||
        symbolic.contribution_assembly.num_child_fronts < 0 ||
        symbolic.contribution_assembly.total_update_indices < 0) {
        symbolic_factorization_destroy(&symbolic);
        csc_destroy(&A_perm_csc);
        csr_destroy(&A_perm);
        csr_destroy(&graph);
        csr_destroy(&A);
        free(perm);
        free(inv_perm);
        return -1;
    }

    symbolic_factorization_print(&symbolic);

    symbolic_factorization_destroy(&symbolic);
    csc_destroy(&A_perm_csc);
    csr_destroy(&A_perm);
    csr_destroy(&graph);
    csr_destroy(&A);
    free(perm);
    free(inv_perm);
    return 0;
}

static int check_inverse_arrays(const int *perm, const int *inv_perm, int n)
{
    if (!perm || !inv_perm || n <= 0) {
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        if (perm[i] < 0 || perm[i] >= n || inv_perm[perm[i]] != i) {
            return -1;
        }
    }
    return 0;
}

static int run_invalid_pattern_test(void)
{
    CSRMatrix A;
    CSRMatrix graph;
    int rc;

    memset(&A, 0, sizeof(A));
    memset(&graph, 0, sizeof(graph));

    rc = create_invalid_pattern(&A);
    if (rc != SDS_OK) {
        return -1;
    }
    rc = build_metis_graph(&A, &graph, 1);
    if (rc == SDS_OK) {
        csr_destroy(&graph);
        csr_destroy(&A);
        return -1;
    }

    csr_destroy(&A);
    return 0;
}
