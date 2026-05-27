#include "matrix/csr_matrix.h"
#include "reordering/metis_nd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int create_structurally_symmetric_matrix(CSRMatrix *A);
static int create_nonsymmetric_pattern_matrix(CSRMatrix *A);
static int check_graph_has_no_diagonal_and_sorted_rows(const CSRMatrix *graph);
static int check_permutation_is_bijection(const int *perm, const int *inv_perm, int n);
static int find_value(const CSRMatrix *A, int row, int col, double *value);
static int test_graph_builder(void);
static int test_invalid_symmetry_validation(void);
static int test_permutation_correctness(void);
static int test_metis_ordering(void);

int main(void)
{
    if (test_graph_builder() != 0) {
        return 1;
    }
    if (test_invalid_symmetry_validation() != 0) {
        return 1;
    }
    if (test_permutation_correctness() != 0) {
        return 1;
    }
    if (test_metis_ordering() != 0) {
        return 1;
    }

    printf("ordering tests passed\n");
    return 0;
}

static int create_structurally_symmetric_matrix(CSRMatrix *A)
{
    int rc = csr_create(A, 4, 4, 12);
    if (rc != SDS_OK) {
        return rc;
    }

    A->rowptr[0] = 0;
    A->rowptr[1] = 3;
    A->rowptr[2] = 6;
    A->rowptr[3] = 10;
    A->rowptr[4] = 12;

    /* Rows are intentionally not sorted. Values are nonsymmetric. */
    {
        int cols[12] = {
            0, 2, 1,
            2, 0, 1,
            3, 0, 1, 2,
            3, 2
        };
        double values[12] = {
            10.0, 3.0, 2.0,
            5.0, 7.0, 20.0,
            17.0, 11.0, 13.0, 30.0,
            40.0, 19.0
        };
        memcpy(A->colind, cols, sizeof(cols));
        memcpy(A->values, values, sizeof(values));
    }

    return SDS_OK;
}

static int create_nonsymmetric_pattern_matrix(CSRMatrix *A)
{
    int rc = csr_create(A, 3, 3, 3);
    if (rc != SDS_OK) {
        return rc;
    }

    A->rowptr[0] = 0;
    A->rowptr[1] = 2;
    A->rowptr[2] = 3;
    A->rowptr[3] = 3;

    A->colind[0] = 0;
    A->colind[1] = 1;  /* Missing matching edge 1 -> 0. */
    A->colind[2] = 1;

    A->values[0] = 1.0;
    A->values[1] = 2.0;
    A->values[2] = 3.0;
    return SDS_OK;
}

static int check_graph_has_no_diagonal_and_sorted_rows(const CSRMatrix *graph)
{
    for (int row = 0; row < graph->nrows; ++row) {
        int previous_col = -1;
        for (int p = graph->rowptr[row]; p < graph->rowptr[row + 1]; ++p) {
            const int col = graph->colind[p];
            if (col == row || col <= previous_col) {
                return -1;
            }
            previous_col = col;
        }
    }
    return 0;
}

static int check_permutation_is_bijection(const int *perm, const int *inv_perm, int n)
{
    int *seen = (int *)calloc((size_t)n, sizeof(int));
    if (!seen) {
        return -1;
    }

    for (int i = 0; i < n; ++i) {
        if (perm[i] < 0 || perm[i] >= n || seen[perm[i]]) {
            free(seen);
            return -1;
        }
        seen[perm[i]] = 1;
    }
    for (int i = 0; i < n; ++i) {
        if (inv_perm[i] < 0 || inv_perm[i] >= n || perm[inv_perm[i]] != i) {
            free(seen);
            return -1;
        }
    }

    free(seen);
    return 0;
}

static int find_value(const CSRMatrix *A, int row, int col, double *value)
{
    for (int p = A->rowptr[row]; p < A->rowptr[row + 1]; ++p) {
        if (A->colind[p] == col) {
            *value = A->values[p];
            return 1;
        }
    }
    return 0;
}

static int test_graph_builder(void)
{
    CSRMatrix A;
    CSRMatrix graph;
    int rc;

    memset(&A, 0, sizeof(A));
    memset(&graph, 0, sizeof(graph));

    rc = create_structurally_symmetric_matrix(&A);
    if (rc != SDS_OK) {
        return -1;
    }

    rc = build_metis_graph(&A, &graph, 1);
    if (rc != SDS_OK) {
        csr_destroy(&A);
        return -1;
    }

    if (graph.nnz != 8 || check_graph_has_no_diagonal_and_sorted_rows(&graph) != 0) {
        csr_destroy(&graph);
        csr_destroy(&A);
        return -1;
    }

    printf("graph pattern:\n");
    for (int row = 0; row < graph.nrows; ++row) {
        printf("  row %d:", row);
        for (int p = graph.rowptr[row]; p < graph.rowptr[row + 1]; ++p) {
            printf(" %d", graph.colind[p]);
        }
        printf("\n");
    }

    csr_destroy(&graph);
    csr_destroy(&A);
    return 0;
}

static int test_invalid_symmetry_validation(void)
{
    CSRMatrix A;
    CSRMatrix graph;
    int rc;

    memset(&A, 0, sizeof(A));
    memset(&graph, 0, sizeof(graph));

    rc = create_nonsymmetric_pattern_matrix(&A);
    if (rc != SDS_OK) {
        return -1;
    }

    rc = build_metis_graph(&A, &graph, 1);
    if (rc == SDS_OK) {
        csr_destroy(&graph);
        csr_destroy(&A);
        return -1;
    }

    rc = build_metis_graph(&A, &graph, 0);
    if (rc != SDS_OK) {
        csr_destroy(&A);
        return -1;
    }

    csr_destroy(&graph);
    csr_destroy(&A);
    return 0;
}

static int test_permutation_correctness(void)
{
    CSRMatrix A;
    CSRMatrix A_perm;
    int perm[4] = {2, 0, 1, 3};
    int rc;

    memset(&A, 0, sizeof(A));
    memset(&A_perm, 0, sizeof(A_perm));

    rc = create_structurally_symmetric_matrix(&A);
    if (rc != SDS_OK) {
        return -1;
    }

    rc = apply_symmetric_permutation(&A, perm, &A_perm);
    if (rc != SDS_OK) {
        csr_destroy(&A);
        return -1;
    }

    for (int row = 0; row < A.nrows; ++row) {
        for (int p = A.rowptr[row]; p < A.rowptr[row + 1]; ++p) {
            double value = 0.0;
            if (!find_value(&A_perm, perm[row], perm[A.colind[p]], &value) ||
                fabs(value - A.values[p]) > 1.0e-12) {
                csr_destroy(&A_perm);
                csr_destroy(&A);
                return -1;
            }
        }
    }

    printf("permuted matrix pattern:\n");
    for (int row = 0; row < A_perm.nrows; ++row) {
        printf("  row %d:", row);
        for (int p = A_perm.rowptr[row]; p < A_perm.rowptr[row + 1]; ++p) {
            printf(" %d", A_perm.colind[p]);
        }
        printf("\n");
    }

    csr_destroy(&A_perm);
    csr_destroy(&A);
    return 0;
}

static int test_metis_ordering(void)
{
    CSRMatrix A;
    CSRMatrix graph;
    int perm[4];
    int inv_perm[4];
    int rc;

    memset(&A, 0, sizeof(A));
    memset(&graph, 0, sizeof(graph));

    rc = create_structurally_symmetric_matrix(&A);
    if (rc != SDS_OK) {
        return -1;
    }
    rc = build_metis_graph(&A, &graph, 1);
    if (rc != SDS_OK) {
        csr_destroy(&A);
        return -1;
    }
    rc = compute_metis_nd_ordering(&graph, perm, inv_perm);
    if (rc != SDS_OK || check_permutation_is_bijection(perm, inv_perm, 4) != 0) {
        csr_destroy(&graph);
        csr_destroy(&A);
        return -1;
    }

    printf("METIS perm:");
    for (int i = 0; i < 4; ++i) {
        printf(" %d", perm[i]);
    }
    printf("\nMETIS inv_perm:");
    for (int i = 0; i < 4; ++i) {
        printf(" %d", inv_perm[i]);
    }
    printf("\n");

    csr_destroy(&graph);
    csr_destroy(&A);
    return 0;
}
