#include "io/matrix_market_io.h"
#include "reordering/metis_nd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_ms(void);
static void print_usage(const char *argv0);
static void print_int_prefix(const char *name, const int *x, int n, int limit);
static void release_resources(CSRMatrix *A,
                              CSRMatrix *graph,
                              CSRMatrix *A_perm,
                              int *perm,
                              int *inv_perm);

int main(int argc, char **argv)
{
    const char *matrix_path;
    int print_limit = 32;
    CSRMatrix A;
    CSRMatrix graph;
    CSRMatrix A_perm;
    int *perm = NULL;
    int *inv_perm = NULL;
    int rc;
    double t0;
    double t_read;
    double t_graph;
    double t_metis;
    double t_reorder;

    if (argc < 2 || argc > 3) {
        print_usage(argv[0]);
        return 2;
    }

    matrix_path = argv[1];
    if (argc == 3) {
        print_limit = atoi(argv[2]);
        if (print_limit < 0) {
            print_usage(argv[0]);
            return 2;
        }
    }

    memset(&A, 0, sizeof(A));
    memset(&graph, 0, sizeof(graph));
    memset(&A_perm, 0, sizeof(A_perm));

    t0 = get_time_ms();
    rc = load_matrix_market_csr(matrix_path, &A);
    t_read = get_time_ms();
    if (rc != SDS_OK) {
        fprintf(stderr, "matrix read failed: %s\n", matrix_path);
        return 1;
    }

    rc = build_metis_graph(&A, &graph, 1);
    t_graph = get_time_ms();
    if (rc != SDS_OK) {
        fprintf(stderr, "METIS graph construction failed; input pattern may be nonsymmetric\n");
        release_resources(&A, &graph, &A_perm, perm, inv_perm);
        return 1;
    }

    perm = (int *)malloc((size_t)A.nrows * sizeof(int));
    inv_perm = (int *)malloc((size_t)A.nrows * sizeof(int));
    if (!perm || !inv_perm) {
        fprintf(stderr, "permutation allocation failed\n");
        release_resources(&A, &graph, &A_perm, perm, inv_perm);
        return 1;
    }

    rc = compute_metis_nd_ordering(&graph, perm, inv_perm);
    t_metis = get_time_ms();
    if (rc != SDS_OK) {
        fprintf(stderr, "METIS_NodeND reordering failed\n");
        release_resources(&A, &graph, &A_perm, perm, inv_perm);
        return 1;
    }

    rc = apply_symmetric_permutation(&A, perm, &A_perm);
    t_reorder = get_time_ms();
    if (rc != SDS_OK) {
        fprintf(stderr, "symmetric matrix reordering failed\n");
        release_resources(&A, &graph, &A_perm, perm, inv_perm);
        return 1;
    }

    printf("matrix: %s\n", matrix_path);
    printf("A: n=%d nnz=%d\n", A.nrows, A.nnz);
    printf("METIS graph from structural pattern: n=%d adjacency_nnz=%d undirected_edges=%d\n",
           graph.nrows, graph.nnz, graph.nnz / 2);
    printf("A_perm = P A P^T: n=%d nnz=%d\n", A_perm.nrows, A_perm.nnz);
    printf("timing_ms: read=%.3f graph=%.3f metis_nd=%.3f reorder=%.3f total=%.3f\n",
           t_read - t0,
           t_graph - t_read,
           t_metis - t_graph,
           t_reorder - t_metis,
           t_reorder - t0);

    print_int_prefix("perm", perm, A.nrows, print_limit);
    print_int_prefix("inv_perm", inv_perm, A.nrows, print_limit);

    release_resources(&A, &graph, &A_perm, perm, inv_perm);
    return 0;
}

static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static void print_usage(const char *argv0)
{
    fprintf(stderr, "usage: %s <matrix.mtx> [print_limit]\n", argv0);
}

static void print_int_prefix(const char *name, const int *x, int n, int limit)
{
    int m;

    if (limit < 0) {
        limit = 0;
    }
    m = n < limit ? n : limit;

    printf("%s[0:%d] =", name, m);
    for (int i = 0; i < m; ++i) {
        printf(" %d", x[i]);
    }
    if (m < n) {
        printf(" ...");
    }
    printf("\n");
}

static void release_resources(CSRMatrix *A,
                              CSRMatrix *graph,
                              CSRMatrix *A_perm,
                              int *perm,
                              int *inv_perm)
{
    csr_destroy(A_perm);
    csr_destroy(graph);
    csr_destroy(A);
    free(perm);
    free(inv_perm);
}
