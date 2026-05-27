#include "io/matrix_market_io.h"
#include "matrix/csc_matrix.h"
#include "matrix/csr_matrix.h"
#include "reordering/metis_nd.h"
#include "symbolic/symbolic_factorize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *case_dir;
    const char *matrix_path;
    int csv;
    int validate_symmetry;
} Options;

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s (--case-dir DIR | --matrix PATH) [--csv] [--no-validate-symmetry]\n",
            prog);
}

static const char *base_name(const char *path)
{
    const char *last = path;
    if (!path) {
        return "";
    }
    for (const char *p = path; *p; ++p) {
        if (*p == '/' && p[1] != '\0') {
            last = p + 1;
        }
    }
    return last;
}

static int parse_args(int argc, char **argv, Options *options)
{
    memset(options, 0, sizeof(*options));
    options->validate_symmetry = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--case-dir") == 0 && i + 1 < argc) {
            options->case_dir = argv[++i];
        } else if (strcmp(argv[i], "--matrix") == 0 && i + 1 < argc) {
            options->matrix_path = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0) {
            options->csv = 1;
        } else if (strcmp(argv[i], "--no-validate-symmetry") == 0) {
            options->validate_symmetry = 0;
        } else {
            return SDS_ERR_BAD_INPUT;
        }
    }

    if ((options->case_dir == NULL) == (options->matrix_path == NULL)) {
        return SDS_ERR_BAD_INPUT;
    }
    return SDS_OK;
}

static char *case_matrix_path(const char *case_dir)
{
    const char suffix[] = "/dump_Ybus.mtx";
    const size_t n = strlen(case_dir);
    char *path = (char *)malloc(n + sizeof(suffix));
    if (!path) {
        return NULL;
    }
    memcpy(path, case_dir, n);
    memcpy(path + n, suffix, sizeof(suffix));
    return path;
}

static void print_csv_header(void)
{
    printf("case_name,n,original_nnz,reordered_nnz,L_nnz,U_nnz,LU_unique_nnz,fill_in_nnz,fill_ratio,extra_fill_ratio\n");
}

int main(int argc, char **argv)
{
    Options options;
    CSRMatrix A;
    CSRMatrix graph;
    CSRMatrix A_reordered;
    CSCMatrix L;
    CSCMatrix U;
    SymbolicFactorizeStats stats;
    int *perm = NULL;
    int *inv_perm = NULL;
    char *owned_matrix_path = NULL;
    const char *matrix_path;
    const char *case_name;
    int rc;

    memset(&A, 0, sizeof(A));
    memset(&graph, 0, sizeof(graph));
    memset(&A_reordered, 0, sizeof(A_reordered));
    memset(&L, 0, sizeof(L));
    memset(&U, 0, sizeof(U));
    memset(&stats, 0, sizeof(stats));

    rc = parse_args(argc, argv, &options);
    if (rc != SDS_OK) {
        print_usage(argv[0]);
        return 2;
    }

    if (options.case_dir) {
        owned_matrix_path = case_matrix_path(options.case_dir);
        if (!owned_matrix_path) {
            fprintf(stderr, "failed to allocate matrix path\n");
            return 2;
        }
        matrix_path = owned_matrix_path;
        case_name = base_name(options.case_dir);
    } else {
        matrix_path = options.matrix_path;
        case_name = base_name(options.matrix_path);
    }

    rc = load_matrix_market_csr(matrix_path, &A);
    if (rc != SDS_OK) {
        fprintf(stderr, "load_matrix_market_csr failed: %d\n", rc);
        goto cleanup;
    }

    perm = (int *)malloc((size_t)A.nrows * sizeof(int));
    inv_perm = (int *)malloc((size_t)A.nrows * sizeof(int));
    if (!perm || !inv_perm) {
        rc = SDS_ERR_ALLOC;
        fprintf(stderr, "failed to allocate permutation arrays\n");
        goto cleanup;
    }

    rc = build_metis_graph(&A, &graph, options.validate_symmetry);
    if (rc != SDS_OK) {
        fprintf(stderr, "build_metis_graph failed: %d\n", rc);
        goto cleanup;
    }
    rc = compute_metis_nd_ordering(&graph, perm, inv_perm);
    if (rc != SDS_OK) {
        fprintf(stderr, "compute_metis_nd_ordering failed: %d\n", rc);
        goto cleanup;
    }
    rc = apply_symmetric_permutation(&A, inv_perm, &A_reordered);
    if (rc != SDS_OK) {
        fprintf(stderr, "apply_symmetric_permutation failed: %d\n", rc);
        goto cleanup;
    }
    rc = symbolic_factorize_reordered_csr_with_stats(&A_reordered, &L, &U, &stats);
    if (rc != SDS_OK) {
        fprintf(stderr, "symbolic_factorize_reordered_csr failed: %d\n", rc);
        goto cleanup;
    }

    {
        const int lu_unique_nnz = L.nnz + U.nnz - A_reordered.nrows;
        const int fill_in = lu_unique_nnz - A_reordered.nnz;
        const double fill_ratio =
            A_reordered.nnz > 0 ? (double)lu_unique_nnz / (double)A_reordered.nnz : 0.0;
        const double extra_ratio =
            A_reordered.nnz > 0 ? (double)fill_in / (double)A_reordered.nnz : 0.0;

        if (options.csv) {
            print_csv_header();
            printf("%s,%d,%d,%d,%d,%d,%d,%d,%.12g,%.12g\n",
                   case_name,
                   A_reordered.nrows,
                   A.nnz,
                   A_reordered.nnz,
                   L.nnz,
                   U.nnz,
                   lu_unique_nnz,
                   fill_in,
                   fill_ratio,
                   extra_ratio);
        } else {
            printf("case: %s\n", case_name);
            printf("n: %d\n", A_reordered.nrows);
            printf("original_nnz: %d\n", A.nnz);
            printf("reordered_nnz: %d\n", A_reordered.nnz);
            printf("L_nnz: %d\n", L.nnz);
            printf("U_nnz: %d\n", U.nnz);
            printf("LU_unique_nnz: %d\n", lu_unique_nnz);
            printf("fill_in_nnz: %d\n", fill_in);
            printf("fill_ratio: %.12g\n", fill_ratio);
            printf("extra_fill_ratio: %.12g\n", extra_ratio);
        }
    }

cleanup:
    csc_destroy(&U);
    csc_destroy(&L);
    csr_destroy(&A_reordered);
    csr_destroy(&graph);
    csr_destroy(&A);
    free(perm);
    free(inv_perm);
    free(owned_matrix_path);
    return rc == SDS_OK ? 0 : 1;
}
