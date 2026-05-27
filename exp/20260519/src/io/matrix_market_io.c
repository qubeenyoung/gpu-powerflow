#include "io/matrix_market_io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int row;
    int col;
    double value;
} MtxEntry;

static int compare_entry(const void *a, const void *b)
{
    const MtxEntry *ea = (const MtxEntry *)a;
    const MtxEntry *eb = (const MtxEntry *)b;
    if (ea->row != eb->row) {
        return ea->row - eb->row;
    }
    return ea->col - eb->col;
}

static void lowercase(char *s)
{
    while (*s) {
        *s = (char)tolower((unsigned char)*s);
        ++s;
    }
}

static int ignored_line(const char *line)
{
    while (*line && isspace((unsigned char)*line)) {
        ++line;
    }
    return *line == '\0' || *line == '%';
}

static int read_data_line(FILE *f, char *line, size_t line_cap)
{
    while (fgets(line, (int)line_cap, f)) {
        if (!ignored_line(line)) {
            return 1;
        }
    }
    return 0;
}

int load_matrix_market_csr(const char *path, CSRMatrix *A)
{
    FILE *f = NULL;
    char line[4096];
    char banner[64];
    char object[64];
    char format[64];
    char field[64];
    char symmetry[64];
    int rows = 0;
    int cols = 0;
    int declared = 0;
    int symmetric = 0;
    int hermitian = 0;
    int pattern = 0;
    int complex_field = 0;
    int k;
    int count = 0;
    int merged_count = 0;
    int rc;
    MtxEntry *entries = NULL;
    MtxEntry *merged = NULL;

    if (!path || !A) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(A, 0, sizeof(*A));

    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "failed to open Matrix Market file: %s\n", path);
        return SDS_ERR_BAD_INPUT;
    }
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return SDS_ERR_BAD_INPUT;
    }
    if (sscanf(line, "%63s %63s %63s %63s %63s",
               banner, object, format, field, symmetry) != 5) {
        fclose(f);
        return SDS_ERR_BAD_INPUT;
    }
    lowercase(banner);
    lowercase(object);
    lowercase(format);
    lowercase(field);
    lowercase(symmetry);
    if (strcmp(banner, "%%matrixmarket") != 0 ||
        strcmp(object, "matrix") != 0 ||
        strcmp(format, "coordinate") != 0) {
        fclose(f);
        return SDS_ERR_BAD_INPUT;
    }
    if (strcmp(field, "real") != 0 &&
        strcmp(field, "integer") != 0 &&
        strcmp(field, "complex") != 0 &&
        strcmp(field, "pattern") != 0) {
        fclose(f);
        return SDS_ERR_BAD_INPUT;
    }
    if (strcmp(symmetry, "general") != 0 &&
        strcmp(symmetry, "symmetric") != 0 &&
        strcmp(symmetry, "hermitian") != 0) {
        fclose(f);
        return SDS_ERR_BAD_INPUT;
    }
    symmetric = strcmp(symmetry, "symmetric") == 0;
    hermitian = strcmp(symmetry, "hermitian") == 0;
    pattern = strcmp(field, "pattern") == 0;
    complex_field = strcmp(field, "complex") == 0;

    if (!read_data_line(f, line, sizeof(line)) ||
        sscanf(line, "%d %d %d", &rows, &cols, &declared) != 3 ||
        rows <= 0 || cols <= 0 || declared < 0) {
        fclose(f);
        return SDS_ERR_BAD_INPUT;
    }

    entries = (MtxEntry *)malloc((size_t)declared * ((symmetric || hermitian) ? 2u : 1u) * sizeof(MtxEntry));
    if (!entries) {
        fclose(f);
        return SDS_ERR_ALLOC;
    }

    for (k = 0; k < declared; ++k) {
        int i = 0;
        int j = 0;
        double value = 1.0;
        double imag = 0.0;
        if (!read_data_line(f, line, sizeof(line))) {
            free(entries);
            fclose(f);
            return SDS_ERR_BAD_INPUT;
        }
        if (pattern) {
            if (sscanf(line, "%d %d", &i, &j) != 2) {
                free(entries);
                fclose(f);
                return SDS_ERR_BAD_INPUT;
            }
        } else if (complex_field) {
            if (sscanf(line, "%d %d %lf %lf", &i, &j, &value, &imag) != 4) {
                free(entries);
                fclose(f);
                return SDS_ERR_BAD_INPUT;
            }
        } else {
            if (sscanf(line, "%d %d %lf", &i, &j, &value) != 3) {
                free(entries);
                fclose(f);
                return SDS_ERR_BAD_INPUT;
            }
        }
        --i;
        --j;
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            free(entries);
            fclose(f);
            return SDS_ERR_BAD_INPUT;
        }
        entries[count].row = i;
        entries[count].col = j;
        entries[count].value = value;
        ++count;
        if ((symmetric || hermitian) && i != j) {
            entries[count].row = j;
            entries[count].col = i;
            entries[count].value = value;
            ++count;
        }
    }
    fclose(f);

    qsort(entries, (size_t)count, sizeof(MtxEntry), compare_entry);
    merged = (MtxEntry *)malloc((size_t)count * sizeof(MtxEntry));
    if (!merged) {
        free(entries);
        return SDS_ERR_ALLOC;
    }
    for (k = 0; k < count; ++k) {
        if (merged_count > 0 &&
            merged[merged_count - 1].row == entries[k].row &&
            merged[merged_count - 1].col == entries[k].col) {
            merged[merged_count - 1].value += entries[k].value;
        } else {
            merged[merged_count] = entries[k];
            ++merged_count;
        }
    }

    rc = csr_create(A, rows, cols, merged_count);
    if (rc != SDS_OK) {
        free(entries);
        free(merged);
        return rc;
    }
    for (k = 0; k < merged_count; ++k) {
        ++A->rowptr[merged[k].row + 1];
    }
    for (k = 0; k < rows; ++k) {
        A->rowptr[k + 1] += A->rowptr[k];
    }
    for (k = 0; k < merged_count; ++k) {
        A->colind[k] = merged[k].col;
        A->values[k] = merged[k].value;
    }

    free(entries);
    free(merged);
    return SDS_OK;
}

int load_vector_file(const char *path, double **values, int *n)
{
    FILE *f;
    int cap = 1024;
    int count = 0;
    double *x;

    if (!path || !values || !n) {
        return SDS_ERR_BAD_INPUT;
    }
    *values = NULL;
    *n = 0;
    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "failed to open vector file: %s\n", path);
        return SDS_ERR_BAD_INPUT;
    }
    x = (double *)malloc((size_t)cap * sizeof(double));
    if (!x) {
        fclose(f);
        return SDS_ERR_ALLOC;
    }
    while (1) {
        double value;
        int got = fscanf(f, "%lf", &value);
        if (got == EOF) {
            break;
        }
        if (got != 1) {
            free(x);
            fclose(f);
            return SDS_ERR_BAD_INPUT;
        }
        if (count == cap) {
            double *next;
            cap *= 2;
            next = (double *)realloc(x, (size_t)cap * sizeof(double));
            if (!next) {
                free(x);
                fclose(f);
                return SDS_ERR_ALLOC;
            }
            x = next;
        }
        x[count++] = value;
    }
    fclose(f);
    if (count == 0) {
        free(x);
        return SDS_ERR_BAD_INPUT;
    }
    *values = x;
    *n = count;
    return SDS_OK;
}

static int read_text_file(const char *path, char **text)
{
    FILE *f = fopen(path, "rb");
    long size;
    char *buf;
    if (!f) {
        return SDS_ERR_BAD_INPUT;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return SDS_ERR_BAD_INPUT;
    }
    size = ftell(f);
    if (size < 0) {
        fclose(f);
        return SDS_ERR_BAD_INPUT;
    }
    rewind(f);
    buf = (char *)malloc((size_t)size + 1u);
    if (!buf) {
        fclose(f);
        return SDS_ERR_ALLOC;
    }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        return SDS_ERR_BAD_INPUT;
    }
    buf[size] = '\0';
    fclose(f);
    *text = buf;
    return SDS_OK;
}

static void json_string_value(const char *text, const char *key, char *out, size_t out_cap)
{
    char needle[128];
    const char *p;
    const char *q;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(text, needle);
    if (!p) {
        if (out_cap) out[0] = '\0';
        return;
    }
    p = strchr(p + strlen(needle), ':');
    if (!p) {
        if (out_cap) out[0] = '\0';
        return;
    }
    p = strchr(p, '"');
    if (!p) {
        if (out_cap) out[0] = '\0';
        return;
    }
    ++p;
    q = strchr(p, '"');
    if (!q) {
        if (out_cap) out[0] = '\0';
        return;
    }
    if ((size_t)(q - p) >= out_cap) {
        q = p + out_cap - 1u;
    }
    memcpy(out, p, (size_t)(q - p));
    out[q - p] = '\0';
}

static int json_int_value(const char *text, const char *key)
{
    char needle[128];
    const char *p;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(text, needle);
    if (!p) return 0;
    p = strchr(p + strlen(needle), ':');
    if (!p) return 0;
    return atoi(p + 1);
}

static double json_double_value(const char *text, const char *key)
{
    char needle[128];
    const char *p;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(text, needle);
    if (!p) return 0.0;
    p = strchr(p + strlen(needle), ':');
    if (!p) return 0.0;
    return atof(p + 1);
}

int load_dump_meta(const char *path, DumpMeta *meta)
{
    char *text = NULL;
    int rc;
    if (!path || !meta) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(meta, 0, sizeof(*meta));
    rc = read_text_file(path, &text);
    if (rc != SDS_OK) {
        return rc;
    }
    json_string_value(text, "case_name", meta->case_name, sizeof(meta->case_name));
    meta->iteration = json_int_value(text, "iteration");
    meta->matrix_rows = json_int_value(text, "matrix_rows");
    meta->matrix_cols = json_int_value(text, "matrix_cols");
    meta->nnz = json_int_value(text, "nnz");
    meta->rhs_norm_2 = json_double_value(text, "rhs_norm_2");
    meta->cpu_reference_residual = json_double_value(text, "cpu_reference_residual");
    free(text);
    return SDS_OK;
}
