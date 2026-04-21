# cuSPARSE API Notes

This experiment uses the cuSPARSE ILU0 path available in the local CUDA
headers.

The relevant local declarations were checked in:

```text
/usr/local/cuda/include/cusparse.h
```

## ILU0 Numeric Factorization

Use the legacy ILU0 API for the numeric factor:

```text
cusparseCreateCsrilu02Info
cusparseDcsrilu02_bufferSize
cusparseDcsrilu02_analysis
cusparseDcsrilu02
cusparseXcsrilu02_zeroPivot
cusparseDestroyCsrilu02Info
```

The first pass keeps `row_ptr` and `col_idx` fixed and calls numeric
factorization after each Jacobian value update.

## Triangular Solves

Use the generic SpSV API for applying the ILU factors:

```text
cusparseSpSV_createDescr
cusparseSpSV_bufferSize
cusparseSpSV_analysis
cusparseSpSV_solve
cusparseSpSV_updateMatrix
cusparseSpSV_destroyDescr
```

Descriptor attributes:

- lower solve: non-unit lower triangular if cuSPARSE expects the diagonal
  from the ILU storage; otherwise test unit-lower behavior explicitly
- upper solve: non-unit upper triangular
- index base: zero
- value type: CUDA_R_64F

The first implementation should include a tiny debug test for the triangular
solve convention because ILU0 stores L and U in one CSR value array.

## Matrix Value Updates

The full-J and block-J value arrays are device arrays. They are updated by the
Jacobian kernel and then reused by cuSPARSE without changing descriptors.

When only values change:

- ILU0: rerun `cusparseDcsrilu02`
- SpSV: call `cusparseSpSV_updateMatrix` if required by the local cuSPARSE
  behavior, then solve

If `updateMatrix` is insufficient for ILU value changes in practice, fall back
to rerunning `cusparseSpSV_analysis` after each numeric ILU factorization for
the first correct version.
