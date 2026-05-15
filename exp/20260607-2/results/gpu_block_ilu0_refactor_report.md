# GPU Block ILU(0) Refactor Summary

## What changed

- `src/tools/gpu_block_ilu0_phase_bench.cu` now uses `GpuBlockILU0` with three public phases:
  - `setup(...)`: symbolic block pattern, sorted block rows, update lists, device buffers, and cuBLAS handle.
  - `factorize(...)`: dense scatter, numeric block ILU(0), and diagonal inverse construction.
  - `apply(...)`: triangular preconditioner apply using existing factor data.
- `factorize()` no longer creates/destroys cuBLAS handles, sorts row blocks, warms up scatter, or performs inner-loop event synchronization unless `--enable-profile` is used.
- `apply()` no longer copies the output vector to host unless `--compute-output-norm` is used.

## Correctness fixes

- `shift_scale` is now applied as a diagonal shift in the cuBLAS diagonal block buffer.
- cuBLAS LU/inverse calls use `n = dim` and `lda = pad`, so the final short block is handled correctly.
- Dense ILU storage remains row-major, while cuBLAS diagonal work buffers are explicitly column-major.
- `getrf` and `getri` status arrays are separated and copied once after factorization.
- A nonsymmetric 2x2 sanity test verifies inverse/application layout and `dim < pad`.

## Fast-path regression

Command:

```bash
./build/gpu_block_ilu0_phase_bench \
    --cases case2383wp,case3120sp,case9241pegase,case13659pegase,case6468rte \
    --block-sizes 16,32 \
    --output-dir results/gpu_block_ilu0_refactor_fast
```

All 10 runs completed with `factor_failed = 0`.

| case | bs16 factor/apply ms | bs32 factor/apply ms |
|---|---:|---:|
| case2383wp | 21.142 / 5.253 | 21.313 / 3.758 |
| case3120sp | 26.793 / 6.734 | 28.057 / 5.038 |
| case6468rte | 51.489 / 14.747 | 53.762 / 10.997 |
| case9241pegase | 81.493 / 24.133 | 74.310 / 15.263 |
| case13659pegase | 98.075 / 30.829 | 104.895 / 21.920 |

## Profile smoke

Command:

```bash
./build/gpu_block_ilu0_phase_bench \
    --cases case2383wp \
    --block-sizes 16 \
    --output-dir results/gpu_block_ilu0_refactor_smoke2 \
    --enable-profile \
    --compute-output-norm
```

Result: `factor_ms = 34.466`, `apply_ms = 18.889`, `factor_failed = 0`.

## Interpretation

The refactor makes the pilot usable as a preconditioner-style component: symbolic setup is separated from repeated numeric factorization and repeated apply. The measured factor/apply times are still much larger than cuDSS, because the numeric block ILU path still launches many tiny scalar kernels and one cuBLAS batched LU/inverse call per diagonal block. The remaining acceleration target is the dense block update/apply work, which is the Tensor Core-friendly part for a future implementation.
