# Extend-add uc-bucket timing diagnostic (2026-06-09)

## Question

After the literature follow-up and the low-mid TF32 force-all check, the remaining plausible
non-GEMM lever for `case8387pegase` was parent assembly / extend-add. Prior all-or-nothing
`CLS_DISABLE_EXTEND_ADD=ON` already suggested this was not enough, but that diagnostic did not
separate many tiny updates from fewer larger updates.

This note adds a bucketed timing-only gate:

- `CLS_EXTEND_SKIP_UC_LE=N`: skip extend-add when `uc <= N`.
- `CLS_EXTEND_SKIP_UC_GT=N`: skip extend-add when `uc > N`.

Default is `0`, so normal builds are unchanged. These options intentionally produce invalid
factorizations and residuals; use them only to isolate timing contribution.

## Implementation

The gate is applied in two places:

- `extend_add_allowed_for_uc(uc)` in `src/factorize/phases.cuh`, called by `extend_add`.
- Every kernel-side extend/fused-extend condition in `src/factorize/kernels.cuh`, so skipped
  buckets avoid the extra synchronization and fused parent drain as well as the atomic scatter.

This matters because several TC paths can fuse trailing drain directly into the parent front;
gating only the `extend_add` helper would under-count parent-update cost.

## 8387 front distribution

Command context:

- case: `/datasets/power_system/nr_linear_systems/case8387pegase`
- cap/seed dump: cap32, seed42
- dump: `/tmp/cls_8387_lowtcforce_cap32_seed42_fronts_20260609_163246.csv`

Extend element distribution:

| bucket | front count | extend elems | fraction |
|---|---:|---:|---:|
| total | 7274 | 224289 | 100.0% |
| `uc <= 8` | 6761 | 99202 | 44.2% |
| `uc <= 16` | 7171 | 155739 | 69.4% |
| `uc > 16` | 103 | 68550 | 30.6% |
| `uc > 32` | 13 | 26982 | 12.0% |

The split is useful: most parent-update elements are small, but there is still a non-trivial
larger-update tail.

## 8387 timing

Builds:

- full: `build-tc-colusolve-respectcap-bighigh512-bigshared512-current`
- `skip_le16`: same policy plus `-DCLS_EXTEND_SKIP_UC_LE=16`
- `skip_gt16`: same policy plus `-DCLS_EXTEND_SKIP_UC_GT=16`

Command shape:

```bash
<build>/custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
  --batch <64|256> --batch-only --repeat 31 \
  --precision <fp32|tf32> --single-precision fp64 \
  --panel-cap <28|32> --metis-seed <7|42>
```

Raw:

- `/tmp/cls_8387_extend_skip_le16_20260609_163835.tsv`
- `/tmp/cls_8387_extend_skip_gt16_20260609_163937.tsv`

| build | seed | cap | B | fp32 ms/sys | tf32 ms/sys | fp32/tf32 |
|---|---:|---:|---:|---:|---:|---:|
| full | 42 | 32 | 64 | 0.024179 | 0.022788 | 1.061 |
| full | 42 | 32 | 256 | 0.020181 | 0.020321 | 0.993 |
| full | 7 | 28 | 64 | 0.023338 | 0.022775 | 1.025 |
| full | 7 | 28 | 256 | 0.020885 | 0.020191 | 1.034 |
| skip `uc<=16` | 42 | 32 | 64 | 0.023286 | 0.022585 | 1.031 |
| skip `uc<=16` | 42 | 32 | 256 | 0.020652 | 0.018801 | 1.098 |
| skip `uc<=16` | 7 | 28 | 64 | 0.023505 | 0.021928 | 1.072 |
| skip `uc<=16` | 7 | 28 | 256 | 0.020503 | 0.019229 | 1.066 |
| skip `uc>16` | 42 | 32 | 64 | 0.024814 | 0.023303 | 1.065 |
| skip `uc>16` | 42 | 32 | 256 | 0.020244 | 0.020019 | 1.011 |
| skip `uc>16` | 7 | 28 | 64 | 0.022798 | 0.023402 | 0.974 |
| skip `uc>16` | 7 | 28 | 256 | 0.019684 | 0.019811 | 0.994 |

Observation:

- Removing `uc<=16` can reduce TF32 B256 time, but the best paired result is still only about
  `1.10x`.
- Removing `uc>16` does not expose a hidden TC win.
- Combined with the older all-extend and small+extend upper-bound notes, parent assembly is not
  the missing 8387 enabler.

## Large-case sanity check

Command shape:

```bash
<build>/custom_linear_solver_run /datasets/power_system/nr_linear_systems/<case> \
  --batch <64|256> --batch-only --repeat 11 \
  --precision <fp32|tf32> --single-precision fp64 \
  --panel-cap 32 --metis-seed 42
```

Raw:

- `/tmp/cls_large_extend_skip_bucket_20260609_164010.tsv`

| case | build | B | fp32 ms/sys | tf32 ms/sys | fp32/tf32 |
|---|---|---:|---:|---:|---:|
| case_ACTIVSg25k | full | 64 | 0.118062 | 0.091685 | 1.288 |
| case_ACTIVSg25k | full | 256 | 0.119704 | 0.086110 | 1.390 |
| case_ACTIVSg25k | skip `uc<=16` | 64 | 0.103377 | 0.085742 | 1.206 |
| case_ACTIVSg25k | skip `uc<=16` | 256 | 0.107091 | 0.082918 | 1.292 |
| case_ACTIVSg25k | skip `uc>16` | 64 | 0.108760 | 0.085710 | 1.269 |
| case_ACTIVSg25k | skip `uc>16` | 256 | 0.109204 | 0.083099 | 1.314 |
| case_SyntheticUSA | full | 64 | 0.457151 | 0.380693 | 1.201 |
| case_SyntheticUSA | full | 256 | 0.451664 | 0.372996 | 1.211 |
| case_SyntheticUSA | skip `uc<=16` | 64 | 0.437218 | 0.362753 | 1.205 |
| case_SyntheticUSA | skip `uc<=16` | 256 | 0.446992 | 0.360867 | 1.239 |
| case_SyntheticUSA | skip `uc>16` | 64 | 0.439549 | 0.362022 | 1.214 |
| case_SyntheticUSA | skip `uc>16` | 256 | 0.432816 | 0.351199 | 1.232 |

The repeat-11 large-case numbers are sanity checks only. The accepted repeat-61 policy evidence
remains:

- 25K B64/B256: `1.222x / 1.275x`
- USA B64/B256: `1.241x / 1.231x`

The bucket skip does not reveal a different enabler for large cases: both FP32 and TF32 move, and
the TF32 win is already explained by mid/big dense-update exposure.

## Decision

Close parent extend-add as the next 8387 Tensor Core lever. The measured 8387 ceiling after
bucketed parent-update removal remains below the requested `1.2..1.4x` range at B64/B256.

Next direction should not be another local extend-add variant. The remaining defensible routes are:

1. Exclude 8387 from the Tensor-Core speedup claim and keep the achieved 25K/USA result.
2. Change 8387 structure upstream: deeper ordering/panelization that creates larger `uc,nc`
   fronts before applying Tensor Cores.
3. Build a separate low-fill sparse-LU branch for 8387-like matrices, accepting that the enabler is
   no longer simply per-front Tensor Core trailing.
