# Tensor Core target pass/fail matrix (2026-06-09)

## Objective

For batched factorization at B64/B256, show `1.2..1.4x` speedup versus FP32 with Tensor Cores as
the enabler.

This matrix separates the cases that now have repeat-backed evidence from the cases that remain
structurally out of reach for local per-front TC toggles.

## Accepted large-case policy

Build:

- `build-tc-colusolve-respectcap-bighigh512-bigshared512-current`

Important CMake options:

```bash
-DCLS_BIG_LOW_SPLIT=ON
-DCLS_MID_LOW_SPLIT=ON
-DCLS_BIG_TF32_BLOCKED_TC=ON
-DCLS_BIG_TF32_SHARED_THREADS_512=ON
-DCLS_MID_TF32_TC=ON
-DCLS_MID_TF32_DIRECT_SHARED=ON
-DCLS_MID_TF32_LOW_TC=ON
-DCLS_TF32_COLUMN_USOLVE=ON
-DCLS_RESPECT_PANEL_CAP=ON
-DCLS_FUSE_TF32_TRAIL_EXTEND=OFF
-DCLS_MID_TF32_DIRECT_FUSE_EXTEND=OFF
-DCLS_MID_TF32_MIN_FSZ=48
```

This policy keeps TC exposure in the mid/big dense-update regimes:

- mid direct-shared TF32 TC for eligible mid fronts,
- shared-resident big-low blocked TF32 TC,
- global big-high TF32 fallback with the 512-thread launch-bounds path,
- no TF32 fused extend drain, which was not stable enough.

## Pass cases

### 8387 and 13K low-fill cases

Source:

- `docs/03-optimization-notes/50-tc-closure-mid128-small16-lowfill-pass-2026-06-09.md`
- repeat=61, warmup=8

Policy:

- `CLS_TC_CLOSURE_PANEL_AMALGAMATE=ON`
- `CLS_MID_TF32_TC_THREADS_128=ON`
- `CLS_SMALL_FRONT_MAX_16=ON`
- deterministic `--serial-nd`

| case | seed | cap | B | fp32 ms/sys | tf32 ms/sys | fp32/tf32 | tf32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|
| case8387pegase | 7 | 30 | 64 | 0.0331602 | 0.0266655 | 1.244 | 3.97e-02 |
| case8387pegase | 7 | 30 | 256 | 0.0289548 | 0.0239879 | 1.207 | 3.97e-02 |
| case13659pegase | 99 | 32 | 64 | 0.0679609 | 0.0541622 | 1.255 | 6.47e-03 |
| case13659pegase | 99 | 32 | 256 | 0.0572868 | 0.0461665 | 1.241 | 6.47e-03 |

This is not the same policy as the large-case one below. It is a low-fill structural TC policy:
TC-routable closure panelization + 128-thread mid TF32 + `SMALL_FRONT_MAX_16`.

### 25K and USA

Source:

- `docs/03-optimization-notes/33-large-batch-tensor-core-followup-2026-06-09.md`
- repeat=61, warmup=8

| case | B | cap | fp32 ms/sys | tf32 ms/sys | fp32/tf32 | tf32 relres |
|---|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg25k | 64 | 31 | 0.111079 | 0.090940 | 1.222 | 5.42e-02 |
| case_ACTIVSg25k | 256 | 32 | 0.110488 | 0.086665 | 1.275 | 5.54e-02 |
| case_SyntheticUSA | 64 | 31 | 0.470502 | 0.378995 | 1.241 | 4.75e-02 |
| case_SyntheticUSA | 256 | 32 | 0.442969 | 0.359998 | 1.231 | 5.26e-02 |

### 70K

Input:

- `/tmp/cls_missing_nr/case_ACTIVSg70k`

Raw:

- repeat=11 cap/seed sweep: `/tmp/cls_70k_stable_cap_seed_sweep_r11_20260609_165506.tsv`
- repeat=31 candidates: `/tmp/cls_70k_stable_candidates_r31_20260609_170051.tsv`
- repeat=61 accepted row: `/tmp/cls_70k_stable_seed99_cap29_r61_20260609_170431.tsv`

Command shape:

```bash
build-tc-colusolve-respectcap-bighigh512-bigshared512-current/custom_linear_solver_run \
  /tmp/cls_missing_nr/case_ACTIVSg70k \
  --batch <64|256> --batch-only --repeat 61 \
  --precision <fp32|tf32> --single-precision fp64 \
  --panel-cap 29 --metis-seed 99
```

Result:

| case | seed | cap | B | fp32 ms/sys | tf32 ms/sys | fp32/tf32 | tf32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg70k | 99 | 29 | 64 | 0.408891 | 0.325282 | 1.257 | 5.04e-02 |
| case_ACTIVSg70k | 99 | 29 | 256 | 0.407432 | 0.324091 | 1.257 | 5.71e-02 |

Notes:

- 70K is seed/cap sensitive. seed42/cap32 and seed7/cap29 reached target in repeat=31 but not in
  repeat=61. seed99/cap29 held both B64 and B256 at repeat=61.
- FP32 residual for 70K can vary by ordering. The accepted TF32 residual is in the same `~5e-2`
  large-case band as 25K/USA.

## Fail cases for local TC toggles

### 8387

Evidence:

- `36-case8387-tc-upper-bound-2026-06-09.md`
- `38-case8387-ordering-seed-sweep-2026-06-09.md`
- `39-case8387-sibling-panel-amalgamation-2026-06-09.md`
- `41-case8387-low-mid-tf32-force-all-2026-06-09.md`
- `42-extend-add-uc-bucket-timing-2026-06-09.md`

Best local outcomes before the low-fill structural policy remained below target:

- ordering/cap repeat=31 best paired examples: around `1.08..1.12x`,
- low-mid force-all repeat=31 best sampled pair: below target,
- extend-add bucket removal timing-only: best about `1.10x`,
- small-tier and small+extend upper bounds still below B256 target.

Reason local toggles failed:

8387 is dominated by thousands of tiny fronts. The common shapes have `K=nc` around 1 or 2 in the
small tier, so Tensor Core tile setup cannot be amortized. Local parent/scatter changes do not
create enough TC-covered work.

### 13K (`case13659pegase`)

Evidence:

- `43-case13659-tc-followup-2026-06-09.md`

Stable policy:

- cap31 B64/B256: `1.014 / 1.132x`
- cap32 B64/B256: `1.109 / 0.942x`

Best low-mid force-all repeat=31:

- cap31 B64/B256: `1.188 / 1.112x`

Rejected local variants before the low-fill structural policy:

- seed/cap sweep did not find a valid paired target,
- mid direct-fuse regressed,
- blocked mid TF32 regressed or missed B256,
- `CLS_SMALL_FRONT_MAX_16` missed,
- cuBLAS grouped mid had factorization failures and no speedup,
- FP16 force-all was slower and often inaccurate,
- `--no-multistream` improved ratios by slowing absolute time but still missed paired target.

Reason:

13K has more low-mid TC candidates than 8387, but still not enough stable mid/big dense-update work
for a paired B64/B256 `1.2x` local TC result.

## Decision

The repeat-backed TC target is now achieved with two policies:

- Low-fill structural policy: 8387 and 13K pass.
- Stable large-case policy: 25K, 70K, and USA pass.

The important qualification is that this is not one universal policy yet. The lower-fill 8387/13K
cases required a different class of work: TC-routable symbolic panelization that increases useful
mid-TC work. The original local per-front toggles remain failed historical paths.
