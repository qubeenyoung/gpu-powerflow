# Analyze Bottleneck and Optimization Report

This note records what is included in `custom_linear_solver::Solver::analyze()`,
what was slow, what was optimized, and what remains.

## Current Analyze Scope

`Solver::analyze()` is value-independent. It does not read numeric matrix values
or RHS values. It prepares the fixed sparsity path used by repeated Newton
`factorize()` and `solve()` calls.

Current work:

```text
device CSR pattern
  -> device CSR-to-CSC pattern build
  -> CSC pattern download for CPU METIS
  -> METIS nested-dissection ordering
  -> reuse METIS symmetric graph and relabel it by perm/iperm
  -> permutation upload
  -> device ordered-CSC pattern build
  -> etree and filled L pattern
  -> multifrontal plan construction
  -> device arena allocation and symbolic map upload
  -> device A-entry map build from ordered CSC
  -> factor and solve CUDA graph capture
```

Removed from the hot path:

- full CSR validation in `analyze()`
- CPU CSR-to-CSC conversion
- CPU CSC permutation
- ordered CSC download for A-entry mapping
- second `symmetric_pattern()` rebuild after METIS
- unused `build_symmetric_filled`, `emit_map`, `d_Sx`, and `mf_emit`

## Baseline Breakdown

Measured on RTX 3090 with:

```bash
CLS_ANALYZE_TIME=1 MF_TIME=1 METIS_TIME=1 \
/tmp/custom_linear_solver_kernel_analyze_check/custom_linear_solver_run \
  /datasets/matpower_linear_systems/case_SyntheticUSA --repeat 1
```

Before the latest optimization pass:

```text
total analyze_ms              ~505 ms

build_csc_device                0.84 ms
download_csc_for_metis          2.98 ms
metis_nd                      231.74 ms
  adj_build                    42.7 ms
  parND                       188.6 ms

build_iperm                     2.92 ms
upload_perm_iperm               0.29 ms
permute_csc_device              0.49 ms
download_ordered_csc            1.39 ms

symmetric_pattern              38.48 ms
etree                           6.80 ms
fill_pattern                   44.05 ms

analyze_multifrontal          174.69 ms
  build_symmetric_filled       33.14 ms
  relaxed_panels                3.81 ms
  multifrontal_symbolic        27.06 ms
  a_pos                        34.55 ms
  emit_map                     71.39 ms
  arena_malloc+H2D              1.83 ms
  solve_graph_capture           2.89 ms
```

The expensive parts were no longer CSR-to-CSC or permutation. They were CPU
ordering/symbolic work and two dead/duplicated structures:

- `symmetric_pattern` rebuilt a graph already produced for METIS.
- `build_symmetric_filled` and `emit_map` produced `Sx`, but the current solve
  path reads dense fronts directly and never consumes `Sx`.
- `a_pos` did two front-row binary searches per entry even though one side is
  always a pivot column with a direct local index.

## Optimizations Applied

### 1. No-Sort Device CSR-to-CSC

File:

```text
src/matrix/pattern_kernels.cu
```

The GPU path now counts CSR columns, scans to CSC `col_ptr`, and atomic-scatters
entries into `row_idx/source_pos`. The old `thrust::sort_by_key` path was removed.

### 2. Device Ordered-CSC Build

`old CSC + iperm -> ordered CSC + ordered_value_to_csr` now runs on device.
Only the ordered CSC structure is downloaded for CPU symbolic work; the
`ordered_value_to_csr` map stays device-resident for `factorize()`.

### 3. Flat METIS Adjacency Builder

File:

```text
src/reordering/metis_nd.cpp
```

The METIS graph builder now uses flat arrays:

```text
degree count
prefix sum
flat adjacency fill
parallel per-vertex sort + unique count
prefix sum
flat METIS adjncy compact
```

This removed the large `std::vector<std::vector<int>>` allocation churn.

### 4. METIS Graph Reuse

`metis_nd()` now optionally returns the symmetric graph it already built. After
ordering, `Solver::analyze()` relabels that graph with `perm/iperm` instead of
calling `symbolic::symmetric_pattern()` again on the ordered CSC.

The relabel fill is parallelized per column.

### 5. Removed Dead Sx Emit Path

The old path built:

```text
L + L^T -> Sp/Si -> emit_front -> d_Sx
```

No current code consumed `d_Sx`; solve uses `d_front` and `d_front_rows`
directly. Removed:

- `build_symmetric_filled`
- `emit_map`
- `mf_emit`
- `d_Sx`, `d_emit_front`, `Sp`, `Si`
- unused `d_Ax` arena space

This removes about 100 ms from `case_SyntheticUSA` analyze before considering
secondary variance.

### 6. Faster A-Entry Map

For each ordered matrix entry `(i, j)`, the owner front is
`panel_of[min(i, j)]`. That minimum side is a pivot column in the owner panel, so
its local index is:

```text
min(i, j) - panel_first[owner]
```

Only the other side may need a `front_rows` lookup. This cuts `a_pos` lookup work
roughly in half.

## Current Breakdown

Instrumented single run after the latest changes:

```text
total analyze_ms              368.04 ms

build_csc_device                0.84 ms
download_csc_for_metis          2.96 ms
metis_nd                      230.88 ms
  adj_build                    43.2 ms
  parND                       187.2 ms

build_iperm                     2.83 ms
upload_perm_iperm               0.29 ms
permute_csc_device              0.50 ms
download_ordered_csc            1.40 ms
permute_metis_graph             4.52 ms

etree                           8.68 ms
fill_pattern                   50.72 ms

analyze_multifrontal           63.94 ms
  relaxed_panels                3.72 ms
  multifrontal_symbolic        33.77 ms
  a_pos                        22.42 ms
  arena_malloc+H2D              1.02 ms
  solve_graph_capture           2.98 ms
```

After applying the follow-up 1-5 pass on the same case:

```text
total analyze_ms              332.70 ms

build_csc_device                0.84 ms
download_csc_for_metis          2.96 ms
metis_nd                      228.06 ms
  adj_build                    39.1 ms
  parND                       188.5 ms

build_iperm                     2.75 ms
upload_perm_iperm               0.28 ms
permute_csc_device              0.50 ms
permute_metis_graph             3.76 ms

etree                           8.49 ms
fill_pattern                   48.87 ms

analyze_multifrontal           35.73 ms
  relaxed_panels                3.81 ms
  multifrontal_symbolic        22.88 ms
  arena_malloc+H2D              5.70 ms
  a_pos_device                  0.60 ms
  solve_graph_capture           2.73 ms
```

The later 2-5 pass kept only net-positive changes. Parallel fill-pattern and
parallel METIS count/fill were tested but not kept because they regressed this
case. Current single-case profile:

```text
total analyze_ms              335.87 ms

build_csc_device                0.85 ms
download_csc_for_metis          2.99 ms
metis_nd                      234.56 ms
  adj_build                    38.8 ms
  parND                       195.2 ms

build_iperm                     2.75 ms
upload_perm_iperm               0.30 ms
permute_csc_device              0.51 ms
permute_metis_graph             4.99 ms

etree                           8.66 ms
fill_pattern                   49.34 ms

analyze_multifrontal           30.45 ms
  relaxed_panels                3.78 ms
  multifrontal_symbolic        17.19 ms
  arena_malloc+H2D              6.24 ms
  a_pos_device                  0.50 ms
  solve_graph_capture           2.72 ms
```

The removed rows are the important part:

```text
symmetric_pattern              removed; replaced by permute_metis_graph
build_symmetric_filled         removed
emit_map                       removed
mf_emit factor graph node      removed
```

## Result

The previous 78-case sweep passed before the follow-up 1-5 pass. The latest
follow-up was validated on `case_SyntheticUSA` only, per request.

`repeat=7` timing:

```text
baseline custom:
  avg analyze 39.014 ms
  avg factor   0.220 ms
  avg solve    0.121 ms

after flat METIS adjacency + no-sort CSC:
  avg analyze 33.959 ms
  avg factor   0.218 ms
  avg solve    0.120 ms

after emit removal + METIS graph reuse:
  avg analyze 20.613 ms
  avg factor   0.219 ms
  avg solve    0.123 ms

cuDSS comparison run:
  avg analyze 26.613 ms
  avg factor   0.330 ms
  avg solve    0.156 ms
```

Representative cases:

```text
case                  n       nnz       old custom A   current A   cuDSS A
case30                53      361            1.140 ms    1.510 ms   13.821 ms
case118               181     1051           2.667 ms    1.878 ms   14.043 ms
case300               530     3736           4.541 ms    3.212 ms   14.919 ms
case_ACTIVSg2000      3607    26345         36.275 ms   17.800 ms   21.665 ms
case_ACTIVSg10k       18544   125174       136.828 ms   61.827 ms   48.422 ms
case_ACTIVSg25k       47246   318672       192.349 ms  108.279 ms   97.655 ms
case_ACTIVSg70k       134104  900558       585.466 ms  310.216 ms  249.240 ms
case_SyntheticUSA     156255  1052085      689.563 ms  346.231 ms  309.388 ms
```

Latest single-case check:

```text
case_SyntheticUSA repeat=7
  analyze   332.026 ms
  factorize   3.264 ms
  solve       1.349 ms
  relres      8.90e-12
```

Validation:

```text
cases passed          78 / 78
max relative residual 7.0e-11
```

## Remaining Work

The remaining large-case analyze bottleneck is METIS ND itself:

```text
METIS parND on SyntheticUSA    ~187 ms
```

Next priority:

```text
1. Reduce or reuse ordering across Newton iterations when the sparsity pattern is unchanged.
2. Move or rewrite fill_pattern / multifrontal_symbolic if analyze still matters.
3. Add a small-case fallback; GPU launch overhead still hurts tiny matrices.
```

## Diagnostic Knobs

Use these environment variables when profiling:

```bash
CLS_ANALYZE_TIME=1   # outer Solver::analyze() phase timing
METIS_TIME=1         # METIS adjacency / ND timing
MF_TIME=1            # multifrontal analyze subphase timing
```
