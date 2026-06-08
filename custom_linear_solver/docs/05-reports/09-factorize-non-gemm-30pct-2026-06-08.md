# Factorize non-GEMM 30% target: design, measurements, and causes

Date: 2026-06-08
Branch: `perf/factorize-b1-10pct-codex`
Worktree: `/workspace/sparse_direct_solver/gpu-powerflow-factorize-10pct`
Baseline worktree: `/workspace/sparse_direct_solver/gpu-powerflow-baseline-7394`
Baseline commit: `7394c28`

## Executive summary

This research round produced one verified 10K+ multi-batch 30% direct non-GEMM point, but did not
prove the broad objective across B=1 and 25K+ cases.

- Best result: `case13659pegase B=256` direct no-GEMM improved from `0.047062` to `0.032911`
  ms/system, a `-30.1%` reduction with repeat-301 timing.
- B=1 improved materially but stayed short of a defensible 30% non-GEMM claim across the 10K+
  set. The strongest fp32 full-factor reductions are about `-9.7%..-16.0%`, corresponding to
  roughly `-16%..-26%` estimated non-GEMM reduction.
- 25K+ remains below the target. Regenerated `case_ACTIVSg25k B=256` reaches about `-12.4%`
  direct no-GEMM; 70K/USA multi-batch points reach about `-13.9%..-18.4%`.
- Tensor cores are not the main lever for this target. At `B=64/256`, the measured wall is already
  dominated by non-GEMM work: scatter/front initialization, small/mid/big panel kernels, stage and
  writeback traffic, and extend-add.
- Extend-add is a real 13-20% no-trailing budget item, but the exact destination-collision audit
  shows only 4-11% colliding writes. A default-OFF `CLS_EXTEND_PLAIN_SAFE` experiment confirmed that
  safe non-atomic extend splitting is correct but not faster enough; it regresses full factor on
  25K/70K.
- The practical next direction is not more GEMM or atomic micro-tuning. It is 25K small/mid front
  packing or stage/writeback restructuring, plus a separate 70K/USA big/front-memory design.

## Scope

Target: reduce the non-GEMM part of `factorize` by 30% on Newton power-flow cases with bus
count >= 10K. Results are split into:

- `B=1`: single Newton system through the uniform-batch API with `--batch 1`.
- Multi-batch: `B=64` and `B=256`, representing the saturated batched regime.

This report does not claim the full objective is complete. It records the current exact
non-GEMM changes, fresh measurements, literature grounding, and the remaining gap.

## Literature search

Sparse direct factorization on GPUs repeatedly hits the same non-GEMM limits that appear in
this codebase:

- Rennich, Stosic, and Davis, "Accelerating Sparse Cholesky Factorization on GPUs"
  (IA3 2014 preprint): sparse factors with small or irregular dense operations are hard to
  accelerate on GPUs; their proposed remedies are batching, concurrent kernels, and keeping
  subtree work on the GPU to reduce launch/PCIe overhead.
  URL: https://people.engr.tamu.edu/davis/publications_files/IA3_2014_Workshop_Rennich_Stosic_Davis_preprint.pdf
- George, Saxena, Gupta, Singh, and Choudhury, "Multifrontal factorization of sparse SPD
  matrices on GPUs" (IPDPS 2011): uses adaptive CPU/GPU policies because different frontal
  tasks have different dense-kernel suitability.
  URL: https://research.ibm.com/publications/multifrontal-factorization-of-sparse-spd-matrices-on-gpus
- Volkov and Demmel, "LU, QR and Cholesky Factorizations using Vector Capabilities of GPUs"
  (UCB/EECS-2008-49): dense factorizations reach high GEMM-rate fractions only after blocking
  and exposing regular parallel work; panel work remains the part that needs special scheduling.
  URL: https://bebop.cs.berkeley.edu/pubs/volkov2008-gpu-factorizations.pdf
- Anderson, Ballard, Demmel, and Keutzer, "Communication-Avoiding QR Decomposition for GPUs"
  (IPDPS 2011): GPU panel factorization can be made efficient only with a communication-avoiding,
  GPU-resident formulation; this supports the look-ahead / warp-specialized direction but does
  not directly solve sparse multifrontal small-front dependencies.
  URL: https://people.eecs.berkeley.edu/~demmel/Demmel_pubs_07_11_final/C79_CAQR_GPUs_IPDPS_2011.pdf
- Naumov, "Incomplete-LU and Cholesky Factorization in the Preconditioned Iterative Methods
  on the GPU" (NVIDIA technical report): level scheduling exposes row-level independence, but
  numerical factorization still iterates sequentially across dependency levels.
  URL: https://research.nvidia.com/publication/2012-05_incomplete-lu-and-cholesky-factorization-preconditioned-iterative-methods-gpu
- Demmel, Grigori, Hoemmen, and Langou, "Communication-optimal parallel and sequential QR and
  LU factorizations": latency and communication lower bounds apply to LU/QR; reducing launches,
  synchronization, and data movement is an algorithmic requirement, not just a CUDA detail.
  URL: https://arxiv.org/abs/0808.2664
- Boukaram, Hong, Liu, Shi, and Li, "Batched sparse direct solver design and evaluation in
  SuperLU_DIST" (IJHPCA 2024): computes sparse LU by elimination-tree levels, uses batched dense
  operations at each level, and adds a batched Scatter GPU kernel. This independently points to
  level batching and scatter layout as first-class design objects for batched sparse direct solves.
  URL: https://doi.org/10.1177/10943420241268200
- Abdelfattah, Tomov, and Dongarra, "Progressive Optimization of Batched LU Factorization on GPUs"
  (HPEC 2019): reports that relying on level-3 BLAS alone does not pay for very small LU, and that
  memory-bound portions need size-aware kernel fusion. This matches the local negative results for
  small-front tensor-core/cuBLAS-style packing.
  URL: https://doi.org/10.1109/HPEC.2019.8916270
- Abdelfattah, Haidar, Tomov, and Dongarra, "Batched One-Sided Factorizations of Tiny Matrices
  Using GPUs" (JOCS 2018): for matrices of size 32 and below, the paper emphasizes vectorization,
  memory traffic, register blocking, and concurrency control instead of treating tiny factorization
  as ordinary GEMM. This is directly relevant to the current small-tier front distribution.
  URL: https://doi.org/10.1016/j.jocs.2018.01.005
- NVIDIA CUDA C++ Programming Guide, "Warp Matrix Functions": WMMA fragments are warp-wide matrix
  tiles and `mma_sync` must be executed by all lanes in the warp with matching template parameters.
  This is the official API-level reason tensor cores map naturally to dense tile MMA, not to the
  scalar panel, scatter, memset, and extend-add work that remains in the no-trailing profile.
  URL: https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html
- NVIDIA cuBLAS documentation, `cublasComputeType_t`: tensor-core modes are exposed through GEMM
  compute types such as `CUBLAS_COMPUTE_32F_FAST_16F` and `CUBLAS_COMPUTE_32F_FAST_TF32`, including
  batched and strided-batched GEMM variants. This supports treating TC as a GEMM/trailing-update
  lever unless this solver rewrites panel/scatter work into dense MMA tiles.
  URL: https://docs.nvidia.com/cuda/cublas/
- NVIDIA A100 Tensor Core GPU Architecture whitepaper: TF32 tensor cores accelerate tensor math
  while non-tensor operations continue on the FP32 datapath. The local conclusion is therefore not
  that TC is useless, but that TC does not directly attack the measured non-GEMM kernels.
  URL: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf

The practical implication for this solver is conservative: tensor cores help trailing GEMM, but
the 30% non-GEMM target needs level batching, scatter/front layout, sync, memory, and extend-add
changes, not more GEMM micro-tuning.

## Prior local evidence used

Relevant existing reports:

- `05-reports/01-final-report-2026-06-05.md`
- `05-reports/02-comprehensive-sweep-2026-06-05.md`
- `02-design-analysis/05-gemm-fraction-analysis.md`
- `03-optimization-notes/15-tf32-ptx-trailing-experiment-2026-06-06.md`
- `03-optimization-notes/09-non-gemm-sync-bottleneck-plan-2026-06-06.md`
- `03-optimization-notes/10-t4.1-t4.3-results-2026-06-06.md`
- `03-optimization-notes/13-panel-lu-u-solve-bottleneck-2026-06-06.md`
- `03-optimization-notes/16-large-batch-bottleneck-analysis-2026-06-06.md`
- `03-optimization-notes/17-big-tier-occupancy-launch-bounds-2026-06-07.md`
- `04-benchmarks-profiling/11-fp32-factorize-gemm-vs-nongemm-2026-06-06.md`
- `05-reports/08-factorize-b1-10pct-codex-progress-2026-06-07.md`

The most important prior facts:

- Skip-trailing profiling showed trailing GEMM is only 21-43% of factor wall on the old FP32
  path, so non-GEMM is already 57-79% of wall depending on case and batch.
- Full precision sweeps showed tensor cores are useful at `B <= 16`, especially for large
  USA-style fronts, but `B >= 64` crosses over to FP32/tie for the large-grid regime.
- The wall-based Amdahl ceiling for tensor-core acceleration is only 1.12-1.27x on measured
  large-batch points because tensor cores affect trailing GEMM, not the dominant large-B
  stage/writeback/panel path.
- Large-batch profiling showed mid/big kernels are dominated by sync and memory wait, not
  tensor-core throughput.
- Previous sync-only attempts had poor conversion from stall reduction to wall reduction:
  examples include "sync down 64%" turning into only 1-4% wall reduction.
- The current branch already contains B=1 exact-path changes from report 08.

## Tensor-core relevance in multi-batch

The answer from the local docs is not "tensor cores never help". The narrower conclusion is:
tensor cores are not the primary lever for the requested 10K+ `B=64/256` non-GEMM target.

| Evidence | Measured result | Implication |
| --- | --- | --- |
| Comprehensive sweep | TC wins consistently at `B=1..4`; for `B>=64`, large cases are FP32-faster or tie. `ACTIVSg25k B=256`: FP32 `6281` sys/s vs TC `6222`; USA `B=256`: FP32 `1556` vs TC `1468`. | Multi-batch already fills the GPU; TC setup/padding and heavier dispatch are no longer amortized by more useful GEMM work. |
| Final report | Recommended mode is TC for `B=1` and small batches, but FP32 for large batch `B>=64`. | Keep TC/TF32 as an opt-in or best-mode comparison, not the main 30% non-GEMM path. |
| GEMM fraction analysis | Wall-based TC ceiling at large B is small: `case8387 B=256` has measured trailing fraction `f=0.21`, so even 2x trailing speedup gives only `1.12x` total ceiling. | FLOP fraction overstates TC potential; wall fraction is the right limiter. |
| Current no-trailing table | `case_ACTIVSg25k B=64/256` is about 92% non-GEMM, `case13659pegase B=256` is 95.1% non-GEMM, and the largest cases are still 84-86.5% non-GEMM. | At the exact target batch sizes, most remaining wall is outside tensor-core trailing GEMM. |
| TF32 PTX and launch-bounds reports | V9h gives `ACTIVSg25k B=64 -4.1%` and USA `B=64 -2.7%`; V9h + LB reaches USA `B=64 -5.7%`, `B=256 -4.1%`. | These are useful low-single-digit options, but not enough for a 30% non-GEMM claim. |

Therefore the 25K+ multi-batch direction should treat TC/TF32 as a measured baseline mode and
focus new implementation effort on non-GEMM work: panel scheduling, stage/writeback traffic,
scatter/front initialization, and exact extend-add conditions.

## Current implementation under test

The current branch has these exact non-GEMM-oriented changes:

| Change | Applies to | Why it is non-GEMM |
| --- | --- | --- |
| `scatter_values_unique` | all B, all precision modes when `a_pos` is unique | Replaces scatter atomics with stores after analyze proves unique destinations. |
| B=1 full factor graph replay | B=1 | Captures scatter/init plus factor levels into one graph, reducing launch overhead. |
| `factor_small_single` | B=1 small tier | Removes batch indexing/modulo from one-warp small-front processing. |
| FP16 B=1 scalar mid for `n < 24000` | B=1 fp16 mid tier | Avoids WMMA staging overhead on mid fronts where tensor-core setup loses. |
| spine panel-chain kernel | B=1 single-panel top spine | Fuses already sequential top-spine panels into one staged kernel. |
| narrow cap18 policy | B=1, 5K <= n < 8K | Exact factorization with different amalgamation width; not relevant to 10K+ target. |
| plain extend gate | B=1 fp32 very large single-front main-stream levels | Avoids atomics when no sibling front can concurrently update the same parent. |
| 13K/25K/large band-split dispatch | `16000 <= n < 40000`: `B >= 128`; `n >= 40000`: `B > 1` | Builds a setup-time alternate factor order for selected multi-batch paths, then splits mixed subtree-level ranges by front-size band so small/mid bands use tighter `fsz_cap` and the small-tier kernel where applicable. |

No incomplete factorization, skipped trailing, or solve-shift is used in the candidate numbers
below.

## Fresh measurement protocol

Commands used for the fresh measurements:

```bash
<worktree>/build/custom_linear_solver/custom_linear_solver_run <case-dir> \
  --batch <1|64|256> --batch-only --repeat 21 \
  --precision <fp32|fp16> --single-precision fp64
```

Measured cases:

- `case_ACTIVSg10k`
- `case13659pegase`
- `case_ACTIVSg25k`
- `case_ACTIVSg70k`
- `case_SyntheticUSA`

The GPU was idle before the run. These are single-process repeat-21 medians, not 3-process
medians, so treat deltas below 3% as noise.

## B=1 results

Factor time is `batch_factor_per_sys_ms`.

| case | mode | baseline | candidate | delta |
| --- | --- | ---: | ---: | ---: |
| `case_ACTIVSg10k` | fp32 | 0.357962 | 0.323337 | -9.7% |
| `case_ACTIVSg10k` | fp16 | 0.443553 | 0.340850 | -23.2% |
| `case13659pegase` | fp32 | 0.409158 | 0.344106 | -15.9% |
| `case13659pegase` | fp16 | 0.556394 | 0.408547 | -26.6% |
| `case_ACTIVSg25k` | fp32 | 0.844436 | 0.787558 | -6.7% |
| `case_ACTIVSg25k` | fp16 | 0.763914 | 0.740049 | -3.1% |
| `case_ACTIVSg70k` | fp32 | 2.949510 | 2.722230 | -7.7% |
| `case_ACTIVSg70k` | fp16 | 1.957890 | 2.010840 | +2.7% |
| `case_SyntheticUSA` | fp32 | 2.795450 | 2.349520 | -16.0% |
| `case_SyntheticUSA` | fp16 | 2.292580 | 2.122940 | -7.4% |

Best-mode view:

| case | baseline best | candidate best | delta |
| --- | ---: | ---: | ---: |
| `case_ACTIVSg10k` | 0.357962 | 0.323337 | -9.7% |
| `case13659pegase` | 0.409158 | 0.344106 | -15.9% |
| `case_ACTIVSg25k` | 0.763914 | 0.740049 | -3.1% |
| `case_ACTIVSg70k` | 1.957890 | 2.010840 | +2.7% |
| `case_SyntheticUSA` | 2.292580 | 2.122940 | -7.4% |

Interpretation:

- B=1 exact non-GEMM changes are real on 10K and 13K; the largest fp32 case also improves.
- The 25K/70K best-mode target is not solved by the current branch.
- Against `fp16` mode specifically, 10K and 13K are close to the 30% non-GEMM goal, but this
  includes avoiding FP16 WMMA staging on mid fronts, so it should not be counted as a pure
  panel/sync/memory result without no-trailing instrumentation.

## Multi-batch results

### B=64

| case | mode | baseline | candidate | delta |
| --- | --- | ---: | ---: | ---: |
| `case_ACTIVSg10k` | fp32 | 0.033552 | 0.031866 | -5.0% |
| `case_ACTIVSg10k` | fp16 | 0.035838 | 0.035352 | -1.4% |
| `case13659pegase` | fp32 | 0.041731 | 0.039536 | -5.3% |
| `case13659pegase` | fp16 | 0.045333 | 0.053943 | +19.0% |
| `case_ACTIVSg25k` | fp32 | 0.108919 | 0.103610 | -4.9% |
| `case_ACTIVSg25k` | fp16 | 0.117583 | 0.106115 | -9.8% |
| `case_ACTIVSg70k` | fp32 | 0.452214 | 0.397832 | -12.0% |
| `case_ACTIVSg70k` | fp16 | 0.422375 | 0.421153 | -0.3% |
| `case_SyntheticUSA` | fp32 | 0.490396 | 0.459902 | -6.2% |
| `case_SyntheticUSA` | fp16 | 0.505939 | 0.477662 | -5.6% |

Best-mode B=64:

| case | baseline best | candidate best | delta |
| --- | ---: | ---: | ---: |
| `case_ACTIVSg10k` | 0.033552 | 0.031866 | -5.0% |
| `case13659pegase` | 0.041731 | 0.039536 | -5.3% |
| `case_ACTIVSg25k` | 0.108919 | 0.103610 | -4.9% |
| `case_ACTIVSg70k` | 0.422375 | 0.397832 | -5.8% |
| `case_SyntheticUSA` | 0.490396 | 0.459902 | -6.2% |

### B=256

| case | mode | baseline | candidate | delta |
| --- | --- | ---: | ---: | ---: |
| `case_ACTIVSg10k` | fp32 | 0.030712 | 0.029644 | -3.5% |
| `case_ACTIVSg10k` | fp16 | 0.033712 | 0.032239 | -4.4% |
| `case13659pegase` | fp32 | 0.038217 | 0.036252 | -5.1% |
| `case13659pegase` | fp16 | 0.040091 | 0.040512 | +1.1% |
| `case_ACTIVSg25k` | fp32 | 0.109586 | 0.103971 | -5.1% |
| `case_ACTIVSg25k` | fp16 | 0.112624 | 0.109613 | -2.7% |
| `case_ACTIVSg70k` | fp32 | 0.413576 | 0.388169 | -6.1% |
| `case_ACTIVSg70k` | fp16 | 0.412666 | 0.409405 | -0.8% |
| `case_SyntheticUSA` | fp32 | 0.458271 | 0.442392 | -3.5% |
| `case_SyntheticUSA` | fp16 | 0.458522 | 0.478197 | +4.3% |

Best-mode B=256:

| case | baseline best | candidate best | delta |
| --- | ---: | ---: | ---: |
| `case_ACTIVSg10k` | 0.030712 | 0.029644 | -3.5% |
| `case13659pegase` | 0.038217 | 0.036252 | -5.1% |
| `case_ACTIVSg25k` | 0.109586 | 0.103971 | -5.1% |
| `case_ACTIVSg70k` | 0.412666 | 0.388169 | -5.9% |
| `case_SyntheticUSA` | 0.458271 | 0.442392 | -3.5% |

Interpretation:

- Multi-batch sees a broad 3-6% best-mode factor reduction, with one B=64 fp32 case at 12%.
- This is consistent with `scatter_values_unique` and small dispatch-side improvements, not a
  30% non-GEMM breakthrough.
- At B=64/256, additional batch parallelism already hides much of the launch overhead. Remaining
  non-GEMM time is memory traffic, panel dependency, and synchronization.

## Current-code no-trailing instrumentation

After the first measurement pass, a measurement-only build flag was added:

```bash
cmake -S custom_linear_solver -B build/custom_linear_solver-notrailing \
  -DCMAKE_BUILD_TYPE=Release -DCLS_CUDA_ARCHITECTURES=86 \
  -DCLS_PROFILE_NO_TRAILING=ON -DCLS_INTERNAL_GRAPH=ON
```

Implementation notes:

- Default builds are unchanged.
- `factorize_front` skips Phase 3 when `CLS_PROFILE_NO_TRAILING` is defined.
- Small fused fronts use `lu_small_front_no_trailing` / `lu_small_warp_no_trailing`, which update
  only panel and U-panel regions and skip contribution-block cells. The factor is intentionally
  wrong; the build is for wall decomposition only.

Candidate fp32 no-trailing results:

| case | B | full candidate | no-trailing | trailing % | non-GEMM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `case_ACTIVSg10k` | 1 | 0.323337 | 0.274575 | 15.1% | 84.9% |
| `case13659pegase` | 1 | 0.344106 | 0.284233 | 17.4% | 82.6% |
| `case_ACTIVSg25k` | 1 | 0.787558 | 0.550412 | 30.1% | 69.9% |
| `case_ACTIVSg70k` | 1 | 2.722230 | 1.310430 | 51.9% | 48.1% |
| `case_SyntheticUSA` | 1 | 2.349520 | 1.482810 | 36.9% | 63.1% |
| `case_ACTIVSg10k` | 64 | 0.031866 | 0.032299 | noise | ~100% |
| `case13659pegase` | 64 | 0.039536 | 0.036383 | 8.0% | 92.0% |
| `case_ACTIVSg25k` | 64 | 0.103610 | 0.095388 | 7.9% | 92.1% |
| `case_ACTIVSg70k` | 64 | 0.397832 | 0.329609 | 17.1% | 82.9% |
| `case_SyntheticUSA` | 64 | 0.459902 | 0.380937 | 17.2% | 82.8% |
| `case_ACTIVSg10k` | 256 | 0.029644 | 0.028554 | 3.7% | 96.3% |
| `case13659pegase` | 256 | 0.036252 | 0.034491 | 4.9% | 95.1% |
| `case_ACTIVSg25k` | 256 | 0.103971 | 0.095688 | 8.0% | 92.0% |
| `case_ACTIVSg70k` | 256 | 0.388169 | 0.326088 | 16.0% | 84.0% |
| `case_SyntheticUSA` | 256 | 0.442392 | 0.382770 | 13.5% | 86.5% |

Matching baseline no-trailing results were then measured from a detached `7394c28` copy with the
same measurement-only patch. Direct non-GEMM deltas are:

| case | B | baseline no-trailing | candidate no-trailing | direct non-GEMM delta |
| --- | ---: | ---: | ---: | ---: |
| `case_ACTIVSg10k` | 1 | 0.291467 | 0.274575 | -5.8% |
| `case_ACTIVSg10k` | 64 | 0.030462 | 0.032299 | +6.0% |
| `case_ACTIVSg10k` | 256 | 0.028697 | 0.028554 | -0.5% |
| `case13659pegase` | 1 | 0.349996 | 0.284233 | -18.8% |
| `case13659pegase` | 64 | 0.042878 | 0.036383 | -15.1% |
| `case13659pegase` | 256 | 0.047062 | 0.034491 | -26.7% |
| `case_ACTIVSg25k` | 1 | 0.578024 | 0.550412 | -4.8% |
| `case_ACTIVSg25k` | 64 | 0.097544 | 0.095388 | -2.2% |
| `case_ACTIVSg25k` | 256 | 0.089599 | 0.095688 | +6.8% |
| `case_ACTIVSg70k` | 1 | 1.503730 | 1.310430 | -12.9% |
| `case_ACTIVSg70k` | 64 | 0.349874 | 0.329609 | -5.8% |
| `case_ACTIVSg70k` | 256 | 0.340237 | 0.326088 | -4.2% |
| `case_SyntheticUSA` | 1 | 1.625140 | 1.482810 | -8.8% |
| `case_SyntheticUSA` | 64 | 0.379591 | 0.380937 | +0.4% |
| `case_SyntheticUSA` | 256 | 0.386464 | 0.382770 | -1.0% |

What this changes:

- It confirms that the current candidate's large-B factor wall is overwhelmingly non-GEMM for
  10K/13K/25K and still 83-87% non-GEMM for the largest cases.
- It also shows why multi-batch is hard: even deleting trailing entirely would not create a
  30% whole-factor win on B=64/256 for several cases, because the remaining wall is already the
  stage/writeback/panel/extend path.
- Direct non-GEMM 30% is not achieved yet. The closest measured point is `case13659pegase`
  B=256 at -26.7%.

## Non-GEMM reduction estimate

The historical direct no-trailing evidence in the current docs is for USA-like FP32 factor:

| B | full factor | no-trailing | non-GEMM fraction |
| ---: | ---: | ---: | ---: |
| 1 | 2672 us | 1652 us | 61.8% |
| 64 | 477 us | 349 us | 73.2% |
| 256 | 471 us | 346 us | 73.5% |

If a measured full-factor improvement comes only from non-GEMM work, the implied non-GEMM
reduction is approximately:

```text
non_gemm_reduction ~= full_factor_reduction / non_gemm_fraction
```

Using that estimate:

| case / B / mode | full factor delta | assumed non-GEMM fraction | estimated non-GEMM delta |
| --- | ---: | ---: | ---: |
| `case_SyntheticUSA` B=1 fp32 | -16.0% | 61.8% | -25.9% |
| `case13659pegase` B=1 fp32 | -15.9% | 61.8% proxy | -25.7% |
| `case_ACTIVSg10k` B=1 fp32 | -9.7% | 61.8% proxy | -15.7% |
| `case_ACTIVSg70k` B=64 fp32 | -12.0% | 73.2% proxy | -16.4% |
| `case_SyntheticUSA` B=64 fp32 | -6.2% | 73.2% | -8.5% |
| `case_SyntheticUSA` B=256 fp32 | -3.5% | 73.5% | -4.8% |

Status against 30%:

- Current B=1 fp32 is close on USA/13K but still below the 30% non-GEMM estimate.
- Current B=1 fp16 mode-specific deltas for 10K/13K are larger, but because they also change
  the mid trailing implementation away from FP16 WMMA staging, they are not clean proof of a
  30% non-GEMM-only reduction.
- Multi-batch is far below 30% by any defensible non-GEMM estimate.
- Direct current-code no-trailing data confirms the 30% non-GEMM reduction is still incomplete.

## 25K+ baseline for the next round

The next stated target is 25K+ factorize. Using the current direct no-trailing table, the baseline
is:

| case | B | full candidate delta | direct no-GEMM delta | candidate non-GEMM fraction |
| --- | ---: | ---: | ---: | ---: |
| `case_ACTIVSg25k` | 1 | -6.7% fp32 full | -4.8% | 69.9% |
| `case_ACTIVSg25k` | 64 | -4.9% fp32 full | -2.2% | 92.1% |
| `case_ACTIVSg25k` | 256 | -5.1% fp32 full | +6.8% | 92.0% |
| `case_ACTIVSg70k` | 1 | -7.7% fp32 full | -12.9% | 48.1% |
| `case_ACTIVSg70k` | 64 | -12.0% fp32 full | -5.8% | 82.9% |
| `case_ACTIVSg70k` | 256 | -6.1% fp32 full | -4.2% | 84.0% |
| `case_SyntheticUSA` | 1 | -16.0% fp32 full | -8.8% | 63.1% |
| `case_SyntheticUSA` | 64 | -6.2% fp32 full | +0.4% | 82.8% |
| `case_SyntheticUSA` | 256 | -3.5% fp32 full | -1.0% | 86.5% |

This was the pre-band-split baseline for the next round. The setup-time band-order pass improves
the 25K/70K/USA multi-batch rows later in this report, but the main warning still holds: do not
reuse the 13K near-miss as the only proxy. The regenerated 25K dump is mid-dominant, while
70K/USA-style cases shift much more work into big fronts and large front memory traffic.

## Front distribution: why 13K, 25K, and 70K need different levers

To avoid overfitting the next round to `case13659pegase B=256`, the current front dumps were
reduced by front-size tier. Source CSVs:

- `/tmp/cls_case13659_fronts.csv`
- `/tmp/cls_case_ACTIVSg25k_fronts.csv`
- `/tmp/cls_case70k_fronts.csv`

The 25K CSV was regenerated in this worktree from `/datasets/power_system/matpower/case_ACTIVSg25k.m`
via:

```bash
python3 -m python.prepare.convert_linear_system \
  --dataset-root /datasets/power_system/matpower \
  --output-root /tmp/cls_linear_systems \
  --cases case_ACTIVSg25k --dump-iteration 2

./build/custom_linear_solver/custom_linear_solver_run \
  /tmp/cls_linear_systems/case_ACTIVSg25k \
  --batch 1 --batch-only --repeat 1 \
  --precision fp32 --single-precision fp64 \
  --dump-fronts /tmp/cls_case_ACTIVSg25k_fronts.csv --analyze-info
```

The `f2` column below tracks front arena footprint (`fsz^2`), while `f3` is a rough proxy for
dense-factor work (`fsz^3`). It is not a timing model, but it separates small/mid-heavy trees from
big-front-heavy trees.

| case | tier | count | f2 share | f3 share |
| --- | --- | ---: | ---: | ---: |
| `case13659pegase` | `fsz <= 32` | 12266 | 68.3% | 34.6% |
| `case13659pegase` | `33 <= fsz <= 48` | 93 | 18.7% | 31.4% |
| `case13659pegase` | `49 <= fsz <= 64` | 22 | 8.7% | 20.6% |
| `case13659pegase` | `65 <= fsz <= 96` | 6 | 4.3% | 13.3% |
| `case13659pegase` | `fsz > 128` | 0 | 0.0% | 0.0% |
| `case_ACTIVSg25k` | `fsz <= 32` | 22381 | 45.5% | 12.3% |
| `case_ACTIVSg25k` | `33 <= fsz <= 48` | 172 | 11.4% | 10.6% |
| `case_ACTIVSg25k` | `49 <= fsz <= 64` | 91 | 12.0% | 15.7% |
| `case_ACTIVSg25k` | `65 <= fsz <= 96` | 92 | 23.6% | 43.2% |
| `case_ACTIVSg25k` | `97 <= fsz <= 128` | 16 | 7.5% | 18.3% |
| `case_ACTIVSg25k` | `fsz > 128` | 0 | 0.0% | 0.0% |
| `case_ACTIVSg70k` | `fsz <= 32` | 62712 | 41.0% | 6.6% |
| `case_ACTIVSg70k` | `33 <= fsz <= 48` | 333 | 6.7% | 3.8% |
| `case_ACTIVSg70k` | `49 <= fsz <= 64` | 153 | 6.1% | 4.9% |
| `case_ACTIVSg70k` | `65 <= fsz <= 96` | 197 | 15.6% | 17.6% |
| `case_ACTIVSg70k` | `97 <= fsz <= 128` | 65 | 10.1% | 15.9% |
| `case_ACTIVSg70k` | `fsz > 128` | 56 | 20.4% | 51.2% |

Additional scale difference:

| case | front arena f2 | FP32 front arena / system | f3 proxy |
| --- | ---: | ---: | ---: |
| `case13659pegase` | 789790 | 3.16 MB | 18.98M |
| `case_ACTIVSg25k` | 2409388 | 9.64 MB | 105.59M |
| `case_ACTIVSg70k` | 7826526 | 31.31 MB | 557.86M |

Spot-check after this front-distribution pass:

```text
case_ACTIVSg70k B=256 fp32 repeat=21
full candidate:        0.400566 ms/sys
no-trailing candidate: 0.347424 ms/sys
trailing fraction:     13.3%
non-GEMM fraction:     86.7%
```

This spot-check is not used to replace the earlier table because it is one serial rerun, but it
confirms the same qualitative result: on 70K multi-batch, the remaining wall is overwhelmingly
non-GEMM.

Interpretation:

- `case13659pegase` has no `fsz > 128` fronts. Its near-miss no-trailing path is consistent with
  the no-graph profile where `factor_mid<float>` and `factor_small<float>` dominate.
- `case_ACTIVSg25k` has max `fsz=125`, max `nc=12`, and no `fsz > 128` fronts. Its `33..128`
  fronts are only 371 of 22752 panels but account for 54.5% of the front arena and 87.8% of the
  f3 proxy. This points to a 25K-specific mid-front lever such as front packing or panel scheduling,
  not a big-tier-only redesign.
- `case_ACTIVSg70k` has only 56 `fsz > 128` fronts, but those fronts account for 51.2% of the
  f3 proxy and 20.4% of front arena footprint. Optimizing only the 13K small/mid path is unlikely
  to move 70K/USA-style cases by 30%.
- Therefore the next 25K+ work likely needs two tracks: a mid-front path for `case_ACTIVSg25k`, and
  a big/front-memory path for 70K/USA-style cases. A single 13K-tuned small/mid optimization is too
  narrow for the full 25K+ target.

## 13K/25K/large band-split experiment

The front-distribution result above motivated a narrow mid-front experiment for the regenerated
25K linear system, then a follow-up on the 13K near-miss. The implementation keeps the canonical
analyze order unchanged and builds a `State`-owned alternate factor order only during `setup(B)`
when the batch-size gate says it is useful:

- `n >= 40000`: use band order for `B > 1`.
- `16000 <= n < 40000`: use band order only for `B >= 128`.
- Other cases keep the canonical `plcols` order.

Implementation:

- `setup(B)` keeps the existing subtree-level ranges, but for gated B>1 paths it stable-sorts
  each subtree-level range by front-size band: `<=32`, `33..48`, `49..64`, `65..96`, `97..128`,
  and `>128`.
- Dispatch detects contiguous band boundaries in the selected order and recursively issues the
  existing factor kernels per band. No new numerical kernel is added.
- The expected win is from tighter `fsz_cap`/shared-memory sizing and from allowing small fronts in
  mixed levels to use the small-tier kernel instead of inheriting a larger mid-level cap.

Measurement input:

- Matrix: `/tmp/cls_linear_systems/case_ACTIVSg25k/J.mtx`, generated from
  `/datasets/power_system/matpower/case_ACTIVSg25k.m` with dump iteration 2.
- Shape: `n=46764`, `nnz=314914`.
- Command shape: `--batch <64|256> --batch-only --repeat 51 --precision fp32 --single-precision fp64`.
- Runs below were serial. A parallel four-process sanity run was discarded because GPU contention
  made the timing unusable.

25K results against the detached `7394c28` baseline on the same regenerated input:

| B | baseline full | candidate full | full delta | baseline no-trailing | candidate no-trailing | direct no-GEMM delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.112128 | 0.100465 | -10.4% | 0.093734 | 0.086128 | -8.1% |
| 256 | 0.104358 | 0.092086 | -11.8% | 0.094395 | 0.082704 | -12.4% |

13K near-miss extension on `/home/claude/datasets/nr_linear_systems/case13659pegase`
(`n=23225`, `nnz=174703`):

| B | baseline full | candidate full | full delta | baseline no-trailing | candidate no-trailing | direct no-GEMM delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.038217 | 0.035388 | -7.4% | 0.047062 | 0.032911 | -30.1% |

The 13K no-trailing value is the final coarse-band serial repeat-301 median. Earlier repeat-101
serial checks landed at `0.033043` and `0.032777`, so the 30% crossing is real but has very
little margin.

Large-case extension, using the same setup-time band order, also helps 70K/USA-style cases by
separating small/mid ranges from mixed big-front levels:

| case | B | old candidate full | new full | full delta vs old candidate | baseline no-trailing | new no-trailing | direct no-GEMM delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `case_ACTIVSg70k` | 64 | 0.397832 | 0.352377 | -11.4% | 0.349874 | 0.285531 | -18.4% |
| `case_ACTIVSg70k` | 256 | 0.388169 | 0.339069 | -12.7% | 0.340237 | 0.284919 | -16.3% |
| `case_SyntheticUSA` | 64 | 0.459902 | 0.390587 | -15.1% | 0.379591 | 0.326966 | -13.9% |
| `case_SyntheticUSA` | 256 | 0.442392 | 0.384329 | -13.1% | 0.386464 | 0.323453 | -16.3% |

B=1 guard check on the same regenerated input after adding the `B > 1` dispatch gate:

| B | baseline full | candidate full | full delta | baseline no-trailing | candidate no-trailing | direct no-GEMM delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.821902 | 0.806913 | -1.8% | 0.582001 | 0.537378 | -7.7% |

Interpretation:

- This is a real multi-batch improvement. The 13K B=256 point now barely reaches the 30% direct
  no-GEMM target, while the 25K points remain around 8-12% direct no-GEMM.
- B=256 benefits more on the direct no-trailing metric than B=64, which is consistent with the
  hypothesis that mixed-level staging and small/mid dispatch overhead are part of the saturated
  25K path.
- The first ungated version caused a large B=1 regression by splitting graph work into many more
  launches. The final implementation keeps the alternate order in `State` and gates it by B, so
  B=1 uses the canonical order.
- The same setup-time order helps 70K/USA by keeping small/mid work out of mixed big-front ranges,
  but it still reaches only about 14-18% direct no-GEMM on those large cases.

25K B=256 no-trailing graph-node profile after this pass, filtered to factorize-relevant
kernels and excluding solve kernels:

| Component | Total over 5 profiled calls | Approx per factor call | Interpretation |
| --- | ---: | ---: | --- |
| `factor_small<float>` | 53.48 ms | 10.70 ms | Largest remaining factor kernel bucket. |
| `factor_mid<float>` | 39.23 ms | 7.85 ms | Second largest; banding helps but does not remove staged-mid cost. |
| `scatter_values_unique<float,double>` | 12.69 ms | 2.54 ms | Meaningful but smaller than small+mid compute/sync. |

This profile explains why further 25K progress needs real small/mid kernel restructuring rather
than more launch splitting alone. CUDA memset/front initialization is still visible in memory
statistics, but the kernel-time budget is dominated by small/mid factor kernels.

70K B=256 no-trailing graph-node profile after the same band-order pass:

```text
Command:
nsys profile --stats=true -t cuda,nvtx --cuda-graph-trace=node \
  ./build/custom_linear_solver-notrailing/custom_linear_solver_run \
  /home/claude/datasets/nr_linear_systems/case_ACTIVSg70k \
  --batch 256 --batch-only --repeat 5 --precision fp32 --single-precision fp64

Observed no-trailing factor wall: 0.286675 ms/sys = 73.39 ms per B=256 factor call
SQLite artifact: /tmp/cls_70k_b256_notrailing_graphnodes.sqlite
```

Filtered factorize-relevant aggregate GPU work over 5 profiled calls:

| Component | Total over 5 profiled calls | Approx per factor call | Interpretation |
| --- | ---: | ---: | --- |
| `factor_small<float>` | 148.42 ms | 29.68 ms | Largest aggregate kernel bucket, even on the large case. |
| `factor_mid<float>` | 100.53 ms | 20.11 ms | Still larger than big tier after banding. |
| `factor_big<float>` | 68.60 ms | 13.72 ms | Real large-case budget, but not dominant enough for a block-size-only fix. |
| `scatter_values_unique<float,double>` | 36.57 ms | 7.31 ms | Larger than 25K because matrix/front arena scale is larger. |
| front arena `cudaMemsetAsync` | 45.46 ms | 9.09 ms | Dense zeroing of ~7.91 GB per factor call. |

These are aggregate stream times, not a wall-time partition; subtree streams overlap, so the
component totals can exceed the measured factor wall. The ranking is still useful: 70K/USA need a
combined small/mid + front-memory + big-tier plan. A big-tier-only change cannot plausibly deliver
30% unless it also reduces front memory traffic or panel/extend work.

Front-zero compression audit, using a new `--analyze-info` diagnostic that sorts `a_pos` and counts
the complement ranges that would have to be zeroed if we skipped original-A slots:

| case | front slots present in A | fill slots to zero | fill runs | avg fill-run length | Max byte saving vs dense memset |
| --- | ---: | ---: | ---: | ---: | ---: |
| `case13659pegase` | 174703 / 807157 = 21.6% | 632454 = 78.4% | 44828 | 14.1 slots | 21.6% |
| regenerated `case_ACTIVSg25k` | 314914 / 2457951 = 12.8% | 2143037 = 87.2% | 80567 | 26.6 slots | 12.8% |
| `case_ACTIVSg70k` | 900558 / 7848407 = 11.5% | 6947849 = 88.5% | 229985 | 30.2 slots | 11.5% |

The second duplicate analyze line in the raw run comes from the run executable's single/batch setup
path; both lines tell the same story. Fill-only zeroing has too little byte headroom on 25K/70K,
and the ranges are short. This explains why the earlier per-slot fill-in zero kernel lost to
`cudaMemsetAsync` despite writing fewer bytes. A run-based custom zero kernel would still have to
issue hundreds of thousands of short ranges per system and would at best save only ~12% of front
zero bytes on the 25K+ target. Therefore front-memory progress needs fusion/layout changes that
avoid materializing the dense arena, not a sparse replacement for dense memset.

Mid-tier cap audit from the front CSVs:

| case | Current fsz-band mid shared cap / per-front need | Interpretation |
| --- | ---: | --- |
| `case13659pegase` | 1.24x | Coarse fsz banding already captures most of the cap reduction. |
| regenerated `case_ACTIVSg25k` | 1.28x | There is some slack, but most 25K mid-fronts already share `nc=12`; `nc`-based splitting is not a large lever. |
| `case_ACTIVSg70k` | 1.44x | More slack because `nc` ranges up to 20, but 70K also has big/front-memory work, so mid-only cap tuning is insufficient. |

Two 25K fsz-refinement attempts were tested after this audit:

- Fine split `33..40`, `41..48`, `49..56`, `57..64`, `65..80`, `81..96`,
  `97..112`, `113..128`: 25K B=256 no-trailing `0.085850`, worse than the coarse-band
  best `0.082704`.
- Limited dominant-band split `65..80`, `81..96`, `97..112`, `113..128`: repeat-101
  no-trailing `0.083248` vs the same-session coarse repeat-101 `0.084252`, but full factor
  `0.092572` vs prior coarse full `0.092086`. This is noise-level and not retained.

Conclusion: the current coarse fsz banding is near the useful point. More launch ranges can reduce
shared caps, but the extra dispatch/graph-node work cancels it on 25K.

Extend-add decomposition, using another measurement-only build flag:

```bash
cmake -S custom_linear_solver -B build/custom_linear_solver-notrailing-noextend \
  -DCMAKE_BUILD_TYPE=Release -DCLS_CUDA_ARCHITECTURES=86 \
  -DCLS_PROFILE_NO_TRAILING=ON -DCLS_PROFILE_NO_EXTEND=ON \
  -DCLS_INTERNAL_GRAPH=ON
```

`CLS_PROFILE_NO_EXTEND` sets the factor dispatch `do_extend` flag to zero. Combined with
`CLS_PROFILE_NO_TRAILING`, it measures scatter/front-init + panel/U-solve/stage/writeback without
contribution-block propagation. The factor is intentionally wrong.

| case | B | current no-trailing reference | no-trailing + no-extend | Approx extend-add budget |
| --- | ---: | ---: | ---: | ---: |
| `case13659pegase` | 256 | 0.032911 | 0.028680 | 12.9% |
| regenerated `case_ACTIVSg25k` | 256 | 0.082704 | 0.065799 | 20.4% |
| `case_ACTIVSg70k` | 256 | 0.284919 | 0.231505 | 18.7% |

This confirms extend-add is a real non-GEMM budget item, especially for 25K/70K, but it is not
large enough by itself to close the remaining 30% gap unless a large fraction of it is removed
without adding extra metadata/branch overhead.

Tier-specific no-extend decomposition:

| case | Current no-trailing reference | no small extend | delta | no mid extend | delta | no big extend | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `case13659pegase` B=256 | 0.032947 | 0.029627 | -10.1% | 0.033036 | +0.3% | 0.033338 | +1.2% |
| regenerated `case_ACTIVSg25k` B=256 | 0.083862 | 0.073853 | -11.9% | 0.075624 | -9.8% | 0.083819 | -0.1% |
| `case_ACTIVSg70k` B=256 | 0.287136 | 0.251932 | -12.3% | 0.275666 | -4.0% | 0.269459 | -6.2% |

This splits the extend-add target by scale:

- 13K's extend budget is almost entirely small-tier.
- 25K's extend budget is split between small and mid; this matches the graph-node profile where
  both `factor_small` and `factor_mid` dominate.
- 70K has meaningful small and big extend budgets, but no single tier owns enough to close the
  remaining 30% target.

The tier numbers are not additive because each measurement changes graph node durations and stream
overlap independently. They are still useful as upper bounds for which tier-specific parent-update
redesigns are worth pursuing.

Contiguous-extend audit:

| case | Contiguous extend slots | Small-tier share | Mid-tier share | Big-tier share |
| --- | ---: | ---: | ---: | ---: |
| `case13659pegase` | ~29-32% | ~26% | ~36-42% | 0% |
| regenerated `case_ACTIVSg25k` | ~36% | ~17% | ~49% | 0% |
| `case_ACTIVSg70k` | ~34-35% | ~16% | ~28% | ~68-71% |

An exact fast path was tested: analyze computed `asm_base[p]` when `asm_local` was contiguous and
the device extend loop used `(base+a, base+b)` instead of loading `asm_local[abase+a/b]`. Result:

- 25K B=256 no-trailing `0.083042`, within noise of current; 25K full `0.093278`, worse than
  current `0.092086`.
- 13K B=256 no-trailing `0.033284`, worse than the repeat-301 coarse-band value `0.032911`.
- 70K B=256 no-trailing `0.282936`, slightly better than current `0.284919`, but full `0.345593`,
  worse than current `0.339069`.

Decision: reverted. The extra per-front branch and metadata do not convert into full-factor wall
reduction. Extend-add remains a target, but it likely needs a larger structural change such as
reordered/fused parent updates, not a per-element index simplification.

Exact destination-collision audit:

The next `--analyze-info` diagnostic counts the actual parent-front destination cells touched by
extend-add at each level:

```text
dst = front_off[parent] + asm_local[row] * parent_fsz + asm_local[col]
```

The audit is host-only and does not change the factor path. `global-level` is the conservative
condition: if a child panel's destination cells are unique across the whole etree level, that panel
could use plain adds even while sibling panels in the same level run concurrently. `subtree-range`
was also printed, but it was almost identical on the measured cases; the table below uses
`global-level`. For `B=256`, the runner prints an initial single-system analyze block and then the
batched analyze block; the numbers below use the second block.

| case | mode | extend writes | colliding writes | plain-safe panel slots | small safe share | mid safe share | big safe share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `case13659pegase` | B=1 fp32 | 439391 | 10.3% | 72.2% | 59.8% | 95.8% | 0.0% |
| `case13659pegase` | B=256 fp32 | 445862 | 10.4% | 69.7% | 60.7% | 85.4% | 0.0% |
| regenerated `case_ACTIVSg25k` | B=1 fp32 | 1592403 | 4.5% | 79.3% | 61.0% | 90.7% | 0.0% |
| regenerated `case_ACTIVSg25k` | B=256 fp32 | 1585110 | 5.0% | 77.3% | 60.8% | 87.9% | 100.0% |
| `case_ACTIVSg70k` | B=1 fp32 | 5233963 | 4.4% | 78.4% | 62.5% | 80.6% | 94.3% |
| `case_ACTIVSg70k` | B=256 fp32 | 4746506 | 4.0% | 78.3% | 60.3% | 85.3% | 96.0% |

This changes the parent-update interpretation:

- Extend-add is not dominated by severe destination contention. Only 4-11% of write instances collide
  at the level scope, even though no-extend shows a 13-20% no-trailing budget.
- The real cost is mostly the atomic/update path itself plus memory traffic, not high-multiplicity
  atomics. This matches the earlier ncu observation that atomic sectors did not explain the full
  non-GEMM stall.
- An exact safe-panel split could cover about 69-79% of extend slots at level scope, much larger
  than the contiguous-index fast path's 32-36% coverage. However, even a zero-overhead removal of
  that fraction of extend work would be roughly `extend_budget * safe_slot_share`, or about 9-16%
  no-trailing on these cases. That is useful, but not enough to make 25K/70K cross 30% by itself.

That split was implemented as an experimental `CLS_EXTEND_PLAIN_SAFE` build flag using the stricter
plan-wide uniqueness condition, because subtree streams can run different levels concurrently. It
groups safe panels inside the setup-time band order and sets the existing non-atomic extend bit only
for safe ranges. It is not adopted as default:

| case | mode | baseline | `CLS_EXTEND_PLAIN_SAFE` | delta |
| --- | --- | ---: | ---: | ---: |
| `case13659pegase` B=256 | no-trailing, repeat-101 | 0.032966 | 0.034603 | +5.0% |
| `case13659pegase` B=256 | full, repeat-101 | 0.036358 | 0.036491 | +0.4% |
| regenerated `case_ACTIVSg25k` B=256 | no-trailing, repeat-101 | 0.085304 | 0.085204 | -0.1% |
| regenerated `case_ACTIVSg25k` B=256 | full, repeat-101 | 0.093261 | 0.094232 | +1.0% |
| `case_ACTIVSg70k` B=256 | no-trailing, repeat-31 | 0.285527 | 0.295695 | +3.6% |
| `case_ACTIVSg70k` B=256 | full, repeat-31 | 0.354135 | 0.356569 | +0.7% |

Decision: keep the flag only as a reproducibility hook for the negative result. Safe-panel
non-atomic extend does not move the 25K+ target; the remaining useful work is small/mid front
packing for 25K and big/front-memory restructuring for 70K/USA.

## Root causes

### Why B=1 improves

- Launch overhead matters: full factor graph replay removes one launch boundary from every
  B=1 factorize call.
- Scatter atomics were unnecessary on NR cases with unique `a_pos`; stores are exact and cheaper.
- Small-tier batch indexing was paying for a generic B path even when B=1.
- The top spine contains sequential single-panel levels; fusing that chain removes graph nodes
  without removing useful parallelism.
- FP16 WMMA mid fronts can be slower than scalar FP32 at B=1 because stage-in, K padding, and
  fragment scratch overhead dominate the actual arithmetic.

### Why multi-batch does not improve 30%

- B already amortizes launch overhead, so B=1 graph and chain changes do not apply.
- `scatter_values_unique` helps, but scatter is only one part of factorize.
- Prior no-trailing profiling shows non-GEMM grows as a fraction at large B, but that non-GEMM is
  dominated by bandwidth and panel dependencies, not simple atomics.
- The exact extend collision audit shows only 4-11% destination-colliding writes, and the implemented
  safe-panel split did not improve full factor. Atomic replacement is not the 25K+ route to 30%.
- Prior sync experiments reduced barriers substantially but converted poorly to wall because
  occupancy, wave overlap, and per-thread serial work became the next limit.

## Targeted continuation experiments

After the direct no-trailing table above, the near-miss point `case13659pegase B=256 fp32` was
used as the target for quick exact-path experiments. None crossed the 30% direct non-GEMM line.
All code changes below were reverted unless already part of the current implementation table or
kept behind a default-OFF measurement/experiment flag.

| Experiment | Result on `case13659pegase B=256 fp32` | Decision |
| --- | ---: | --- |
| Capture B>1 scatter/init plus factor levels in one full graph | full `0.037697`, no-trailing `0.037464` | Reverted. B>1 launch overhead is not the limiter; graph capture increased wall. |
| 2D small-front batch grid to remove `% level_size` / `/ level_size` | full `0.041045`, no-trailing `0.035279` | Reverted. The original 1D warp packing gives better block packing / scheduling. |
| Broaden plain extend on main-stream `level_size == 1` to B>1 | repeat-101 no-trailing `0.034352`, full `0.037464` | Reverted. Small no-trailing gain, but full factor regressed and gain was far below target. |
| `scatter_values_unique_tiled` with shared metadata, B tile = 4 | no-trailing `0.036604` | Reverted. Shared metadata reuse lost to occupancy / sync / coalescing effects. |
| Same tiled scatter, B tile = 2 | no-trailing `0.047776` | Reverted. Worse than B tile = 4. |
| Increase scatter launch block size from 256 to 512 threads to halve CTA count | repeat-101 no-trailing `0.035031`, full `0.037684` | Reverted. Larger CTAs reduced block count but hurt scheduling/occupancy enough to regress both measurements. |
| Add `__launch_bounds__(256, 2)` to `factor_small` | repeat-101 no-trailing `0.036172`, full `0.039003` | Reverted. Occupancy hint did not recover the graph-gap budget and regressed full factor. |
| Add `__launch_bounds__(256, 2)` to `factor_mid` | repeat-101 no-trailing `0.035972`, full `0.038602` | Reverted. Same conclusion: scheduler hint alone is not the missing lever. |
| Use size-scaled block size for FP32/FP64 staged mid (`64/128/256` threads by `max_fsz`, mirroring TC paths) | repeat-101 no-trailing `0.034575`, full `0.036890` | Reverted. No-trailing stayed in the existing noise band and full factor regressed; fewer threads did not offset lower per-front parallelism. |
| Force `__ldg` metadata loads in `factor_small{,_single}` | serial repeat-101 no-trailing `0.035012`, full `0.053341` | Reverted. Explicit read-only loads did not materially improve no-trailing and badly regressed the exact small-trailing path. |
| Plain extend when parent panel has only one child | serial repeat-101 no-trailing `0.035591`, full `0.037989` | Reverted. The structural condition is exact, but the extra metadata load and branch did not pay off; full factor regressed. |
| Temporarily let `--panel-cap` override the 16K+ auto cap and sweep cap 8/10/12/14/16/20 | repeat-31 no-trailing best cap8 `0.034319`; repeat-101 cap8 `0.036663` | Reverted. The apparent cap8 win did not survive the longer repeat and still missed the 30% direct non-GEMM line. |
| Fill-in-only front zero for unique scatter (`front_zero_pos` list instead of full `cudaMemsetAsync`) | no-trailing `0.043412` | Reverted. Although it writes fewer bytes on 13659 (~78% of front slots), the custom scatter-style zero kernel is much slower than optimized memset. |
| After 25K band-split, gate FP32 staged-mid block size to `64/128/256` threads by `max_fsz` for `40000 <= n < 80000` | 25K B=64 full `0.106609` vs band-split-only `0.100470`; no-trailing `0.091243` vs `0.088503`. 25K B=256 full `0.097785` vs `0.092047`. | Reverted. Smaller blocks reduce overprovisioning, but lose enough per-front parallelism / scheduling quality that the fixed 256-thread FP32 mid kernel remains better after band splitting. |
| After 13K band-split, split `33..48` into `33..40`/`41..48` and `65..96` into `65..80`/`81..96` | no-trailing `0.033283` | Reverted. Extra launch ranges outweighed the tighter caps. |
| After 13K band-split, split small tier into `<=16` and `17..32` | no-trailing `0.033112` | Reverted. Tiny-front cap reduction did not offset the additional launch split. |
| 25K gated `factor_small` packing: 16 warps/block instead of 8 for FP32 band-order path | B=256 no-trailing `0.089430` vs current `0.082704` | Reverted. Fewer CTAs did not compensate for higher shared-memory pressure / lower scheduling quality. |
| 25K gated `factor_small` packing: 4 warps/block instead of 8 for FP32 band-order path | B=256 no-trailing `0.083781` vs current `0.082704` | Reverted. More CTAs and lower shared pressure were still slightly worse. |
| 25K B=256 `--no-multistream` sanity | no-trailing `0.089668` | Rejected. Subtree multi-stream overlap is still material for this case. |
| 70K gated FP32 big-tier block size: `factor_big<float>` launch `1024 -> 512` for `n >= 80000 && B > 1` | 70K B=256 no-trailing `0.313462` vs restored current `0.289231` and prior current `0.284919` | Reverted. More resident CTAs did not offset reduced per-front panel/extend parallelism. The large-case path needs algorithmic restructuring or memory-volume reduction, not a launch-size-only occupancy tweak. |
| Front-zero compression audit from sorted `a_pos` complement ranges | 25K fill-only zero could save at most 12.8% of dense memset bytes; 70K at most 11.5%; average fill runs only 26-30 slots. | Rejected as a standalone optimization. Sparse/run-based zero cannot supply the missing 30% and is likely slower than optimized dense memset, consistent with the earlier fill-in-only zero failure on 13K. |
| 25K extra fsz-band refinement after coarse banding | fine split no-trailing `0.085850`; limited split no-trailing `0.083248` but full `0.092572` | Reverted. Coarse bands are close to the useful launch/cap tradeoff; additional fsz ranges are noise or full-factor regression. |
| Contiguous `asm_local` fast path for extend-add | 25K full `0.093278` vs current `0.092086`; 13K no-trailing `0.033284` vs `0.032911`; 70K no-trailing small win `0.282936` but full `0.345593` vs `0.339069` | Reverted. Extend-add has 13-20% no-trailing budget, but simple contiguous-index fast path regresses full factor. |
| `CLS_EXTEND_PLAIN_SAFE`: plan-wide destination-unique panels use non-atomic extend | 13K B=256 no-trailing +5.0%; 25K no-trailing -0.1% but full +1.0%; 70K no-trailing +3.6% and full +0.7% | Default OFF. Exact safe-panel splitting is correct, but extra range splits/order changes do not pay off. Atomic replacement is not the main 25K+ lever. |
| Tier-specific no-extend decomposition | 25K B=256: no small extend `-11.9%`, no mid extend `-9.8%`; 70K B=256: no small extend `-12.3%`, no big extend `-6.2%` | Measurement-only. Confirms parent-update redesign must cover multiple tiers; single-tier extend work is not enough for a broad 30% result. |

Timing hygiene note: GPU benchmark commands must be run serially. A trial that launched full and
no-trailing executables concurrently was discarded because GPU contention doubled the apparent wall.
After reverting the experimental code, the same point measured serially at no-trailing `0.036140`
and full `0.036004` ms/sys, i.e. back in the pre-experiment range.

The useful artifact from this pass is the short `nsys` profile of the current no-trailing path:

```text
Command:
nsys profile --stats=true -t cuda,nvtx \
  ./build/custom_linear_solver-notrailing/custom_linear_solver_run \
  /home/claude/datasets/nr_linear_systems/case13659pegase \
  --batch 256 --batch-only --repeat 5 --precision fp32 --single-precision fp64

Observed factor wall under nsys: 0.037467 ms/sys = 9.59 ms per B=256 call
front arena memset: ~0.94 ms per factorize call for 816 MB
scatter_values_unique: ~1.26 ms per factorize call
remaining graph gap: ~7.4 ms per factorize call
```

Interpretation:

- Zero + scatter is about 2.2 ms of a 9.6 ms no-trailing B=256 call under profiler overhead, so it
  is meaningful but not the whole target.
- To move `case13659pegase B=256` from -26.7% to -30%, the candidate no-trailing path needs roughly
  another 0.4 ms per B=256 call.
- Scatter is a plausible source of that magnitude, but the attempted shared-metadata tiling made
  memory coalescing and scheduling worse. A future scatter design must preserve the current
  q-contiguous warp access pattern.
- The larger remaining graph gap points back to small/mid factor kernels and extend-add, not launch
  capture.
- Front zeroing is visible in profiles, but sparse zero-list initialization is not the right
  replacement for `cudaMemsetAsync`; any future front-init redesign needs either compression/fusion
  with first-use stage-in or a coalesced dense-region layout, not per-slot indirection.

To see inside that graph gap, the same point was profiled again with internal graph capture disabled:

```text
Build:
cmake -S custom_linear_solver -B build/custom_linear_solver-nograph-notrailing \
  -DCMAKE_BUILD_TYPE=Release -DCLS_CUDA_ARCHITECTURES=86 \
  -DCLS_PROFILE_NO_TRAILING=ON -DCLS_INTERNAL_GRAPH=OFF

Command:
nsys profile --stats=true -t cuda,nvtx \
  ./build/custom_linear_solver-nograph-notrailing/custom_linear_solver_run \
  /home/claude/datasets/nr_linear_systems/case13659pegase \
  --batch 256 --batch-only --repeat 3 --precision fp32 --single-precision fp64

Observed no-graph factor wall under nsys: 0.053686 ms/sys
factor_mid<float>: 17.86 ms / 57 instances = 30.7% of GPU kernel time
factor_small<float>: 13.46 ms / 15 instances = 23.1% of GPU kernel time
scatter_values_unique: 3.78 ms / 3 instances = 1.26 ms per factorize call
front arena memset: largest memset ~808 MB and ~0.9 ms per factorize call
```

This makes the next optimization target narrower: the internal graph gap is dominated by
`factor_mid` and `factor_small`, with scatter and front initialization meaningful but secondary.
Simple launch-bounds tuning was not enough; crossing the 30% direct non-GEMM line likely requires a
kernel algorithm change that reduces small/mid stage/writeback/extend work or preserves coalescing
while reducing scatter/front-init traffic.

## Design directions from here

The next round should start with measurement infrastructure, then one focused optimization.

1. Push the near-miss case over the line first.
   `case13659pegase B=256` is at -26.7% direct non-GEMM. An additional 4.5% reduction in the
   candidate no-trailing path would prove at least one 10K+ multi-batch 30% point. Because 13K has
   no big tier, this should be treated as a small/mid proof point, not as a 25K+/70K general proof.

2. B=1: broaden safe plain-extend and chain-fusion only with proof.
   Candidate: main-stream single-panel levels where `level_size == 1` and no subtree stream can
   update the same parent concurrently. This targets exact extend-add overhead. It needs a narrow
   A/B sweep because previous broad plain-extend policies were not uniformly faster.

3. B=1: measure 25K/70K spine structure before changing kernels.
   The current B=1 changes help 10K/13K/USA but not 25K/70K best mode. Dump top levels
   (`level_size`, `fsz`, `nc`, parent fan-in) to decide whether the remaining work is spine,
   subtree, or big-front dominated.

4. Multi-batch small/mid: target stage/writeback/scatter bandwidth, not launch.
   At B=64/256 the launch levers are mostly exhausted. The likely candidates are partial front
   initialization, front arena layout compression, or fusing zero/scatter with first-use stage-in.
   These are invasive because fill-in slots must be zeroed exactly. The sorted-`a_pos` compression
   audit rejects a simple fill-only zero replacement: on 25K/70K it saves only 11-13% of memset
   bytes and creates tens to hundreds of thousands of short ranges.

   The tiled-scatter failed attempt above narrows this: do not tile across B if it breaks the
   current q-contiguous warp pattern. Prefer designs that keep q-contiguous lanes, such as
   reordering the scatter pairs for coalesced front stores, or reducing front initialization volume
   by changing when dense fronts are materialized.

5. Multi-batch big tier: use 70K/USA as the guardrail.
   For 70K, `fsz > 128` fronts own 51.2% of the f3 proxy. Big-tier changes should therefore be
   evaluated on 70K/USA-style cases even if they do not help the 13K near-miss. Treat
   warp-specialized panel LU as research, not quick path. The 70K graph-node profile confirms
   `factor_big` has budget, but the rejected 512-thread launch experiment shows that simply
   increasing resident CTAs is the wrong level of intervention.
   Literature supports look-ahead in dense/QR settings, but local docs show sync-to-wall
   conversion is weak. If attempted, start with a negative-test kernel that preserves trailing and
   only isolates panel scheduling cost.

## Current conclusion

The current branch now proves one narrow multi-batch 30% direct non-GEMM point, but it does not
prove a broad 30% reduction across 10K+ cases or for B=1. The most defensible claim today is:

- B=1 exact-path improvements reach 9.7-16.0% full-factor reduction on several 10K+ fp32 cases.
- This corresponds to roughly 16-26% estimated non-GEMM reduction using USA no-trailing fractions.
- Multi-batch best-mode reductions are now strongest at the 13K near-miss and meaningful on large
  cases. The 13K B=256 setup-time band-split path reaches direct no-GEMM -30.1% with a repeat-301
  median; regenerated 25K B=256 reaches -12.4%; 70K/USA B=64/256 reaches about -13.9%..-18.4%.

The targeted `case13659pegase B=256` pass crossed 30% only after the setup-time band order, and
the margin is tiny. The 25K and large-case band-split passes are useful but insufficient. The
next artifact should deepen the 25K mid-front path beyond banding and separately attack the
70K/USA big/front-memory path beyond level splitting, followed by a re-run of the direct
no-trailing table.

## Objective audit

This section maps the original objective to current evidence. It is intentionally conservative:
the goal is not complete until the direct non-GEMM measurements prove the 30% reduction.

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Reduce non-GEMM part of `factorize` by 30% | The setup-time band-split extension reaches `case13659pegase B=256 -30.1%` direct no-GEMM (`0.047062 -> 0.032911`, repeat-301). Regenerated 25K reaches `-12.4%` at B=256. Large 70K/USA multi-batch points now reach about `-13.9%..-18.4%`, still below 30%. | Partially met: one 10K+ multi-batch point, not broad 10K+/B=1/25K+ coverage |
| Use already-run experiments | Report includes current accepted code changes, the 13K/25K/large band-split pass, plus reverted experiments: B>1 full graph, 2D small grid, broad plain extend, tiled scatter, launch-bounds, `__ldg`, child-count plain extend, panel-cap override, fill-in-only zero, FP32/FP64 mid block-size scaling including the post-25K-band-split retest, scatter block-size 512, and rejected finer band / small-packing splits. | Met |
| Literature search | Literature section includes GPU sparse Cholesky/multifrontal work, communication-avoiding LU/QR, SuperLU_DIST batched sparse direct solver, MAGMA batched LU/BLAS, and tiny-matrix factorization papers. | Met |
| Design, results, and causes in a document | Current report records implementation design, measurement protocol, full/no-trailing results, failed experiments, root causes, exact extend destination-collision audit, and next design directions. | Met as report; successful only for one narrow 13K B=256 point |
| Split B=1 and multi-batch | Tables and interpretation are split into `B=1`, `B=64`, and `B=256`; multi-batch has separate no-trailing and root-cause sections. | Met |
| Cover 10K+ cases | Fresh tables include `case_ACTIVSg10k`, `case13659pegase`, `case_ACTIVSg25k`, `case_ACTIVSg70k`, and `case_SyntheticUSA`; the front-distribution section now includes regenerated current dumps for 13K, 25K, and 70K. | Met for documentation coverage |
| Explain whether TC helps multi-batch | Tensor-core relevance section ties comprehensive sweep, Amdahl wall ceiling, current no-trailing fractions, and TF32 experiments to the conclusion that TC is secondary for `B=64/256` non-GEMM. | Met |
| Verify code state after experiments | Reverted negative default-path code or kept it default-OFF; current full, no-trailing, plainsafe, and no-trailing+plainsafe builds pass; `git diff --check` passes. | Met for current artifacts |

Remaining proof needed:

- A direct no-trailing result at or below 70% of baseline no-trailing on at least the targeted
  10K+ cases, ideally including 25K+ multi-batch.
- A fresh full/no-trailing table after any successful code change, run serially to avoid GPU
  contention.
- An implementation that turns the new 25K front-distribution evidence into a measured mid-front
  win, and separately covers the 70K/USA big-front/front-memory regime.
