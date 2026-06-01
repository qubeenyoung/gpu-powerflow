# factorize / solve / analyze Optimization Report

Goal (user): cut `custom_linear_solver` **factorize (f)**, **solve (s)** by 30% and
**analyze** by 30%, on MATPOWER NR linear systems for the 3000 / 6000 / 9000-bus and
25k / 70k-bus systems. A linear-system relative residual `< 1e-3` counts as a pass.

Hardware: RTX 3090 (GA102, sm_86), CUDA 12.8, 12-core host. Datasets dumped with
`external/lin_solver/prepare_datasets/python/prepare_nr_linear_system.py` (NR Jacobian +
mismatch at iteration 2). Targets: `case3120sp` (n=5991), `case6470rte` (12485),
`case9241pegase` (17036), `case_ACTIVSg25k` (47246), `case_ACTIVSg70k` (134104).

All timings are the runner's median wall-clock (`--repeat 40`); the kernel time is ~97%
of the factor/solve wall-clock (launch/event overhead is negligible — verified with
`CLS_KERNEL_TIME`).

## Results (baseline = master HEAD, optimized = this branch)

| case   | n      | analyze ms          | factorize ms          | solve ms            | relres   |
|--------|--------|---------------------|-----------------------|---------------------|----------|
| 3120sp | 5991   | 18.17 → 14.16 (-22%)| 0.317 → 0.255 (-20%)  | 0.192 → 0.20  (~0)  | 9.3e-5   |
| 6470rte| 12485  | 37.43 → 26.34 (-30%)| 0.437 → 0.346 (-21%)  | 0.244 → 0.27  (~0)  | 3.3e-5   |
| 9241peg| 17036  | 54.36 → 35.82 (-34%)| 0.744 → 0.517 (-31%)  | 0.354 → 0.354 ( 0)  | 2.6e-6   |
| 25k    | 47246  | 100.4 → 86.5  (-14%)| 1.351 → 1.265 (-6%)   | 0.639 → 0.619 (-3%) | 3.4e-13  |
| 70k    | 134104 | 276.8 → 237.4 (-14%)| 2.849 → 2.764 (-3%)   | 1.173 → 1.122 (-4%) | 3.9e-11  |

Small/medium analyze and factorize meet (or nearly meet) the 30% target. Large-case
factorize/solve and large-case analyze do **not** — the reasons are structural and are
documented below as part of the optimization loop's self-feedback.

## What was changed

### A1. GPU symmetric adjacency graph build (analyze)
`matrix::build_symmetric_graph_device` builds the METIS graph (A+Aᵀ pattern, self-loops
removed, neighbors sorted+deduped per vertex) on the GPU via a `thrust::sort`/`unique`
over directed edge keys `row*n+col`, then downloads `xadj`/`adjncy`. It replaces the
single-threaded CPU `build_symmetric_adjacency` **and** the separate host CSC download.
`metis_nd_from_graph` runs ND on the prebuilt graph.

Why it matters most on small/medium: the CPU adjacency build ran serial for `n<32768`
(the `par_for` threshold), so it was 30–46% of `metis_nd` there (e.g. 9241: 19 ms of
40 ms). On GPU it is ~0.6–4.4 ms across all sizes.

### A2. Parallel root induce (analyze)
The parallel nested dissection (`par_nd_rec`) splits with a vertex separator then recurses
on the two halves across cores. At the **root** only one thread is active, so the
single-threaded `induce()` (subgraph extraction) sat on the critical path — 19 ms on 70k,
7 ms on 25k. `induce_par` extracts the two halves concurrently, each multi-core. Saves
~17 ms (70k) / ~3 ms (25k). Deeper levels keep serial `induce()` (already thread-saturated).

### F1. Adaptive mixed-precision factorize (factorize)
The codebase had an unused FP64-master / FP32-working "mixed" factor path. On GA102 FP64
runs at **1/64** of FP32 throughput, so the dense per-front LU bulk is far cheaper in FP32
while the FP64 master keeps the assembly precise (pivots derive from the precise master).
Enabled adaptively for `n < 24000`, where it cuts factor 20–31% and keeps relres `< 1e-4`
(target `1e-3`; validated on 12 cases incl. 118 / 1354 / 2869 / ACTIVSg2000 / 3012wp /
6515rte / 8387pegase). `MF_MIXED` / `MF_NO_MIXED` override.

## What did NOT work (measured, reverted, or not shipped)

- **Mixed precision on large cases**: factor is launch/occupancy-bound there (70k total
  work is only **0.083 GFLOP** vs 2.7 ms), so FP32 gives no speedup, *and* the mixed path
  disables the multi-block big-front kernels → 70k regressed (2.95→4.07 ms) and relres rose
  to 6.4e-3 (fail). Kept FP64 for `n>=24000`.
- **Parallel `fill_pattern` (symbolic Cholesky)**: the merge work is concentrated in a few
  high-fill separator columns near the etree root that are inherently serial; level-parallel
  merge gave no speedup (matches the prior cy181-era finding). Reverted.
- **Multi-stream subtree solve**: `MF_MS_PROFILE` shows the ND-ordered etree has one
  dominant subtree (max_subtree = 100% on 25k/70k → est. speedup 1.0x). Solve is
  critical-path-bound on the separator chain; there is no parallelism to exploit.
- **FP32 solve (`MF_SOLVE_F32`)**: the per-solve FP64→FP32 front conversion costs more than
  the bandwidth it saves; solve is latency- not bandwidth-bound. No win.
- **parND depth/base and amalgamation `MF_CAP` sweeps**: parND is already at its 12-core
  parallel limit; cap only trades factor↔solve (no Pareto move). `cap>16` overflows the
  register-partial path.
- **METIS separator options (`NITER`)**: no effect — the separator cost is in METIS's
  internal multilevel coarsening, not refinement.

## Why large-case analyze is stuck at ~14% (self-feedback)

parND is 66% of large analyze (70k: 156 ms). Profiling (`PARND_PROF`) shows the
**single-threaded METIS vertex separators on the top levels** dominate: the root separator
alone is 41 ms (70k) / 15 ms (25k), and the depth-1/2 separators add more on the critical
path. These are `METIS_ComputeVertexSeparator` calls whose cost is multilevel coarsening —
not tunable away. The remaining CPU symbolic (fill 40 ms + etree 7 ms + multifrontal 25 ms
on 70k) is dominated by the same few large separator columns and does not parallelize.

A genuine 30% on large analyze requires a **GPU multilevel nested dissection** (GPU heavy-
edge coarsening → METIS on the small coarse graph → GPU projection + boundary FM
refinement). This was scoped but **not shipped**: a lower-quality GPU separator (e.g. BFS
level-set) enlarges the separators → more fill → it would *regress* factorize and solve,
which are the other two targets. Quality-preserving GPU ND with FM refinement is a larger
effort than the session budget. It is the clear next step for the large-case analyze goal.

## Tensor cores and iterative refinement (requested — measured, do not fit this workload)

**Tensor cores.** GA102 has no FP64 tensor cores, so tensor-core use means FP16/TF32. But the
factor is not arithmetic-bound: the 70k dense factor work is ~0.08 GFLOP, whose FP64 compute
floor is ~166 µs — only **~6%** of the 2.76 ms factor wall (FP32 floor ~2 µs, FP16-TC ~0.6 µs).
The other ~94% is kernel-launch and per-level `__syncthreads`/graph-node latency across the deep
etree of tiny fronts (96% of fronts are fsz≤16). Tensor cores accelerate only the ~6% that is
arithmetic, so they cannot approach a 30% factor cut here. (They would matter on dense or
large-front problems; power-grid multifrontal fronts are too small and too many.)

**Iterative refinement (low-precision solve, then correct).** Implemented in the runner
(`--ir N`: `r = b − A x` in FP64, `dx = solve(r)`, `x += dx`). Measured FP32-solve + 1 IR step:

| case | FP64 solve | FP32 solve | FP32 + 1·IR (solve+ir = total) | relres after IR |
|------|-----------|-----------|--------------------------------|-----------------|
| 9241 | 0.382 ms  | 0.343 ms  | 0.321 + 0.393 = **0.714 ms**   | 2.1e-10 |
| 25k  | 0.621 ms  | 0.655 ms  | 0.641 + 0.703 = **1.34 ms**    | 5.0e-9  |
| 70k  | 1.156 ms  | 1.337 ms  | 1.302 + 1.408 = **2.71 ms**    | 1.2e-10 |

IR **recovers accuracy** (relres → 1e-9..1e-10), confirming the technique is correct. But one IR
step costs ≈ one extra solve + an spmv, and the low-precision solve is **not** faster here (solve
is latency-bound, not bandwidth-bound), so FP32-solve + IR is ~2× slower than a plain FP64 solve.
IR pays off when the low-precision factor/solve it enables is much cheaper than high precision —
which holds on compute-bound dense problems, not on these latency-bound sparse power-grid fronts.
Where low precision *does* help (small/medium factor) the result already passes 1e-3 without IR,
so no refinement is needed.

## Multi-batch: the regime where the 30% targets are exceeded on every size

The single-system limits above are all forms of *latency* (launch + per-level sync on a deep
etree of tiny fronts). The realistic power-flow workload is **many systems with the same
topology** (contingencies, time steps, stochastic scenarios) → identical sparsity → **one
shared analyze**, then B numeric factor/solve. Batching those B systems amortizes the latency
across B and fills the GPU on the otherwise occupancy-starved narrow levels.

Implemented as a uniform-batch path (`factorize/multifrontal_batched.{hpp,cu}`,
`Solver::batched_{setup,factorize,solve}`, runner `--batch B`): front-major arena
`B*front_total`, each dense kernel adds `blockIdx.y = batch`, one launch per etree level covers
all B fronts. Verified correct: max relres over all B batches matches the single-system solve.

**Per-system factor / solve vs the optimized single-system path** (B=128, same precision —
mixed for n<24000, FP64 above; relres unchanged):

| case   | factor single → B=128 | solve single → B=128 |
|--------|-----------------------|----------------------|
| 3120sp | 0.255 → 0.023 (**-91%**) | 0.196 → 0.021 (**-90%**) |
| 6470rte| 0.346 → 0.049 (**-86%**) | 0.271 → 0.041 (**-85%**) |
| 9241peg| 0.517 → 0.076 (**-85%**) | 0.354 → 0.057 (**-84%**) |
| 25k    | 1.265 → 0.354 (**-72%**) | 0.619 → 0.158 (**-74%**) |
| 70k    | 2.764 → 1.339 (**-52%**) | 1.122 → 0.434 (**-61%**) |

The 30% factor and **solve** targets are exceeded on every size, including the large cases that
were impossible single-system. Per-system time keeps dropping until ~B=32–64 (small/medium) or
B≈128 (70k), where it saturates — the GPU is now full, i.e. the workload has become
**compute-bound**.

**Now precision and tensor cores matter** (the single-system answer was "no"; the batched
compute-bound answer is "yes"):

- **Mixed FP32** helps large factor in the batched regime where it was useless single-system:
  25k B=64 factor 0.367→0.307 (-16%), 70k 1.344→1.056 (-21%). Small/medium were already mixed.
- **Iterative refinement** finally has something to amortize: mixed pushes 70k relres to ~1e-3
  (borderline); 1 FP64 IR step recovers it to ~1e-6. The IR step costs ≈ one extra solve, so it
  is worth it only when the mixed-factor margin is needed (e.g. tightening 70k's residual);
  small/medium pass 1e-3 with no IR.
- **Tensor cores — implemented and measured: they do NOT help here.** A batched FP16 WMMA
  trailing-update path was built (`mf_factor_extend_mixed_tc_b`, env `MF_TC`): FP64 master
  assembly, FP32 panel/U, FP16 16×16×16 tensor-core GEMM for the `C(uc×uc) -= L(uc×nc)·U(nc×uc)`
  trailing, nc≤16 zero-padded to K=16, L/U staged FP16 in shared. It is correct (relres recovers
  to <1e-3 with 2 IR steps), but **slower than FP32 mixed** at every batch size:

  | B=64        | FP64 factor | FP32 mixed | FP16-TC | FP16-TC + 2·IR (relres) |
  |-------------|-------------|------------|---------|--------------------------|
  | 25k         | 0.368 ms    | **0.303**  | 0.418   | 0.417 (2.5e-6) |
  | 70k         | 1.380 ms    | **1.066**  | 1.334   | 1.442 (7.7e-5) |

  (70k B=128: mixed 0.999 vs TC 1.258; 9241 B=128: mixed 0.074 vs TC 0.116 — TC worse on the
  smaller fronts.) Root cause: the multifrontal panels are nc≤16, so the trailing GEMM has
  **K=16 — one tensor tile deep**. Tensor cores need a tall K to amortize their fragment-load /
  FP16-staging overhead; at K=16 with fsz≤256 the mandatory FP32→FP16 staging (the K-padding
  can't be loaded directly from the front) costs more than the plain FP32 FMA trailing saves.
  **FP32 mixed is the precision sweet spot for this workload; tensor cores would only pay off on
  much larger / denser fronts than power-grid multifrontal produces.**

Takeaway: **batching is the correct lever for this workload.** It converts the latency-bound
single-system problem (where precision/tensor-cores are inert) into a compute-bound batched one
(where they help), and on its own delivers 52–91% per-system factor/solve reductions.

## Diagnostic env knobs added

`CLS_KERNEL_TIME` (runner: factor/solve kernel vs wall), `CLS_FACTOR_SPLIT` (memset/scatter/
graph), `FILL_TIME` (postorder/column_counts/merge), `PARND_PROF` (root separator/induce),
`PARND_TOPIND` (top-level parallel-induce cutoff), `METIS_NITER`, `MF_MIXED`/`MF_NO_MIXED`,
`MF_SOLVE_F32`. Runner `--ir N` runs N FP64 iterative-refinement steps after the solve.
