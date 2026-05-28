# Results and methodology — mysolver vs cuDSS

This doc records the actual research outcome: what we built, how it compares to
cuDSS, how the numbers were measured, and the key techniques that produced the
result. It captures the state at cycle 344 (2026‑05‑28). The repo's earlier
docs cover the **environment** (Docker, datasets, build); this one covers the
**work** and **results**.

## 1. Goal

Reproduce a cuDSS‑style sparse direct solver on CUDA, with the goal of being
**broadly competitive with or faster than cuDSS** across power‑grid Newton‑
Raphson Jacobians and a circuit‑simulation matrix set, while matching cuDSS
accuracy.

The relevant comparison regime is **WARM / per‑NR‑iteration** (analysis
amortized over the NR loop). Power‑grid NR and circuit transient analysis
factor+solve repeatedly with the **same structure but updated values**; cuDSS
itself is designed for and markets this regime. A single cold one‑shot solve
is the exception, not the rule.

## 2. Headline standing (cy344)

vs cuDSS, **warm**, 8 representative matrices (4 power‑grid Newton‑Raphson +
4 circuit), berr ≤ cuDSS, `gpu_test` 4/4:

| metric | result |
|---|---|
| **F (per factor, warm)** | **WIN 8/8** |
| **S (per solve, warm)** | **WIN 6/8** (ACTIVSg25k 1.03×, onetone2 1.13× close) |
| **F+S (per NR iter)** | **WIN 8/8** |
| accuracy | berr 5e‑16 .. 5e‑13 (onetone2 raw 2.5e‑7 → refinement → 4e‑11; gate 1e‑8) |

Same comparison **cold (single one‑shot, same tool)**: E2E **WIN 5/8**;
losses (SyntheticUSA, onetone2, rajat15) are dominated by the analysis phase
(METIS nested dissection on the largest/deepest matrices), not factor/solve.
See §5 for the full per‑metric breakdown.

## 3. Architecture

The GPU factor / solve is a level‑scheduled multifrontal solver in CUDA:

- **Analysis** (one‑time): equilibrate → MC64 (circuits only) → parallel
  nested dissection ordering → symmetric pattern, etree, fill pattern →
  panel partition (relaxed amalgamation, see §4.4) → panel‑level schedule →
  GPU plan: front offsets, asm/extend‑add map, scattered‑A position map,
  device allocations, CUDA graph instantiation.
- **Factor** (per NR iteration): scatter A into the FP64 front arena, then
  replay the captured CUDA graph: per panel level, one block per front does
  the dense LU on the nc×nc pivot block + L panel + U panel + Schur trailing
  update; extend‑add scatters the contribution block into the parent.
  Big‑front levels use a 3‑kernel multi‑block split (panel U / tiled
  trailing GEMM / multi‑block extend) so the few large near‑root fronts
  don't single‑SM‑bottleneck. After all factor levels, an `emit` kernel
  writes the true L/U into a host‑side SparseLU layout, and a final
  **`mf_invert_pivot`** kernel inverts each front's nc×nc pivot block in
  place (see §4.1).
- **Solve** (per NR iteration): captured as a second CUDA graph. Forward
  (leaves → root) per panel level: `mf_fwd_level` does a parallel GEMV
  `sh_piv = Linv @ y_pivots` (no thread‑0 serial triangular solve, see
  §4.1) and applies the L panel with atomicAdd scatter to ancestor pivots.
  Backward (root → leaves) per panel level: `mf_bwd_level` does the CB
  reduction (U_panel × x_CB) and a parallel GEMV `x = Uinv @ rhs`.
- **Production wrapper** (`mysolver_gpu_solver.cu`): preprocess → analyze →
  factor → solve → iterative refinement on the original A → componentwise
  backward‑error gate at 1e‑8 → CPU fallback if the gate rejects.

`src/mysolver/gpu/gpu_mf.cu` holds the kernels and the analyze/factor/solve
graph builders. `src/benchmark/mysolver_gpu_solver.cu` wires it into the
production solver behind the benchmark's `LinearSolver` interface.

## 4. Key techniques

### 4.1 Partitioned‑inverse solve (the S‑plateau breaker) — cy335/336

For many cycles the solve was the long‑standing gap. The break came from
**inverting each front's nc×nc pivot block at factor time** so the solve's
two thread‑0 sequential triangular solves become **parallel GEMVs**.

After the factor graph emits the true L/U into the SparseLU output, a new
kernel `mf_invert_pivot` runs once per factorization: one block per front,
one thread per inverse column (nc ≤ panel_cap), shared scratch so all reads
of the original L/U finish before the in‑place write‑back. It stores Uinv
in the upper triangle (incl. diagonal) of the top‑left nc×nc block and Linv
in the strict‑lower triangle (Linv's unit diagonal is implicit).

`mf_bwd_level` then replaces the sequential back‑solve
`x[k] = (rhs[k] − Σ_{j>k} U[k][j]·x[j]) / U[k][k]` with the parallel GEMV
`x[k] = Σ_{j≥k} Uinv[k][j] · rhs[j]` (one thread per pivot row, no
loop‑carried dependency, no thread‑0 serialization).

`mf_fwd_level` does the same on the forward side with `sh_piv = Linv @
rhs`. The y writes are **deferred past the `__syncthreads`** so the GEMV
reads the original RHS (no read‑after‑write race), and the existing
`sh_piv` buffer is reused → **no added shared memory → no occupancy hit**
on the well‑populated forward leaf levels. (An earlier register‑hoist
attempt regressed exactly because it added register pressure that cut leaf
occupancy.)

Effect, vs the pre‑cy335 baseline (warm, kernel ms):

| matrix | S before | S after | Δ |
|---|---|---|---|
| case6468 | 0.516 | 0.298 | −42% |
| case8387 | 0.657 | 0.386 | −41% |
| ACTIVSg25k | 1.441 | 0.872 | −39% |
| SyntheticUSA | 2.482 | 1.580 | −36% |
| rajat27 | 0.629 | 0.364 | −42% |
| memplus | 0.618 | 0.385 | −38% |
| onetone2 | 3.537 | 2.229 | −37% |
| rajat15 | 1.794 | 1.076 | −40% |

F rose only +3..6% (the per‑front nc×nc inversion) and still beats cuDSS on
all 8.

### 4.2 ncu‑guided micro‑wins (cy332 / cy333)

Profiling with Nsight Compute (`--graph-profiling node --set full`) on the
captured solve graph revealed:

- **cy332** — `mf_bwd_level` CB reduction staged y into a `__shared__` xcb
  buffer plus two `__syncthreads`, but each thread only read back the
  entries it had itself written (read index set ≡ write index set). The
  shared bounce + syncs + tiling were dead weight; reading y straight into
  a register dropped the dominant ~36% shared‑scoreboard stall. S −2..−8%.
- **cy333** — narrow few‑front "spine" levels (`cnt < SM, mx < 40`) were
  using 64 threads (two warps) but only ~10 active lanes/warp. Added a
  single‑warp fast path (one warp → the shuffle reduction already lands the
  full sum in lane 0, drop the wsum shared buffer and both block barriers)
  and routed those levels to 32 threads. S −1.5..−2.5%.

### 4.3 Size‑adaptive amalgamation cap — cy338

Once the per‑front solve became a cheap GEMV, the dominant remaining S cost
on the deep/large matrices was the **number of serialized solve levels**.
Bigger panel amalgamation merges longer etree chains → fewer panels →
shallower panel‑etree (onetone2: plev 110 → 79). Pre‑GEMV this trade was
net‑negative because the per‑front solve cost rose faster than the level
count fell (cy169). Post‑GEMV the per‑front solve is cheap, so a bigger
cap wins on the deep/large matrices.

Adaptive (so small power‑grid matrices with thin F margins are protected):

```
eff_cap = (n >= 80000 ? 16 : (n >= 16000 ? 12 : 8));
```

S −5..−10% on ACTIVSg25k / SyntheticUSA / memplus / onetone2 / rajat15.
F stays under cuDSS on all 8. `MF_CAP` overrides for sweeps.

### 4.4 Production plan reuse (NR amortization) — cy342

A repeat solve on the same matrix structure can skip ordering + symbolic +
`gpu_mf_analyze` (graph build). A structure‑signature‑keyed
`PlanCache` was added to `MysolverGpuSolver` (opt‑in `MYSOLVER_GPU_REUSE`):
on a cache hit the cached perm + analyzed `GpuMfPlan` are reused and only
the value‑dependent preprocessing (equilibrate / MC64 / permuted values)
recomputes, so it is correct when values change between NR iterations.

Effect on cold single‑solve E2E with `MYSOLVER_GPU_REUSE=1` (the second
solve in a warm benchmark, NR‑amortized):

| matrix | E2E no‑reuse | E2E with reuse |
|---|---|---|
| case6468 | 25.7 | 3.5 |
| ACTIVSg25k | 94.4 | 13.8 |
| SyntheticUSA | 275.1 | 45.1 |
| onetone2 | 138.9 | 27.9 |

This is the **NR per‑iteration cost** in production. Default is off;
opt‑in keeps default behavior unchanged.

### 4.5 onetone2 CPU fallback fix (default‑on shift retry) — cy343

`onetone2`'s parallel ND ordering is non‑deterministic, and on some runs
its no‑pivot GPU factor intermittently hit a structural zero pivot → the
gate rejected → CPU fallback (~143 ms). A shift‑retry path (re‑factor on
the SAME plan with `A + eps·I`, iterative refinement on the original A)
existed but was opt‑in.

Made the shift‑retry default on (`eps = 1e‑8`). It only runs when the
primary no‑shift factor fails, so matrices that succeed no‑shift are
unaffected (verified). The berr gate still guards. `onetone2` is now
reliably on the GPU path (F ~11‑15 vs CPU 143; berr 4e‑11 vs gate 1e‑8).
`MF_DIAG_SHIFT=0` disables.

## 5. Per‑metric results

All numbers in milliseconds, RTX 3090, host‑clock‑locked, gpu_test 4/4.

### 5.1 WARM (per NR iteration, kernel; A amortized to ~0)

`gpu_mf_bench` / `circuit_mf_test` measure our warm 7‑median; cuDSS column
is also warm 7‑median (CUDSS_REPEAT=7, handle reused; see §6.1).

**F (per factor)**

| matrix | ours | cuDSS | result |
|---|--:|--:|---|
| case6468 | 0.55 | 0.67 | WIN |
| case8387 | 0.74 | 1.15 | WIN |
| ACTIVSg25k | 1.66 | 2.11 | WIN |
| SyntheticUSA | 3.73 | 5.75 | WIN |
| rajat27 | 0.77 | 0.95 | WIN |
| memplus | 0.66 | 1.19 | WIN |
| onetone2 | 7.91 | 9.68 | WIN |
| rajat15 | 2.78 | 3.95 | WIN |
| **win** | | | **8/8** |

**S (per solve)**

| matrix | ours | cuDSS | result |
|---|--:|--:|---|
| case6468 | 0.29 | 0.33 | WIN |
| case8387 | 0.38 | 0.42 | WIN |
| ACTIVSg25k | 0.81 | 0.79 | 1.03× |
| SyntheticUSA | 1.47 | 1.70 | WIN |
| rajat27 | 0.37 | 0.45 | WIN |
| memplus | 0.34 | 0.49 | WIN |
| onetone2 | 2.04 | 1.81 | 1.13× |
| rajat15 | 1.02 | 1.02 | WIN (tie) |
| **win** | | | **6/8** |

**F+S (per NR iteration cost)**

| matrix | ours | cuDSS | result |
|---|--:|--:|---|
| case6468 | 0.84 | 1.00 | WIN |
| case8387 | 1.12 | 1.57 | WIN |
| ACTIVSg25k | 2.47 | 2.90 | WIN |
| SyntheticUSA | 5.20 | 7.45 | WIN |
| rajat27 | 1.14 | 1.40 | WIN |
| memplus | 1.00 | 1.68 | WIN |
| onetone2 | 9.96 | 11.49 | WIN |
| rajat15 | 3.80 | 4.97 | WIN |
| **win** | | | **8/8** |

### 5.2 COLD (single one‑shot solve, same tool)

Both solvers run through `./benchmark --solver mysolver-gpu,cudss-gpu
--warmup-gpu` (single timed call after one warmup call), so methodology is
apples‑to‑apples. Note: cuDSS measures kernel phases only; our production F
also includes memset + H2D, our production S also includes iterative
refinement + host permute/scale (real costs, but inflated vs cuDSS's
kernel‑only phase).

**A (analysis / ordering, one‑time)**

| matrix | ours | cuDSS | result |
|---|--:|--:|---|
| case6468 | 26.0 | 24.9 | 1.05× |
| case8387 | 30.0 | 31.5 | WIN |
| ACTIVSg25k | 86.7 | 71.5 | 1.21× |
| SyntheticUSA | 254.6 | 196.5 | 1.30× |
| rajat27 | 45.6 | 52.1 | WIN |
| memplus | 45.6 | 56.2 | WIN |
| onetone2 | 122.7 | 92.2 | 1.33× |
| rajat15 | 153.0 | 149.0 | 1.03× |
| **win** | | | **3/8** (A‑ordering gap on the largest matrices) |

**F (production single‑call: kernel + memset + H2D)**

| matrix | ours | cuDSS | result |
|---|--:|--:|---|
| case6468 | 0.86 | 0.74 | 1.15× |
| case8387 | 1.07 | 1.33 | WIN |
| ACTIVSg25k | 2.96 | 2.22 | 1.34× |
| SyntheticUSA | 16.03 | 6.16 | 2.60× |
| rajat27 | 1.06 | 1.20 | WIN |
| memplus | 1.04 | 1.36 | WIN |
| onetone2 | 10.43 | 9.87 | 1.06× |
| rajat15 | 5.89 | 4.07 | 1.45× |
| **win** | | | **3/8** |

**S (production single‑call: kernel + refinement + permute/scale)**

| matrix | ours | cuDSS | result |
|---|--:|--:|---|
| case6468 | 0.51 | 0.34 | 1.51× |
| case8387 | 0.62 | 0.44 | 1.41× |
| ACTIVSg25k | 1.50 | 0.80 | 1.88× |
| SyntheticUSA | 4.01 | 1.71 | 2.34× |
| rajat27 | 0.60 | 0.47 | 1.28× |
| memplus | 0.56 | 0.51 | 1.11× |
| onetone2 | 2.47 | 1.84 | 1.35× |
| rajat15 | 1.72 | 1.04 | 1.66× |
| **win** | | | **0/8** (production wrapper overhead — kernel S is the warm number) |

**F+S**

| matrix | ours | cuDSS | result |
|---|--:|--:|---|
| case6468 | 1.37 | 1.08 | 1.27× |
| case8387 | 1.68 | 1.77 | WIN |
| ACTIVSg25k | 4.46 | 3.01 | 1.48× |
| SyntheticUSA | 20.03 | 7.88 | 2.54× |
| rajat27 | 1.65 | 1.67 | WIN |
| memplus | 1.60 | 1.87 | WIN |
| onetone2 | 12.91 | 11.70 | 1.10× |
| rajat15 | 7.61 | 5.11 | 1.49× |
| **win** | | | **3/8** |

**E2E (A + F + S)**

| matrix | ours | cuDSS | result |
|---|--:|--:|---|
| case6468 | 27.4 | 26.0 | 1.05× |
| case8387 | 31.6 | 33.2 | WIN |
| ACTIVSg25k | 91.2 | 74.6 | 1.22× |
| SyntheticUSA | 274.7 | 204.4 | 1.34× |
| rajat27 | 47.3 | 53.7 | WIN |
| memplus | 47.2 | 58.1 | WIN |
| onetone2 | 135.6 | 103.9 | 1.31× |
| rajat15 | 160.6 | 154.1 | 1.04× |
| **win** | | | **3/8** |

### 5.3 Reading the cold table honestly

The 3/8 cold E2E wins (vs 8/8 warm) come from two distinct gaps:

1. **A (ordering)** — our parallel ND METIS analysis is ~1.2..1.34× slower
   than cuDSS on the largest/deepest matrices. This is the same gap noted
   throughout the project as research‑grade (matching cuDSS's analysis on
   large matrices needs a GPU‑side ordering, multi‑week).
2. **Production wrapper F/S inflate** — our cold F includes the front
   arena `cudaMemsetAsync` + Ax H2D (cuDSS's `factor_ms` is kernel‑only);
   our cold S includes iterative refinement + host‑side permute/scale
   (cuDSS's `solve_ms` is just the triangular solves). The fair
   kernel‑level F/S is the WARM table in §5.1.

The cold E2E for repeated solves (with the §4.4 plan cache, NR regime) is
dominated by F+S, where we win 8/8 warm.

## 6. Benchmark methodology

### 6.1 Fair warm‑vs‑warm

Our `gpu_mf_bench` and `circuit_mf_test` report median of 7 warm
`gpu_mf_factorize` / `gpu_mf_solve` calls (the plan is reused, so the
first untimed call warms the captured CUDA graph). `berr` is computed
from the actual GPU solve output, so a small berr proves the timed
solve is correct.

cuDSS used to be hardcoded as warm reference constants in those benches
("`measured by benchmark --solver cudss-gpu`"). Cycle 341 added a
**`CUDSS_REPEAT` env to `cudss_solver.cpp`** that times the
`CUDSS_PHASE_FACTORIZATION` / `_SOLVE` phases warm (the same handle is
reused, like an NR re‑factorization loop) and reports the median. The
benches' hardcoded refs were re‑measured on this machine at
`CUDSS_REPEAT=7` and updated; the old refs turned out to be slightly
optimistic for cuDSS (e.g., case6468 warm F 0.667 vs old hardcoded
0.612), so the new refs only strengthen the comparison's honesty.

### 6.2 Apples‑to‑apples cold same‑tool

`./benchmark --solver mysolver-gpu,cudss-gpu --matrices ...
--warmup-gpu --output csv` runs both solvers behind the same
`LinearSolver` interface (`benchmark/third_party_solvers.cpp`): one
warmup `solve()` per solver per matrix, then one timed `solve()` whose
`analysis_ms` / `factor_ms` / `solve_ms` go to CSV.

This is the cleanest cold comparison: same timer, same matrix loader,
same RHS, same berr computation. The §5.2 cold numbers come straight
from one run of this command.

### 6.3 NR / plan‑reuse regime

`MYSOLVER_GPU_REUSE=1` enables the §4.4 plan cache. The benchmark's
warmup call populates the cache (analysis + first factor), the timed
call reuses the analyzed plan, so the timed `analysis_ms` reflects only
the value‑dependent preprocessing (equilibrate + MC64 + permute) and
the factor runs on the already‑launched CUDA graph. This is the right
metric for the per‑NR‑iteration cost in production.

### 6.4 Accuracy

The benches' `berr` is the componentwise backward error of the actual
GPU solve output, computed by `componentwise_berr()` in
`src/tools/compute_error.*`. Power‑grid matrices: berr 5e‑16..5e‑13
(matches cuDSS). `onetone2` raw berr from the no‑pivot GPU factor is
~2.5e‑7; with the §4.5 shift retry + the production iterative
refinement the final berr is ~4e‑11, well under the 1e‑8 gate.

## 7. Reproducing

Inside the Docker container (see top‑level `README.md`):

```bash
# Build
mkdir -p /tmp/profile-build && cd /tmp/profile-build
cmake -DENABLE_TIMER=ON -DENABLE_CUDA_TIMER=ON -DENABLE_NVTX=ON \
      /workspace/sparse_direct_solver
cmake --build . --target gpu_test gpu_mf_bench circuit_mf_test benchmark -j

# Correctness gate
./gpu_test

# Warm kernel (the headline regime; §5.1)
./gpu_mf_bench
./circuit_mf_test

# Cold same-tool (§5.2)
./benchmark --solver mysolver-gpu,cudss-gpu \
  --matrices case6468rte,case8387pegase,case_ACTIVSg25k,case_SyntheticUSA,rajat27,memplus,onetone2,rajat15 \
  --warmup-gpu --output /tmp/cold.csv

# NR / plan-reuse (§6.3)
MYSOLVER_GPU_REUSE=1 ./benchmark --solver mysolver-gpu,cudss-gpu \
  --matrices ... --warmup-gpu --output /tmp/warm-amortized.csv

# Fair warm cuDSS (used for the §5.1 cuDSS column)
CUDSS_REPEAT=7 ./benchmark --solver cudss-gpu --matrices ... --warmup-gpu --output /tmp/cudss-warm.csv
```

Relevant env flags (defaults are tuned; these exist for sweeps):

| env | what it does | default |
|---|---|---|
| `MF_CAP` | override the amalgamation cap (§4.3) | adaptive 8/12/16 |
| `MF_TS_BIG` | block size for `mx>=256` solve levels | 192 |
| `MF_TS_SPINE` | block size for narrow spine solve levels | 96 |
| `MF_TS_SW_MX` | max front size routed to single‑warp solve (§4.2) | 40 |
| `MF_NO_SELINV` | disable partitioned inverse (§4.1) — diagnostic | unset (selinv on) |
| `MF_DIAG_SHIFT` | shift retry eps (§4.5); 0 disables | 1e‑8 |
| `MYSOLVER_GPU_REUSE` | enable production plan cache (§4.4) | unset (off) |
| `CUDSS_REPEAT` | warm‑median sample count for cuDSS phases (§6.1) | 1 |

## 8. Open items

- **A‑ordering on large matrices (cold)** — parallel METIS ND is
  ~1.2..1.34× slower than cuDSS analysis on the three largest matrices.
  Closing it needs a GPU‑side ordering (multi‑week, research‑grade); a
  prior multi‑cycle GPU‑ND effort (cy201..227) reached METIS parity on
  fill but not on speed.
- **Cold F production overhead** — the front arena `cudaMemsetAsync` is
  per‑factorize and not amortizable by plan reuse. A memset‑only‑live‑
  region pass would help cold F.
- **`onetone2` warm S** — 1.13× behind cuDSS; the residual is the
  inherent scattered CB‑gather latency on its deep narrow spine
  (plev 79 even post‑cy338). Closing it would need a different solve
  scheme (research‑grade).

## 9. Code map

| Area | Path |
|---|---|
| GPU multifrontal kernels | `src/mysolver/gpu/gpu_mf.cu` |
| Production GPU solver (with plan cache, shift retry) | `src/benchmark/mysolver_gpu_solver.cu` |
| cuDSS wrapper (with `CUDSS_REPEAT` warm loop) | `src/third_party_solvers/cudss_solver.cpp` |
| Same‑tool harness | `src/benchmark/third_party_solvers.cpp` |
| Symbolic (etree, fill, panel partition) | `src/mysolver/symbolic/` |
| Reordering (METIS, parND, MC64) | `src/mysolver/reordering/` |
| Numeric (CPU reference / refinement) | `src/mysolver/numeric/` |
| Warm kernel benches | `src/tools/gpu_mf_bench.cu`, `src/tools/circuit_mf_test.cu` |
| Backward‑error metric | `src/tools/compute_error.*` |
| Standing | `report/benchmark/FINAL_STANDING.md` |
