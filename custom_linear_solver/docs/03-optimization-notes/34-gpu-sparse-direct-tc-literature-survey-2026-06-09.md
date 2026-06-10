# GPU Sparse Direct / Tensor Core Literature Survey

**Date**: 2026-06-09

**Scope**: Broad literature pass for the B64/B256 Tensor Core speedup work. Sources are primary
papers, arXiv preprints from the authors, and NVIDIA official documentation/blogs. The goal is not
to collect citations for presentation only; it is to decide which implementation directions are
credible after the current measurements.

## Current Evidence To Explain

- Large cases: the best stable policy is default 512-thread global big-high TF32 fallback +
  shared-resident big-low 512 threads, no TF32 trail+extend fuse, with `CLS_RESPECT_PANEL_CAP=ON`.
  Repeat=61 speedups are 25K `B64/B256 = 1.222/1.275x` and USA `B64/B256 = 1.241/1.231x`.
- 8387: repeat=31 remains below target. The same policy gives about `1.11x` at B64 and `1.06x`
  at B256 on the best valid cap points.
- 8387 front shape: dominant tiny fronts are `(fsz,nc,uc)=(6,2,4)`, `(4,2,2)`, `(5,2,3)`, and
  `(3,1,2)`. The useful K dimension is `nc=1/2`, far below TF32 `mma.m16n8k8`.
- Rejected local experiments already match the literature: tiny-front TC, small bucket splitting,
  unique scatter, variable-lane small packing, and low-mid blocked TF32 do not create a stable
  FP32-relative TC speedup on 8387.

## Source Map

| Area | Main references | Why it matters here |
|---|---|---|
| Multifrontal foundations | Duff/Reid, Liu, Davis/Duff UMFPACK | Defines fronts, contribution blocks, assembly tree, and why dense kernels are the intended compute unit. |
| Supernodal LU | SuperLU, SuperLU_DIST | Shows when grouping columns exposes Level-3 BLAS, and why level/batch scheduling is natural. |
| Circuit/power-grid sparse LU | KLU, Basker | Explains the opposite regime: low-fill, low-flop matrices where dense kernels are weak. |
| GPU sparse direct solvers | CPU-GPU multifrontal, STRUMPACK GPU, SuperLU_DIST batched, cuDSS | Supports thresholded offload, batched levels, scatter/assembly kernels, and GPU-resident solve pipelines. |
| Batched small dense kernels | MAGMA batched, batched GEMM, batched LU | Directly relevant to 8387 tiny fronts and launch amortization. |
| Tensor Core mechanics | PTX ISA, Ampere whitepaper, cuBLAS grouped GEMM | Sets hard tile-shape constraints and possible grouped-GEMM escape hatches. |
| Mixed precision accuracy | Buttari et al., TC iterative refinement, Ootomo/Yokota | Separates raw factor-time speedup from acceptable end-to-end NR/refinement behavior. |
| Power-grid / ACOPF | GPU ACOPF and GPU-resident sparse direct papers | Confirms sparse linear solve dominance and justifies GPU-resident batched NR design. |

## 2026-06-09 Addendum: Verified References And Decision Matrix

This addendum broadens the first pass with up-to-date official docs, recent papers, and a
decision-oriented classification. It uses current external sources checked on 2026-06-09; preprints
are marked as such where relevant.

### High-signal references by role

| Reference | Status | What it says | Direct use in this project |
|---|---:|---|---|
| [cuDSS documentation](https://docs.nvidia.com/cuda/cudss/index.html) | NVIDIA official | cuDSS supports uniform/non-uniform batching, multi-stage analysis/factor/solve, optional refactorization/refinement, pivot controls, memory/flop estimates, hybrid modes, and multi-GPU/multi-node modes. | Confirms that our phase API, analysis reuse, batch path, and refactorization-style loop are aligned with vendor sparse-direct design. |
| [cuDSS Getting Started](https://docs.nvidia.com/cuda/cudss/getting_started.html) | NVIDIA official | The workflow is explicitly split into analysis, factorization, and solve. | Supports reporting factorize separately from setup/analyze, and keeping `setup_ms` outside the Newton loop. |
| [cuDSS engineering blog, v0.4/v0.5](https://developer.nvidia.com/blog/nvidia-cudss-advances-solver-technologies-for-engineering-and-scientific-computing/) | NVIDIA official | Batching helps when individual systems are too small to saturate the GPU; host/hybrid execution helps tiny/small matrices; dense regions in factors drive big speedups. | Supports two conclusions: batch B64/B256 is a real enabler, but 8387 tiny-front GPU/TC pressure is structurally weak unless work is aggregated or moved to a different branch. |
| [cuDSS Tips and Tricks](https://docs.nvidia.com/cuda/archive/13.1.1/cudss/tips_and_tricks.html) | NVIDIA official | Reuse analysis/refactorization, warm up factor/solve, and measure asynchronous GPU phases with synchronization. It also notes sparse solve/factor phases can be memory- or synchronization-bound. | Matches our measurement protocol (`--warmup`, repeat medians, device sync) and explains why multi-batch does not automatically remove non-GEMM bottlenecks. |
| [Batched sparse direct solver in SuperLU_DIST](https://doi.org/10.1177/10943420241268200) | Peer-reviewed, 2024 | Factorization by elimination-tree levels, batched dense operations at each level, and a new batched Scatter GPU kernel. | Closest external match to our level-batched multifrontal path. It validates treating scatter/extend as first-class, not as incidental overhead. |
| [New SuperLU_DIST capabilities](https://doi.org/10.1145/3577197) | Peer-reviewed, 2023 | Communication-avoiding 3D sparse LU, multi-GPU support, and mixed-precision LU plus double-precision iterative refinement. | Supports the "communication/latency + mixed precision correction" axis for next work, not endless local per-front TC toggles. |
| [STRUMPACK GPU BLR sparse direct solver](https://doi.org/10.1177/10943420241288567) | Peer-reviewed, 2024 | GPU multifrontal solver uses vendor BLAS/solver libraries, variable-size batched GEMM/TRSM/LU, and BLR compression; reported exact-GPU and BLR speedups. | Strong support for variable-size batched dense kernels on sufficiently large fronts; weak support for 8387 tiny leaves. BLR is large-front memory/computation work, not a tiny-front TC rescue. |
| [Parallel Sparse and Data-Sparse Factorization-based Linear Solvers](https://arxiv.org/abs/2602.14289) | 2026 review/preprint | Modern sparse-direct progress is organized around reducing communication/latency and reducing complexity through low-rank/hierarchical compression. | Reframes the next search: assembly-tree latency, parent update traffic, ordering/panelization, and possibly compression for large cases. |
| [Towards Efficient ACOPF on GPUs](https://arxiv.org/abs/2302.08656) | Conference/preprint | ACOPF GPU work identifies the linear solver as the dominant component and evaluates synthetic Northeast 25K and Eastern US 70K grids. | Confirms our 25K/USA target cases are application-realistic, not arbitrary. |
| [GPU-resident sparse direct linear solvers for ACOPF](https://www.ornl.gov/publication/gpu-resident-sparse-direct-linear-solvers-alternating-current-optimal-powerflow) | Peer-reviewed, 2024 | Sparse linear systems commonly take more than half of economic-dispatch/ACOPF runtime; GPU-native sparse direct solvers can improve end-to-end GPU workflows. | Supports device-resident solve and batched NR as the system-level objective. |
| [SABLE: GPU-Based Power Flow Accelerator](https://arxiv.org/abs/2606.07099) | Very recent preprint, 2026-06-05 | Uses reusable sparse templates, custom GPU kernels, cuDSS-based sparse-direct LU, and mixed precision for batched differentiable power flow. | Latest external confirmation that fixed-pattern reuse + sparse-direct LU + mixed precision is the active direction for batched power-flow workloads. Treat as unreviewed until peer review. |
| [MAGMA Batched](https://icl.utk.edu/files/publications/2016/icl-utk-909-2016.pdf) | Technical report/paper family | Small matrix factorizations need batched, size-specialized kernels and careful launch amortization. | Supports specialized small-front kernels for absolute time, but does not imply TC ratio improves. |
| [Fast batched small HGEMM using Tensor Cores](https://doi.org/10.1109/IPDPS.2019.00022) | Peer-reviewed, 2019 | Tensor Cores can help very small GEMMs only with custom low-level designs that overcome hardware/API tile restrictions. | If tiny TC is revisited, it must be many-front packing or shape-changing panelization. One-front padded `m16n8k8` is the wrong baseline. |
| [cuBLAS grouped GEMM blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) and [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html) | NVIDIA official | Grouped GEMM combines different GEMM shapes in one launch; C outputs in batched GEMM must not overlap. | Useful for independent 49..128/big-high trailing updates. Not directly usable for overlapping parent accumulation. |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) and [A100 Ampere whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf) | NVIDIA official | TF32 Tensor Cores operate on MMA tile shapes and produce FP32 outputs from FP32-range/TF32-precision inputs. | Hard constraint: 8387 `nc=1/2` is not merely "small"; it underfills the TC K granularity. |
| [Mixed-precision iterative refinement with Tensor Cores](https://pmc.ncbi.nlm.nih.gov/articles/PMC7735315/) | Peer-reviewed, 2020 | Low-precision factor/solve plus high-precision residual/correction can recover accuracy and model when TC-IR is beneficial. | Accuracy story should move to NR/refinement wall time, not one-shot residual only. |

### What the widened literature changes

The extra references do not reverse the current interpretation. They sharpen it:

- **Large 25K/USA**: literature strongly supports the current path. Supernodal/multifrontal GPU
  solvers and cuDSS both exploit dense regions, phase reuse, batching, and vendor dense kernels.
  Our stable 1.22-1.28x factorize speedups at B64/B256 are in the right regime.
- **8387**: literature makes the negative result less surprising. cuDSS explicitly adds host/hybrid
  execution for small matrices; batched small-GEMM papers require custom aggregation to use TC well;
  KLU/Basker/circuit-solver work exists because low-fill sparse problems are not naturally dense
  BLAS problems.
- **Non-GEMM**: SuperLU_DIST batched and cuDSS both treat scatter, level scheduling, and triangular
  solve dependencies as real design targets. Multi-batch helps launch amortization and parallelism,
  but it does not remove parent assembly, panel LU/U-solve, pointer setup, or synchronization.
- **Accuracy**: the mixed-precision literature suggests a separate metric: `TF32 factor + correction`
  end-to-end NR/refinement wall time. Raw TF32 residual is a diagnostic, not the whole story.

### Decision matrix for next work

| Candidate | Literature support | Local evidence | Decision |
|---|---|---|---|
| Keep current mid/big TF32 policy for 25K/USA | Strong: multifrontal/supernodal dense updates, cuDSS dense-factor improvements, grouped/batched dense kernels. | Repeat=61: 25K/USA B64/B256 all above 1.2x. | Keep as canonical large-case policy. |
| More one-front-at-a-time tiny TC on 8387 | Weak: TC small-GEMM literature says special aggregation is needed; cuDSS treats small matrices as hybrid/host candidates. | Tiny TC, small bucket, variable-lane, tiny-lane, small-TC repeats all failed or unstable. | Stop. Do not spend more cycles here unless the shape changes. |
| Many-front packed TC for 8387 tiny leaves | Conditional: supported in spirit by batched small HGEMM, but only if tile utilization is acceptable. | docs/37 shows `small<=16` packed efficiency about 6.5% and useful share about 21-22%. | Reject current-shape implementation. Revisit only after ordering/panelization increases `uc,nc`. |
| Parent update / assembly redesign | Strong: multifrontal assembly is algorithmic core; SuperLU_DIST adds batched Scatter; cuDSS tips discuss bandwidth/sync bounds. | docs/35 says fan-in alone is not enough for 8387, but large-case parent update sizes are meaningful. | Profile timing by parent/update bucket before implementing a reduce kernel. |
| Grouped GEMM for 49..128/big-high | Moderate to strong: cuBLAS grouped GEMM handles variable shapes in one launch. | Previous grouped attempt did not beat custom path broadly; current policy still leaves some mid/big trailing work. | Revisit narrowly for independent outputs only, with pointer metadata amortized and graph-safe. |
| Low-fill sparse-LU branch for 8387 | Strong from KLU/Basker/circuit-solver line. | 8387 dominated by tiny fronts; TC ratio target may not improve because FP32 common work improves too. | Consider only if absolute factor/solve time matters more than TC-enabler ratio. |
| TF32 factor plus correction/refinement | Strong: Buttari/Haidar/Ootomo line and SuperLU_DIST mixed precision. | Current TF32 residuals around `~5e-2` on large cases; NR tolerance impact not yet measured. | Add end-to-end NR/refinement experiment as separate success metric. |

### Reading order for follow-up

1. **Implementation architecture**: cuDSS docs + SuperLU_DIST batched sparse direct.
2. **Large-front GPU solver design**: STRUMPACK GPU BLR + SuperLU_DIST 3D/mixed precision.
3. **Tiny/mid dense kernel mechanics**: MAGMA Batched + IPDPS 2019 small HGEMM + cuBLAS grouped.
4. **Power-grid application validity**: ACOPF GPU papers + SABLE preprint.
5. **Accuracy story**: sparse mixed precision + Tensor Core iterative refinement + corrected TC GEMM.

## Sparse Direct Solver Foundations

### Duff and Reid: multifrontal as dense updates plus assembly

Duff and Reid's multifrontal work established the basic view used by this solver: local dense
frontal matrices, update/contribution blocks, and an assembly tree. The unsymmetric paper states
the core performance motivation clearly: analysis should be small relative to factorization, storage
should be predictable, and vector/parallel factorization should have extra performance opportunity.
Sources:

- [The Multifrontal Solution of Indefinite Sparse Symmetric Linear Equations](https://doi.org/10.1145/356044.356047)
- [The Multifrontal Solution of Unsymmetric Sets of Linear Equations](https://doi.org/10.1137/0905045)

Implication for us: parent `extend_add` is not an incidental store path; it is the assembly of
contribution blocks into parent fronts. A serious redesign should reason about contribution-block
scheduling and parent fan-in, not only atomic instructions.

### Liu: assembly tree, update matrix, supernodes

Liu's SIAM Review article is the best compact conceptual reference for fronts, update matrices,
assembly trees, reorderings, and supernodes. Source:

- [The Multifrontal Method for Sparse Matrix Solution: Theory and Practice](https://doi.org/10.1137/1034004)

Implication for us: the next useful profile should be tree-aware. Per-kernel timing is not enough;
we need parent fan-in, update-block size, and level/subtree locality distributions.

### Davis/Duff UMFPACK: unsymmetric-pattern multifrontal dense kernels

Davis and Duff's unsymmetric-pattern multifrontal paper is directly relevant to unsymmetric sparse
LU. It frames the issue we keep seeing: general sparse factorization has irregular access, while
multifrontal methods try to push the innermost work into regular dense kernels. Source:

- [An Unsymmetric-Pattern Multifrontal Method for Sparse LU Factorization](https://doi.org/10.1137/S0895479894246905)
- [Author PDF](https://people.engr.tamu.edu/davis/publications_files/Unsymmetric_Pattern_Multifrontal_Method_for_Sparse_LU_Factorization.pdf)

Implication for us: Tensor Cores only help where the multifrontal transformation actually creates
large enough dense GEMM-like work. 8387 leaves largely fail that condition.

### SuperLU: supernodes make dense kernels first-class

SuperLU's sparse partial pivoting paper introduces unsymmetric supernodes, panel updates, and 2D
partitioning for memory hierarchy. Source:

- [A Supernodal Approach to Sparse Partial Pivoting](https://doi.org/10.1137/S0895479895291765)
- [SuperLU users' guide](https://doi.org/10.2172/751785)

Implication for us: the large-case policy is aligned with supernodal/multifrontal literature:
increase dense update residency and use tuned matrix kernels when the front shape permits it. It
does not imply tiny fronts should be forced through TC.

### KLU: circuit matrices are a different regime

KLU is the counterexample to the "more dense BLAS is always better" story. It targets circuit
simulation matrices, uses BTF, fill-reducing orderings, and Gilbert/Peierls sparse left-looking LU.
Source:

- [Algorithm 907: KLU, a Direct Sparse Solver for Circuit Simulation Problems](https://doi.org/10.1145/1824801.1824814)

Implication for us: 8387 may be structurally closer to a KLU/circuit regime than to a PDE-like
large-front regime. A KLU-like branch could improve absolute 8387 time, but it will probably reduce
FP32 and TF32 similarly and therefore may not satisfy the requested TC-enabler ratio.

## GPU Sparse Direct Solver Literature

### Early CPU-GPU multifrontal work: thresholded GPU offload

The CPU-GPU unsymmetric multifrontal paper analyzes when GPU BLAS helps and emphasizes thresholding,
communication, and memory reuse. Source:

- [A CPU-GPU hybrid approach for the unsymmetric multifrontal method](https://doi.org/10.1016/j.parco.2011.09.002)

Implication for us: our current tiering policy is consistent with the literature. The GPU path must
be shape-thresholded; not every front should be pushed into the most sophisticated GPU kernel.

### STRUMPACK GPU multifrontal: offload enough dense work, control movement

Ghysels and Synk's STRUMPACK GPU work ports numerical factorization and triangular solves to GPUs
for multifrontal sparse LU. The recurring lesson is to offload/coarsen fronts with enough dense
linear algebra while managing movement and launch overhead. Sources:

- [High performance sparse multifrontal solvers on modern GPUs](https://escholarship.org/uc/item/7tv84567)
- [GPU STRUMPACK PDF](https://escholarship.org/content/qt7tv84567/qt7tv84567_noSplash_d41501c8913db5b2aa4fc426284a01c2.pdf)

Implication for us: the large fronts deserve more work; 8387's leaves do not automatically become a
Tensor Core workload just because the solver is multifrontal.

### STRUMPACK BLR / compressed fronts: large-front direction, not 8387 rescue

Recent STRUMPACK BLR GPU work uses block low-rank compression inside a GPU-capable multifrontal
solver and surveys modern sparse direct solvers with GPU support. Source:

- [A graphics processing unit accelerated sparse direct solver and preconditioner with block low rank compression](https://doi.org/10.1177/10943420241288567)

Implication for us: compression is a plausible future path for USA/70K-style large fronts and memory
traffic, but it is not a near-term 8387 fix because 8387 is dominated by tiny dense blocks, not
large compressible fronts.

### SuperLU_DIST batched: closest match to B64/B256 sparse direct batching

Boukaram et al. redesign SuperLU_DIST into a batched GPU sparse direct solver. Key parallels:
factorization by elimination-tree levels, batched dense operations at each level, and a batched
scatter kernel. Source:

- [Batched sparse direct solver design and evaluation in SuperLU_DIST](https://doi.org/10.1177/10943420241268200)

Implication for us: batched scatter/assembly and level-wise dense grouping are first-class solver
design elements. This supports profiling `extend_add` and scatter by parent/update shape, not only
retuning TC instructions.

### cuDSS: vendor baseline and API evidence

cuDSS is NVIDIA's direct sparse solver library. Its docs list non-uniform and uniform batching,
analysis/numerical factorization/solve phases, optional refactorization and iterative refinement,
pivoting controls, and GPU/multi-GPU modes. NVIDIA's cuDSS blog also explicitly notes that tiny and
small matrices may not have enough parallelism to saturate a GPU and can pay non-negligible overhead.
Sources:

- [cuDSS documentation](https://docs.nvidia.com/cuda/cudss/index.html)
- [cuDSS engineering/scientific computing blog](https://developer.nvidia.com/blog/nvidia-cudss-advances-solver-technologies-for-engineering-and-scientific-computing/)

Implication for us: black-box vendor evidence agrees with our 8387 experience. Tiny problems need
either batching/aggregation or CPU/host-like treatment; GPU TC per tiny front is unlikely to win.

## Batched Dense / Tensor Core References

### MAGMA batched BLAS/LAPACK: many small matrices need custom kernels

MAGMA batched work targets many small matrices and designs batched BLAS/LAPACK kernels around that
regime. Source:

- [MAGMA Batched: A Batched BLAS Approach for Small Matrix Factorizations and Applications on GPUs](https://icl.utk.edu/files/publications/2016/icl-utk-909-2016.pdf)

Implication for us: for small fronts, launch amortization and size-specific kernels matter more than
calling a large-matrix primitive. Our failed small bucket split and variable-lane experiment tested
two local variants, but they did not change enough common work to improve the FP32-relative TC ratio.

### Batched GEMM for small sizes: Tensor Cores can help only after packing/aggregation

Abdelfattah, Tomov, and Dongarra's batched HGEMM paper is the closest dense-kernel analog for tiny
fronts. It focuses on small matrices that cannot fully occupy Tensor Cores and shows that special
packing/multi-problem mapping is needed. Source:

- [Fast Batched Matrix Multiplication for Small Sizes using Half-Precision Arithmetic on GPUs](https://netlib.org/utk/people/JackDongarra/PAPERS/ipdps-batched-2019.pdf)

Implication for us: if we ever revisit tiny TC, it should be a multi-front packed TC kernel, not
one-front-at-a-time padding into `m16n8k8`. But for 8387, `nc=1/2` means the K dimension is so thin
that even packing has to be very carefully justified.

### Batched LU: Level-3 BLAS can be the wrong emphasis for very small matrices

The progressive batched LU paper states the small-matrix lesson directly: relying on Level-3 BLAS
does not pay off when the problem is very small; memory-bound parts and kernel fusion matter. Source:

- [Progressive Optimization of Batched LU Factorization on GPUs](https://doi.org/10.1109/HPEC.2019.8916270)

Implication for us: 8387's small-front wall is probably not fixed by more GEMM. A small-front
structural path should target staging/writeback/control and assembly, with the expectation that the
TC ratio target may remain hard.

### Batched GEMM autotuning and variable sizes

Abdelfattah et al. also show that variable-size batched GEMM requires specialized design and
autotuning. Source:

- [Performance, Design, and Autotuning of Batched GEMM for GPUs](https://icl.utk.edu/publications/performance-design-and-autotuning-batched-gemm-gpus)

Implication for us: cuBLAS/CUTLASS grouped paths are plausible for 49..128 and big-high trailing
updates, where shapes are variable but still large enough. They are not a credible tiny-front
shortcut by themselves.

### cuBLAS grouped GEMM: useful, but only for independent dense updates

NVIDIA cuBLAS 12.5 introduced grouped GEMM APIs that group different matrix sizes/transposes/scales
into one launch, including FP32/TF32 support. Official cuBLAS docs also warn that batched GEMM
matrices must be independent and non-overlapping. Sources:

- [Grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html)

Implication for us: grouped GEMM is worth revisiting for mid/big independent trailing updates, but
it does not solve panel LU, U-solve, parent `extend_add`, graph-capture pointer setup, or overlapping
parent accumulation.

### PTX / Ampere Tensor Core constraints

The PTX ISA documents TF32 `mma.sync.aligned.m16n8k8...tf32...f32` and related sm_80 requirements.
The A100/Ampere whitepaper places TF32 in the Tensor Core/HPC path and notes cuSOLVER support for
Tensor Core formats. Sources:

- [PTX ISA warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [NVIDIA A100 Tensor Core GPU Architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)

Implication for us: the hardware tile shape is a hard constraint. An `nc=1/2` trailing update is not
just "small"; it is below the K granularity that TF32 MMA wants to amortize.

## Mixed Precision And Accuracy

### Sparse mixed precision: factor low, refine high

Buttari et al. studied mixed precision for sparse matrix computations using direct and iterative
solvers, including iterative refinement around sparse direct factors. Source:

- [Using Mixed Precision for Sparse Matrix Computations to Enhance the Performance while Achieving 64-bit Accuracy](https://icl.utk.edu/projectsfiles/rib/pubs/a17-buttari.pdf)

Implication for us: raw TF32 factor residual `~5e-2` is not the final accuracy question. The right
question is whether NR convergence or an explicit correction/refinement loop absorbs it at lower
end-to-end cost.

### Tensor Core iterative refinement

Modern TC iterative-refinement work uses low precision factors/solves and high precision residual
correction to recover target accuracy. Source:

- [Mixed-precision iterative refinement using tensor cores on GPUs to accelerate solution of linear systems](https://icl.utk.edu/files/publications/2020/icl-utk-1439-2020.pdf)

Implication for us: evaluate `TF32 factor + FP32/FP64 residual correction` separately from the
factor-time speedup metric. It may be the better story for accuracy, but it is a different target.

### Corrected Tensor Core GEMM

Ootomo and Yokota show that Tensor Core GEMM can recover FP32-like accuracy with correction schemes
while still exceeding FP32 SIMT throughput on A100. Source:

- [Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak performance](https://arxiv.org/abs/2203.03341)

Implication for us: correction is interesting for large dense updates, but applying it to 8387 tiny
fronts is likely overhead-heavy. It is more plausible for big-high trailing updates or a refinement
path than for tiny per-front LU.

## Power Grid / ACOPF GPU Context

### ACOPF linear solver dominance

The GPU ACOPF paper identifies the sparse linear solver as the core bottleneck and reports GPU
ACOPF experiments on synthetic Northeast 25K and Eastern U.S. 70K grids. Source:

- [Towards Efficient Alternating Current Optimal Power Flow Analysis on GPUs](https://arxiv.org/abs/2302.08656)

Implication for us: the 25K/USA targets are well chosen. They are exactly the regime where a
GPU-native direct solver matters.

### GPU-resident sparse direct solver for ACOPF

The GPU-resident sparse direct solver paper argues for keeping the sparse solver and surrounding
analysis on GPU and reports that linear systems typically take more than half of the computation in
economic dispatch/ACOPF workflows. Source:

- [GPU-Resident Sparse Direct Linear Solvers for ACOPF](https://arxiv.org/abs/2306.14337)

Implication for us: graph capture, analysis reuse, device-resident factor/solve, and batching are
not optional details. They are the application-level reason the custom solver can be competitive
even when individual kernels have non-GEMM bottlenecks.

### Batched Newton-Raphson power flow

GPU/CPU batched NR power-flow work uses simultaneous calculations and LU refactorization to exploit
many related systems. Source:

- [Fast parallel Newton-Raphson power flow solver for large number of system calculations with CPU and GPU](https://doi.org/10.1016/j.segan.2021.100483)

Implication for us: B64/B256 is application-realistic, not a synthetic benchmark trick. Shared
sparsity and repeated numeric refactorization are the correct assumptions.

## Recent Review / Broader Trend

Li and Liu's 2026 review organizes modern sparse direct solvers around two axes: reducing
communication/latency in task/data parallel settings and reducing complexity through low-rank or
hierarchical compression. Source:

- [Parallel Sparse and Data-Sparse Factorization-based Linear Solvers](https://arxiv.org/abs/2602.14289)

Implication for us: our next work should stop looking like isolated kernel toggles. The credible
next directions are latency/communication reduction in the assembly tree, and possibly compression
or grouped dense updates for large fronts.

## Implications For Current Direction

### 1. The large-case TC policy is literature-aligned

The stable 25K/USA result is not accidental. It follows the multifrontal/supernodal playbook:
make enough dense trailing update visible, keep enough front state resident, and use Tensor Core
MMA only where the tile shape is not absurdly underfilled.

### 2. 8387 is not a per-front Tensor Core problem

KLU/circuit sparse LU and batched-small-LU literature explain 8387 better than dense GEMM
literature. The matrix has many low-K tiny fronts. A smaller TC gate, direct-shared read path, or
variable-lane warp packing changes local overhead but does not create enough Tensor Core-covered
work. A one-thread-per-front `fsz<=8` tiny-lane path was also measured and removed: it only reached
about `1.03..1.06x` at B64 and regressed or tied B256. Our failed experiments are consistent with
the literature.

### 3. Parent assembly deserves the next design pass

The multifrontal literature treats update matrices and assembly as core algorithmic objects. For our
code, the first fan-in profile (`docs/35`) already rules out a broad "high-conflict parent reduce"
as an obvious 8387 fix: fan-in 9+ accounts for only about 21% of 8387 `uc^2` update elems. The
next useful work is therefore timing-gated:

- profile `extend_add` time by parent fan-in, child count, update size, and conflict rate,
- identify parents where siblings can be reduced before global writes,
- test two-stage parent accumulation only where it reduces TF32 wall more than FP32 wall,
- avoid repeating `scatter_values_unique`: improving common FP32 work can lower the reported TC
  speedup ratio even if absolute time improves.

### 4. Grouped GEMM should be restricted to mid/big

Grouped GEMM is credible only for independent 49..128 and big-high trailing updates. It should not
be used as a 8387 tiny-front rescue path. The key implementation risk remains graph-captured pointer
setup and independence of output blocks.

### 5. Mixed precision should be evaluated at NR/refinement level

If TF32 residuals around `5e-2` are too high for one-shot linear solves, the literature suggests
iterative refinement or NR-level correction, not abandoning TC outright. This should be measured as
end-to-end NR convergence and wall time, separate from raw factorize speedup.

## Recommended Shortlist

1. **Parent-update timing gate**: the fan-in/update-size histogram now exists; measure whether the
   large enough buckets consume at least `~10%` of factor wall before writing a new reduce kernel.
2. **Mid/big grouped GEMM A/B**: revisit grouped GEMM only for 49..128 and big-high trailing, with
   graph-captured pointer arrays prepared outside capture or via stable device metadata.
3. **TF32 factor + correction path**: measure one NR loop or explicit refinement with TF32 factors,
   FP32/FP64 residual, and correction solve. Report convergence and end-to-end wall, not just
   factor residual.
4. **8387 structural branch**: local 8387 routes are now heavily constrained by docs 35..37. If
   8387 must remain in scope, the credible work is deeper ordering/panelization that creates
   `uc,nc` large enough for TC, or a separate low-fill sparse-LU path if absolute time matters more
   than the TC-enabler ratio.

## Bottom Line

The literature supports the current large-case direction and explains why 8387 remains stubborn.
Tensor Cores are a valid enabler for 25K/USA once enough mid/big trailing work is exposed. For 8387,
the credible path is not more per-front TC; it is either parent-assembly/small-front structural work
or a separate low-fill sparse-LU branch, with the caveat that such a branch may improve absolute
time without improving the FP32-relative TC speedup ratio.
