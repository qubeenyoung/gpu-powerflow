# Literature Deep Dive Follow-up: Sparse Direct, Tensor Cores, and Power Flow

**Date**: 2026-06-09

**Purpose**: Follow-up to `34-gpu-sparse-direct-tc-literature-survey-2026-06-09.md`.
The previous survey already established the main position: the current mid/big TF32 Tensor Core
path is credible for 25K/USA, while 8387 is structurally hostile to one-front-at-a-time Tensor
Core acceleration. This note widens the reference set and turns it into a decision map for what to
try next.

## Executive Read

- The literature now strongly supports the large-case path: GPU sparse direct solvers get most of
  their wins by exposing dense regions, reusing analysis/symbolic data, batching systems, and
  keeping factor/solve phases device-resident.
- 8387 looks less like a "small dense GEMM" problem and more like the circuit/power-grid low-fill
  sparse-LU regime. In that regime, fine-grained dependencies, tiny supernodes/fronts, and memory
  movement dominate.
- Tensor Core papers do not say "use TC on every small matrix." They say the opposite: for very
  small or sparse-derived shapes, useful TC work appears only after aggregation, shape regularity,
  or high "TCU synergy."
- cuDSS documentation and blog posts are especially relevant because they independently converge on
  our current split: GPU/device path for dense-factor regions and batch reuse; hybrid/host execution
  or different dispatch for small/medium matrices.
- One local caveat remains: current `CLS_MID_TF32_LOW_TC` dispatch is size-gated and does not
  actually enable the low-mid 33..48 bucket on 8387 (`n=14908`). A force-all diagnostic is still
  worth one current-state measurement, but literature suggests it is a narrow diagnostic, not a new
  main direction.

## Reference Clusters

### A. Vendor baseline: cuDSS as current sparse-direct design evidence

Sources:

- [NVIDIA cuDSS documentation](https://docs.nvidia.com/cuda/cudss/index.html)
- [cuDSS Getting Started](https://docs.nvidia.com/cuda/archive/13.0.1/cudss/getting_started.html)
- [cuDSS Advanced Features](https://docs.nvidia.com/cuda/cudss/advanced_features.html)
- [cuDSS Tips and Tricks](https://docs.nvidia.com/cuda/cudss/tips_and_tricks.html)
- [NVIDIA cuDSS Advances Solver Technologies](https://developer.nvidia.com/blog/nvidia-cudss-advances-solver-technologies-for-engineering-and-scientific-computing/)
- [Solving Large-Scale Linear Sparse Problems with NVIDIA cuDSS](https://developer.nvidia.com/blog/solving-large-scale-linear-sparse-problems-with-nvidia-cudss/)

Observed points:

- cuDSS exposes analysis, factorization, solve, optional refactorization, optional iterative
  refinement, batching, pivot controls, memory/flop estimates, hybrid host/device modes, and
  multi-GPU modes.
- cuDSS has both uniform and non-uniform batching, matching our B64/B256 use case more closely than
  single-system sparse-direct benchmarks.
- cuDSS explicitly recommends hybrid host/device execution for small and medium matrices where GPU
  parallelism is insufficient.
- cuDSS performance guidance separates phase timing, requires synchronization for asynchronous
  factor/solve timing, and warns that factorization can be memory-bandwidth bound depending on
  factor structure while solve is often memory/synchronization bound.
- The advanced docs expose nested-dissection partition tree reuse, not just permutation reuse. This
  is directly relevant to our seed/panelization work: the tree is a performance object, not only a
  row/column order.

Implication for our code:

- The current phase split and repeated numeric factorization model are not just implementation
  choices; they match vendor API structure.
- 25K/USA should keep using the stable mid/big GPU path as canonical.
- 8387 should not be judged by "can TC be forced into every front?" The external baseline itself
  uses size-based dispatch and hybrid execution for small matrices.
- Analysis/order/panel tree data deserves more attention than seed-only permutation sweeps.

### B. Batched sparse direct solvers: level batching and scatter are first-class

Sources:

- [Batched sparse direct solver design and evaluation in SuperLU_DIST](https://doi.org/10.1177/10943420241268200)
- [A distributed CPU-GPU sparse direct solver](https://portal.nersc.gov/project/sparse/xiaoye-web/europar14.pdf)
- [New SuperLU_DIST capabilities](https://doi.org/10.1145/3577197)

Observed points:

- The SuperLU_DIST batched sparse-direct paper is the closest external analog to this work: it
  factors by elimination-tree levels, uses batched dense operations per level, and adds a batched
  Scatter GPU kernel.
- Earlier distributed CPU-GPU SuperLU_DIST work already identified the problem that small dense
  operations dominate parts of the workload and need aggregation/scheduling to use GPUs well.
- Newer SuperLU_DIST work emphasizes communication-avoiding sparse LU and mixed precision with
  iterative refinement.

Implication for our code:

- `extend_add`, scatter, and parent update are not secondary cleanup. They are algorithmic kernels
  that external solvers optimize explicitly.
- Multi-batch helps only when it aggregates enough independent work. It does not erase panel
  dependencies, parent assembly, or small-front scheduling overhead.
- A parent-update redesign should be timing-gated and shape-gated, not assumed from fan-in alone.

### C. Multifrontal/supernodal GPU solvers: dense regions decide the win

Sources:

- [High performance sparse multifrontal solvers on modern GPUs](https://www.sciencedirect.com/science/article/pii/S0167819122000059)
- [A graphics processing unit accelerated sparse direct solver and preconditioner with block low rank compression](https://doi.org/10.1177/10943420241288567)
- [MUMPS solver documentation](https://mumps-solver.org/index.php)
- [Parallel Sparse and Data-Sparse Factorization-based Linear Solvers](https://arxiv.org/abs/2602.14289)

Observed points:

- STRUMPACK GPU work reinforces the standard multifrontal condition for success: good ordering
  should move most numerical work into dense supernodes/frontal matrices.
- STRUMPACK BLR pushes large-front work further by using block low-rank compression and GPU dense
  kernels such as variable-size GEMM/TRSM/LU.
- The 2026 sparse-direct review organizes modern progress around two axes: reducing
  communication/latency and reducing complexity through low-rank/hierarchical compression.

Implication for our code:

- Large cases are in the right regime: current repeat=61 large-case speedups above 1.2x are
  consistent with dense-region exposure.
- BLR/compression is not a near-term 8387 rescue. It is relevant to USA/70K-style large-front memory
  and compute pressure.
- For 8387, a structural change must create larger useful fronts or change the algorithmic branch;
  it cannot just add more TC code to the existing tiny leaves.

### D. Regular sparse-block alternatives: PanguLU, Caracal, and circuit-style GPU LU

Sources:

- [PanguLU SC23 proceedings page](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap248.html)
- [Caracal Zenodo artifact](https://zenodo.org/records/16778900)
- [GLU3.0: Fast GPU-based Parallel Sparse LU Factorization for Circuit Simulation](https://arxiv.org/abs/1908.00204)
- [Basker: A Threaded Sparse LU Factorization](https://arxiv.org/abs/1601.05725)
- [KLU: A Direct Sparse Solver for Circuit Simulation Problems](https://doi.org/10.1145/1824801.1824814)
- [State-of-The-Art Sparse Direct Solvers](https://arxiv.org/abs/1907.05309)

Observed points:

- PanguLU deliberately avoids the classic multifrontal/supernodal dense-fill path: it uses regular
  2D blocking, stores blocks in sparse form to avoid extra fill, and chooses block-wise sparse BLAS
  methods on local GPUs.
- Caracal is a recent GPU-resident sparse LU line with fine-grained scheduling; its artifact and
  SC25 metadata place it in the same low-fill/direct-LU space as circuit simulation.
- GLU3.0 and related circuit GPU LU work emphasize irregular dependencies, stage-adaptive kernels,
  and data-dependency detection, not Tensor Core dense-GEMM dominance.
- Basker/KLU explain why circuit and power-grid matrices often benefit from low-fill sparse LU
  strategies rather than aggressive dense-kernel transformation.

Implication for our code:

- If 8387 remains mandatory, the credible alternative is a low-fill sparse-LU branch or a deeper
  panelization/order transformation, not another local TC gate.
- Such a branch may improve absolute time while failing the requested "TF32 vs FP32 factorize
  speedup" ratio, because it will reduce common scalar work for both precisions.
- This is a project-direction decision: a low-fill branch optimizes the application, not the
  Tensor-Core-as-enabler claim.

### E. Tensor Core mechanics and grouped dense kernels

Sources:

- [PTX ISA warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [NVIDIA A100 architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
- [cuBLAS grouped GEMM blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [CUTLASS grouped kernel scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)
- [Fast batched matrix multiplication for small sizes using half-precision arithmetic on GPUs](https://doi.org/10.1109/IPDPS.2019.00022)
- [Performance, Design, and Autotuning of Batched GEMM for GPUs](https://icl.utk.edu/publications/performance-design-and-autotuning-batched-gemm-gpus)

Observed points:

- TF32 MMA tile shapes (`m16n8k8` and related shapes on sm_80+) impose a real K granularity.
- cuBLAS grouped GEMM and CUTLASS grouped kernels are designed to batch heterogeneous GEMMs in one
  launch, using persistent scheduling over tiles/problems.
- Small Tensor Core GEMM papers do show wins, but only after careful low-level designs that deal
  with the fact that tiny matrices cannot fill Tensor Cores naturally.

Implication for our code:

- The 8387 dominant shapes `(6,2,4)`, `(4,2,2)`, `(5,2,3)` are below useful TC granularity. This is
  a hardware-shape mismatch, not just an implementation inefficiency.
- Grouped GEMM is still worth a narrow revisit for independent 49..128 and big-high trailing
  updates, especially where pointer setup can be amortized or captured.
- Grouped GEMM does not solve overlapping parent accumulation, panel LU, U-solve, or graph metadata
  setup.

### F. Tensor Cores for sparse-derived kernels: synergy must be measured

Sources:

- [cuTeSpMM: Accelerating Sparse-Dense Matrix Multiplication using GPU Tensor Cores](https://arxiv.org/abs/2504.06443)
- [Accelerating sparse matrix-matrix multiplication with GPU Tensor Cores](https://doi.org/10.1016/j.compeleceng.2020.106848)
- [Spatula: A Hardware Accelerator for Sparse Matrix Factorization](https://doi.org/10.1145/3613424.3623783)

Observed points:

- Recent Tensor Core SpMM work introduces the idea that only some sparse patterns have enough
  "TCU synergy" to beat scalar GPU kernels.
- Tensor Core sparse-kernel work typically succeeds by finding dense-enough tiles or by processing
  sparse tiles with enough useful payload.
- Spatula is a hardware-architecture paper, but it is useful as a negative control for generic GPU
  expectations: sparse factorization combines structured small dense work with long dependency
  chains and irregular reuse, which existing GPUs often underutilize.

Implication for our code:

- We should require a front/update "TC synergy" metric before adding TC kernels. For this codebase,
  candidate metrics are `nc`, `uc`, `useful_mma_flop / padded_mma_flop`, front count per grouped
  launch, and parent-output independence.
- The existing docs/36 and docs/37 analyses are in the right spirit. They should be treated as
  go/no-go filters, not after-the-fact explanations.
- 8387's small-front TC efficiency estimate is far below the threshold implied by the sparse-TC
  literature.

### G. Mixed precision and accuracy: move the metric to NR/refinement

Sources:

- [Mixed-precision iterative refinement using tensor cores on GPUs](https://pmc.ncbi.nlm.nih.gov/articles/PMC7735315/)
- [Using Mixed Precision for Sparse Matrix Computations](https://icl.utk.edu/projectsfiles/rib/pubs/a17-buttari.pdf)
- [Performance impact of precision reduction in sparse linear systems solvers](https://pmc.ncbi.nlm.nih.gov/articles/PMC8771784/)
- [Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak performance](https://arxiv.org/abs/2203.03341)

Observed points:

- Mixed-precision direct-solver literature usually does not stop at raw low-precision factorization.
  It factors low, computes residual/corrections higher, and reports final accuracy and end-to-end
  time.
- cuDSS also exposes iterative refinement controls and discusses residual-based accuracy checks.
- Corrected Tensor Core GEMM is plausible for large dense updates, but correction overhead is
  probably too high for 8387 tiny fronts.

Implication for our code:

- Large-case TF32 residuals around `~5e-2` should trigger an NR/refinement-level experiment, not an
  immediate rejection.
- The success metric should be: `TF32 factor + correction` end-to-end wall time and convergence
  versus FP32/FP64, with repeated sparsity pattern reuse.
- This metric is separate from the current "factorize speedup over FP32" goal.

### H. Power-flow and ACOPF application context

Sources:

- [Linear solvers for power grid optimization problems: review of GPU-accelerated linear solvers](https://www.pnnl.gov/publications/linear-solvers-power-grid-optimization-problems-review-gpu-accelerated-linear-solvers)
- [Towards Efficient ACOPF on GPUs](https://arxiv.org/abs/2302.08656)
- [GPU-resident sparse direct linear solvers for ACOPF](https://www.ornl.gov/publication/gpu-resident-sparse-direct-linear-solvers-alternating-current-optimal-powerflow)
- [Accelerating Optimal Power Flow with GPUs](https://arxiv.org/abs/2307.16830)
- [SABLE: GPU-Based Power Flow Accelerator for Sparsity-Aware Batched Learning](https://arxiv.org/abs/2606.07099)

Observed points:

- The 2021/2022 power-grid linear-solver review reported that existing tested packages did not
  deliver significant GPU acceleration for those power-grid matrices, which frames why custom
  GPU-resident work matters.
- The 25K and Eastern US 70K cases appear in external GPU ACOPF literature, so our benchmark cases
  are application-relevant.
- GPU-resident ACOPF work reports that the sparse linear systems often dominate runtime and that
  keeping the computation on GPU is important.
- SABLE is very recent and directly relevant: it combines reusable sparse templates, custom GPU
  kernels, cuDSS-based sparse-direct LU, batched power-flow, and mixed precision.

Implication for our code:

- The B64/B256 target is not artificial. Batched repeated power-flow/ACOPF workloads are now an
  explicit application direction.
- Device-resident setup, warm reused analysis, and fixed sparsity templates are central to the
  application story.
- SABLE strengthens the argument for mixed precision plus sparse-direct reuse, but it does not imply
  that every front should be a Tensor Core front.

## How This Reclassifies Our Current Options

| Option | Literature support | Current local evidence | Decision |
|---|---|---|---|
| Keep stable large-case mid/big TF32 policy | Strong: cuDSS dense-factor regions, multifrontal GPU, grouped/batched dense kernels. | 25K/USA repeat=61 are above 1.2x at B64/B256. | Keep canonical. |
| Force low-mid 33..48 TF32 on 8387 | Weak-to-moderate as a diagnostic only. TC shape is still thin, but current code did not truly test this for `n=14908`. | Current dispatch size gate excludes 8387 low-mid despite `CLS_MID_TF32_LOW_TC=ON`. | Run one force-all diagnostic; do not treat as main bet. |
| More one-front tiny TC | Weak. Small-TC literature requires aggregation; sparse-TC literature requires measured synergy. | Tiny-TC, small bucket split, packed tiny estimate all failed or low-ceiling. | Stop. |
| Many-front packed TC for 8387 | Conditional in theory, weak for current shapes. | docs/37 shows very low packed tile efficiency for `small<=16`. | Reject current-shape version. Revisit only if panelization increases `nc/uc`. |
| Parent/update redesign | Strong, because multifrontal assembly/scatter is core. | docs/35 fan-in does not prove enough 8387 headroom; large cases more promising. | Profile time by update shape before implementing. |
| Grouped GEMM for mid/big | Moderate-to-strong for independent outputs. | Prior broad grouped attempt was not enough; stable policy still has candidate mid/big work. | Revisit narrowly for 49..128/big-high independent trailing only. |
| Low-fill sparse-LU branch for 8387 | Strong for circuit/power-grid low-fill matrices. | 8387 dominated by tiny fronts and low K. | Good for absolute time, risky for TC-vs-FP32 ratio. |
| TF32 + correction/refinement | Strong for mixed precision accuracy. | Large-case TF32 factor speed exists; residual needs application-level handling. | Add NR/refinement wall-time experiment. |

## Reading Order

1. cuDSS docs, advanced features, and tips: phase split, batch/reuse, hybrid execution, measurement
   protocol, partition tree reuse.
2. SuperLU_DIST batched sparse direct: level scheduling, batched dense work, and scatter as a
   first-class GPU kernel.
3. STRUMPACK GPU/BLR and 2026 sparse-direct review: dense-front success conditions and large-front
   future directions.
4. PanguLU, Caracal, GLU3.0, Basker/KLU: low-fill sparse-LU branch for circuit/power-grid-like
   matrices.
5. PTX/cuBLAS/CUTLASS/small batched GEMM: hard Tensor Core tile constraints and grouped dense
   implementation options.
6. Power-grid GPU papers and SABLE: application-level justification for batch, sparse template
   reuse, mixed precision, and GPU-resident solve.

## Concrete Next Experiments Suggested By Literature

1. **Low-mid force-all diagnostic for 8387**
   - Add a default-off `CLS_MID_TF32_LOW_TC_FORCE_ALL` dispatch flag.
   - Test cap `{24,28,32}`, B `{64,256}`, repeat 31.
   - If it does not reach `>=1.2x` on both B64/B256, close the low-mid TC path.

2. **Parent update time attribution**
   - Use current dump data to bucket `extend_add` by parent fan-in, `uc`, `asm_len`, and output
     overlap/conflict.
   - Only implement a new parent reduce kernel if a bucket accounts for at least about 10% of
     factor wall and is not equally beneficial to FP32.

3. **Grouped GEMM narrow revisit**
   - Restrict to independent 49..128 and big-high trailing updates.
   - Keep pointer metadata stable and graph-safe.
   - Compare against the current custom path, not against a disabled/slow FP32 baseline.

4. **NR/refinement metric**
   - Measure `TF32 factor + correction` wall time and final NR convergence.
   - Report this separately from raw factorize speedup.

5. **8387 strategic branch decision**
   - If the project requires 8387 plus TC factor ratio, the remaining credible path is deeper
     ordering/panelization that increases useful `nc`, not a local kernel tweak.
   - If the project requires 8387 absolute performance, evaluate low-fill sparse-LU/circuit-style
     algorithms even if the TC ratio goal is abandoned for that case.

## Bottom Line

The expanded literature does not overturn the current direction. It narrows it.

- For 25K/USA, continue the current mid/big TF32 policy and refine accuracy through correction or
  NR-level measurement.
- For 8387, run the one missing low-mid force-all diagnostic because the current dispatch gate
  leaves a small ambiguity. After that, stop treating 8387 as a per-front Tensor Core problem unless
  panelization/order changes create materially larger `nc`.
- The next serious design work should be tree/assembly aware: parent update timing, grouped
  independent trailing for mid/big, and explicit low-fill branch tradeoffs.
