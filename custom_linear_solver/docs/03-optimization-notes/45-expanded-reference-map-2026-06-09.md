# Expanded Reference Map: GPU Sparse Direct Solvers, Tensor Cores, and Batched Power Flow

**Date**: 2026-06-09

**Purpose**: Broaden the literature base behind the current Tensor Core direction. This is not a
new experiment log. It is an annotated reference map that ties external evidence to our local
decisions: keep the large-case mid/big TF32 path, stop treating 8387/13K as a local per-front TC
kernel problem, and shift the remaining uncertainty to panelization/assembly/reuse or to a
low-fill sparse-LU branch.

## Short Read

- Modern sparse-direct literature agrees on one core condition for GPU success: expose enough
  dense work, batch independent work, and keep the symbolic/numeric reuse path cheap.
- Vendor cuDSS documentation independently supports our split. It exposes uniform/non-uniform
  batching, phase reuse, refactorization, iterative refinement, memory/flop estimates, and hybrid
  host/device execution. NVIDIA explicitly recommends hybrid execution for small and medium
  matrices with low GPU parallelism.
- Tensor Core literature does not support "turn TC on for all small fronts." TC wins require tile
  regularity, enough `K`, enough useful payload per padded tile, or grouped scheduling that
  amortizes metadata and launch costs.
- Circuit/power-grid low-fill sparse LU literature is the closest external analogy for 8387/13K.
  It repeatedly moves away from dense supernodal/multifrontal dominance and toward BTF,
  Gilbert-Peierls-style LU, regular sparse blocking, or fine-grained GPU scheduling.
- The newest power-flow evidence, especially SABLE (submitted 2026-06-05), validates batched
  sparse template reuse, cuDSS-based sparse-direct LU, custom GPU kernels, and mixed precision as
  an application direction. It does not imply every frontal update should use Tensor Cores.

## 1. Sparse-Direct Foundations

### Liu 1992, multifrontal theory

Source: [The Multifrontal Method for Sparse Matrix Solution: Theory and Practice](https://epubs.siam.org/doi/10.1137/1034004)

Relevant content:

- Formalizes frontal matrices, update matrices, assembly tree, reorderings, supernodes, and
  implementation details.
- For our code, this validates that `extend_add` and assembly are algorithmic core operations, not
  cleanup after GEMM.
- It also explains why changing panel/tree structure changes both work and dependency shape. A
  "bigger front" can expose dense kernels, but it can also add fill and solve cost.

Local implication:

- The 8387/13K failures after seed/cap, sibling amalgamation, chain amalgamation, and extend bucket
  timing are not surprising. The tree and assembly structure determine the useful dense work.
- Further local TC kernels are only justified after a structural metric shows more useful `nc` or
  `uc` per front.

### Davis/Duff/unsymmetric multifrontal and supernodal lineage

Sources:

- [A combined unifrontal/multifrontal method for unsymmetric sparse matrices](https://dl.acm.org/doi/10.1145/305658.287640)
- [An Unsymmetric-Pattern Multifrontal Method for Sparse LU Factorization](https://epubs.siam.org/doi/10.1137/S0895479894246905)
- [State-of-The-Art Sparse Direct Solvers](https://arxiv.org/abs/1907.05309)

Relevant content:

- Sparse elimination performance depends on preprocessing, ordering, fill control, dense
  submatrix detection, and concurrency.
- Dense level-3 BLAS is powerful only when the symbolic structure actually creates dense
  subproblems.

Local implication:

- For 25K/70K/USA, the local accepted repeat=61 speedups are aligned with dense-subproblem
  exposure.
- For 8387/13K, the dominant tiny shapes mean the code is in a low-fill regime, not a dense-front
  regime.

## 2. Vendor Baseline: cuDSS

Sources:

- [NVIDIA cuDSS documentation](https://docs.nvidia.com/cuda/cudss/)
- [cuDSS Advanced Features](https://docs.nvidia.com/cuda/cudss/advanced_features.html)
- [cuDSS Tips and Tricks](https://docs.nvidia.com/cuda/cudss/tips_and_tricks.html)
- [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/release_notes.html)
- [NVIDIA cuDSS advances solver technologies](https://developer.nvidia.com/blog/nvidia-cudss-advances-solver-technologies-for-engineering-and-scientific-computing/)

Relevant content:

- cuDSS exposes uniform batching, non-uniform batching, analysis, factorization, solve,
  refactorization, iterative refinement, memory/flop estimates, pivot controls, hybrid memory,
  hybrid execution, multi-GPU, and multi-node execution.
- Hybrid host/device execution is recommended for small and medium matrices when GPU parallelism
  is low. It is also a documented split from hybrid memory mode.
- The API treats analysis/reordering/symbolic data and repeated numeric phases as first-class
  objects.

Local implication:

- The current B64/B256 benchmark model is not artificial. It matches uniform-batch, repeated
  fixed-pattern sparse-direct use.
- The large-case mid/big TF32 path is a credible GPU/device path.
- 8387/13K should not be forced into a GPU-only Tensor Core story. Even NVIDIA's sparse-direct
  library exposes a hybrid route for the small/medium regime.

## 3. GPU Multifrontal and Supernodal Solvers

### STRUMPACK GPU

Sources:

- [High performance sparse multifrontal solvers on modern GPUs](https://www.osti.gov/pages/biblio/1960514)
- [GPU sparse direct solver with BLR compression](https://www.osti.gov/biblio/2499469)

Relevant content:

- STRUMPACK GPU offloads dense frontal work and sparse scatter-gather work.
- Large frontal matrices use vendor BLAS/solver libraries; smaller fronts use custom kernels to
  reduce launch overhead.
- High performance depends on identifying subtrees whose factorization data fit on GPU.
- BLR/compression work targets large-front cost and memory pressure.

Local implication:

- Our three-tier dispatch mirrors a published GPU multifrontal pattern: library-like dense kernels
  where fronts are big enough, custom kernels where they are not.
- BLR/compression is more relevant to USA/70K memory and large-front cost than to rescuing 8387.

### SuperLU_DIST GPU and batched sparse direct

Sources:

- [A distributed CPU-GPU sparse direct solver](https://portal.nersc.gov/project/sparse/xiaoye-web/europar14.pdf)
- [Newly released SuperLU_DIST capabilities](https://www.ornl.gov/publication/newly-released-capabilities-distributed-memory-superlu-sparse-direct-solver)
- [Batched sparse direct solver design and evaluation in SuperLU_DIST](https://dl.acm.org/doi/abs/10.1177/10943420241268200)

Relevant content:

- The 2014 SuperLU_DIST GPU paper already identified the problem: small BLAS operations dominate
  enough of sparse LU that aggregation and scheduling are needed to use GPUs well.
- Recent SuperLU_DIST adds multi-GPU support and mixed-precision routines with single-precision LU
  plus double-precision iterative refinement.
- The batched sparse-direct paper factors by elimination-tree levels, uses batched dense operations,
  and adds a batched Scatter GPU kernel.

Local implication:

- Multi-batch helps only when it creates enough independent work at a level. It does not erase
  parent assembly, dependency depth, scatter, or tiny-front metadata costs.
- `extend_add`/scatter timing should be treated as a first-class research target. But the local
  bucket-skip measurements show it is not a hidden 8387 Tensor Core enabler by itself.

## 4. Low-Fill Sparse-LU Alternatives for Circuit/Power-Grid Regimes

### KLU and circuit sparse matrices

Sources:

- [KLU ACM TOMS paper](https://dl.acm.org/doi/10.1145/1824801.1824814)
- [Sparse Matrix Methods for Circuit Simulation Problems](https://link.springer.com/chapter/10.1007/978-3-642-22453-9_1)

Relevant content:

- Circuit matrices are often extremely sparse even after factorization and can decompose into many
  small BTF blocks.
- The Springer chapter states the direct contrast plainly: dense-submatrix supernodal and
  multifrontal methods are not effective for extreme sparsity, while KLU exploits BTF and
  Gilbert-Peierls sparse left-looking LU.

Local implication:

- 8387/13K look closer to this low-fill family than to a dense-front multifrontal family.
- A low-fill sparse-LU branch may improve absolute time, but it weakens the specific "Tensor Core
  is the enabler" claim for those cases.

### Basker, GLU3.0, PanguLU, Caracal

Sources:

- [Basker: threaded sparse LU](https://arxiv.org/abs/1601.05725)
- [GLU3.0: GPU sparse LU for circuit simulation](https://arxiv.org/abs/1908.00204)
- [PanguLU SC23 proceedings](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap248.html)
- [Caracal artifact](https://zenodo.org/records/16778900)
- [Caracal ACM paper](https://dl.acm.org/doi/10.1145/3712285.3759792)

Relevant content:

- Basker targets circuit and power-grid matrices using hierarchical sparse LU data layouts.
- GLU3.0 frames GPU sparse LU as a dependency and irregular-memory problem, not primarily a dense
  GEMM problem.
- PanguLU explicitly rejects classic multifrontal/supernodal dense aggregation for its setting. It
  uses regular 2D sparse blocks, block-wise sparse BLAS, mapping changes for balance, and
  synchronization reduction.
- Caracal continues the GPU-resident sparse-LU direction with lightweight fine-grained scheduling.

Local implication:

- If 8387/13K are mandatory benchmark targets, the credible alternative is not another per-front
  TC threshold. It is either stronger panelization that changes the symbolic shape or a separate
  low-fill sparse-LU branch.

## 5. Tensor Core Mechanics and Grouped Dense Kernels

Sources:

- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA A100 architecture whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS grouped GEMM blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [CUTLASS grouped scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)
- [Fast batched matrix multiplication for small sizes using half precision on GPUs](https://www.netlib.org/utk/people/JackDongarra/PAPERS/ipdps-batched-2019.pdf)
- [Performance, design, and autotuning of batched GEMM for GPUs](https://www.netlib.org/utk/people/JackDongarra/PAPERS/performance-design-and-autotuning.pdf)

Relevant content:

- TF32 MMA has fixed warp-level tile shapes; our sm_80-style path uses shapes like `m16n8k8`.
- Small batched GEMM papers show that small matrix TC wins require specialized kernels and careful
  amortization.
- cuBLAS grouped GEMM groups variable shapes, transpositions, and scaling factors into one launch.
- CUTLASS grouped kernels are persistent kernels over multiple problems; scheduling overhead and
  load balance matter more as problem intensity drops.

Local implication:

- Dominant 8387 tiny shapes such as `(fsz,nc,uc)=(6,2,4),(4,2,2),(5,2,3)` are below useful TC
  granularity. The issue is hardware shape mismatch, not a missing switch.
- Grouped GEMM remains plausible only for a narrow region: independent 49..128 and big-high
  trailing updates where pointer setup can be amortized and output accumulation is simple.

## 6. Tensor Cores for Sparse-Derived Kernels

Sources:

- [cuTeSpMM](https://arxiv.org/abs/2504.06443)
- [tSparse / Tensor Core SpGEMM](https://arxiv.org/abs/2009.14600)
- [NVIDIA block sparse Tensor Core blog](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)
- [Spatula sparse factorization accelerator](https://dl.acm.org/doi/10.1145/3613424.3623783)

Relevant content:

- cuTeSpMM introduces a TCU-synergy concept: Tensor Cores help sparse-derived kernels only when
  the sparse structure has enough modeled operational intensity and useful payload.
- tSparse and block-sparse TC approaches succeed by operating on tile/block structure, not arbitrary
  tiny sparse fragments.
- Spatula is a useful negative control: sparse factorization combines structured math with
  dependency-heavy scheduling and irregular reuse; specialized hardware is proposed because generic
  GPUs leave utilization on the table.

Local implication:

- We should require a local "TC synergy" gate before adding TC kernels. Candidate metrics:
  `useful_mma_flop / padded_mma_flop`, `nc`, `uc`, number of same-shape fronts per launch,
  independent output count, and parent accumulation complexity.
- Current 8387/13K metrics do not pass that gate.

## 7. Mixed Precision and Accuracy

Sources:

- [Using mixed precision for sparse matrix computations](https://graal.ens-lyon.fr/~abuttari/mypapers/paper_mumps_toms.pdf)
- [Harnessing GPU Tensor Cores for mixed-precision iterative refinement](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf)
- [Mixed-precision iterative refinement using tensor cores on GPUs](https://royalsocietypublishing.org/rspa/article/476/2243/20200110/81022/Mixed-precision-iterative-refinement-using-tensor)
- [Recovering single precision accuracy from Tensor Cores](https://arxiv.org/abs/2203.03341)

Relevant content:

- Mixed-precision sparse/direct literature usually evaluates a lower-precision factorization inside
  an iterative refinement or correction loop.
- The sparse mixed-precision work explicitly applies the idea to multifrontal/supernodal direct
  methods and sparse iterative methods.
- Tensor Core iterative refinement literature uses low precision for the expensive factor/solve
  work and higher precision for residuals/corrections.
- Corrected Tensor Core GEMM can recover FP32-like accuracy but adds overhead that is likely only
  worth it for larger dense updates.

Local implication:

- Large-case TF32 residuals around `O(5e-2)` should be judged at NR/refinement level, not just raw
  factor residual.
- A separate experiment should measure `TF32 factor + correction/refinement` wall time and final
  convergence versus FP32/FP64.

## 8. Power-Flow and ACOPF Application Literature

Sources:

- [Linear solvers for power-grid optimization problems](https://www.pnnl.gov/publications/linear-solvers-power-grid-optimization-problems-review-gpu-accelerated-linear-solvers)
- [Towards Efficient ACOPF on GPUs](https://arxiv.org/abs/2302.08656)
- [GPU-resident sparse direct linear solvers for ACOPF](https://arxiv.org/abs/2306.14337)
- [Accelerating Optimal Power Flow with GPUs](https://arxiv.org/abs/2307.16830)
- [SABLE: GPU-Based Power Flow Accelerator for Sparsity-Aware Batched Learning](https://arxiv.org/abs/2606.07099)
- [NREL OPF-derived sparse linear solver benchmarks](https://www.nrel.gov/docs/fy24osti/86931.pdf)

Relevant content:

- The PNNL review found that tested GPU-capable direct solvers did not deliver significant GPU
  acceleration on those power-grid optimization matrices, motivating custom and GPU-resident work.
- ACOPF GPU papers identify the sparse linear solver as a dominant component and use 25K/70K-scale
  synthetic U.S. grids as relevant benchmarks.
- SABLE is the most directly aligned recent paper: batched PF, fixed sparse templates, zero-copy
  interoperability, custom GPU kernels, cuDSS sparse-direct LU, and mixed precision.

Local implication:

- B64/B256 repeated fixed-pattern factorization is the right application abstraction.
- 25K/70K/USA are defensible benchmark anchors.
- The application-level direction should emphasize reusable sparse templates and batched numerical
  phases. The Tensor Core claim is strongest for cases where the factorization structure exposes
  enough mid/big dense trailing work.

## Decision Map

| Direction | External support | Local evidence | Status |
|---|---|---|---|
| Stable mid/big TF32 for 25K/70K/USA | Strong: cuDSS batching/reuse, STRUMPACK dense-front GPU, ACOPF GPU papers. | repeat=61 passes: 25K, 70K, USA all `>=1.22x` at B64/B256 under stable policy. | Keep canonical. |
| More per-front tiny TC for 8387/13K | Weak: TC/sparse-TC papers require synergy and amortization. | tiny shapes and packed-TC estimate fail; low-mid force-all and local toggles fail paired target. | Stop. |
| Parent/assembly redesign | Strong: multifrontal and SuperLU_DIST treat scatter/assembly as first-class. | bucket skip does not close 8387 gap; large cases have more large update mass. | Profile and target absolute time, not TC ratio. |
| Grouped GEMM for mid/big independent trailing | Moderate: cuBLAS/CUTLASS support heterogeneous grouped GEMM. | Prior broad grouped attempt failed; shape-specific route still plausible. | Revisit narrowly only. |
| Stronger panelization/tree change | Strong in theory: fronts/tree decide dense work. | sibling/chain safe amalgamation too weak so far. | Only continue with a measurable `nc/uc` shift target. |
| Low-fill sparse-LU branch for 8387/13K | Strong for circuit/power-grid low-fill regimes. | 8387/13K look structurally low-fill. | Good absolute-time path; not a TC-enabler path. |
| TF32 factor + refinement/NR metric | Strong mixed-precision precedent. | Large-case factor speed exists; residual needs end-to-end validation. | Add as separate accuracy/performance track. |

## Practical Reading Order

1. cuDSS docs and advanced features: API design, batching, phase reuse, hybrid mode.
2. Liu 1992 plus STRUMPACK/SuperLU_DIST: multifrontal tree, dense fronts, scatter/assembly.
3. KLU/Basker/GLU/PanguLU/Caracal: low-fill sparse-LU alternative for 8387/13K.
4. PTX/cuBLAS/CUTLASS/small batched GEMM: hard TC tile constraints and grouped scheduling cost.
5. cuTeSpMM/tSparse/Spatula: sparse-derived TC synergy as a go/no-go metric.
6. Power-flow/ACOPF/SABLE: why batch/reuse/mixed precision matter at the application level.

## Bottom Line

The broader references make the current direction more conservative, not more speculative.

- Keep the accepted large-case Tensor Core story for 25K/70K/USA.
- Do not spend more time on per-front tiny Tensor Core kernels for 8387/13K without a new symbolic
  structure that materially increases useful `nc`/`uc`.
- Treat non-GEMM work as important for absolute performance, but not as a proven way to make
  Tensor Cores the speedup enabler on low-fill cases.
- Add an application-level mixed-precision/refinement metric before making accuracy claims from raw
  TF32 factor residuals.
