# GPU Block ILU(0) Phase Pilot

This is a GPU numeric phase pilot, not a production block ILU solver. It runs block ILU(0) factor/apply on GPU for the block-coloring order and records phase timings. The kernels are intentionally simple scalar CUDA kernels; no Tensor Core kernel is used here.

## Timing

| case | bs | blocks | block nnz | factor ms | right+update ms | diag inv ms | factor unacct ms | apply ms | offdiag apply ms | diag apply ms | apply unacct ms | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|case6468rte|32|396|2944|61.4789123535|16.5832962268|28.1507524326|16.7448636941|33.1494407654|14.2674240111|2.2380160084|16.6440007458|0|

## Work shares

| block size | mean factor offdiag work share | mean apply offdiag work share | mean factor/BJ work | mean apply/BJ work |
|---:|---:|---:|---:|---:|
|32|0.895767552153|0.865623619646|9.5939414324|7.44178402011|

## Interpretation

- GPU block ILU(0) exists here only as a phase-timing pilot.
- The implementation verifies that factor/apply can be moved to GPU, but it is not optimized.
- The measured pilot is still slow because it launches many small kernels/cuBLAS calls. The `unacct` columns mostly represent launch/scheduling gaps between the individually timed subphases.
- The symbolic work shares show that the mathematical work is dominated by dense off-diagonal block update/apply.
- Those update/apply phases are the Tensor Core target. The current kernels do not use Tensor Cores, so this report should be read as a bottleneck map and optimization target, not as the final accelerated result.
