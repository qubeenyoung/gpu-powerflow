# GPU Block ILU(0) Phase Pilot

This is a GPU numeric phase pilot, not a production block ILU solver. It runs block ILU(0) factor/apply on GPU for the block-coloring order and records phase timings. The kernels are intentionally simple scalar CUDA kernels; no Tensor Core kernel is used here.

## Timing

| case | bs | blocks | block nnz | setup ms | factor ms | right+update ms | diag inv ms | factor remainder ms | apply ms | offdiag apply ms | diag apply ms | apply remainder ms | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|case2383wp|16|279|1905|66.881187439|112.254272461|7.11974399118|15.8026557192|89.3318727505|18.1866569519|6.35168002779|1.23897599266|10.5960009315|0|

## Work shares

| block size | mean factor offdiag work share | mean apply offdiag work share | mean factor/BJ work | mean apply/BJ work |
|---:|---:|---:|---:|---:|
|16|0.869872021456|0.853757294764|7.68474244498|6.83794790577|

## Interpretation

- GPU block ILU(0) exists here only as a phase-timing pilot.
- The implementation verifies that factor/apply can be moved to GPU, but it is not optimized.
- The measured pilot is still slow because it launches many small kernels/cuBLAS calls. The remainder columns mostly represent launch/scheduling gaps between the individually timed subphases.
- The symbolic work shares show that the mathematical work is dominated by dense off-diagonal block update/apply.
- Those update/apply phases are the Tensor Core target. The current kernels do not use Tensor Cores, so this report should be read as a bottleneck map and optimization target, not as the final accelerated result.
