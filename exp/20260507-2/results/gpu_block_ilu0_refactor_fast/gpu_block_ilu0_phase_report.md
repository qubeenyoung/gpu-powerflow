# GPU Block ILU(0) Phase Pilot

This is a GPU numeric phase pilot, not a production block ILU solver. It runs block ILU(0) factor/apply on GPU for the block-coloring order and records phase timings. The kernels are intentionally simple scalar CUDA kernels; no Tensor Core kernel is used here.

This run used the fast path, so inner-loop phase timing is disabled. The `factor ms` and `apply ms` columns are the useful runtime numbers; subphase columns are zero by design. Re-run with `--enable-profile` for detailed phase attribution.

## Timing

| case | bs | blocks | block nnz | setup ms | factor ms | right+update ms | diag inv ms | factor remainder ms | apply ms | offdiag apply ms | diag apply ms | apply remainder ms | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|case2383wp|16|279|1905|56.2984008789|21.1415042877|0|0|21.1415042877|5.25267219543|0|0|5.25267219543|0|
|case2383wp|32|139|973|41.2497901917|21.3125114441|0|0|21.3125114441|3.75843191147|0|0|3.75843191147|0|
|case3120sp|16|376|2438|75.8374404907|26.7929592133|0|0|26.7929592133|6.73436784744|0|0|6.73436784744|0|
|case3120sp|32|188|1284|55.0082550049|28.0565757751|0|0|28.0565757751|5.03817605972|0|0|5.03817605972|0|
|case9241pegase|16|1077|8381|264.858795166|81.4932479858|0|0|81.4932479858|24.1327037811|0|0|24.1327037811|0|
|case9241pegase|32|534|4080|209.649658203|74.3096313477|0|0|74.3096313477|15.2629117966|0|0|15.2629117966|0|
|case13659pegase|16|1467|12159|392.79208374|98.0746231079|0|0|98.0746231079|30.828704834|0|0|30.828704834|0|
|case13659pegase|32|730|5894|294.458374023|104.895484924|0|0|104.895484924|21.9199047089|0|0|21.9199047089|0|
|case6468rte|16|794|6110|187.286270142|51.4887695312|0|0|51.4887695312|14.7467517853|0|0|14.7467517853|0|
|case6468rte|32|396|2944|138.868667603|53.762046814|0|0|53.762046814|10.9972801208|0|0|10.9972801208|0|

## Work shares

| block size | mean factor offdiag work share | mean apply offdiag work share | mean factor/BJ work | mean apply/BJ work |
|---:|---:|---:|---:|---:|
|16|0.89158957355|0.863745525321|9.84371107888|7.39522997889|
|32|0.890417717095|0.864276356289|9.4510880256|7.39341953355|

## Interpretation

- GPU block ILU(0) exists here only as a phase-timing pilot.
- The implementation verifies that factor/apply can be moved to GPU, but it is not optimized.
- Fast-path runs intentionally avoid inner-loop event synchronization, so phase columns are not populated.
- The symbolic work shares show that the mathematical work is dominated by dense off-diagonal block update/apply.
- Those update/apply phases are the Tensor Core target. The current kernels do not use Tensor Cores, so this report should be read as a bottleneck map and optimization target, not as the final accelerated result.
