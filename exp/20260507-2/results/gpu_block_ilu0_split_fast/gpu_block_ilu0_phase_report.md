# GPU Block ILU(0) Phase Pilot

This is a GPU numeric phase pilot, not a production block ILU solver. It runs block ILU(0) factor/apply on GPU for the block-coloring order and records phase timings. The kernels are intentionally simple scalar CUDA kernels; no Tensor Core kernel is used here.

This run used the fast path, so inner-loop phase timing is disabled. The `factor ms` and `apply ms` columns are the useful runtime numbers; subphase columns are zero by design. Re-run with `--enable-profile` for detailed phase attribution.

## Timing

| case | bs | blocks | block nnz | setup ms | factor ms | right+update ms | diag inv ms | factor remainder ms | apply ms | offdiag apply ms | diag apply ms | apply remainder ms | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|case2383wp|16|279|1905|56.4504013062|21.1649284363|0|0|21.1649284363|5.25897598267|0|0|5.25897598267|0|
|case2383wp|32|139|973|41.2618865967|21.2654075623|0|0|21.2654075623|3.78089594841|0|0|3.78089594841|0|
|case3120sp|16|376|2438|75.8915557861|26.8298244476|0|0|26.8298244476|6.71033620834|0|0|6.71033620834|0|
|case3120sp|32|188|1284|55.1421432495|27.9910392761|0|0|27.9910392761|5.0562877655|0|0|5.0562877655|0|
|case9241pegase|16|1077|8381|265.653259277|81.4919662476|0|0|81.4919662476|24.278591156|0|0|24.278591156|0|
|case9241pegase|32|534|4080|216.297439575|80.5212173462|0|0|80.5212173462|16.4583034515|0|0|16.4583034515|0|
|case13659pegase|16|1467|12159|406.028289795|98.6695709229|0|0|98.6695709229|31.0115833282|0|0|31.0115833282|0|
|case13659pegase|32|730|5894|298.551300049|105.485313416|0|0|105.485313416|21.9388809204|0|0|21.9388809204|0|
|case6468rte|16|794|6110|188.143707275|51.8451194763|0|0|51.8451194763|14.8726396561|0|0|14.8726396561|0|
|case6468rte|32|396|2944|139.446273804|54.0293121338|0|0|54.0293121338|11.0033922195|0|0|11.0033922195|0|

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
