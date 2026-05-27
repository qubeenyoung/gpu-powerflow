# GPU Block ILU(0) Phase Pilot

This is a GPU numeric phase pilot, not a production block ILU solver. It runs block ILU(0) factor/apply on GPU for the block-coloring order and records phase timings. The kernels are intentionally simple scalar CUDA kernels; no Tensor Core kernel is used here.

This run used the fast path, so inner-loop phase timing is disabled. The `factor ms` and `apply ms` columns are the useful runtime numbers; subphase columns are zero by design. Re-run with `--enable-profile` for detailed phase attribution.

## Timing

| case | bs | blocks | block nnz | setup ms | factor ms | right+update ms | diag inv ms | factor remainder ms | apply ms | offdiag apply ms | diag apply ms | apply remainder ms | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|case2383wp|16|279|1905|57.3119049072|21.9258880615|0|0|21.9258880615|5.86892795563|0|0|5.86892795563|0|
|case2383wp|32|139|973|41.9424972534|27.2558078766|0|0|27.2558078766|5.54105615616|0|0|5.54105615616|0|
|case3120sp|16|376|2438|77.5346908569|26.9967365265|0|0|26.9967365265|6.78656005859|0|0|6.78656005859|0|
|case3120sp|32|188|1284|56.6302719116|28.3330554962|0|0|28.3330554962|5.74428796768|0|0|5.74428796768|0|
|case9241pegase|16|1077|8381|268.609527588|83.6638717651|0|0|83.6638717651|25.1102085114|0|0|25.1102085114|0|
|case9241pegase|32|534|4080|210.304000854|86.2822418213|0|0|86.2822418213|18.0749759674|0|0|18.0749759674|0|
|case13659pegase|16|1467|12159|394.515869141|117.969917297|0|0|117.969917297|36.5070381165|0|0|36.5070381165|0|
|case13659pegase|32|730|5894|294.376251221|123.064323425|0|0|123.064323425|24.0418243408|0|0|24.0418243408|0|
|case6468rte|16|794|6110|191.711227417|57.7628173828|0|0|57.7628173828|21.3791999817|0|0|21.3791999817|0|
|case6468rte|32|396|2944|140.737533569|61.5557136536|0|0|61.5557136536|13.1110715866|0|0|13.1110715866|0|

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
