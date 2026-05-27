# GPU Block ILU(0) Phase Pilot

This is a GPU numeric phase pilot, not a production block ILU solver. It runs block ILU(0) factor/apply on GPU for the block-coloring order and records phase timings. The kernels are intentionally simple scalar CUDA kernels; no Tensor Core kernel is used here.

## Timing

| case | bs | blocks | block nnz | factor ms | right+update ms | diag inv ms | factor unacct ms | apply ms | offdiag apply ms | diag apply ms | apply unacct ms | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|case2383wp|16|279|1905|31.8402557373|6.13776004175|15.2073917501|10.4951039455|17.6080951691|6.09305603849|1.02992000803|10.4851191225|0|
|case2383wp|32|139|973|18.6204166412|4.38249593973|8.87027214095|5.36764856055|9.96928024292|3.89187195338|0.665695989039|5.4117123005|0|
|case3120sp|16|376|2438|40.0506896973|7.82131204847|18.9042236805|13.3251539683|22.5631999969|7.73334404384|1.4124480125|13.4174079406|0|
|case3120sp|32|188|1284|24.4593925476|5.99273594655|11.3536322117|7.11302438937|13.1946239471|5.19686393207|0.910047985613|7.08771202946|0|
|case9241pegase|16|1077|8381|133.438339233|30.6807040537|54.725087177|48.0325480027|78.5383682251|28.2865601254|4.07750402112|46.1743040786|0|
|case9241pegase|32|534|4080|76.3658218384|21.6070401249|32.2177606747|22.5410210388|42.7347183228|17.7934397114|2.69811195368|22.2431666576|0|
|case13659pegase|16|1467|12159|184.626205444|43.9542399498|73.7185908519|66.9533746426|114.137153625|41.8013121735|5.63241602527|66.7034254267|0|
|case13659pegase|32|730|5894|108.444671631|31.9325122545|44.069216866|32.4429425104|61.5817298889|25.885215567|3.67795193358|32.0185623884|0|
|case6468rte|16|794|6110|94.7568664551|20.942112061|39.915487349|33.8992670451|57.3438720703|20.4042880952|2.97062402102|33.9689599541|0|
|case6468rte|32|396|2944|55.0387840271|14.7507839962|23.8893765099|16.398623521|30.9346885681|12.7554557854|2.00748796621|16.1717448165|0|

## Work shares

| block size | mean factor offdiag work share | mean apply offdiag work share | mean factor/BJ work | mean apply/BJ work |
|---:|---:|---:|---:|---:|
|16|0.89158957355|0.863745525321|9.84371107888|7.39522997889|
|32|0.890417717095|0.864276356289|9.4510880256|7.39341953355|

## Interpretation

- GPU block ILU(0) exists here only as a phase-timing pilot.
- The implementation verifies that factor/apply can be moved to GPU, but it is not optimized.
- The measured pilot is still slow because it launches many small kernels/cuBLAS calls. The `unacct` columns mostly represent launch/scheduling gaps between the individually timed subphases.
- The symbolic work shares show that the mathematical work is dominated by dense off-diagonal block update/apply.
- Those update/apply phases are the Tensor Core target. The current kernels do not use Tensor Cores, so this report should be read as a bottleneck map and optimization target, not as the final accelerated result.
