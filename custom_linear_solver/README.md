# custom_linear_solver

`custom_linear_solver` is the cuDSS-like wrapper layer planned for the copied
CUDA multifrontal solver code.

The copied operation code is flattened directly under `src`: `plan`,
`factorize`, `solve`, `symbolic`, `reordering`, and `matrix`. It is a small subset of
`external/lin_solver`: only the GPU multifrontal kernels, symbolic analysis,
METIS nested-dissection ordering, and the minimal matrix/API state are kept
here. Benchmark drivers, matching experiments, GPU-ND experiments, and
third-party solver adapters are not part of this module.

The public API is planned around the same phase model as cuDSS:

```text
set_data      upload/register matrix descriptors
set_rhs       register RHS vector
set_solution  register output vector
analyze       one-time symbolic analysis for a fixed sparsity pattern
factorize     numeric factorization for current values
solve         solve current RHS into the output vector
get_data      inspect currently registered descriptors
```

Dataset/script flow:

```bash
python3 -m python.converters.convert_m_to_mat \
  --input-root /datasets/matpower \
  --output-root /datasets/matpower_mat

python3 -m python.converters.convert_mat_to_cupf_input \
  --input-root /datasets/matpower_mat \
  --output-root /datasets/matpower_cupf

python3 -m python.converters.convert_mat_to_linear_system \
  --input-root /datasets/matpower_mat \
  --output-root /datasets/matpower_linear_systems

cmake -S custom_linear_solver -B build/custom_linear_solver \
  -DCLS_BUILD_CUDA_OPS=ON \
  -DCLS_BUILD_SCRIPTS=ON
cmake --build build/custom_linear_solver -j

build/custom_linear_solver/custom_linear_solver_run \
  /datasets/matpower_linear_systems/case30 \
  --solution-out /tmp/case30_cls_solution.mtx
```

See `docs/api-and-build-design.md` for the design and source inventory, and
`docs/analyze-bottleneck-and-optimization.md` for the current analyze
bottleneck/optimization notes.
