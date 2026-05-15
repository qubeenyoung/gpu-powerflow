# cuPF Native Backward Validation and Benchmark

This experiment validates cuPF pybind forward results and the native cuPF
implicit-adjoint backward path. The benchmark no longer labels the old
Python/SciPy adjoint helper as `cupf_pybind` backward.

## Build

```bash
CUPF_WITH_CUDA=ON python3 -m pip install -e ./cuPF
```

For the paper-grade PyTorch integration path, build with Torch support:

```bash
CUPF_WITH_CUDA=ON CUPF_WITH_TORCH=ON python3 -m pip install -e ./cuPF --force-reinstall
```

If CUDA import fails because the Python NVIDIA cuDSS/cuBLAS wheels are not on
the dynamic linker path:

```bash
export LD_LIBRARY_PATH=/root/.local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH}
```

For `CUPF_WITH_TORCH=ON`, import `torch` before `cupf` in ad-hoc Python
snippets if the environment has multiple NVIDIA wheel library sets. The
benchmark scripts already do this.

CPU-only smoke tests can be built with:

```bash
CUPF_WITH_CUDA=OFF python3 -m pip install -e ./cuPF
```

## Implementations

Forward CPU reference:

- `benchmarks/utils.py::cpu_newton_pf`
- PYPOWER-style Newton equations over dumped `Ybus`, `Sbus`, `V0`, `pv`, `pq`
- Jacobian assembled by the PYPOWER `dSbus_dV` formula and solved with
  `scipy.sparse.linalg.spsolve`

cuPF forward:

- `cupf.NewtonSolver.initialize(...)`
- `cupf.NewtonSolver.solve(...)`
- `cupf.NewtonSolver.solve_batch(...)`

cuPF native backward:

- `cupf.NewtonSolver.solve_adjoint(...)`
- cached path: `cupf.NewtonSolver.solve(..., solve_options.prepare_adjoint_cache=True)`
- C++ API: `NewtonSolver::solve_adjoint(...)`
- This solves `J^T lambda = dL/dx` inside cuPF, then projects load gradients.
- Python/SciPy is not used in the `cupf_native_adjoint` or `cupf_pybind`
  backward path. The old `implicit_load_gradients` function remains only as a
  `python_implicit_adjoint_reference` helper.
- Torch extension zero-copy API:
  `cupf.NewtonSolver.solve_with_adjoint_cache_torch(...)` and
  `cupf.NewtonSolver.solve_adjoint_torch(...)`
- These APIs accept `torch::Tensor` CUDA tensors directly in C++, validate
  dtype/shape/device/contiguity, use `tensor.data_ptr<T>()`, and run on the
  PyTorch current CUDA stream.
- Dynamic `load_p/load_q`, `grad_va/grad_vm`, and `grad_load_p/q` outputs stay
  in device memory. Output tensors are preallocated by the caller.
- The raw pointer API remains as `solve_adjoint_cuda_raw_unsafe(...)` for
  debug only. It is reported as `cupf_raw_pointer_unsafe` and is not a
  paper-grade performance path.

Torch baseline:

- `benchmark_cupf_vs_torch.py::torch_newton_pf`
- Dense torch tensor `Ybus`
- Torch tensor Jacobian assembly
- `torch.linalg.solve`

## Backward Math and Ordering

cuPF residual convention:

```text
F(x, Sbus) = S_calc(x) - Sbus
```

State vector:

```text
x = [Va[pv], Va[pq], Vm[pq]]
```

Residual rows:

```text
F = [Pmis[pv], Pmis[pq], Qmis[pq]]
```

Slack buses are excluded. PV buses have active-power rows only. PQ buses have
both active- and reactive-power rows.

For loss `L(x*)`:

```text
J = dF/dx at x*
J^T lambda = dL/dx
dL/dp = -lambda^T dF/dp
```

The checked parameters are demand variables `load_p` and `load_q`. Increasing
load decreases net injection `Sbus`, so with `F = S_calc - Sbus` the projected
gradients are:

```text
grad_load_p[pv/pq bus] = -lambda[P row]
grad_load_q[pq bus]    = -lambda[Q row]
```

This sign convention is written into `backward_summary.csv` as
`F=S_calc-S_spec; load increase decreases Sbus; grad_load=-lambda`.

The scalar validation loss is deterministic:

```text
loss = sum(w_vm * Vm) + sum(w_va * Va)
```

`w_vm` and `w_va` are arange/modulo-based deterministic weights from
`deterministic_weights(...)`.

## Transpose Solve Backend

Backward exactness requires `J(x*)`, the Jacobian at the final converged state.
The last Newton factorization is generally `J(x_k)`, not necessarily `J(x*)`,
so it must not be blindly reused. Therefore `prepare_adjoint_cache=True`
performs the exact final-state preparation at the end of forward:

- rebuilds the Jacobian at final `x*`
- factorizes `J(x*)` for the adjoint cache
- marks the cache as matching final state
- reports `reused_forward_factorization=False`
- later backward reports `used_adjoint_cache=True`
- later backward reports `reused_final_state_factorization=True`
- later backward reports `refactorized_for_backward=False`

Backends:

- CPU: KLU factorization of `J(x*)`, then `klu_tsolve` on the same
  factorization for `J^T lambda = grad_state`
- CUDA: cuDSS transpose solve mode is unavailable in the installed header.
  cuPF therefore builds the `J^T` CSR pattern and `J value -> J^T value`
  permutation during solver initialization. With
  `allow_explicit_transpose_fallback=True`, forward cache preparation fills
  final `J(x*)` values, launches a device kernel to populate `J^T` values, and
  factorizes that explicit transpose once. Cached backward then reuses that
  transpose factorization and does not transpose, host-copy, or refactorize.

The installed cuDSS header marks `CUDSS_CONFIG_SOLVE_MODE` as not supported for
transpose/conjugate-transpose solves. CUDA rows therefore set
`used_explicit_transpose=True`. This fallback is allowed only when
`allow_explicit_transpose_fallback=True`.

## Timing Semantics

Runtime output includes `timing_scope`:

- `forward_only`: only forward solve
- `forward_with_adjoint_cache`: forward solve plus final-state `J(x*)`
  adjoint cache factorization
- `backward_only_cached`: cache is prepared before timing; timed work is
  `J^T` solve plus load-gradient projection only
- `forward_plus_backward_cached`: forward with adjoint cache plus cached
  backward solve

CUDA timings use `torch.cuda.Event` with `torch.cuda.synchronize()` around each
measured repeat. CPU timings use `time.perf_counter()`.

Python itself is not the issue for timing; crossing the pybind boundary through
NumPy/list/std::vector host containers is. Runtime rows are split:

- `cupf_pybind_numpy`: NumPy host path, `includes_host_device_transfer=True`
- `cupf_torch_extension_zero_copy`: paper-grade Torch C++ extension path,
  `torch_extension_zero_copy=True`, `raw_pointer_api_used=False`,
  `current_stream_integrated=True`
- `cupf_raw_pointer_unsafe`: debug-only raw `uintptr_t` path,
  `raw_pointer_api_used=True`

The paper-grade cuPF PyTorch runtime number is
`cupf_torch_extension_zero_copy`. Do not use `cupf_pybind_numpy` or
`cupf_raw_pointer_unsafe` as final paper performance numbers.

The Torch extension uses `c10::cuda::getCurrentCUDAStream()` and sets the cuPF
active stream before launching cuPF kernels or cuDSS phases. cuDSS stream
integration uses `cudssSetStream`. CUDA timing uses `torch.cuda.Event` on the
same current stream.

For `cupf_torch_extension_zero_copy/backward_only_cached`, expected metadata:

```text
torch_extension_zero_copy=True
raw_pointer_api_used=False
current_stream_integrated=True
used_adjoint_cache=True
jt_symbolic_analyzed_at_initialize=True
jt_values_transposed_on_device=True
jt_factorized_during_forward_cache=True
jt_refactorized_during_backward=False
host_roundtrip_for_jt_transpose=False
includes_host_device_transfer=False
used_python_scipy=False
```

If a future fallback is added, it must not be reported as `cupf_pybind`
backward. Use names such as `python_implicit_adjoint_reference`,
`cupf_forward_plus_python_adjoint_reference`, or
`fallback_cpu_native`.

## Validation

Full 78-case validation:

```bash
python3 exp/20260511/benchmarks/validate_cupf_forward_backward.py \
  --dataset-dir datasets/matpower8.1/cupf_all_dumps \
  --num-datasets 78 \
  --device cuda \
  --cupf-compute fp64 \
  --fd-params load_p load_q \
  --fd-h 1e-4 \
  --fd-samples 128 \
  --output-dir results/cupf_validation
```

Smoke:

```bash
python3 exp/20260511/benchmarks/validate_cupf_forward_backward.py \
  --dataset-dir datasets/matpower8.1/cupf_all_dumps \
  --cases case10ba \
  --device cuda \
  --cupf-compute fp64 \
  --fd-samples 2 \
  --output-dir /tmp/cupf_validation_native_smoke
```

Outputs:

```text
forward_summary.csv
backward_summary.csv
summary.json
summary.md
```

Forward aggregate relative errors include only cases where both CPU and cuPF
converged. Convergence counts are reported separately.

## Runtime

Full benchmark:

```bash
python3 exp/20260511/benchmarks/benchmark_cupf_vs_torch.py \
  --dataset-dir datasets/matpower8.1/cupf_all_dumps \
  --batch-sizes 1 64 256 \
  --warmup 20 \
  --repeats 100 \
  --device cuda \
  --cupf-compute mixed \
  --output-dir results/cupf_benchmark
```

Smoke:

```bash
python3 exp/20260511/benchmarks/benchmark_cupf_vs_torch.py \
  --dataset-dir datasets/matpower8.1/cupf_all_dumps \
  --case-name case10ba \
  --batch-sizes 1 \
  --warmup 1 \
  --repeats 3 \
  --device cuda \
  --cupf-compute mixed \
  --output-dir /tmp/cupf_benchmark_native_smoke
```

Outputs:

```text
runtime_summary.csv
runtime_summary.json
runtime_summary.md
```
