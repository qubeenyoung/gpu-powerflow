# cuPF Implementation Audit Workspace

This directory contains the implementation audit for annual-report writing.

Scope:

- Inspect current cuPF implementation beyond the linear solver.
- Map Newton-Raphson operations to Python, C++, CUDA, cuDSS, or mixed paths.
- Record source evidence, implementation details, and missing evidence.
- Keep all generated audit artifacts under this directory only.

Constraints:

- Production cuPF source code is read-only for this audit.
- No invasive timers are added to production code.
- Any helper scripts or generated summaries must remain under this directory.

Required outputs:

- `results/source_map.csv`
- `results/operator_summary.csv`
- `results/research_progress_summary.csv`
- `results/benchmark_summary.csv`
- `report/cupf_implementation_audit.md`

Repository notes:

- Audit workspace root: `/workspace/gpu-powerflow`
- cuPF source tree inspected: `/workspace/gpu-powerflow-master/cuPF`
- The `/workspace/gpu-powerflow` tree contains datasets and experiment outputs used by benchmark commands.
- The source tree had pre-existing uncommitted changes before this audit. This audit treats the working tree as the current cuPF implementation and does not modify it.

Environment observed during the audit:

- GPU: NVIDIA GeForce RTX 3090, driver 580.126.09, 24576 MiB
- CUDA compiler: nvcc 12.8.93
- CMake: 3.22.1
- cuDSS library path used for smoke tests: `/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss.so`
- Benchmark binary used: `/workspace/gpu-powerflow-master/cuPF/build/bench-operators/benchmarks/cupf_case_benchmark`

Benchmark scope:

- Only small smoke tests were run.
- Tested profile: `cuda_mixed_edge`
- Tested cases: `case14` with batch size 1 and 4, and `case118` with batch size 1
- No new production timers or benchmark harnesses were added.
