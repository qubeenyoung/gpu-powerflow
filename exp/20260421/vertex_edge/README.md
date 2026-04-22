# 20260421 Vertex/Edge Kernel Experiment

Standalone copy of the Jacobian edge/vertex fill kernels and voltage update code.

## Contents

- `src/edge/edge_fill_jac.*`: edge-based Jacobian fill kernels, including the batch entry point.
- `src/vertex/vertex_fill_jac.cu`: vertex-per-thread and vertex-per-warp Jacobian fill kernels.
- `src/common/*`: Ybus/Jacobian pattern data types and host-side Jacobian pattern builder.
- `src/update/update_voltage.*`: batched voltage initialization and Newton voltage update kernels.
- `benchmark_vertex_edge.cu`: standalone timing/check harness for the copied kernels.

## Build

```bash
cmake -S exp/20260421/vertex_edge -B exp/20260421/vertex_edge/build
cmake --build exp/20260421/vertex_edge/build -j
```

## Run

```bash
./exp/20260421/vertex_edge/build/vertex_edge_bench \
  --data datasets/texas_univ_cases/cuPF_datasets \
  --case case_ACTIVSg200 \
  --mode all \
  --warmup 10 \
  --iters 100 \
  --check-vertex
```

Supported `--mode` values include `edge`, `vertex`, `vertex_thread`, `vertex_warp`, `vertex_both`, `both`, and `all`.
