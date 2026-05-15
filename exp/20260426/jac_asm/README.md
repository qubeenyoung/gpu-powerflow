# Jacobian Assembly CSR/COO Preprocessing Split

This experiment keeps Jacobian symbolic analysis common and makes the
edge-only CSR-to-COO cost explicit.

- `src/common`: CSR Ybus types, Jacobian pattern build, and value-slot maps
  shared by vertex and edge fills.
- `src/vertex`: vertex fill kernels that consume `YbusCsr`.
- `src/edge`: edge fill kernels that consume `YbusCoo`, plus
  `buildEdgeYbusMap()` for the extra `row[k]` materialization.

The benchmark reports:

- `analyze_ms`: common Jacobian pattern/value-map construction.
- `edge_map_ms`: edge-only `row_ptr -> row[k]` materialization.
- `analyze_fused_edge_map_ms`: Jacobian construction with `row[k]`
  materialization fused into the value-map pass.
- `edge_fill_ms`, `vertex_fill_ms`: CUDA kernel fill times.

Build and run one case:

```bash
cmake -S exp/20260426/jac_asm -B exp/20260426/jac_asm/build
cmake --build exp/20260426/jac_asm/build
exp/20260426/jac_asm/build/jac_asm_bench --case case118 --mode both
```
