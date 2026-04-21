# 2026-04-16 Jacobian Vertex/Edge Metrics

This experiment measures two structural reasons why cuPF Jacobian assembly can
behave differently in vertex-based and edge-based kernels.

## Metrics

- Load balance: extract every `row_ptr[i+1] - row_ptr[i]` and report `max/mean`,
  `p95/mean`, and exact histograms.
- Atomic collision pressure: count how many edge-kernel writes land on each
  reduced-Jacobian target index.
- Diagonal target fan-in: report the same fan-in statistics specifically for
  `diagJ**[i]` targets.

The main load-balance metric is `ybus_active_pvpq_rows`, because the cuPF vertex
Jacobian kernel launches one warp per active `pvpq` bus and each warp walks that
bus's Ybus row. The script also emits `ybus_all_rows` and `jacobian_csr_rows`
for reference.

## Run

```bash
python3 exp/20260416/jac_vertex_edge/scripts/analyze_jac_vertex_edge.py
```

Useful options:

```bash
python3 exp/20260416/jac_vertex_edge/scripts/analyze_jac_vertex_edge.py \
  --dataset-root exp/20260414/amgx/cupf_dumps \
  --output-dir exp/20260416/jac_vertex_edge/results \
  --case case_ACTIVSg500
```

## Outputs

- `results/case_dimensions.csv`: case size and reconstructed Jacobian size.
- `results/load_balance_summary.csv`: row length `max/mean`, `p95/mean`, and percentiles.
- `results/row_lengths.csv`: all extracted row lengths.
- `results/row_length_histogram.csv`: exact row-length histograms.
- `results/atomic_fanin_summary.csv`: target fan-in summaries.
- `results/target_fanin.csv`: all edge-kernel atomic writes counted by target index.
- `results/diag_target_fanin.csv`: diagonal target fan-in counts.
- `results/fanin_histogram.csv`: exact fan-in histograms.
- `results/SUMMARY.md`: compact table for the selected cases.

## Operator Speed And NCU

The focused cases are listed in `cases_ncu_speed.txt`:

```text
MemphisCase2026_Mar7
Texas7k_20220923
Base_West_Interconnect_121GW
case_ACTIVSg25k
Base_Eastern_Interconnect_515GW
```

Build and run a Jacobian-operator-only speed probe:

```bash
python3 exp/20260416/jac_vertex_edge/scripts/run_operator_speed.py \
  --cuda-visible-devices 0 \
  --warmup 5 \
  --repeats 30
```

This writes:

- `results/operator_speed/<run>/operator_speed_raw.csv`
- `results/operator_speed/<run>/operator_speed_summary.csv`
- `results/operator_speed/<run>/operator_speed_comparison.csv`
- `results/operator_speed/<run>/SUMMARY.md`

`operator_speed_comparison.csv` and `SUMMARY.md` report edge/vertex operator
times in microseconds, edge speedup over vertex, estimated atomic time share
from the `edge_atomic` vs `edge_noatomic` delta, and the measured vertex/edge
lane utilization for the focused cases.

The checked run in this folder is:

- `results/operator_speed/requested_cases_r30/SUMMARY.md`
- `results/operator_speed/requested_cases_r30/operator_speed_summary.csv`
- `results/operator_speed/requested_cases_r30/operator_speed_comparison.csv`

The probe modes are:

- `edge_atomic`: current cuPF mixed edge kernel using `atomicAdd`.
- `edge_noatomic`: same edge computation but plain load/add/store instead of `atomicAdd`.
- `vertex`: current cuPF mixed vertex kernel.

`edge_noatomic` is a performance probe only. It intentionally has data races on
diagonal fan-in targets and is not a numerically valid solver path.

Generate Nsight Compute reports for a selected Jacobian kernel:

```bash
python3 exp/20260416/jac_vertex_edge/scripts/run_vertex_ncu.py \
  --cuda-visible-devices 0 \
  --run-name vertex_full \
  --ncu-set full
```

For the current edge atomic kernel:

```bash
python3 exp/20260416/jac_vertex_edge/scripts/run_vertex_ncu.py \
  --mode edge_atomic \
  --cuda-visible-devices 0 \
  --run-name edge_atomic_full \
  --ncu-set full
```

Use `--dry-run` to only write `ncu_<mode>_commands.sh` without launching `ncu`.
The default kernel filters are:

- `vertex`: `regex:update_jacobian_vertex_fp32_kernel`
- `edge_atomic`: `regex:update_jacobian_edge_fp32_kernel`
- `edge_noatomic`: `regex:update_jacobian_edge_noatomic_fp32_kernel`

The script writes `*.ncu-rep` files under `results/ncu_<mode>/<run>/<case>/`.

The generated command files for the five focused cases are:

- `results/ncu_vertex/requested_cases_vertex_full/ncu_vertex_commands.sh`
- `results/ncu_edge_atomic/requested_cases_edge_atomic_full/ncu_edge_atomic_commands.sh`
