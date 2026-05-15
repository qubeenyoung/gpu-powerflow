# Jacobian Assembly MATPOWER Lane And Coalescing Metrics

This experiment is copied from `exp/20260420/jac_asm` and retargeted to
`datasets/matpower8.1/cupf_all_dumps`.

Run the full timing and NCU lane-utilization measurement with:

```bash
exp/20260423/jac_asm/scripts/run_matpower_lane_occ.sh
```

The script writes:

- `results/timing_matpower_all.csv`
- `results/ncu_matpower_all_lane_occ.ncu-rep`
- `results/ncu_matpower_all_lane_occ.csv`
- `results/ncu_matpower_all_lane_occ.txt`
- `results/ncu_matpower_all_lane_occ_summary.csv`
- `results/jac_asm_matpower_summary.md`

The lane utilization definition matches the 20260420 experiment:

- `lane util % = smsp__thread_inst_executed_per_inst_executed.pct`
- `avg lanes = smsp__thread_inst_executed_per_inst_executed.ratio`

The coalescing proxy metrics are:

- `global load sectors/request`, from L1TEX global-load sectors over requests
- `global write sectors/request`, combining global store, atomic, and reduction operations
- `sector util % = smsp__sass_average_data_bytes_per_sector / 32 * 100`
