# Linear Solver Measurement Audit

This workspace audits the previously generated GPU sparse linear solver benchmark
without modifying production cuPF code or replacing the v1/v2 reports.

The audit checks:

- whether each wrapper solved the intended system orientation and sign convention,
- whether reported timing phases are comparable or monolithic,
- whether near-zero Newton residual systems were misclassified by relative residual
  alone,
- whether the iterative solver configurations were only fixed-configuration evidence
  or reasonable best-effort evidence,
- whether the SuperLU_DIST `Invalid ISPEC` failure was a wrapper/runtime
  configuration issue.

Reproduction entry point:

```bash
cd /workspace/gpu-powerflow
python3 exp/20260510/lin_sol/measurement_audit/scripts/run_measurement_audit.py
```

The script writes CSVs, audit raw JSON, SuperLU_DIST diagnostics, and the v3 report
under `exp/20260510/lin_sol/measurement_audit/` and
`exp/20260510/lin_sol/report/linear_solver_measurement_audit_v3.md`.
