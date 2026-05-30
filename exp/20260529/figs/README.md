# 2026-05-29 Figures

- Linear solver source: `lin_solver/comparison.csv`
- cuPF source: `cupf/comparison.csv`
- Cases: 78 linear-system rows, 78 cuPF rows
- X-axis uses log scale because case sizes span several orders of magnitude.
- Time figures use mean of 10 independent process runs per case/backend.

## Linear Solver Files

- `lin_solver_phase_time_vs_n.png/pdf`: analyze, factorize, solve vs Jacobian dimension
- `lin_solver_total_time_vs_n.png/pdf`: analyze + factorize + solve vs Jacobian dimension
- `lin_solver_speedup_vs_n.png/pdf`: cuDSS/custom speedup vs Jacobian dimension
- `lin_solver_phase_time_vs_nnz.png/pdf`: same phase comparison vs nonzeros
- `lin_solver_speedup_vs_nnz.png/pdf`: speedup vs nonzeros

## cuPF Files

- `cupf_time_vs_n_bus.png`: initialize, solve, total runtime vs bus count

## Linear Solver Median Speedup

- Analyze: 8.81x
- Factorize: 158.37x
- Solve: 44.80x
- Total: 17.40x

## cuPF Newton Iterations

- Source: `cupf/raw_runs.csv` (`iterations` per measured run)
- Identical cuDSS/custom per-run iteration sequences: 78 / 78 cases
- Case distribution: 2 iter: 2, 3 iter: 7, 4 iter: 33, 5 iter: 22, 6 iter: 7, 7 iter: 7
