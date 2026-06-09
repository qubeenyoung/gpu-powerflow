# Tutorial C++ Pandapower Reference B=1

- Executable source: `gpu-powerflow/python/tutorial/cpp/pandapower_jacobian_reference.cpp`.
- Cases: `case3012wp`, `case6468rte`, `case8387pegase`, `case_ACTIVSg25k`, `case_SyntheticUSA`.
- Run: `B=1`, tolerance `1e-8`, max iteration `10`, warmup `1`, repeats `5`.
- Reported values are median over repeats.
- All requested cases fit and converged in this run.

## Init And Solve

| case | init ms | solve ms | iterations | max mismatch |
|---|---:|---:|---:|---:|
| `case3012wp` | 5.272 | 18.227 | 4 | 8.755e-12 |
| `case6468rte` | 11.892 | 38.964 | 4 | 7.306e-12 |
| `case8387pegase` | 17.663 | 57.330 | 4 | 1.807e-11 |
| `case_ACTIVSg25k` | 47.932 | 243.788 | 5 | 9.108e-11 |
| `case_SyntheticUSA` | 225.664 | 1771.593 | 7 | 7.221e-11 |

## Dominant Solve Operators

| case | solve total | jacobian | factorize | triangular solve | ibus | mismatch+norm | voltage update |
|---|---:|---:|---:|---:|---:|---:|---:|
| `case3012wp` | 18.227 | 12.714 | 4.408 | 0.220 | 0.097 | 0.076 | 0.565 |
| `case6468rte` | 38.964 | 26.623 | 10.292 | 0.484 | 0.245 | 0.166 | 1.225 |
| `case8387pegase` | 57.330 | 39.560 | 14.727 | 0.675 | 0.366 | 0.224 | 1.569 |
| `case_ACTIVSg25k` | 243.788 | 156.658 | 73.628 | 3.117 | 1.391 | 0.938 | 5.991 |
| `case_SyntheticUSA` | 1771.593 | 1172.170 | 531.231 | 24.844 | 6.702 | 4.512 | 29.658 |

## Files

- Init/solve CSV: `data/tutorial_cpp_reference_b1_init_solve.csv`
- Operator CSV: `data/tutorial_cpp_reference_b1_ops_ms.csv`
- Operator pie charts: `figures/tutorial_cpp_operator_pies/*_operator_pie.png`
- Raw run CSV: `tmp/tutorial_cpp_ref/raw_b1.csv`
