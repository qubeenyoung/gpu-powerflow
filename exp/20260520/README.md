# GLU power-flow Jacobian experiment

This experiment wires the `exp/20260519` power-flow case loader to the GLU 3.0
sparse LU solver under `third_party/glu`.

The GLU/NICSLU sources needed for analyze, symbolic, numeric factorization, and
solve are built directly from `third_party/glu`; this experiment tree keeps only
the wrapper, benchmark, scripts, and reports.

The pipeline is intentionally split to expose the GLU phases:

1. Load `dump_Ybus.mtx`, `dump_V.txt`, `dump_Sbus.txt`, `dump_pv.txt`, and
   `dump_pq.txt`.
2. Build the Newton power-flow Jacobian and RHS using the same formulas as the
   20260519 cuDSS benchmark.
3. Convert CSR to CSC and call NICSLU `Analyze`.
4. Run GLU symbolic fill prediction, CSR transpose, value prediction, and
   level scheduling.
5. Run GLU GPU numeric factorization.
6. Solve the RHS with the GLU factors and report residuals.

Build:

```bash
cmake -S exp/20260520 -B exp/20260520/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build exp/20260520/build -j
```

Run:

```bash
exp/20260520/build/glu_pf_benchmark \
  --case-dir path/to/case_dir \
  --rhs-mode synthetic
```

Use `--perturb` to enable GLU's static pivot perturbation path, `--csv` for a
single CSV row, and `--print-glu-log` to print captured GLU stage logs.
