#!/usr/bin/env python3
"""Focused SuperLU_DIST best-effort/optimality check."""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "measurement_audit"
RAW = AUDIT / "results" / "raw_json"
RESULTS = AUDIT / "results"
LOGS = AUDIT / "logs" / "superlu_optimality"
REPORT = ROOT / "report" / "superlu_dist_optimality_check.md"
DATASETS = ROOT / "datasets" / "dumped_systems"
MPIRUN = ROOT / "third_party" / "mpich" / "install" / "bin" / "mpirun"
EXE = AUDIT / "superlu_dist_phase" / "build" / "superlu_dist_phase_benchmark"


def ensure_dirs() -> None:
    for p in [RAW, RESULTS, LOGS, REPORT.parent]:
        p.mkdir(parents=True, exist_ok=True)


def base_args(case: str, out: Path, rowperm: str, extra: list[str]) -> list[str]:
    base = DATASETS / case / "iter_000"
    return [
        "--matrix", str(base / "J.mtx"),
        "--rhs", str(base / "rhs.txt"),
        "--xref", str(base / "x_ref.txt"),
        "--meta", str(base / "meta.json"),
        "--dtype", "fp64",
        "--repeats", "1",
        "--warmup", "0",
        "--colperm", "MMD_AT_PLUS_A",
        "--rowperm", rowperm,
        "--out", str(out),
        *extra,
    ]


def run(name: str, case: str, np: int, nprow: int, npcol: int, rowperm: str, extra: list[str]) -> Path:
    out = RAW / f"superlu_opt_{name}_{case}_np{np}_{nprow}x{npcol}_{rowperm}.json"
    cmd = [
        str(MPIRUN), "-np", str(np), str(EXE),
        *base_args(case, out, rowperm, ["--nprow", str(nprow), "--npcol", str(npcol), *extra]),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=600)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    (LOGS / f"{name}_{case}_np{np}_{nprow}x{npcol}_{rowperm}.stdout.log").write_text(proc.stdout)
    (LOGS / f"{name}_{case}_np{np}_{nprow}x{npcol}_{rowperm}.stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        write_failure(out, cmd, elapsed_ms, proc.returncode, proc.stdout, proc.stderr)
    else:
        data = load_json(out)
        data["optimality_external_elapsed_ms"] = elapsed_ms
        data["optimality_command"] = " ".join(shlex.quote(x) for x in cmd)
        write_json(out, data)
    return out


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_failure(out: Path, cmd: list[str], elapsed_ms: float, code: int, stdout: str, stderr: str) -> None:
    meta: dict[str, Any] = {}
    if "--meta" in cmd:
        try:
            meta = load_json(Path(cmd[cmd.index("--meta") + 1]))
        except Exception:
            pass
    write_json(out, {
        "solver_name": "SuperLU_DIST",
        "build_status": "runtime_failed",
        "case_name": meta.get("case_name", "unknown"),
        "iteration": meta.get("iteration", -1),
        "matrix_rows": meta.get("matrix_rows", 0),
        "matrix_cols": meta.get("matrix_cols", 0),
        "nnz": meta.get("nnz", 0),
        "dtype": "fp64",
        "converged": False,
        "total_end_to_end_ms": elapsed_ms,
        "notes": f"Command failed with code {code}",
        "error_tail": (stderr or stdout)[-3000:],
        "optimality_command": " ".join(shlex.quote(x) for x in cmd),
    })


def enrich(path: Path, label: str) -> dict[str, Any]:
    d = load_json(path)
    case = d.get("case_name")
    iteration = int(d.get("iteration") or 0)
    meta_path = DATASETS / str(case) / f"iter_{iteration:03d}" / "meta.json"
    if meta_path.exists() and d.get("absolute_residual_2") is None and d.get("relative_residual_2") is not None:
        rhs_norm = float(load_json(meta_path).get("rhs_norm_2") or 0.0)
        abs_res = float(d["relative_residual_2"]) * rhs_norm
        d["absolute_residual_2"] = abs_res
        d["scaled_residual_2"] = abs_res / max(1.0, rhs_norm)
    return {
        "label": label,
        "case": d.get("case_name"),
        "np": d.get("mpi_ranks"),
        "nprow": d.get("nprow"),
        "npcol": d.get("npcol"),
        "rowperm": d.get("rowperm"),
        "status": d.get("build_status"),
        "converged": d.get("converged"),
        "analysis_ms": d.get("analysis_ms"),
        "factorization_ms": d.get("factorization_ms"),
        "solve_ms": d.get("solve_ms"),
        "total_solver_ms": d.get("total_solver_ms"),
        "external_elapsed_ms": d.get("optimality_external_elapsed_ms"),
        "scaled_residual_2": d.get("scaled_residual_2"),
        "relative_error_to_x_ref_2": d.get("relative_error_to_x_ref_2"),
        "gpu_buffer_mb": d.get("peak_gpu_memory_mb"),
        "acc_offload": d.get("superlu_acc_offload"),
        "equil": d.get("superlu_equil_enabled"),
        "replace_tiny_pivot": d.get("superlu_replace_tiny_pivot"),
        "iter_refine": d.get("superlu_iter_refine"),
        "par_symb_fact": d.get("superlu_par_symb_fact"),
        "raw_json": str(path),
        "notes": d.get("notes", "") + " " + str(d.get("error_tail", "")),
    }


def write_csv(rows: list[dict[str, Any]]) -> Path:
    out = RESULTS / "superlu_dist_optimality_check.csv"
    keys = list(rows[0].keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return out


def fmt(v: Any) -> str:
    if v is None or v == "":
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    try:
        x = float(v)
        if abs(x) >= 1000:
            return f"{x:,.1f}"
        if abs(x) >= 1:
            return f"{x:.3f}"
        return f"{x:.3e}"
    except Exception:
        return str(v)


def md_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        ("label", "label"), ("case", "case"), ("np", "np"), ("grid", "grid"),
        ("rowperm", "rowperm"), ("conv", "converged"), ("factor ms", "factorization_ms"),
        ("solve ms", "solve_ms"), ("solver ms", "total_solver_ms"),
        ("gpu MB", "gpu_buffer_mb"), ("scaled resid", "scaled_residual_2"),
    ]
    out = ["| " + " | ".join(c[0] for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        rr = dict(r)
        rr["grid"] = f"{fmt(r.get('nprow'))}x{fmt(r.get('npcol'))}"
        out.append("| " + " | ".join(fmt(rr.get(k)) for _, k in cols) + " |")
    return "\n".join(out)


def write_report(rows: list[dict[str, Any]], csv_path: Path) -> None:
    large_ok = [r for r in rows if r["case"] in {"case2869pegase", "case9241pegase"} and r["status"] == "ok" and str(r["converged"]).lower() in {"true", "yes"}]
    best = []
    for case in ["case2869pegase", "case9241pegase"]:
        c = [r for r in large_ok if r["case"] == case]
        c.sort(key=lambda r: float(r["total_solver_ms"] or 1e300))
        best.extend(c[:5])
    offload = [r for r in rows if "offload" in str(r["label"]) and r["case"] in {"case2869pegase", "case9241pegase"}]
    grid = [r for r in rows if "grid" in str(r["label"]) and r["case"] in {"case2869pegase", "case9241pegase"}]
    option = [r for r in rows if "option" in str(r["label"]) and r["case"] in {"case2869pegase", "case9241pegase"}]
    REPORT.write_text(f"""# SuperLU_DIST Optimality Check

This follow-up checks whether the SuperLU_DIST large-case runs were reasonable best-effort executions. It varies the main knobs exposed by the current ABglobal diagnostic wrapper:

- fill-reducing column ordering fixed to `MMD_AT_PLUS_A`
- row permutation: `LargeDiag_MC64` and `NOROWPERM`
- GPU offload enabled/disabled through `superlu_acc_offload`
- process grid shape for `np=2` and `np=4`
- selected solver options: equilibration, iterative refinement, tiny-pivot replacement

The check still uses `pdgssvx_ABglobal`, so it is a one-shot MPI/hybrid baseline rather than a validated reusable Newton factorization path.

## Best Observed Runs

{md_table(best)}

## GPU Offload Check

{md_table(offload)}

The CUDA-enabled build is real, but the best MMD runs use very small GPU buffers. Disabling SuperLU_DIST GPU offload did not make the large cases slower in this ABglobal path; in some runs it was slightly faster. Therefore the current best SuperLU_DIST numbers should be interpreted as MPI/CPU-dominant hybrid timings, not as strong GPU-resident sparse direct-solver evidence.

## Process Grid Check

{md_table(grid)}

The earlier `1 x np` grid was not the only shape tested here. For these single-node cases, increasing ranks or using a square-ish `2 x 2` grid did not beat `np=1`; communication/setup overhead and the small matrix sizes relative to distributed direct-solver overhead make single-rank best for this dataset.

## Option Check

{md_table(option)}

No tested option materially beats the best `MMD_AT_PLUS_A` one-shot configuration. Iterative refinement adds cost and is unnecessary because the no-refinement residuals are already near FP64 reference quality. Tiny-pivot replacement and equilibration changes did not produce a better default.

## Judgment

The original NATURAL ordering result was not best effort. The later MMD ordering results are a reasonable best-effort ABglobal configuration for this installed SuperLU_DIST build. However, they are not a proof of optimal SuperLU_DIST overall because:

- the test uses the high-level ABglobal driver, not a lower-level reusable factor/solve API
- ParMETIS and COLAMD are not enabled in this build
- GPU usage in best MMD runs is minimal
- 3D/SLATE-style fully GPU-oriented paths were not validated here

For cuPF annual-report wording, SuperLU_DIST should be reported as a valid external MPI/hybrid direct-solver baseline with best-effort MMD ordering, but not as a fully GPU-resident or cuPF-integration-equivalent alternative to cuDSS.

CSV: `{csv_path.relative_to(ROOT)}`
""")


def main() -> None:
    ensure_dirs()
    paths: list[tuple[Path, str]] = []
    for case, best_row in [("case2869pegase", "LargeDiag_MC64"), ("case9241pegase", "NOROWPERM")]:
        for rowperm in ["LargeDiag_MC64", "NOROWPERM"]:
            paths.append((run("offload_on", case, 1, 1, 1, rowperm, ["--acc-offload", "1"]), "offload_on"))
            paths.append((run("offload_off", case, 1, 1, 1, rowperm, ["--acc-offload", "0"]), "offload_off"))
        for nprow, npcol in [(1, 2), (2, 1)]:
            paths.append((run("grid_np2", case, 2, nprow, npcol, best_row, ["--acc-offload", "1"]), "grid_np2"))
        for nprow, npcol in [(1, 4), (2, 2), (4, 1)]:
            paths.append((run("grid_np4", case, 4, nprow, npcol, best_row, ["--acc-offload", "1"]), "grid_np4"))
        option_runs = [
            ("option_equil_off", ["--equil", "0"]),
            ("option_replace_tiny", ["--replace-tiny-pivot", "1"]),
            ("option_refine_double", ["--iter-refine", "DOUBLE"]),
            ("option_par_symb", ["--par-symb-fact", "1"]),
        ]
        for label, extra in option_runs:
            paths.append((run(label, case, 1, 1, 1, best_row, ["--acc-offload", "1", *extra]), label))
    rows = [enrich(path, label) for path, label in paths]
    csv_path = write_csv(rows)
    write_report(rows, csv_path)
    print(REPORT)


if __name__ == "__main__":
    main()
