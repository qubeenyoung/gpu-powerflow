"""Run representative cuPF variants through the native C++ evaluator."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from . import eval_common, matpower_data
from .cupf_variants import BUILD_DIRS, filter_variants, find_cpp_executable, gpu_available, variants_for_entrypoint


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prepare_dumps(args: argparse.Namespace, variant: dict[str, Any], out: Path) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    paths = eval_common.selected_case_paths(args)
    dump_root = out / "dumps"
    dump_root.mkdir(parents=True, exist_ok=True)
    by_case: dict[str, dict[str, Any]] = {}
    for path in paths:
        case, reference = eval_common.load_case_and_reference(path, args)
        case_dir = matpower_data.save_dump_case(case, dump_root, reference)
        by_case[case.case_name] = {
            "case": case,
            "reference": reference,
            "case_dir": case_dir,
            "case_path": path,
        }
    eval_common.write_json(
        out / "prepare_manifest.json",
        {
            "variant": variant["variant"],
            "dump_root": str(dump_root),
            "cases": {name: {"case_path": str(item["case_path"]), "case_dir": str(item["case_dir"])} for name, item in by_case.items()},
        },
    )
    return paths, by_case


def _cpp_command(args: argparse.Namespace, variant: dict[str, Any], executable: Path, dump_root: Path, raw_out: Path) -> list[str]:
    cmd = [
        str(executable),
        "--case-root",
        str(dump_root),
        "--output-dir",
        str(raw_out),
        "--backend",
        variant["backend"],
        "--compute",
        variant["compute"],
        "--cpu-jacobian",
        variant.get("cpu_jacobian", "native"),
        "--cpu-linear-solver",
        variant.get("cpu_linear_solver", "klu"),
        "--cuda-jacobian",
        variant.get("cuda_jacobian", "edge"),
        "--cuda-linear-solver",
        variant["cuda_linear_solver"],
        "--tolerance",
        str(args.tolerance),
        "--max-iter",
        str(args.max_iter),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
    ]
    if args.cases:
        cmd += ["--cases", *[Path(case).stem for case in args.cases]]
    return cmd


def _normalize_rows(raw_runs: Path, out: Path, variant: dict[str, Any], case_info: dict[str, dict[str, Any]]) -> None:
    with eval_common.CsvSink(out / "runs.csv") as sink:
        for raw in _read_csv(raw_runs):
            name = raw.get("case_name", "")
            info = case_info.get(name)
            case = info["case"] if info else None
            reference = info["reference"] if info else None
            init_ms = raw.get("initialize_ms", "")
            solve_ms = raw.get("solve_ms", "")
            try:
                total_ms = float(init_ms) + float(solve_ms)
            except (TypeError, ValueError):
                total_ms = ""
            row = {
                "mode": "cupf",
                "variant": variant["variant"],
                "case_name": name,
                "case_path": str(info["case_path"]) if info else raw.get("case_path", ""),
                "backend": variant["backend"],
                "compute": variant["compute"],
                "linear_solver": variant["linear_solver"],
                "jacobian": variant.get("jacobian", ""),
                "entrypoint": "native",
                "repeat_idx": raw.get("repeat_idx", ""),
                "warmup": "0",
                "success": raw.get("success", "0"),
                "converged": raw.get("cupf_converged", ""),
                "iterations": raw.get("cupf_iterations", ""),
                "error_message": raw.get("error_message", ""),
                "initialize_ms": init_ms,
                "solve_ms": solve_ms,
                "total_ms": total_ms,
                "output_mismatch": raw.get("output_mismatch", raw.get("cupf_final_mismatch", "")),
                "max_abs_v_error": raw.get("max_abs_v_error", ""),
                "rms_abs_v_error": raw.get("rms_abs_v_error", ""),
            }
            if case is not None:
                row.update(eval_common.dimensions(case))
            else:
                row.update(
                    {
                        "n_bus": raw.get("n_bus", ""),
                        "ybus_nnz": raw.get("ybus_nnz", ""),
                        "n_ref": raw.get("n_ref", ""),
                        "n_pv": raw.get("n_pv", ""),
                        "n_pq": raw.get("n_pq", ""),
                    }
                )
            if reference is not None:
                row.update(eval_common.reference_fields(reference))
            sink.write(row)


def run_variant(args: argparse.Namespace, variant: dict[str, Any]) -> None:
    out = eval_common.variant_dir(args, variant["variant"])
    paths = eval_common.selected_case_paths(args)
    manifest = eval_common.manifest(
        args,
        "cupf",
        variant["variant"],
        paths,
        backend=variant["backend"],
        compute=variant["compute"],
        linear_solver=variant["linear_solver"],
        jacobian=variant.get("jacobian", ""),
        entrypoint="native",
        build_key=variant["build_key"],
    )

    if variant["requires_gpu"] and not gpu_available():
        eval_common.write_skip(out, "requires a CUDA GPU device (none available on this host)", manifest)
        print(f"[{variant['variant']}][SKIP] no CUDA GPU", flush=True)
        return
    executable = find_cpp_executable(BUILD_DIRS[variant["build_key"]])
    if executable is None:
        eval_common.write_skip(out, f"cupf_cpp_evaluate not found under {BUILD_DIRS[variant['build_key']]}", manifest)
        print(f"[{variant['variant']}][SKIP] missing C++ evaluator", flush=True)
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "SKIPPED.txt").unlink(missing_ok=True)
    eval_common.write_json(out / "run.json", manifest)
    paths, case_info = _prepare_dumps(args, variant, out)

    with tempfile.TemporaryDirectory(prefix=f"{variant['variant']}-") as tmp:
        raw_out = Path(tmp)
        cmd = _cpp_command(args, variant, executable, out / "dumps", raw_out)
        print(f"[{variant['variant']}][RUN] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        _normalize_rows(raw_out / "runs.csv", out, variant, case_info)
        if (raw_out / "timing.csv").exists():
            shutil.copy2(raw_out / "timing.csv", out / "timing.csv")
    print(f"[{variant['variant']}] wrote {out / 'runs.csv'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cuPF native C++ benchmark variants.")
    eval_common.add_common_args(parser)
    parser.add_argument("--variants", nargs="*", help="Variant IDs. Defaults to all representative native variants.")
    parser.add_argument("--include-diagnostic-fp32", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    variants = filter_variants(variants_for_entrypoint(args.include_diagnostic_fp32, "native"), args.variants)
    eval_common.run_root(args).mkdir(parents=True, exist_ok=True)
    for variant in variants:
        run_variant(args, variant)
    if not args.no_aggregate:
        from .aggregate_results import aggregate

        aggregate(eval_common.run_root(args))


if __name__ == "__main__":
    main()
