"""Run MATLAB/MATPOWER baseline variants and normalize them to runs.csv."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
from typing import Any

from . import eval_common

MATPOWER_VARIANTS = [
    {
        "variant": "matpower-default",
        "backend": "cpu",
        "compute": "fp64",
        "linear_solver": "",
        "entrypoint": "matlab-runpf",
    },
    {
        "variant": "matpower-lu5",
        "backend": "cpu",
        "compute": "fp64",
        "linear_solver": "LU5",
        "entrypoint": "matlab-runpf",
    },
]

MATLAB_DIR = eval_common.REPO_ROOT / "matlab"
MATLAB_RUNNER = MATLAB_DIR / "run_matpower_case.m"


def _matlab_bin(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env_bin = os.environ.get("MATLAB_BIN")
    if env_bin:
        return env_bin
    default = Path("/opt/matlab/bin/matlab")
    if default.exists():
        return str(default)
    return shutil.which("matlab")


def _matpower_home(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return Path(os.environ.get("MATPOWER_HOME", "/opt/matpower"))


def _load_repo_env(env: dict[str, str]) -> dict[str, str]:
    """Load simple KEY=VALUE entries from local .env files without logging secrets."""
    repo_root = Path(__file__).resolve().parents[2]
    for env_path in (repo_root / ".env", repo_root / "matlab" / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            try:
                parts = shlex.split(line, comments=True, posix=True)
            except ValueError:
                parts = [line]
            if len(parts) != 1 or "=" not in parts[0]:
                continue
            key, value = parts[0].split("=", 1)
            if key and key not in env:
                env[key] = value
    return env


def _matlab_env() -> dict[str, str]:
    env = _load_repo_env(dict(os.environ))
    if env.get("MATLAB_LICENSE_FILE") and not env.get("MLM_LICENSE_FILE"):
        env["MLM_LICENSE_FILE"] = env["MATLAB_LICENSE_FILE"]
    if (
        not env.get("MATLAB_LICMODE")
        and env.get("MATLAB_USER_ID")
        and env.get("MATLAB_PASSWORD")
        and not env.get("MLM_LICENSE_FILE")
    ):
        env["MATLAB_LICMODE"] = "onlinelicensing"
    return env


def _matlab_license_args(env: dict[str, str]) -> list[str]:
    licmode = env.get("MATLAB_LICMODE")
    if licmode:
        return ["-licmode", licmode]
    return []


def _ensure_online_login(env: dict[str, str], matlab_bin: str, out: Path, manifest: dict[str, Any]) -> bool:
    if env.get("MATLAB_LICMODE") != "onlinelicensing":
        return True
    if not (env.get("MATLAB_USER_ID") and env.get("MATLAB_PASSWORD")):
        return True
    repo_root = Path(__file__).resolve().parents[2]
    login_script = repo_root / "matlab" / "login_online.bash"
    if not login_script.exists():
        return True
    login_env = dict(env)
    login_env["MATLAB_BIN"] = matlab_bin
    try:
        subprocess.run(
            [str(login_script)],
            check=True,
            text=True,
            capture_output=True,
            timeout=180,
            env=login_env,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        reason = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            reason = (exc.stderr or exc.stdout or reason)[-4000:]
        eval_common.write_skip(out, f"MATLAB online licensing login failed: {reason}", manifest)
        print("[matpower][SKIP] MATLAB online licensing login failed", flush=True)
        return False
    return True


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matlab_quote(value: str | Path) -> str:
    """Return a MATLAB single-quoted string literal without exposing shell quoting."""
    return str(value).replace("'", "''")


def _matlab_benchmark_call(cases_file: Path, raw_csv: Path, matpower_home: Path, variant: dict[str, Any], args: argparse.Namespace) -> str:
    """Build the MATLAB -batch expression that invokes matlab/run_matpower_case.m.

    The substantial MATLAB logic lives in the checked-in .m file.  Python only
    passes paths and benchmark parameters, which keeps this runner readable and
    makes manual MATLAB debugging use the same code path as automated runs.
    """
    matlab_dir = _matlab_quote(MATLAB_DIR.as_posix())
    return (
        f"addpath('{matlab_dir}'); "
        "run_matpower_case('', '', "
        "'Mode', 'benchmark', "
        f"'CasesFile', '{_matlab_quote(cases_file.as_posix())}', "
        f"'OutputCsv', '{_matlab_quote(raw_csv.as_posix())}', "
        f"'MatpowerHome', '{_matlab_quote(matpower_home.as_posix())}', "
        f"'Variant', '{_matlab_quote(variant['variant'])}', "
        f"'LinearSolver', '{_matlab_quote(variant['linear_solver'])}', "
        f"'Warmup', {int(args.warmup)}, "
        f"'Repeats', {int(args.repeats)}, "
        f"'Tolerance', {float(args.tolerance):.17g}, "
        f"'MaxIter', {int(args.max_iter)});"
    )


def _normalize_rows(raw_csv: Path, out: Path, variant: dict[str, Any], case_info: dict[str, dict[str, Any]]) -> None:
    with eval_common.CsvSink(out / "runs.csv") as sink:
        for raw in _read_csv(raw_csv):
            info = case_info.get(raw.get("case_name", ""))
            case = info["case"] if info else None
            init_ms = raw.get("initialize_ms", "")
            solve_ms = raw.get("solve_ms", "")
            try:
                total_ms = float(init_ms) + float(solve_ms)
            except (TypeError, ValueError):
                total_ms = ""
            row = {
                "mode": "matpower",
                "variant": variant["variant"],
                "case_name": raw.get("case_name", ""),
                "case_path": raw.get("case_path", ""),
                "backend": variant["backend"],
                "compute": variant["compute"],
                "linear_solver": variant["linear_solver"] or "default",
                "entrypoint": variant["entrypoint"],
                "repeat_idx": raw.get("repeat_idx", ""),
                "warmup": raw.get("warmup", "0"),
                "success": raw.get("success", "0"),
                "converged": raw.get("converged", ""),
                "iterations": raw.get("iterations", ""),
                "error_message": raw.get("error_message", ""),
                "initialize_ms": init_ms,
                "solve_ms": solve_ms,
                "total_ms": total_ms,
                "reported_solve_ms": raw.get("reported_solve_ms", ""),
                "output_mismatch": raw.get("output_mismatch", ""),
            }
            if case is not None:
                row.update(eval_common.dimensions(case))
            sink.write(row)


def run_variant(args: argparse.Namespace, variant: dict[str, Any]) -> None:
    paths = eval_common.selected_case_paths(args)
    out = eval_common.variant_dir(args, variant["variant"])
    manifest = eval_common.manifest(
        args,
        "matpower",
        variant["variant"],
        paths,
        backend=variant["backend"],
        compute=variant["compute"],
        linear_solver=variant["linear_solver"] or "default",
        entrypoint=variant["entrypoint"],
    )
    matlab_bin = _matlab_bin(args.matlab_bin)
    matpower_home = _matpower_home(args.matpower_home)
    matlab_env = _matlab_env()

    if matlab_bin is None:
        eval_common.write_skip(out, "MATLAB executable not found", manifest)
        print(f"[{variant['variant']}][SKIP] MATLAB executable not found", flush=True)
        return
    if not MATLAB_RUNNER.exists():
        eval_common.write_skip(out, f"MATLAB runner not found: {MATLAB_RUNNER}", manifest)
        print(f"[{variant['variant']}][SKIP] missing MATLAB runner={MATLAB_RUNNER}", flush=True)
        return
    if not matpower_home.exists():
        eval_common.write_skip(out, f"MATPOWER_HOME not found: {matpower_home}", manifest)
        print(f"[{variant['variant']}][SKIP] missing MATPOWER_HOME={matpower_home}", flush=True)
        return
    if not _ensure_online_login(matlab_env, matlab_bin, out, manifest):
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "SKIPPED.txt").unlink(missing_ok=True)
    eval_common.write_json(out / "run.json", manifest)
    case_info = {}
    for path in paths:
        case = eval_common.load_case_and_reference(path, args)[0]
        case_info[case.case_name] = {"case": case}

    with tempfile.TemporaryDirectory(prefix=f"{variant['variant']}-") as tmp:
        tmp_dir = Path(tmp)
        cases_file = tmp_dir / "cases.txt"
        raw_csv = tmp_dir / "matpower_raw.csv"
        cases_file.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
        cmd = [
            matlab_bin,
            *_matlab_license_args(matlab_env),
            "-batch",
            _matlab_benchmark_call(cases_file, raw_csv, matpower_home, variant, args),
        ]
        print(f"[{variant['variant']}][RUN] {' '.join(cmd)}", flush=True)
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=True,
                timeout=args.matlab_timeout_sec,
                env=matlab_env,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            reason = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                reason = (exc.stderr or exc.stdout or reason)[-4000:]
            eval_common.write_skip(out, f"MATLAB/MATPOWER failed: {reason}", manifest)
            print(f"[{variant['variant']}][SKIP] MATLAB/MATPOWER failed", flush=True)
            return
        if proc.stdout:
            print(proc.stdout, end="", flush=True)
        if not raw_csv.exists():
            eval_common.write_skip(out, "MATLAB completed but did not write raw CSV", manifest)
            return
        _normalize_rows(raw_csv, out, variant, case_info)
    print(f"[{variant['variant']}] wrote {out / 'runs.csv'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MATLAB/MATPOWER benchmark variants.")
    eval_common.add_common_args(parser)
    parser.add_argument("--variants", nargs="*", help="matpower-default and/or matpower-lu5.")
    parser.add_argument("--matlab-bin", default=None)
    parser.add_argument("--matpower-home", type=Path, default=None)
    parser.add_argument("--matlab-timeout-sec", type=int, default=7200)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    requested = set(args.variants or [])
    variants = [v for v in MATPOWER_VARIANTS if not requested or v["variant"] in requested]
    eval_common.run_root(args).mkdir(parents=True, exist_ok=True)
    for variant in variants:
        run_variant(args, variant)
    if not args.no_aggregate:
        from .aggregate_results import aggregate

        aggregate(eval_common.run_root(args))


if __name__ == "__main__":
    main()
