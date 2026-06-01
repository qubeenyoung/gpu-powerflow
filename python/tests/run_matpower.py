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
    """Load simple KEY=VALUE entries from the repo-local .env without logging secrets."""
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return env
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


def _matlab_script(cases_file: Path, raw_csv: Path, matpower_home: Path, variant: dict[str, Any], args: argparse.Namespace) -> str:
    solver = variant["linear_solver"]
    return f"""
cases_file = "{cases_file.as_posix()}";
raw_csv = "{raw_csv.as_posix()}";
matpower_home = "{matpower_home.as_posix()}";
variant = "{variant['variant']}";
linear_solver = "{solver}";
warmup = {int(args.warmup)};
repeats = {int(args.repeats)};
tol = {float(args.tolerance):.17g};
max_it = {int(args.max_iter)};

if exist(matpower_home, "dir") ~= 7
    error("MATPOWER_HOME not found: %s", matpower_home);
end
addpath(genpath(matpower_home));
define_constants;
case_lines = readlines(cases_file);
case_lines = case_lines(strlength(strtrim(case_lines)) > 0);
fid = fopen(raw_csv, "w");
cleanup = onCleanup(@() fclose(fid));
csv_row(fid, {{"case_name","case_path","repeat_idx","warmup","success","converged","iterations","error_message","initialize_ms","solve_ms","reported_solve_ms","output_mismatch"}});

for c = 1:numel(case_lines)
    case_path = char(case_lines(c));
    [~, case_name, ~] = fileparts(case_path);
    for repeat = 0:(warmup + repeats - 1)
        is_warmup = repeat < warmup;
        repeat_idx = repeat;
        if ~is_warmup
            repeat_idx = repeat - warmup;
        end
        try
            t_init = tic;
            mpc = loadcase(case_path);
            mpopt = mpoption("verbose", 0, "out.all", 0, "pf.alg", "NR", "pf.tol", tol, "pf.nr.max_it", max_it);
            if strlength(linear_solver) > 0
                mpopt = mpoption(mpopt, "pf.nr.lin_solver", char(linear_solver));
            end
            initialize_ms = toc(t_init) * 1000.0;

            t_solve = tic;
            results = runpf(mpc, mpopt);
            solve_ms = toc(t_solve) * 1000.0;
            reported_solve_ms = NaN;
            if isfield(results, "et")
                reported_solve_ms = results.et * 1000.0;
            end
            iterations = NaN;
            if isfield(results, "iterations")
                iterations = results.iterations;
            end
            output_mismatch = NaN;
            try
                int_results = ext2int(results);
                [Ybus, ~, ~] = makeYbus(int_results.baseMVA, int_results.bus, int_results.branch);
                Sbus = makeSbus(int_results.baseMVA, int_results.bus, int_results.gen);
                [~, pv, pq] = bustypes(int_results.bus, int_results.gen);
                V = int_results.bus(:, VM) .* exp(1j * pi / 180 * int_results.bus(:, VA));
                mis = V .* conj(Ybus * V) - Sbus;
                F = [real(mis(pv)); real(mis(pq)); imag(mis(pq))];
                if isempty(F)
                    output_mismatch = 0.0;
                else
                    output_mismatch = norm(F, inf);
                end
            catch mismatch_exc
                output_mismatch = NaN;
            end
            csv_row(fid, {{case_name, case_path, repeat_idx, double(is_warmup), 1, double(results.success), iterations, "", initialize_ms, solve_ms, reported_solve_ms, output_mismatch}});
            if ~is_warmup
                fprintf("[%s][OK] %s repeat=%d init_ms=%.3f solve_ms=%.3f resid=%.3e\\n", variant, case_name, repeat_idx, initialize_ms, solve_ms, output_mismatch);
            end
        catch exc
            csv_row(fid, {{case_name, case_path, repeat_idx, double(is_warmup), 0, 0, NaN, getReport(exc, "basic", "hyperlinks", "off"), NaN, NaN, NaN, NaN}});
            fprintf(2, "[%s][FAIL] %s: %s\\n", variant, case_name, exc.message);
        end
    end
end

function csv_row(fid, values)
    for k = 1:numel(values)
        if k > 1
            fprintf(fid, ",");
        end
        fprintf(fid, "%s", csv_cell(values{{k}}));
    end
    fprintf(fid, "\\n");
end

function out = csv_cell(value)
    if isnumeric(value) || islogical(value)
        if isempty(value) || any(isnan(value(:)))
            out = "";
        else
            out = sprintf("%.17g", value);
        end
    else
        out = char(string(value));
    end
    if contains(out, '"')
        out = strrep(out, '"', '""');
    end
    if contains(out, ',') || contains(out, '"') || contains(out, newline)
        out = ['"' out '"'];
    end
end
"""


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
        script = tmp_dir / "run_matpower_benchmark.m"
        cases_file.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
        script.write_text(_matlab_script(cases_file, raw_csv, matpower_home, variant, args), encoding="utf-8")
        cmd = [matlab_bin, *_matlab_license_args(matlab_env), "-batch", f"run('{script.as_posix()}')"]
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
