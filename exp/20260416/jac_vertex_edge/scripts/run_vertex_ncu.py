#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = EXP_ROOT.parents[2]
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "exp" / "20260414" / "amgx" / "cupf_dumps"
DEFAULT_BUILD_DIR = EXP_ROOT / "build" / "operator_probe"
DEFAULT_CASE_LIST = EXP_ROOT / "cases_ncu_speed.txt"
MODE_CONFIGS = {
    "edge_atomic": {
        "kernel_name": "regex:update_jacobian_edge_fp32_kernel",
        "results_dir": "ncu_edge_atomic",
        "report_stem": "edge_atomic_jacobian",
    },
    "edge_noatomic": {
        "kernel_name": "regex:update_jacobian_edge_noatomic_fp32_kernel",
        "results_dir": "ncu_edge_noatomic",
        "report_stem": "edge_noatomic_jacobian",
    },
    "vertex": {
        "kernel_name": "regex:update_jacobian_vertex_fp32_kernel",
        "results_dir": "ncu_vertex",
        "report_stem": "vertex_jacobian",
    },
}


def read_case_list(path: Path) -> list[str]:
    cases: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            cases.append(stripped)
    return cases


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("[RUN]", " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def configure_and_build(build_dir: Path, cuda_arch: str | None, command_log: list[dict[str, Any]]) -> Path:
    configure_cmd = [
        "cmake",
        "-S", str(EXP_ROOT),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if cuda_arch:
        configure_cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
    build_cmd = [
        "cmake",
        "--build", str(build_dir),
        "--target", "jacobian_operator_probe",
        "-j", str(os.cpu_count() or 1),
    ]
    command_log.append({"name": "cmake_configure", "cmd": configure_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(configure_cmd, cwd=WORKSPACE_ROOT)
    command_log.append({"name": "cmake_build", "cmd": build_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(build_cmd, cwd=WORKSPACE_ROOT)
    binary = build_dir / "jacobian_operator_probe"
    if not binary.exists():
        raise FileNotFoundError(f"probe binary was not built: {binary}")
    return binary


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Nsight Compute on a selected Jacobian kernel for selected cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--case-list", type=Path, default=DEFAULT_CASE_LIST)
    parser.add_argument("--cases", nargs="*", help="Override case-list with explicit case names.")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="vertex")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--results-root", type=Path, help="Defaults to results/ncu_<mode>.")
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--ncu-binary", default="ncu")
    parser.add_argument("--ncu-set", default="full", help="Nsight Compute --set value. Use empty string to omit.")
    parser.add_argument("--kernel-name", help="Defaults to the kernel regex for the selected mode.")
    parser.add_argument("--warmup", type=int, default=0, help="Probe warmup inside ncu. Keep 0 to profile one launch.")
    parser.add_argument("--repeats", type=int, default=1, help="Probe repeats inside ncu. Keep 1 for one report per case.")
    parser.add_argument("--cuda-visible-devices", help="Set CUDA_VISIBLE_DEVICES for ncu.")
    parser.add_argument("--cuda-arch", help="Override CMAKE_CUDA_ARCHITECTURES, for example 86.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write command files but do not invoke ncu.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode_config = MODE_CONFIGS[args.mode]
    kernel_name = args.kernel_name or mode_config["kernel_name"]
    results_root = args.results_root or EXP_ROOT / "results" / mode_config["results_dir"]

    cases = args.cases if args.cases is not None else read_case_list(args.case_list)
    if not cases:
        raise SystemExit("no cases selected")

    command_log: list[dict[str, Any]] = []
    binary = args.build_dir / "jacobian_operator_probe"
    if args.skip_build:
        if not binary.exists():
            raise FileNotFoundError(f"--skip-build was set but probe binary is missing: {binary}")
    else:
        binary = configure_and_build(args.build_dir, args.cuda_arch, command_log)

    ncu_path = shutil.which(args.ncu_binary) or args.ncu_binary
    if not args.dry_run and shutil.which(args.ncu_binary) is None and not Path(args.ncu_binary).exists():
        raise FileNotFoundError(f"Nsight Compute CLI not found: {args.ncu_binary}")

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    run_root = results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    shell_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(WORKSPACE_ROOT))}",
    ]
    if args.cuda_visible_devices is not None:
        shell_lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_visible_devices)}")
    for case in cases:
        case_dir = args.dataset_root / case
        if not case_dir.exists():
            raise FileNotFoundError(f"case directory not found: {case_dir}")

        report_stem = run_root / case / mode_config["report_stem"]
        report_stem.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(ncu_path),
            "--target-processes", "all",
            "--force-overwrite",
            "--kernel-name", kernel_name,
            "-o", str(report_stem),
        ]
        if args.ncu_set:
            cmd.extend(["--set", args.ncu_set])
        cmd.extend([
            str(binary),
            "--case-dir", str(case_dir),
            "--mode", args.mode,
            "--warmup", str(args.warmup),
            "--repeats", str(args.repeats),
        ])
        command_log.append({"name": f"ncu_{args.mode}_{case}", "cmd": cmd, "cwd": str(WORKSPACE_ROOT)})
        shell_lines.append(shell_join(cmd))
        if not args.dry_run:
            run_command(cmd, cwd=WORKSPACE_ROOT, env=env)

    commands_path = run_root / f"ncu_{args.mode}_commands.sh"
    commands_path.write_text("\n".join(shell_lines) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)
    (run_root / "command_log.json").write_text(
        json.dumps({
            "dataset_root": str(args.dataset_root),
            "cases": cases,
            "mode": args.mode,
            "kernel_name": kernel_name,
            "ncu_set": args.ncu_set,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            "dry_run": args.dry_run,
            "commands": command_log,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] wrote {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
