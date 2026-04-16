#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parent
WORKSPACE_ROOT = EXP_ROOT.parents[2]
CUPF_ROOT = WORKSPACE_ROOT / "cuPF"
BENCHMARK_SCRIPT = CUPF_ROOT / "benchmarks" / "run_benchmarks.py"

DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "texas_univ_cases" / "cuPF_datasets"
DEFAULT_RESULTS_ROOT = EXP_ROOT / "results"
DEFAULT_BUILD_ROOT = EXP_ROOT / "build"
DEFAULT_CASE_LIST = EXP_ROOT / "cases.txt"
DEFAULT_PROFILES = ["cuda_edge", "cuda_edge_modified"]


def default_threading_lib() -> Path:
    candidates = [
        Path("/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0"),
        Path("/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so.0"),
        Path("/usr/local/lib/libcudss_mtlayer_gomp.so.0"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find libcudss_mtlayer_gomp.so.0. Pass --cudss-threading-lib explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standard vs modified Newton solver on Texas University cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--case-list", type=Path, default=DEFAULT_CASE_LIST)
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--cuda-visible-devices", default="3")
    parser.add_argument("--cudss-threading-lib", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threading_lib = args.cudss_threading_lib or default_threading_lib()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["CUDSS_THREADING_LIB"] = str(threading_lib)

    cudss_lib_dir = threading_lib.parent
    ld_library_path = [entry for entry in env.get("LD_LIBRARY_PATH", "").split(":") if entry]
    if str(cudss_lib_dir) not in ld_library_path:
        env["LD_LIBRARY_PATH"] = ":".join([str(cudss_lib_dir), *ld_library_path])

    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--dataset-root", str(args.dataset_root),
        "--results-root", str(args.results_root),
        "--run-name", args.run_name,
        "--mode", "both",
        "--end2end-build-dir", str(args.build_root / "end2end"),
        "--operators-build-dir", str(args.build_root / "operators"),
        "--case-list", str(args.case_list),
        "--profiles", *args.profiles,
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--tolerance", str(args.tolerance),
        "--max-iter", str(args.max_iter),
        "--with-cuda",
        "--cudss-enable-mt",
        "--cudss-host-nthreads", "AUTO",
        "--cudss-threading-lib", str(threading_lib),
        "--cudss-nd-nlevels", "AUTO",
    ]
    if args.skip_build:
        cmd.append("--skip-build")

    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "wrapper_environment.json").write_text(
        json.dumps({
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "physical_gpu_note": "CUDA_VISIBLE_DEVICES=3 exposes physical GPU 3 as local device 0.",
            "cudss_threading_lib": env["CUDSS_THREADING_LIB"],
            "cudss_host_nthreads": "AUTO",
            "cudss_mt_enabled": True,
            "command": cmd,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[RUN]", " ".join(cmd), flush=True)
    print(f"[ENV] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    print(f"[ENV] CUDSS_THREADING_LIB={env['CUDSS_THREADING_LIB']}", flush=True)
    subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
