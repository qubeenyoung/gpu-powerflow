#!/usr/bin/env python3
"""Inspect hardware, CUDA, compiler, Python, and solver-library availability."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = EXP_ROOT / "results"


def run(cmd: List[str], timeout: int = 10) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": repr(exc)}


def first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text.strip() else ""


def read_first_matching(path: Path, prefix: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        return ""
    return ""


def cpu_model() -> str:
    return read_first_matching(Path("/proc/cpuinfo"), "model name") or platform.processor()


def ram_gb() -> Optional[float]:
    raw = read_first_matching(Path("/proc/meminfo"), "MemTotal")
    match = re.search(r"([0-9]+)", raw)
    if not match:
        return None
    return round(int(match.group(1)) / 1024.0 / 1024.0, 2)


def command_version(cmd: str, args: List[str]) -> Dict[str, Any]:
    path = shutil.which(cmd)
    if not path:
        return {"path": "", "version": "", "available": False}
    out = run([path] + args)
    return {
        "path": path,
        "version": first_line(out["stdout"] or out["stderr"]),
        "available": out["returncode"] == 0,
        "raw": out,
    }


def nvidia_smi() -> Dict[str, Any]:
    path = shutil.which("nvidia-smi")
    if not path:
        return {"available": False, "gpus": []}
    out = run(
        [
            path,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    gpus = []
    if out["returncode"] == 0:
        for line in out["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(
                    {
                        "name": parts[0],
                        "memory_total": parts[1],
                        "driver_version": parts[2],
                    }
                )
    return {"available": out["returncode"] == 0, "gpus": gpus, "raw": out}


def nvcc_toolkit_version(nvcc: Dict[str, Any]) -> str:
    raw = nvcc.get("raw", {})
    text = (raw.get("stdout", "") + "\n" + raw.get("stderr", "")).strip()
    match = re.search(r"release\s+([0-9.]+),\s+V([0-9.]+)", text)
    if match:
        return f"CUDA {match.group(1)} (nvcc V{match.group(2)})"
    return nvcc.get("version", "")


def find_existing(paths: List[Path]) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return ""


def find_library(name_patterns: List[str], roots: List[Path]) -> str:
    for root in roots:
        if not root.exists():
            continue
        for pattern in name_patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return str(matches[0])
    return ""


def grep_version(header: str, names: List[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not header:
        return values
    try:
        text = Path(header).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return values
    for name in names:
        match = re.search(rf"#define\s+{re.escape(name)}\s+([0-9]+)", text)
        if match:
            values[name] = match.group(1)
    return values


def solver_availability() -> Dict[str, Any]:
    cuda_root = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    cuda_include = cuda_root / "targets/x86_64-linux/include"
    cuda_lib = cuda_root / "targets/x86_64-linux/lib"
    cudss_include = find_existing(
        [
            Path("/root/.local/lib/python3.10/site-packages/nvidia/cu12/include/cudss.h"),
            cuda_include / "cudss.h",
            Path("/usr/local/include/cudss.h"),
        ]
    )
    cudss_lib = find_library(
        ["libcudss.so", "libcudss.so.*"],
        [
            Path("/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib"),
            cuda_lib,
            Path("/usr/local/lib"),
        ],
    )
    cudss_version = grep_version(
        cudss_include,
        ["CUDSS_VERSION_MAJOR", "CUDSS_VERSION_MINOR", "CUDSS_VERSION_PATCH"],
    )
    cusolver_header = find_existing([cuda_include / "cusolverSp.h"])
    cusolver_lib = find_library(["libcusolver.so", "libcusolver.so.*"], [cuda_lib])
    cusolver_version = grep_version(
        str(cuda_include / "cusolver_common.h"),
        [
            "CUSOLVER_VER_MAJOR",
            "CUSOLVER_VER_MINOR",
            "CUSOLVER_VER_PATCH",
            "CUSOLVER_VER_BUILD",
        ],
    )
    amgx_header = find_existing([Path("/usr/local/include/amgx_c.h"), Path("/usr/include/amgx_c.h")])
    amgx_lib = find_library(["libamgxsh.so", "libamgx.a"], [Path("/usr/local/lib"), Path("/usr/lib")])
    ginkgo_config = find_library(
        ["**/GinkgoConfig.cmake", "**/libginkgo.so"],
        [Path("/usr/local"), Path("/usr"), EXP_ROOT / "third_party"],
    )
    superlu_dist = find_library(
        ["**/*superlu_dist*", "**/libsuperlu_dist*"],
        [Path("/usr/local"), Path("/usr"), EXP_ROOT / "third_party"],
    )
    strumpack = find_library(
        ["**/*STRUMPACKConfig.cmake", "**/libstrumpack*"],
        [Path("/usr/local"), Path("/usr"), EXP_ROOT / "third_party"],
    )
    return {
        "cudss": {
            "available": bool(cudss_include and cudss_lib),
            "include": cudss_include,
            "library": cudss_lib,
            "version_macros": cudss_version,
        },
        "cusolver": {
            "available": bool(cusolver_header and cusolver_lib),
            "include": cusolver_header,
            "library": cusolver_lib,
            "version_macros": cusolver_version,
            "cusolverRf_header": find_existing([cuda_include / "cusolverRf.h"]),
        },
        "amgx": {
            "available": bool(amgx_header and amgx_lib),
            "include": amgx_header,
            "library": amgx_lib,
        },
        "ginkgo": {"available": bool(ginkgo_config), "path": ginkgo_config},
        "superlu_dist": {"available": bool(superlu_dist), "path": superlu_dist},
        "strumpack": {"available": bool(strumpack), "path": strumpack},
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    nvcc = command_version("nvcc", ["--version"])
    cmake = command_version("cmake", ["--version"])
    gcc = command_version("gcc", ["--version"])
    gxx = command_version("g++", ["--version"])
    mpicc = command_version("mpicc", ["--version"])
    mpicxx = command_version("mpicxx", ["--version"])
    env: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu_model": cpu_model(),
        "ram_gb": ram_gb(),
        "gpu": nvidia_smi(),
        "cuda": {
            "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
            "nvcc": nvcc,
            "toolkit_version": nvcc_toolkit_version(nvcc),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "compilers": {
            "gcc": gcc,
            "g++": gxx,
            "mpicc": mpicc,
            "mpicxx": mpicxx,
            "cmake": cmake,
        },
        "solver_libraries": solver_availability(),
    }
    out = RESULTS_DIR / "environment.json"
    out.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
