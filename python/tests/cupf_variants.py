"""Representative cuPF benchmark variant definitions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .eval_common import REPO_ROOT

BUILD_DIRS = {
    "cpu": REPO_ROOT / "cuPF" / "build" / "eval-cpu",
    "gpu": REPO_ROOT / "cuPF" / "build" / "eval-gpu",
    "gpu-custom": REPO_ROOT / "cuPF" / "build" / "eval-gpu-custom",
}

BASE_CUPF_VARIANTS: list[dict[str, Any]] = [
    {
        "variant": "cupf-cpu-klu",
        "backend": "cpu",
        "compute": "fp64",
        "cpu_linear_solver": "klu",
        "cuda_jacobian": "edge",
        "cuda_linear_solver": "cudss",
        "linear_solver": "klu",
        "jacobian": "native_fixed_pattern",
        "build_key": "cpu",
        "requires_gpu": False,
    },
    {
        "variant": "cupf-cpu-umfpack",
        "backend": "cpu",
        "compute": "fp64",
        "cpu_linear_solver": "umfpack",
        "cuda_jacobian": "edge",
        "cuda_linear_solver": "cudss",
        "linear_solver": "umfpack",
        "jacobian": "native_fixed_pattern",
        "build_key": "cpu",
        "requires_gpu": False,
    },
    {
        "variant": "cupf-fp64-cudss",
        "backend": "cuda",
        "compute": "fp64",
        "cpu_linear_solver": "klu",
        "cuda_jacobian": "edge",
        "cuda_linear_solver": "cudss",
        "linear_solver": "cudss",
        "jacobian": "edge",
        "build_key": "gpu",
        "requires_gpu": True,
    },
    {
        "variant": "cupf-fp64-cudss-edge-atomic",
        "backend": "cuda",
        "compute": "fp64",
        "cpu_linear_solver": "klu",
        "cuda_jacobian": "edge_atomic",
        "cuda_linear_solver": "cudss",
        "linear_solver": "cudss",
        "jacobian": "edge_atomic",
        "build_key": "gpu",
        "requires_gpu": True,
    },
    {
        "variant": "cupf-fp64-cudss-vertex-warp",
        "backend": "cuda",
        "compute": "fp64",
        "cpu_linear_solver": "klu",
        "cuda_jacobian": "vertex_warp",
        "cuda_linear_solver": "cudss",
        "linear_solver": "cudss",
        "jacobian": "vertex_warp",
        "build_key": "gpu",
        "requires_gpu": True,
    },
    {
        "variant": "cupf-mixed-cudss",
        "backend": "cuda",
        "compute": "mixed",
        "cpu_linear_solver": "klu",
        "cuda_jacobian": "edge",
        "cuda_linear_solver": "cudss",
        "linear_solver": "cudss",
        "jacobian": "edge",
        "build_key": "gpu",
        "requires_gpu": True,
    },
    {
        "variant": "cupf-fp64-custom",
        "backend": "cuda",
        "compute": "fp64",
        "cpu_linear_solver": "klu",
        "cuda_jacobian": "edge",
        "cuda_linear_solver": "custom",
        "linear_solver": "custom",
        "jacobian": "edge",
        "build_key": "gpu-custom",
        "requires_gpu": True,
    },
]

DIAGNOSTIC_CUPF_VARIANTS: list[dict[str, Any]] = [
    {
        "variant": "cupf-fp32-cudss",
        "backend": "cuda",
        "compute": "fp32",
        "cpu_linear_solver": "klu",
        "cuda_jacobian": "edge",
        "cuda_linear_solver": "cudss",
        "linear_solver": "cudss",
        "jacobian": "edge",
        "build_key": "gpu",
        "requires_gpu": True,
    },
]


def all_cupf_variants(include_diagnostic_fp32: bool = False) -> list[dict[str, Any]]:
    variants = [dict(v) for v in BASE_CUPF_VARIANTS]
    if include_diagnostic_fp32:
        variants.extend(dict(v) for v in DIAGNOSTIC_CUPF_VARIANTS)
    return variants


def with_entrypoint(variant: dict[str, Any], entrypoint: str) -> dict[str, Any]:
    out = dict(variant)
    out["entrypoint"] = entrypoint
    out["variant"] = f"{variant['variant']}-{entrypoint}"
    return out


def variants_for_entrypoint(include_diagnostic_fp32: bool, entrypoint: str) -> list[dict[str, Any]]:
    return [with_entrypoint(v, entrypoint) for v in all_cupf_variants(include_diagnostic_fp32)]


def filter_variants(variants: list[dict[str, Any]], requested: list[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return variants
    wanted = set(requested)
    return [
        variant
        for variant in variants
        if variant["variant"] in wanted
        or variant["variant"].removesuffix(f"-{variant.get('entrypoint', '')}") in wanted
    ]


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def find_cupf_module(build_dir: Path) -> Path | None:
    if not build_dir.exists():
        return None
    for module in build_dir.rglob("_cupf*.so"):
        return module.parent
    return None


def find_cpp_executable(build_dir: Path) -> Path | None:
    if not build_dir.exists():
        return None
    for executable in build_dir.rglob("cupf_cpp_evaluate"):
        if executable.is_file():
            return executable
    return None
