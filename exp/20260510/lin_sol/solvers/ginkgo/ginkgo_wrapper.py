#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "common" / "python"))

from benchmark_json import parse_common_args, unavailable_result, write_result


def main() -> None:
    args = parse_common_args()
    log_path = EXP_ROOT / "logs" / "install_ginkgo.log"
    notes = "Ginkgo CUDA benchmark wrapper placeholder. Ginkgo was not found as an installed CUDA-enabled library in this environment."
    if log_path.exists():
        notes += " Install log: " + str(log_path)
    result = unavailable_result(
        solver_name="Ginkgo",
        build_status="unavailable",
        args=args,
        notes=notes,
        gpu_resident_after_initial_load="unavailable",
        extra={
            "solver_configuration": "GMRES/BiCGSTAB with standard preconditioners planned after CUDA-enabled Ginkgo build succeeds",
            "preconditioner": "unavailable",
            "actual_iterations": -1,
            "final_residual": None,
        },
    )
    write_result(args.out, result)


if __name__ == "__main__":
    main()
