#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "common" / "python"))

from benchmark_json import parse_common_args, unavailable_result, write_result


def main() -> None:
    args = parse_common_args()
    log_path = EXP_ROOT / "logs" / "install_strumpack.log"
    notes = "STRUMPACK sparse direct candidate is unavailable until a CUDA-enabled STRUMPACK build is present."
    if log_path.exists():
        notes += " Install log: " + str(log_path)
    result = unavailable_result(
        solver_name="STRUMPACK",
        build_status="unavailable",
        args=args,
        notes=notes,
        gpu_resident_after_initial_load="unavailable",
        extra={
            "solver_type": "sparse direct / multifrontal",
            "mpi_openmp_requirement": "MPI and OpenMP dependencies required for representative builds",
            "launch_command": "mpirun -np <ranks> strumpack_benchmark ...",
        },
    )
    write_result(args.out, result)


if __name__ == "__main__":
    main()
