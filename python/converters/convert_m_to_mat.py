from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
from matpowercaseframes import CaseFrames


MATPOWER_M_ROOT = Path("/datasets/pglib-opf")
DEFAULT_OUTPUT_ROOT = Path("/workspace/datasets/pf_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MATPOWER .m files to SciPy/PYPOWER .mat files.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases", nargs="*", help="Optional .m filenames or stems under /datasets/pglib-opf.")
    return parser.parse_args()


def normalize_mpc(ppc: dict) -> dict:
    normalized = {}
    for key, value in ppc.items():
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)
        normalized[key] = value
    return normalized


def input_paths(cases: list[str] | None) -> list[Path]:
    if not cases:
        return sorted(MATPOWER_M_ROOT.glob("*.m"))

    paths = []
    for case_name in cases:
        filename = case_name if case_name.endswith(".m") else f"{case_name}.m"
        path = MATPOWER_M_ROOT / filename
        if not path.exists():
            raise FileNotFoundError(f"MATPOWER .m case not found: {path}")
        paths.append(path)
    return paths


def convert_case(input_path: Path, output_root: Path) -> Path:
    case_frames = CaseFrames(str(input_path))
    ppc = normalize_mpc(case_frames.to_mpc())
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{input_path.stem}.mat"
    scipy.io.savemat(output_path, {"mpc": ppc})
    return output_path


def main() -> None:
    args = parse_args()
    inputs = input_paths(args.cases)

    success = 0
    failures = 0
    for input_path in inputs:
        try:
            output_path = convert_case(input_path, args.output_root)
            success += 1
            print(f"[OK] {input_path.name} -> {output_path}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {input_path}: {exc}")

    print(f"Processed={success + failures} Success={success} Failures={failures}")


if __name__ == "__main__":
    main()
