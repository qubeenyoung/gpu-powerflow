from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import scipy.io
import matpowercaseframes.reader as mpc_reader
import matpowercaseframes.utils as mpc_utils
from matpowercaseframes import CaseFrames


DEFAULT_INPUT_ROOT = Path("/workspace/datasets/pglib-opf")
DEFAULT_OUTPUT_ROOT = Path("/workspace/datasets/pglib-opf/pf_dataset")


def _robust_number_or_string(value: str) -> int | float | str:
    try:
        float_value = float(value)
    except ValueError:
        expression = value.replace("^", "**")
        if not re.fullmatch(r"[0-9eE+\-*/(). _sqrtnanifINFNaN]+", expression):
            return value
        try:
            float_value = float(
                eval(
                    expression,
                    {"__builtins__": {}},
                    {
                        "sqrt": math.sqrt,
                        "Inf": math.inf,
                        "inf": math.inf,
                        "NaN": math.nan,
                        "nan": math.nan,
                    },
                )
            )
        except Exception:
            return value
    if not np.isfinite(float_value):
        return float_value
    int_value = int(float_value)
    return int_value if int_value == float_value else float_value


mpc_reader.int_else_float_except_string = _robust_number_or_string
mpc_utils.int_else_float_except_string = _robust_number_or_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MATPOWER .m files to SciPy/PYPOWER .mat files.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases", nargs="*", help="Optional .m filenames or stems under input-root.")
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


def apply_matpower_postprocessing(ppc: dict, input_path: Path) -> dict:
    text = input_path.read_text(encoding="cp1252", errors="replace")

    if "bus" in ppc:
        ppc["bus"] = np.asarray(ppc["bus"], dtype=float)
    if "branch" in ppc:
        ppc["branch"] = np.asarray(ppc["branch"], dtype=float)
    if "gen" in ppc:
        ppc["gen"] = np.asarray(ppc["gen"], dtype=float)
    if "gencost" in ppc and ppc["gencost"] is not None:
        ppc["gencost"] = np.asarray(ppc["gencost"], dtype=float)

    # Common MATPOWER distribution-case epilogue:
    #   Vbase = mpc.bus(1, BASE_KV) * 1e3;
    #   Sbase = mpc.baseMVA * 1e6;
    #   mpc.branch(:, [BR_R BR_X]) = ... / (Vbase^2 / Sbase);
    if "mpc.branch(:, [BR_R BR_X])" in text:
        bus = np.asarray(ppc["bus"], dtype=float)
        branch = np.asarray(ppc["branch"], dtype=float)
        vbase = bus[0, 9] * 1e3
        sbase = float(ppc["baseMVA"]) * 1e6
        branch[:, [2, 3]] = branch[:, [2, 3]] / (vbase**2 / sbase)
        ppc["branch"] = branch

    # Common MATPOWER distribution-case epilogue:
    #   mpc.bus(:, [PD, QD]) = mpc.bus(:, [PD, QD]) / 1e3;
    if "mpc.bus(:, [PD, QD])" in text and "/ 1e3" in text:
        bus = np.asarray(ppc["bus"], dtype=float)
        bus[:, [2, 3]] = bus[:, [2, 3]] / 1e3
        ppc["bus"] = bus

    # case141 has an additional power-factor conversion after the kW to MW conversion.
    pf_match = re.search(r"pf\s*=\s*([0-9.]+)\s*;", text)
    if pf_match and "sin(acos(pf))" in text:
        pf = float(pf_match.group(1))
        bus = np.asarray(ppc["bus"], dtype=float)
        bus[:, 3] = bus[:, 2] * math.sin(math.acos(pf))
        bus[:, 2] = bus[:, 2] * pf
        ppc["bus"] = bus

    return ppc


def input_paths(input_root: Path, cases: list[str] | None) -> list[Path]:
    if not cases:
        return sorted(input_root.glob("*.m"))

    paths = []
    for case_name in cases:
        raw_path = Path(case_name)
        if raw_path.exists():
            paths.append(raw_path)
            continue
        filename = case_name if case_name.endswith(".m") else f"{case_name}.m"
        path = input_root / filename
        if not path.exists():
            raise FileNotFoundError(f"MATPOWER .m case not found: {path}")
        paths.append(path)
    return paths


def convert_case(input_path: Path, output_root: Path) -> Path:
    try:
        case_frames = CaseFrames(str(input_path), update_index=False)
    except UnicodeDecodeError:
        text = input_path.read_text(encoding="cp1252")
        with tempfile.NamedTemporaryFile("w", suffix=".m", encoding="utf-8", delete=False) as tmp_file:
            tmp_file.write(text)
            tmp_path = Path(tmp_file.name)
        try:
            case_frames = CaseFrames(str(tmp_path), update_index=False)
        finally:
            tmp_path.unlink(missing_ok=True)
    ppc = normalize_mpc(case_frames.to_mpc())
    ppc = apply_matpower_postprocessing(ppc, input_path)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{input_path.stem}.mat"
    scipy.io.savemat(output_path, {"mpc": ppc})
    return output_path


def main() -> None:
    args = parse_args()
    inputs = input_paths(args.input_root, args.cases)

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
