#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path("/workspace")
ANALYSIS_ROOT = ROOT / "exp/20260416/analysis"
DATASET_ROOT = ROOT / "datasets/matpower8.1/cuPF_datasets"
DEFAULT_TARGETS = (100, 500, 1000, 5000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump cuDSS reorder permutation vectors and elimination trees for representative MATPOWER cases."
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--binary", type=Path, default=ANALYSIS_ROOT / "build/dump_cudss_reorder_data")
    parser.add_argument("--output-root", type=Path, default=ANALYSIS_ROOT / "dumps")
    parser.add_argument("--targets", nargs="*", type=int, default=list(DEFAULT_TARGETS))
    parser.add_argument("--reordering-alg", default="DEFAULT", choices=("DEFAULT", "ALG_1", "ALG_2"))
    parser.add_argument("--nd-nlevels", default="AUTO")
    return parser.parse_args()


def load_cases(dataset_root: Path) -> list[dict]:
    cases: list[dict] = []
    for meta_path in sorted(dataset_root.glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["case"] = meta_path.parent.name
        meta["case_dir"] = str(meta_path.parent)
        cases.append(meta)
    if not cases:
        raise RuntimeError(f"No cuPF cases found under {dataset_root}")
    return cases


def select_cases(cases: list[dict], targets: list[int]) -> list[tuple[int, dict]]:
    selected: list[tuple[int, dict]] = []
    for target in targets:
        above = [case for case in cases if int(case["n_bus"]) >= target]
        if above:
            case = min(above, key=lambda c: (int(c["n_bus"]), c["case"]))
        else:
            case = max(cases, key=lambda c: (int(c["n_bus"]), c["case"]))
        selected.append((target, case))
    return selected


def run_one(args: argparse.Namespace, target: int, case: dict) -> dict:
    case_name = case["case"]
    output_dir = args.output_root / f"target_{target}" / case_name
    cmd = [
        str(args.binary),
        "--case-dir",
        case["case_dir"],
        "--output-dir",
        str(output_dir),
        "--case-label",
        case_name,
        "--target-bus-label",
        str(target),
        "--reordering-alg",
        args.reordering_alg,
        "--nd-nlevels",
        str(args.nd_nlevels),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"dump failed for target={target} case={case_name}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "target_bus": target,
        "case": case_name,
        "n_bus": metadata["n_bus"],
        "n_pv": metadata["n_pv"],
        "n_pq": metadata["n_pq"],
        "jacobian_dim": metadata["jacobian_dim"],
        "jacobian_nnz": metadata["jacobian_nnz"],
        "perm_reorder_row_len": metadata["perm_reorder_row_len"],
        "perm_reorder_col_len": metadata["perm_reorder_col_len"],
        "elimination_tree_len": metadata["elimination_tree_len"],
        "output_dir": str(output_dir),
    }


def write_manifest(args: argparse.Namespace, selected: list[tuple[int, dict]], rows: list[dict]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_root": str(args.dataset_root),
        "binary": str(args.binary),
        "reordering_alg": args.reordering_alg,
        "nd_nlevels": args.nd_nlevels,
        "selection_policy": "smallest n_bus >= target_bus",
        "selected_cases": [
            {
                "target_bus": target,
                "case": case["case"],
                "n_bus": int(case["n_bus"]),
                "case_dir": case["case_dir"],
            }
            for target, case in selected
        ],
        "outputs": rows,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cases = load_cases(args.dataset_root)
    selected = select_cases(cases, args.targets)
    rows = [run_one(args, target, case) for target, case in selected]
    write_manifest(args, selected, rows)
    for row in rows:
        print(
            f"{row['target_bus']}: {row['case']} "
            f"n_bus={row['n_bus']} dim={row['jacobian_dim']} "
            f"etree={row['elimination_tree_len']} -> {row['output_dir']}"
        )


if __name__ == "__main__":
    main()
