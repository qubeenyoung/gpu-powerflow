from __future__ import annotations

import argparse
from pathlib import Path

from .common import NR_DATASET_ROOT, all_mat_case_names, preprocess_case, save_nr_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .mat cases to legacy nr_dataset format.")
    parser.add_argument("--output-root", type=Path, default=NR_DATASET_ROOT)
    parser.add_argument("--cases", nargs="*", help="Case names such as 118_ieee or 1354_pegase.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_names = args.cases if args.cases else all_mat_case_names()

    for case_name in case_names:
        case_data = preprocess_case(case_name)
        output_dir = save_nr_data(case_data, output_root=args.output_root)
        print(f"[OK] {case_data.case_stem} -> {output_dir}")


if __name__ == "__main__":
    main()
