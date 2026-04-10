from __future__ import annotations

import argparse
from pathlib import Path

from .common import CUPF_DUMP_ROOT, TARGET_CASES, preprocess_case, save_cupf_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .mat cases to cuPF dump_case_loader input format.")
    parser.add_argument("--output-root", type=Path, default=CUPF_DUMP_ROOT)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=list(TARGET_CASES),
        help="Case names such as 118_ieee or 2746wop_k. Defaults to benchmark target cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for case_name in args.cases:
        case_data = preprocess_case(case_name)
        output_dir = save_cupf_dump(case_data, output_root=args.output_root)
        print(f"[OK] {case_data.case_stem} -> {output_dir}")


if __name__ == "__main__":
    main()
