#!/usr/bin/env python3
"""Summarize multifrontal front-size dumps for the 2606012 lab meeting."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
FRONT_DIR = ROOT / "data" / "fronts"
OUT_DIR = ROOT / "data"

CASES = [
    ("1xxx", "case1197", "case1197.csv"),
    ("2xxx", "case_ACTIVSg2000", "case_ACTIVSg2000.csv"),
    ("3xxx", "case3012wp", "case3012wp.csv"),
    ("6xxx", "case6468rte", "case6468rte.csv"),
    ("8xxx", "case8387pegase", "case8387pegase.csv"),
    ("10K", "case_ACTIVSg10k", "case_ACTIVSg10k.csv"),
    ("13K", "case13659pegase", "case13659pegase.csv"),
    ("25K", "case_ACTIVSg25k", "case_ACTIVSg25k.csv"),
    ("70K", "case_ACTIVSg70k", "case_ACTIVSg70k.csv"),
    ("usa", "case_SyntheticUSA", "case_SyntheticUSA.csv"),
]

BUCKETS = [
    ("<=8", 0, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-48", 33, 48),
    ("49-64", 49, 64),
    ("65-80", 65, 80),
    ("81-96", 81, 96),
    ("97-128", 97, 128),
    ("129-192", 129, 192),
    ("193-256", 193, 256),
    (">256", 257, None),
]


def read_rows(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: int(v) for k, v in row.items()})
    return rows


def quantile(values: Iterable[int], q: float) -> int:
    xs = sorted(values)
    if not xs:
        return 0
    idx = round((len(xs) - 1) * q)
    return xs[idx]


def average(values: Iterable[int]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


def bucket_of(fsz: int) -> str:
    for label, lo, hi in BUCKETS:
        if fsz >= lo and (hi is None or fsz <= hi):
            return label
    raise ValueError(f"unbucketed fsz={fsz}")


def tier_of(fsz: int) -> str:
    if fsz <= 32:
        return "small"
    if fsz <= 128:
        return "mid"
    return "big"


def pct(count: int, total: int) -> str:
    return f"{(100.0 * count / total) if total else 0.0:.2f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_case(group: str, case: str, rows: list[dict[str, int]]) -> dict[str, object]:
    n = len(rows)
    fsz = [r["fsz"] for r in rows]
    nc = [r["nc"] for r in rows]
    uc = [r["uc"] for r in rows]
    tiers = Counter(tier_of(v) for v in fsz)
    levels = {r["level"] for r in rows}
    return {
        "case_group": group,
        "case": case,
        "fronts": n,
        "levels": len(levels),
        "fsz_min": min(fsz) if fsz else 0,
        "fsz_p50": quantile(fsz, 0.50),
        "fsz_p90": quantile(fsz, 0.90),
        "fsz_p95": quantile(fsz, 0.95),
        "fsz_p99": quantile(fsz, 0.99),
        "fsz_max": max(fsz) if fsz else 0,
        "fsz_avg": f"{average(fsz):.2f}",
        "nc_max": max(nc) if nc else 0,
        "uc_max": max(uc) if uc else 0,
        "small_cnt": tiers["small"],
        "mid_cnt": tiers["mid"],
        "big_cnt": tiers["big"],
        "small_pct": pct(tiers["small"], n),
        "mid_pct": pct(tiers["mid"], n),
        "big_pct": pct(tiers["big"], n),
        "sum_fsz2": sum(v * v for v in fsz),
        "sum_uc2": sum(v * v for v in uc),
        "sum_trailing_work": sum(r["uc"] * r["uc"] * r["nc"] for r in rows),
    }


def summarize_level(group: str, case: str, level: int, rows: list[dict[str, int]]) -> dict[str, object]:
    n = len(rows)
    fsz = [r["fsz"] for r in rows]
    nc = [r["nc"] for r in rows]
    uc = [r["uc"] for r in rows]
    tiers = Counter(tier_of(v) for v in fsz)
    buckets = Counter(bucket_of(v) for v in fsz)
    dominant_bucket = buckets.most_common(1)[0][0] if buckets else ""
    return {
        "case_group": group,
        "case": case,
        "level": level,
        "fronts": n,
        "fsz_min": min(fsz) if fsz else 0,
        "fsz_p50": quantile(fsz, 0.50),
        "fsz_p90": quantile(fsz, 0.90),
        "fsz_p99": quantile(fsz, 0.99),
        "fsz_max": max(fsz) if fsz else 0,
        "fsz_avg": f"{average(fsz):.2f}",
        "nc_max": max(nc) if nc else 0,
        "uc_max": max(uc) if uc else 0,
        "small_cnt": tiers["small"],
        "mid_cnt": tiers["mid"],
        "big_cnt": tiers["big"],
        "dominant_bucket": dominant_bucket,
        "sum_fsz2": sum(v * v for v in fsz),
        "sum_trailing_work": sum(r["uc"] * r["uc"] * r["nc"] for r in rows),
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def write_markdown(case_rows: list[dict[str, object]], hist_rows: list[dict[str, object]]) -> None:
    summary_rows = [
        [
            r["case_group"],
            r["case"],
            r["fronts"],
            r["levels"],
            r["fsz_p50"],
            r["fsz_p90"],
            r["fsz_p99"],
            r["fsz_max"],
            f'{r["small_pct"]}%',
            f'{r["mid_pct"]}%',
            f'{r["big_pct"]}%',
        ]
        for r in case_rows
    ]

    hist_by_case: dict[str, dict[str, str]] = defaultdict(dict)
    for row in hist_rows:
        hist_by_case[str(row["case"])][str(row["bucket"])] = f'{row["pct"]}%'

    hist_table_rows = []
    for _, case, _ in CASES:
        hist_table_rows.append([case] + [hist_by_case[case].get(label, "0.00%") for label, _, _ in BUCKETS])

    text = [
        "# 2606012 Lab Meeting - Front Size Tracker",
        "",
        "This page is generated by `scripts/summarize_fronts.py` from the raw front dumps in `data/fronts/`.",
        "The target set is one representative case from each requested size band: 1xxx, 2xxx, 3xxx, 6xxx, 8xxx, 10K, 13K, 25K, 70K, and usa.",
        "",
        "## Case-Level Summary",
        "",
        markdown_table(
            ["group", "case", "fronts", "levels", "p50", "p90", "p99", "max", "small", "mid", "big"],
            summary_rows,
        ),
        "",
        "Tier definition: small `fsz <= 32`, mid `33 <= fsz <= 128`, big `fsz > 128`.",
        "",
        "## Front-Size Bucket Percentages",
        "",
        markdown_table(["case"] + [label for label, _, _ in BUCKETS], hist_table_rows),
        "",
        "## Data Products",
        "",
        "- `data/case_front_summary.csv`: one row per case.",
        "- `data/front_size_histogram.csv`: front-size histogram by case.",
        "- `data/level_front_summary.csv`: one row per `(case, level)`.",
        "- `data/level_front_size_histogram.csv`: front-size histogram by `(case, level)`.",
        "",
        "Use the level-level CSVs to track where small/mid/big fronts appear in the etree and where underfilled upper levels are likely.",
        "",
    ]
    (ROOT / "SUMMARY.md").write_text("\n".join(text))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    case_summary: list[dict[str, object]] = []
    case_hist: list[dict[str, object]] = []
    level_summary: list[dict[str, object]] = []
    level_hist: list[dict[str, object]] = []

    for group, case, filename in CASES:
        path = FRONT_DIR / filename
        rows = read_rows(path)
        case_summary.append(summarize_case(group, case, rows))

        by_bucket = Counter(bucket_of(r["fsz"]) for r in rows)
        for label, _, _ in BUCKETS:
            count = by_bucket[label]
            case_hist.append({
                "case_group": group,
                "case": case,
                "bucket": label,
                "count": count,
                "pct": pct(count, len(rows)),
            })

        by_level: dict[int, list[dict[str, int]]] = defaultdict(list)
        for row in rows:
            by_level[row["level"]].append(row)

        for level in sorted(by_level):
            level_rows = by_level[level]
            level_summary.append(summarize_level(group, case, level, level_rows))
            level_buckets = Counter(bucket_of(r["fsz"]) for r in level_rows)
            for label, _, _ in BUCKETS:
                count = level_buckets[label]
                if count == 0:
                    continue
                level_hist.append({
                    "case_group": group,
                    "case": case,
                    "level": level,
                    "bucket": label,
                    "count": count,
                    "pct": pct(count, len(level_rows)),
                })

    write_csv(
        OUT_DIR / "case_front_summary.csv",
        [
            "case_group", "case", "fronts", "levels", "fsz_min", "fsz_p50", "fsz_p90",
            "fsz_p95", "fsz_p99", "fsz_max", "fsz_avg", "nc_max", "uc_max", "small_cnt",
            "mid_cnt", "big_cnt", "small_pct", "mid_pct", "big_pct", "sum_fsz2",
            "sum_uc2", "sum_trailing_work",
        ],
        case_summary,
    )
    write_csv(
        OUT_DIR / "front_size_histogram.csv",
        ["case_group", "case", "bucket", "count", "pct"],
        case_hist,
    )
    write_csv(
        OUT_DIR / "level_front_summary.csv",
        [
            "case_group", "case", "level", "fronts", "fsz_min", "fsz_p50", "fsz_p90",
            "fsz_p99", "fsz_max", "fsz_avg", "nc_max", "uc_max", "small_cnt",
            "mid_cnt", "big_cnt", "dominant_bucket", "sum_fsz2", "sum_trailing_work",
        ],
        level_summary,
    )
    write_csv(
        OUT_DIR / "level_front_size_histogram.csv",
        ["case_group", "case", "level", "bucket", "count", "pct"],
        level_hist,
    )
    write_markdown(case_summary, case_hist)


if __name__ == "__main__":
    main()
