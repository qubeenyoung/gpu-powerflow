#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
NCU_DIR = EXP / "results" / "ncu"
CASES = EXP / "cases.csv"
OUT = NCU_DIR / "ncu_phase_kernel_summary.csv"


def load_case_meta() -> dict[str, dict[str, str]]:
    with CASES.open(newline="", encoding="utf-8") as fh:
        return {row["case"]: row for row in csv.DictReader(fh)}


def parse_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith('"'):
            continue
        parsed = next(csv.reader([line]))
        if parsed and parsed[0] == "Process ID":
            header = parsed
            continue
        if "header" not in locals():
            continue
        rows.append(dict(zip(header, parsed)))
    return rows


def short_kernel(name: str) -> str:
    name = name.replace("void cudss::", "")
    name = name.split("<", 1)[0]
    name = name.split("(", 1)[0]
    return name


def file_case_phase(path: Path) -> tuple[str, str]:
    match = re.match(r"(.+)_(factorize|solve)0_basic\.csv$", path.name)
    if not match:
        return path.stem, ""
    return match.group(1), match.group(2)


def main() -> None:
    meta = load_case_meta()
    fields = [
        "case",
        "tier",
        "n",
        "nnz",
        "phase",
        "kernel",
        "invocations",
        "grid_size",
        "block_size",
        "duration_ns",
        "achieved_occupancy_pct",
        "sm_throughput_pct",
        "dram_throughput_pct",
        "waves_per_sm",
    ]
    out_rows: list[dict[str, object]] = []
    for path in sorted(NCU_DIR.glob("*_basic.csv")):
        case, phase = file_case_phase(path)
        if phase not in {"factorize", "solve"}:
            continue
        by_kernel: dict[str, dict[str, str]] = {}
        for row in parse_csv(path):
            kernel = row["Kernel Name"]
            entry = by_kernel.setdefault(
                kernel,
                {
                    "case": case,
                    "tier": meta.get(case, {}).get("tier", ""),
                    "n": meta.get(case, {}).get("n", ""),
                    "nnz": meta.get(case, {}).get("nnz", ""),
                    "phase": phase,
                    "kernel": short_kernel(kernel),
                    "invocations": row["Invocations"],
                    "grid_size": row["Grid Size"],
                    "block_size": row["Block Size"],
                },
            )
            metric = row["Metric Name"]
            if metric == "Duration":
                entry["duration_ns"] = row["Average"]
            elif metric == "Achieved Occupancy":
                entry["achieved_occupancy_pct"] = row["Average"]
            elif metric == "Compute (SM) Throughput":
                entry["sm_throughput_pct"] = row["Average"]
            elif metric == "DRAM Throughput":
                entry["dram_throughput_pct"] = row["Average"]
            elif metric == "Waves Per SM":
                entry["waves_per_sm"] = row["Average"]
        for entry in by_kernel.values():
            out_rows.append({field: entry.get(field, "") for field in fields})

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {OUT}")

    agg = NCU_DIR / "ncu_phase_summary.csv"
    agg_fields = [
        "case",
        "tier",
        "n",
        "nnz",
        "phase",
        "kernel_rows",
        "invocations_total",
        "duration_ms_sum",
        "max_duration_kernel",
        "max_duration_ns",
        "mean_achieved_occupancy_pct",
        "mean_sm_throughput_pct",
        "mean_dram_throughput_pct",
        "max_grid_size",
    ]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in out_rows:
        grouped.setdefault((str(row["case"]), str(row["phase"])), []).append(row)
    with agg.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=agg_fields)
        writer.writeheader()
        for (case, phase), items in sorted(grouped.items(), key=lambda kv: (int(kv[1][0]["n"]), kv[0][1])):
            durations = [float(item["duration_ns"]) for item in items if item["duration_ns"] != ""]
            occ = [float(item["achieved_occupancy_pct"]) for item in items if item["achieved_occupancy_pct"] != ""]
            sm = [float(item["sm_throughput_pct"]) for item in items if item["sm_throughput_pct"] != ""]
            dram = [float(item["dram_throughput_pct"]) for item in items if item["dram_throughput_pct"] != ""]
            max_item = max(items, key=lambda item: float(item["duration_ns"] or 0.0))
            grids = []
            for item in items:
                match = re.search(r"\((\d+),", str(item["grid_size"]))
                if match:
                    grids.append(int(match.group(1)))
            writer.writerow(
                {
                    "case": case,
                    "tier": items[0]["tier"],
                    "n": items[0]["n"],
                    "nnz": items[0]["nnz"],
                    "phase": phase,
                    "kernel_rows": len(items),
                    "invocations_total": sum(int(item["invocations"] or 0) for item in items),
                    "duration_ms_sum": sum(durations) / 1e6,
                    "max_duration_kernel": max_item["kernel"],
                    "max_duration_ns": max_item["duration_ns"],
                    "mean_achieved_occupancy_pct": sum(occ) / len(occ) if occ else "",
                    "mean_sm_throughput_pct": sum(sm) / len(sm) if sm else "",
                    "mean_dram_throughput_pct": sum(dram) / len(dram) if dram else "",
                    "max_grid_size": max(grids) if grids else "",
                }
            )
    print(f"wrote {agg}")


if __name__ == "__main__":
    main()
