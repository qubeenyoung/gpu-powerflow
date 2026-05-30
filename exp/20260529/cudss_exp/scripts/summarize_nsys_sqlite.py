#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
NSYS_DIR = EXP / "results" / "nsys"
CASES = EXP / "cases.csv"
OUT = NSYS_DIR / "nsys_phase_summary.csv"


def load_case_meta() -> dict[str, dict[str, str]]:
    with CASES.open(newline="", encoding="utf-8") as fh:
        return {row["case"]: row for row in csv.DictReader(fh)}


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return (
        con.execute(
            "select count(*) from sqlite_master where type='table' and name=?", (table,)
        ).fetchone()[0]
        > 0
    )


def sum_inside(con: sqlite3.Connection, table: str, start: int, end: int) -> tuple[int, int]:
    if not table_exists(con, table):
        return 0, 0
    count, total = con.execute(
        f"select count(*), coalesce(sum(end-start), 0) from {table} where start >= ? and end <= ?",
        (start, end),
    ).fetchone()
    return int(count), int(total)


def runtime_breakdown(con: sqlite3.Connection, start: int, end: int) -> tuple[int, int, int, int]:
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return 0, 0, 0, 0
    rows = con.execute(
        """
        select coalesce(s.value, ''), r.end - r.start
        from CUPTI_ACTIVITY_KIND_RUNTIME r
        left join StringIds s on s.id = r.nameId
        where r.start >= ? and r.end <= ?
        """,
        (start, end),
    ).fetchall()
    api_total = sum(int(duration) for _, duration in rows)
    sync_total = sum(int(duration) for name, duration in rows if "Synchronize" in name)
    execute_total = sum(int(duration) for name, duration in rows if "cudss" in name.lower())
    return len(rows), api_total, sync_total, execute_total


def classify_phase(text: str) -> str | None:
    if text == "cudss_analysis":
        return "analysis"
    if text.startswith("cudss_factorize_"):
        return "factorize"
    if text.startswith("cudss_solve_"):
        return "solve"
    return None


def case_from_sqlite(path: Path) -> str:
    match = re.match(r"(.+)_repeat\d+\.sqlite$", path.name)
    if not match:
        return path.stem
    return match.group(1)


def summarize_file(path: Path, meta: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    case = case_from_sqlite(path)
    con = sqlite3.connect(path)
    rows: list[dict[str, object]] = []
    ranges = con.execute(
        """
        select text, start, end
        from NVTX_EVENTS
        where text in ('cudss_analysis')
           or text like 'cudss_factorize_%'
           or text like 'cudss_solve_%'
        order by start
        """
    ).fetchall()
    for text, start, end in ranges:
        phase = classify_phase(text)
        if phase is None or end is None:
            continue
        runtime_count, api_ns, sync_api_ns, cudss_api_ns = runtime_breakdown(con, start, end)
        kernel_count, kernel_ns = sum_inside(con, "CUPTI_ACTIVITY_KIND_KERNEL", start, end)
        memcpy_count, memcpy_ns = sum_inside(con, "CUPTI_ACTIVITY_KIND_MEMCPY", start, end)
        memset_count, memset_ns = sum_inside(con, "CUPTI_ACTIVITY_KIND_MEMSET", start, end)
        sync_count, sync_activity_ns = sum_inside(
            con, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", start, end
        )
        wall_ns = int(end) - int(start)
        rows.append(
            {
                "case": case,
                "tier": meta.get(case, {}).get("tier", ""),
                "n": meta.get(case, {}).get("n", ""),
                "nnz": meta.get(case, {}).get("nnz", ""),
                "range": text,
                "phase": phase,
                "wall_ms": wall_ns / 1e6,
                "runtime_api_ms": api_ns / 1e6,
                "cuda_sync_api_ms": sync_api_ns / 1e6,
                "cudss_api_ms": cudss_api_ns / 1e6,
                "gpu_kernel_ms": kernel_ns / 1e6,
                "gpu_kernel_count": kernel_count,
                "memcpy_ms": memcpy_ns / 1e6,
                "memcpy_count": memcpy_count,
                "memset_ms": memset_ns / 1e6,
                "memset_count": memset_count,
                "sync_activity_ms": sync_activity_ns / 1e6,
                "sync_activity_count": sync_count,
            }
        )
    con.close()
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    meta = load_case_meta()
    files = sorted(
        path for path in NSYS_DIR.glob("*_repeat*.sqlite") if not path.name.startswith("smoke_")
    )
    rows: list[dict[str, object]] = []
    for path in files:
        rows.extend(summarize_file(path, meta))

    fields = [
        "case",
        "tier",
        "n",
        "nnz",
        "range",
        "phase",
        "wall_ms",
        "runtime_api_ms",
        "cuda_sync_api_ms",
        "cudss_api_ms",
        "gpu_kernel_ms",
        "gpu_kernel_count",
        "memcpy_ms",
        "memcpy_count",
        "memset_ms",
        "memset_count",
        "sync_activity_ms",
        "sync_activity_count",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case"]), str(row["phase"])), []).append(row)
    agg_path = NSYS_DIR / "nsys_phase_summary_agg.csv"
    agg_fields = [
        "case",
        "tier",
        "n",
        "nnz",
        "phase",
        "ranges",
        "wall_ms_mean",
        "runtime_api_ms_mean",
        "cuda_sync_api_ms_mean",
        "cudss_api_ms_mean",
        "gpu_kernel_ms_mean",
        "gpu_kernel_count_mean",
        "memcpy_ms_mean",
        "memcpy_count_mean",
        "memset_ms_mean",
        "memset_count_mean",
        "sync_activity_ms_mean",
        "sync_activity_count_mean",
        "kernel_over_wall",
        "api_over_wall",
        "sync_api_over_wall",
    ]
    with agg_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=agg_fields)
        writer.writeheader()
        for (case, phase), items in sorted(grouped.items(), key=lambda kv: (int(kv[1][0]["n"]), kv[0][1])):
            wall = mean([float(item["wall_ms"]) for item in items])
            kernel = mean([float(item["gpu_kernel_ms"]) for item in items])
            api = mean([float(item["runtime_api_ms"]) for item in items])
            sync_api = mean([float(item["cuda_sync_api_ms"]) for item in items])
            out = {
                "case": case,
                "tier": items[0]["tier"],
                "n": items[0]["n"],
                "nnz": items[0]["nnz"],
                "phase": phase,
                "ranges": len(items),
                "wall_ms_mean": wall,
                "runtime_api_ms_mean": api,
                "cuda_sync_api_ms_mean": sync_api,
                "cudss_api_ms_mean": mean([float(item["cudss_api_ms"]) for item in items]),
                "gpu_kernel_ms_mean": kernel,
                "gpu_kernel_count_mean": mean([float(item["gpu_kernel_count"]) for item in items]),
                "memcpy_ms_mean": mean([float(item["memcpy_ms"]) for item in items]),
                "memcpy_count_mean": mean([float(item["memcpy_count"]) for item in items]),
                "memset_ms_mean": mean([float(item["memset_ms"]) for item in items]),
                "memset_count_mean": mean([float(item["memset_count"]) for item in items]),
                "sync_activity_ms_mean": mean([float(item["sync_activity_ms"]) for item in items]),
                "sync_activity_count_mean": mean([float(item["sync_activity_count"]) for item in items]),
                "kernel_over_wall": kernel / wall if wall else "",
                "api_over_wall": api / wall if wall else "",
                "sync_api_over_wall": sync_api / wall if wall else "",
            }
            writer.writerow(out)
    print(f"wrote {OUT}")
    print(f"wrote {agg_path}")


if __name__ == "__main__":
    main()
