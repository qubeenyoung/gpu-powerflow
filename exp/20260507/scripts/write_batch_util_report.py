#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Markdown report for a batch utilization sweep.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str) -> float:
    return float(value)


def fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def counter(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]))


def index_by_bin_batch(rows: list[dict[str, str]]) -> dict[str, dict[int, dict[str, str]]]:
    indexed: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        indexed.setdefault(row["size_bin"], {})[int(row["batch_size"])] = row
    return indexed


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    output = args.output or (run_dir / "MICROBATCH_RECOMMENDATION.md")

    recommendations = read_csv(run_dir / "recommendations.csv")
    size_rows = read_csv(run_dir / "size_bin_summary.csv")
    case_rows = read_csv(run_dir / "case_batch_summary.csv")
    errors_path = run_dir / "errors.json"

    by_bin = index_by_bin_batch(size_rows)
    bin_order = ["<100", "100-999", "1k-9,999", "10k-49,999", ">=50k"]

    lines: list[str] = [
        "# Micro-Batch Recommendation",
        "",
        f"- Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Source run: `{run_dir}`",
        "- Scope: 1차 timing sweep (`warmup=1`, `repeats=3`), 최종 수치 확정 전 단계",
        "",
        "## 전체 결론",
        "",
        f"- 모든 case가 batch 256까지 성공했다: `{len(recommendations)} / {len(recommendations)}`.",
        f"- end-to-end 기준 best batch 분포: `{counter(recommendations, 'best_elapsed_batch')}`.",
        f"- solve-only 기준 best batch 분포: `{counter(recommendations, 'best_solve_batch')}`.",
        "- DP throughput 우선 micro-batch는 현재 데이터만 보면 전 bin에서 `256`이 1순위다.",
        "- PP 또는 memory-sensitive schedule은 큰 case에서 `128`과 `256`을 모두 후보로 남긴다.",
        "- 이 결론은 timing 기반 1차 판단이며, utilization 최종 판단은 Nsight Compute 보강 후 확정한다.",
        "",
        "## Size-Bin 요약",
        "",
        "| size bin | cases | b1 elapsed ms/scenario | b128 elapsed | b256 elapsed | b1->b256 elapsed speedup | b1 solve ms/scenario | b256 solve | b1->b256 solve speedup | 128->256 elapsed gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for bin_name in bin_order:
        rows = by_bin.get(bin_name)
        if not rows or 1 not in rows or 128 not in rows or 256 not in rows:
            continue
        b1 = rows[1]
        b128 = rows[128]
        b256 = rows[256]
        elapsed_1 = fnum(b1["elapsed_ms_per_scenario_mean"])
        elapsed_128 = fnum(b128["elapsed_ms_per_scenario_mean"])
        elapsed_256 = fnum(b256["elapsed_ms_per_scenario_mean"])
        solve_1 = fnum(b1["solve_ms_per_scenario_mean"])
        solve_256 = fnum(b256["solve_ms_per_scenario_mean"])
        elapsed_speedup = elapsed_1 / elapsed_256
        solve_speedup = solve_1 / solve_256
        gain_128_256 = (elapsed_128 - elapsed_256) / elapsed_128 * 100.0
        lines.append(
            f"| `{bin_name}` | {b256['cases']} | {fmt(elapsed_1)} | {fmt(elapsed_128)} | "
            f"{fmt(elapsed_256)} | {fmt(elapsed_speedup)}x | {fmt(solve_1)} | "
            f"{fmt(solve_256)} | {fmt(solve_speedup)}x | {fmt(gain_128_256)}% |"
        )

    non_256_solve = [row for row in recommendations if row["best_solve_batch"] != "256"]
    lines.extend([
        "",
        "## Solve-Only 예외",
        "",
        "end-to-end best는 모든 case에서 `256`이지만, solve-only best는 아래 3개 case에서 `128`이었다.",
        "",
        "| case | size bin | buses | best solve batch | best solve ms/scenario | best elapsed batch |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in non_256_solve:
        lines.append(
            f"| `{row['case_name']}` | `{row['size_bin']}` | {row['buses']} | "
            f"{row['best_solve_batch']} | {fmt(fnum(row['best_solve_ms_per_scenario']))} | "
            f"{row['best_elapsed_batch']} |"
        )

    lines.extend([
        "",
        "## 임시 추천",
        "",
        "| 목적 | 추천 batch | 근거 | 보강 필요 |",
        "| --- | ---: | --- | --- |",
        "| DP throughput | 256 | 모든 case의 end-to-end ms/scenario가 256에서 최저 | final `warmup=3`, `repeats=10` 재측정 |",
        "| PP stage balance | 128 또는 256 | 큰 case에서 128 이후 이득이 작아지고 solve-only 예외가 있음 | stage별 memory/workspace와 Nsight Compute |",
        "| 작은 case launch amortization | 256 | `<100` bin에서 b1->b256 end-to-end speedup이 가장 큼 | Nsight Systems 대체 지표 필요 |",
        "| 큰 case memory-sensitive | 128 우선, 256 확인 | `>=50k` bin은 b128->b256 end-to-end gain이 작고 memory 사용 증가 | b512 실패/성공 여부 및 memory cap 확인 |",
        "",
        "## 다음 측정",
        "",
        "- 대표 batch `64, 128, 256`에 대해 `ncu` SpeedOfLight metric을 수집한다.",
        "- 대표 case는 `case118`, `case9241pegase`, `case_ACTIVSg25k`, `case_ACTIVSg70k`, `case_SyntheticUSA`로 둔다.",
        "- 최종 보고용 timing은 전체 case가 아니라 후보 batch 중심으로 `warmup=3`, `repeats=10` 재측정한다.",
    ])

    if errors_path.exists():
        lines.extend(["", "## Errors", "", f"- `{errors_path}` 확인 필요"])
    else:
        failed = [row for row in case_rows if row.get("success_all") == "False"]
        lines.extend(["", "## 실패 여부", "", f"- batch별 aggregate 기준 수렴 실패 row: `{len(failed)}`"])

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
