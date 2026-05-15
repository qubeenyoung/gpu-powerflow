#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS = Path("results")
MODES = [
    ("mr1_bj", "MR1 block-Jacobi"),
    ("mr1_coarse", "MR1 block-Jacobi + coarse merged"),
    ("mr2_coarse", "2D-MR block-Jacobi + coarse separated"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def with_mode(rows: list[dict[str, str]], mode: str, label: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        copied["mode"] = mode
        copied["mode_label"] = label
        out.append(copied)
    return out


def avg_metric(rows: list[dict[str, str]], case_key: str, metric: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        values[(row.get(case_key, ""), row["mode_label"])].append(as_float(row, metric))
    return {key: mean(vals) for key, vals in values.items() if vals}


def main() -> None:
    summary: list[dict[str, str]] = []
    iters: list[dict[str, str]] = []
    shadow: list[dict[str, str]] = []
    timing: list[dict[str, str]] = []
    for mode, label in MODES:
        summary += with_mode(read_rows(RESULTS / f"mr2_compare_summary_{mode}.csv"), mode, label)
        iters += with_mode(read_rows(RESULTS / f"mr2_compare_iters_{mode}.csv"), mode, label)
        shadow += with_mode(read_rows(RESULTS / f"mr2_compare_shadow_{mode}.csv"), mode, label)
        timing += with_mode(read_rows(RESULTS / f"mr2_compare_timing_{mode}.csv"), mode, label)

    write_rows(RESULTS / "mr2_compare_summary.csv", summary)
    write_rows(RESULTS / "mr2_compare_iters.csv", iters)
    write_rows(RESULTS / "mr2_compare_shadow_dx.csv", shadow)
    write_rows(RESULTS / "mr2_compare_timing_breakdown.csv", timing)

    middle_total = avg_metric(
        [row for row in timing if as_float(row, "middle_solver_total_ms") > 0.0],
        "case_name",
        "middle_solver_total_ms",
    )
    dx_ratio = avg_metric(shadow, "case", "dx_norm_ratio")
    dx_cosine = avg_metric(shadow, "case", "dx_cosine")
    mismatch_ratio = avg_metric(shadow, "case", "gmres_nonlinear_ratio_inf")

    cases = sorted({row["case_name"] for row in summary})
    labels = [label for _, label in MODES]
    summary_by_case_mode = {(row["case_name"], row["mode_label"]): row for row in summary}

    lines = [
        "# 2D-MR coarse separated comparison",
        "",
        "## 읽는 법",
        "",
        "- **MR1 block-Jacobi**: local block-Jacobi correction만 사용한다.",
        "- **MR1 + coarse merged**: local correction과 coarse correction을 먼저 더하고 scalar alpha 하나로 최소잔차 보정한다.",
        "- **2D-MR + coarse separated**: local 방향과 coarse 방향에 서로 다른 계수 `a0`, `a1`을 둔 2D 최소잔차 보정이다.",
        "- **중간 step mismatch ratio**: middle solver step 후 mismatch_inf / step 전 mismatch_inf. 작을수록 좋다.",
        "- **dx 크기비**: `||dx_iterative||2 / ||dx_cuDSS||2`. 1에 가까울수록 cuDSS Newton step만큼 크다.",
        "- **dx 방향유사도**: `cos(dx_iterative, dx_cuDSS)`. 1에 가까울수록 방향이 비슷하다.",
        "",
        "## 1. NR 결과",
        "",
        "| 케이스 | 모드 | 수렴 | Newton 반복 | Fallback | Hybrid 시간(s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        for label in labels:
            row = summary_by_case_mode.get((case, label), {})
            lines.append(
                f"| {case} | {label} | {row.get('converged', '')} | "
                f"{row.get('nr_iters', '')} | {row.get('fallback_calls', '')} | "
                f"{as_float(row, 'total_seconds'):.4f} |"
            )

    lines += [
        "",
        "## 2. Middle step quality",
        "",
        "| 케이스 | 모드 | middle solver 평균(ms) | dx 크기비 | dx 방향유사도 | 중간 step mismatch ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        for label in labels:
            lines.append(
                f"| {case} | {label} | {middle_total.get((case, label), 0.0):.4f} | "
                f"{dx_ratio.get((case, label), 0.0):.4g} | "
                f"{dx_cosine.get((case, label), 0.0):.4g} | "
                f"{mismatch_ratio.get((case, label), 0.0):.4g} |"
            )

    lines += [
        "",
        "## 3. 판단",
        "",
        "- 2D-MR가 성공하려면 merged coarse 대비 Fallback 또는 Newton 반복이 줄고, dx 크기비/방향유사도 또는 mismatch ratio가 개선되어야 한다.",
        "- middle solver 평균 시간이 크게 증가하면, NR 반복 감소가 없을 때 총 시간에서 불리하다.",
    ]
    mode_avg_middle = {
        label: mean(
            value for (case, mode_label), value in middle_total.items() if mode_label == label
        )
        for label in labels
        if any(mode_label == label for (case, mode_label) in middle_total)
    }
    mode_avg_dx = {
        label: mean(value for (case, mode_label), value in dx_ratio.items() if mode_label == label)
        for label in labels
        if any(mode_label == label for (case, mode_label) in dx_ratio)
    }
    mode_avg_cos = {
        label: mean(value for (case, mode_label), value in dx_cosine.items() if mode_label == label)
        for label in labels
        if any(mode_label == label for (case, mode_label) in dx_cosine)
    }
    if "2D-MR block-Jacobi + coarse separated" in mode_avg_middle:
        lines += [
            "- 이번 결과에서는 **2D-MR separated가 성공하지 못했다.**",
            "- Fallback 감소가 없고, 여러 케이스에서 Newton 반복이 merged coarse보다 늘었다.",
            f"- 평균 middle solver 시간은 MR1 block-Jacobi `{mode_avg_middle.get('MR1 block-Jacobi', 0.0):.3f} ms`, "
            f"merged coarse `{mode_avg_middle.get('MR1 block-Jacobi + coarse merged', 0.0):.3f} ms`, "
            f"2D-MR separated `{mode_avg_middle.get('2D-MR block-Jacobi + coarse separated', 0.0):.3f} ms`이다.",
            f"- 평균 dx 크기비는 merged coarse `{mode_avg_dx.get('MR1 block-Jacobi + coarse merged', 0.0):.4g}`, "
            f"2D-MR separated `{mode_avg_dx.get('2D-MR block-Jacobi + coarse separated', 0.0):.4g}`이다.",
            f"- 평균 dx 방향유사도는 merged coarse `{mode_avg_cos.get('MR1 block-Jacobi + coarse merged', 0.0):.4g}`, "
            f"2D-MR separated `{mode_avg_cos.get('2D-MR block-Jacobi + coarse separated', 0.0):.4g}`이다.",
        ]
    (RESULTS / "mr2_compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
