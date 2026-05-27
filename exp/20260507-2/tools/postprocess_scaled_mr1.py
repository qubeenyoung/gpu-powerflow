#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


RESULTS = Path("results")
MODES = [
    ("baseline", "MR1 + coarse"),
    ("scaled", "scaled MR1 + coarse"),
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


def middle_attempt_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if as_float(row, "gmres_trial_solve_seconds") > 0.0
        or row.get("solver_used") in {"mr1_middle", "cudss_fallback"}
    ]


def avg_by_case_mode(rows: list[dict[str, str]], key: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row, key)
        if value > 0.0:
            values[(row.get("case_name", row.get("case", "")), row["mode_label"])].append(value)
    return {key_: mean(vals) for key_, vals in values.items() if vals}


def shadow_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        out[(row.get("case", ""), row.get("nr_iter", ""), row["mode_label"])] = row
    return out


def min_positive(values: list[float]) -> float:
    positives = [value for value in values if value > 0.0]
    return min(positives) if positives else 0.0


def trial_mismatch_ratio(row: dict[str, str], shadow: dict[tuple[str, str, str], dict[str, str]]) -> float:
    if row["mode"] == "scaled":
        return min_positive([
            as_float(row, "mismatch_ratio_gamma_4"),
            as_float(row, "mismatch_ratio_gamma_2"),
            as_float(row, "mismatch_ratio_gamma_1"),
        ])
    shadow_row = shadow.get((row.get("case_name", ""), row.get("nr_iter", ""), row["mode_label"]), {})
    return as_float(shadow_row, "gmres_nonlinear_ratio_inf")


def scaled_dx_ratio(row: dict[str, str], shadow: dict[tuple[str, str, str], dict[str, str]]) -> float:
    shadow_row = shadow.get((row.get("case_name", ""), row.get("nr_iter", ""), row["mode_label"]), {})
    ratio = as_float(shadow_row, "dx_norm_ratio")
    gamma = as_float(row, "chosen_gamma") if row["mode"] == "scaled" else 1.0
    if gamma <= 0.0:
        gamma = 1.0
    return gamma * ratio


def shadow_cosine(row: dict[str, str], shadow: dict[tuple[str, str, str], dict[str, str]]) -> float:
    shadow_row = shadow.get((row.get("case_name", ""), row.get("nr_iter", ""), row["mode_label"]), {})
    return as_float(shadow_row, "dx_cosine")


def case_mode_mean(values: dict[tuple[str, str], list[float]]) -> dict[tuple[str, str], float]:
    return {key: mean(vals) for key, vals in values.items() if vals}


def main() -> None:
    summary: list[dict[str, str]] = []
    iters: list[dict[str, str]] = []
    shadow: list[dict[str, str]] = []
    timing: list[dict[str, str]] = []
    for mode, label in MODES:
        summary += with_mode(read_rows(RESULTS / f"scaled_mr1_summary_{mode}.csv"), mode, label)
        iters += with_mode(read_rows(RESULTS / f"scaled_mr1_iters_{mode}.csv"), mode, label)
        shadow += with_mode(read_rows(RESULTS / f"scaled_mr1_shadow_{mode}.csv"), mode, label)
        timing += with_mode(read_rows(RESULTS / f"scaled_mr1_timing_{mode}.csv"), mode, label)

    write_rows(RESULTS / "scaled_mr1_summary.csv", summary)
    write_rows(RESULTS / "scaled_mr1_iters.csv", iters)
    write_rows(RESULTS / "scaled_mr1_shadow_dx.csv", shadow)
    write_rows(RESULTS / "scaled_mr1_timing_breakdown.csv", timing)

    shadow_by_iter = shadow_lookup(shadow)
    attempts = middle_attempt_rows(iters)
    timing_attempts = middle_attempt_rows(timing)
    middle_total = avg_by_case_mode(timing_attempts, "total_middle_time_ms")
    extra_eval = avg_by_case_mode(timing_attempts, "extra_mismatch_eval_ms")

    dx_ratio_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    dx_cos_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    trial_ratio_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    gamma_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    gamma_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in attempts:
        key = (row.get("case_name", ""), row["mode_label"])
        dx_ratio = scaled_dx_ratio(row, shadow_by_iter)
        dx_cos = shadow_cosine(row, shadow_by_iter)
        trial_ratio = trial_mismatch_ratio(row, shadow_by_iter)
        gamma = as_float(row, "chosen_gamma")
        if dx_ratio > 0.0:
            dx_ratio_values[key].append(dx_ratio)
        if dx_cos != 0.0:
            dx_cos_values[key].append(dx_cos)
        if trial_ratio > 0.0:
            trial_ratio_values[key].append(trial_ratio)
        if row["mode"] == "scaled" and gamma > 0.0:
            gamma_values[key].append(gamma)
            gamma_counts[key][f"{gamma:g}"] += 1

    dx_ratio = case_mode_mean(dx_ratio_values)
    dx_cosine = case_mode_mean(dx_cos_values)
    trial_ratio = case_mode_mean(trial_ratio_values)
    gamma_mean = case_mode_mean(gamma_values)

    cases = sorted({row["case_name"] for row in summary})
    labels = [label for _, label in MODES]
    summary_by_case_mode = {(row["case_name"], row["mode_label"]): row for row in summary}

    lines = [
        "# Scaled MR1 + coarse experiment",
        "",
        "## 읽는 법",
        "",
        "- **MR1 + coarse**: 기존 방식이다. MR1이 만든 `dx`를 그대로 적용한다.",
        "- **scaled MR1 + coarse**: 같은 `dx`에 대해 `gamma = 4, 2, 1`을 임시 적용해 보고, mismatch_inf가 가장 작은 gamma만 선택한다.",
        "- **middle trial ratio**: middle solver 후보 step 후 `mismatch_inf_after / mismatch_inf_before`이다. 작을수록 Newton 진행이 좋다.",
        "- **scaled dx 크기비**: shadow 진단의 `||dx_mr1|| / ||dx_cuDSS||`에 선택된 gamma를 곱한 값이다.",
        "",
        "## 1. NR 결과",
        "",
        "| 케이스 | 모드 | 수렴 | NR 반복 | cuDSS 호출 | MR1 호출 | Accepted | Rejected | Fallback | 시간(s) | Pure cuDSS(s) | Speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        for label in labels:
            row = summary_by_case_mode.get((case, label), {})
            lines.append(
                f"| {case} | {label} | {row.get('converged', '')} | "
                f"{row.get('nr_iters', '')} | {row.get('cudss_calls', '')} | "
                f"{row.get('gmres_calls', '')} | {row.get('accepted_gmres_steps', '')} | "
                f"{row.get('rejected_gmres_steps', '')} | {row.get('fallback_calls', '')} | "
                f"{as_float(row, 'total_seconds'):.4f} | "
                f"{as_float(row, 'pure_cudss_total_seconds'):.4f} | "
                f"{as_float(row, 'speedup_vs_pure_cudss'):.3f} |"
            )

    lines += [
        "",
        "## 2. Middle step",
        "",
        "| 케이스 | 모드 | total middle 평균(ms) | 추가 mismatch 평가(ms) | 선택 gamma 평균 | gamma count | scaled dx 크기비 | dx 방향유사도 | middle trial ratio |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for case in cases:
        for label in labels:
            key = (case, label)
            counts = ", ".join(
                f"{gamma}:{count}" for gamma, count in sorted(gamma_counts.get(key, {}).items())
            )
            lines.append(
                f"| {case} | {label} | {middle_total.get(key, 0.0):.4f} | "
                f"{extra_eval.get(key, 0.0):.4f} | {gamma_mean.get(key, 0.0):.3g} | "
                f"{counts} | {dx_ratio.get(key, 0.0):.4g} | "
                f"{dx_cosine.get(key, 0.0):.4g} | {trial_ratio.get(key, 0.0):.4g} |"
            )

    baseline_time = [
        as_float(row, "total_seconds")
        for row in summary
        if row["mode"] == "baseline" and as_float(row, "total_seconds") > 0.0
    ]
    scaled_time = [
        as_float(row, "total_seconds")
        for row in summary
        if row["mode"] == "scaled" and as_float(row, "total_seconds") > 0.0
    ]
    baseline_fallback = [
        as_float(row, "fallback_calls") for row in summary if row["mode"] == "baseline"
    ]
    scaled_fallback = [
        as_float(row, "fallback_calls") for row in summary if row["mode"] == "scaled"
    ]
    lines += [
        "",
        "## 3. 판단",
        "",
        f"- 평균 hybrid 시간: baseline `{mean(baseline_time):.4f}s`, scaled `{mean(scaled_time):.4f}s`.",
        f"- 평균 fallback 수: baseline `{mean(baseline_fallback):.2f}`, scaled `{mean(scaled_fallback):.2f}`.",
        "- 성공 판단은 scaled가 fallback 또는 NR 반복을 줄이고, 추가 mismatch 평가 비용을 감수해도 총 시간이 개선되는지로 본다.",
    ]
    (RESULTS / "scaled_mr1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
