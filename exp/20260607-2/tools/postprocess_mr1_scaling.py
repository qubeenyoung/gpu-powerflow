#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS = Path("results")
MODES = [
    ("none", "none"),
    ("ruiz", "ruiz"),
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


def with_mode(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        copied["scaling"] = mode
        out.append(copied)
    return out


def mean_by_case_mode(rows: list[dict[str, str]], case_key: str, metric: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row, metric)
        if value != 0.0:
            values[(row.get(case_key, ""), row["scaling"])].append(value)
    return {key: mean(vals) for key, vals in values.items() if vals}


def middle_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if as_float(row, "gmres_trial_solve_seconds") > 0.0
        or row.get("solver_used") in {"mr1_middle", "cudss_fallback"}
    ]


def main() -> None:
    summary: list[dict[str, str]] = []
    iters: list[dict[str, str]] = []
    shadow: list[dict[str, str]] = []
    timing: list[dict[str, str]] = []
    for mode, _ in MODES:
        summary += with_mode(read_rows(RESULTS / f"mr1_scaling_summary_{mode}.csv"), mode)
        iters += with_mode(read_rows(RESULTS / f"mr1_scaling_iters_{mode}.csv"), mode)
        shadow += with_mode(read_rows(RESULTS / f"mr1_scaling_shadow_{mode}.csv"), mode)
        timing += with_mode(read_rows(RESULTS / f"mr1_scaling_timing_{mode}.csv"), mode)

    write_rows(RESULTS / "mr1_scaling_summary.csv", summary)
    write_rows(RESULTS / "mr1_scaling_iters.csv", iters)
    write_rows(RESULTS / "mr1_scaling_shadow_dx.csv", shadow)
    write_rows(RESULTS / "mr1_scaling_timing.csv", timing)

    cases = sorted({row["case_name"] for row in summary})
    summary_map = {(row["case_name"], row["scaling"]): row for row in summary}

    dx_norm = mean_by_case_mode(shadow, "case", "dx_norm_ratio")
    dx_cos = mean_by_case_mode(shadow, "case", "dx_cosine")
    theta_norm = mean_by_case_mode(shadow, "case", "theta_norm_ratio")
    theta_cos = mean_by_case_mode(shadow, "case", "theta_cosine")
    vmag_norm = mean_by_case_mode(shadow, "case", "vmag_norm_ratio")
    vmag_cos = mean_by_case_mode(shadow, "case", "vmag_cosine")
    trial_ratio = mean_by_case_mode(shadow, "case", "gmres_nonlinear_ratio_inf")

    middle = middle_rows(timing)
    setup_ms = mean_by_case_mode(middle, "case_name", "linear_setup_ms")
    middle_ms = mean_by_case_mode(middle, "case_name", "middle_solver_total_ms")
    scaling_total_ms = mean_by_case_mode(middle, "case_name", "scaling_total_ms")
    scaled_rel = mean_by_case_mode(middle, "case_name", "scaled_linear_rel_res")
    unscaled_rel = mean_by_case_mode(middle, "case_name", "unscaled_linear_rel_res")

    lines = [
        "# MR1 row/column scaling experiment",
        "",
        "## 읽는 법",
        "",
        "- **none**: 기존 MR1 + METIS block-Jacobi, coarse/gamma 없이 사용한다.",
        "- **ruiz**: `A_s = Dr A_perm Dc`, `b_s = Dr b_perm`에서 MR1을 수행하고 `dx_perm = Dc y`로 되돌린다.",
        "- **middle trial ratio**는 shadow 진단의 `gmres_nonlinear_ratio_inf` 평균이다. 작을수록 middle step이 mismatch를 더 줄인다.",
        "- **scaled/unscaled linear residual**은 각각 scaled system과 원래 permuted system 기준이다.",
        "",
        "## 1. NR 결과",
        "",
        "| case | scaling | converged | NR iters | cuDSS calls | MR1 calls | accepted | fallback | hybrid time(ms) | pure cuDSS(ms) | speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        for mode, _ in MODES:
            row = summary_map.get((case, mode), {})
            lines.append(
                f"| {case} | {mode} | {row.get('converged', '')} | "
                f"{row.get('nr_iters', '')} | {row.get('cudss_calls', '')} | "
                f"{row.get('gmres_calls', '')} | {row.get('accepted_gmres_steps', '')} | "
                f"{row.get('fallback_calls', '')} | {1000.0 * as_float(row, 'total_seconds'):.2f} | "
                f"{1000.0 * as_float(row, 'pure_cudss_total_seconds'):.2f} | "
                f"{as_float(row, 'speedup_vs_pure_cudss'):.3f} |"
            )

    lines += [
        "",
        "## 2. dx quality",
        "",
        "| case | dx_norm none | dx_norm ruiz | dx_cos none | dx_cos ruiz | theta_norm none | theta_norm ruiz | vmag_norm none | vmag_norm ruiz |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case} | {dx_norm.get((case, 'none'), 0.0):.4g} | "
            f"{dx_norm.get((case, 'ruiz'), 0.0):.4g} | "
            f"{dx_cos.get((case, 'none'), 0.0):.4g} | {dx_cos.get((case, 'ruiz'), 0.0):.4g} | "
            f"{theta_norm.get((case, 'none'), 0.0):.4g} | {theta_norm.get((case, 'ruiz'), 0.0):.4g} | "
            f"{vmag_norm.get((case, 'none'), 0.0):.4g} | {vmag_norm.get((case, 'ruiz'), 0.0):.4g} |"
        )

    lines += [
        "",
        "## 3. middle quality",
        "",
        "| case | middle trial ratio none | middle trial ratio ruiz | scaled linear rel ruiz | unscaled linear rel ruiz |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case} | {trial_ratio.get((case, 'none'), 0.0):.4g} | "
            f"{trial_ratio.get((case, 'ruiz'), 0.0):.4g} | "
            f"{scaled_rel.get((case, 'ruiz'), 0.0):.4g} | "
            f"{unscaled_rel.get((case, 'ruiz'), 0.0):.4g} |"
        )

    lines += [
        "",
        "## 4. timing",
        "",
        "| case | scaling total ruiz(ms) | middle total none(ms) | middle total ruiz(ms) | setup none(ms) | setup ruiz(ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case} | {scaling_total_ms.get((case, 'ruiz'), 0.0):.4f} | "
            f"{middle_ms.get((case, 'none'), 0.0):.4f} | "
            f"{middle_ms.get((case, 'ruiz'), 0.0):.4f} | "
            f"{setup_ms.get((case, 'none'), 0.0):.4f} | "
            f"{setup_ms.get((case, 'ruiz'), 0.0):.4f} |"
        )

    ruiz_better_dx = [
        case for case in cases
        if dx_norm.get((case, "ruiz"), 0.0) >= 2.0 * max(dx_norm.get((case, "none"), 0.0), 1.0e-300)
    ]
    fallback_changed = [
        case for case in cases
        if as_float(summary_map.get((case, "ruiz"), {}), "fallback_calls") <
        as_float(summary_map.get((case, "none"), {}), "fallback_calls")
    ]
    lines += [
        "",
        "## 5. 판단",
        "",
        f"- dx_norm_ratio가 2배 이상 오른 케이스: {', '.join(ruiz_better_dx) if ruiz_better_dx else 'none'}.",
        f"- fallback이 줄어든 케이스: {', '.join(fallback_changed) if fallback_changed else 'none'}.",
        "- 성공 기준은 dx 크기/방향, middle trial ratio, fallback/NR 반복, 총 시간을 함께 본다.",
    ]
    if not ruiz_better_dx and not fallback_changed:
        lines.append("- 이번 결과가 이 상태라면 raw scaling이 주된 원인이라는 가설은 약하다.")

    (RESULTS / "mr1_scaling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
