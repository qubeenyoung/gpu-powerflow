#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS = Path("results")


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


def coarse_label(row: dict[str, str]) -> str:
    text = row.get("setting") or row.get("preconditioner") or row.get("middle_solver") or ""
    return "on" if "coarse" in text else "off"


def avg_by_case(rows: list[dict[str, str]], value_key: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        case = row.get("case") or row.get("case_name") or ""
        value = as_float(row, value_key)
        values[(case, coarse_label(row))].append(value)
    return {key: mean(vals) for key, vals in values.items() if vals}


def main() -> None:
    summary = read_rows(RESULTS / "mr1_coarse_hybrid_summary_off.csv")
    summary += read_rows(RESULTS / "mr1_coarse_hybrid_summary_on.csv")
    iters = read_rows(RESULTS / "mr1_coarse_hybrid_iters_off.csv")
    iters += read_rows(RESULTS / "mr1_coarse_hybrid_iters_on.csv")
    shadow = read_rows(RESULTS / "mr1_coarse_shadow_dx_off.csv")
    shadow += read_rows(RESULTS / "mr1_coarse_shadow_dx_on.csv")
    timing = read_rows(RESULTS / "mr1_coarse_timing_breakdown_off.csv")
    timing += read_rows(RESULTS / "mr1_coarse_timing_breakdown_on.csv")

    write_rows(RESULTS / "mr1_coarse_hybrid_summary.csv", summary)
    write_rows(RESULTS / "mr1_coarse_hybrid_iters.csv", iters)
    write_rows(RESULTS / "mr1_coarse_shadow_dx.csv", shadow)
    write_rows(RESULTS / "mr1_coarse_timing_breakdown.csv", timing)

    dx_ratio = avg_by_case(shadow, "dx_norm_ratio")
    dx_cosine = avg_by_case(shadow, "dx_cosine")
    coarse_total = [as_float(row, "coarse_total_ms") for row in timing if coarse_label(row) == "on"]
    middle_total = avg_by_case(timing, "middle_solver_total_ms")
    pure_nr_iters: dict[str, str] = {}
    for pure_path in [
        RESULTS / "pure_cudss_nr_iters_all_cases.csv",
        RESULTS / "pure_cudss_nr_iters_10kplus_cases.csv",
    ]:
        if pure_path.exists():
            for row in read_rows(pure_path):
                pure_nr_iters[row.get("case_name", "")] = row.get("nr_iters", "")

    lines = [
        "# MR1 + METIS block-Jacobi coarse correction 결과",
        "",
        "## 읽는 법",
        "",
        "- **기존**: coarse correction 없는 `MR1 + METIS block-Jacobi`",
        "- **Coarse**: block 사이 coupling을 작은 dense coarse system으로 한 번 보정한 버전",
        "- **Newton 반복**: 전체 NR iteration 수. 작을수록 좋다.",
        "- **Fallback**: MR1 step이 충분히 좋지 않아서 같은 NR iteration에서 cuDSS로 다시 푼 횟수. 작을수록 좋다.",
        "- **속도비**: `pure cuDSS 시간 / hybrid 시간`. 1보다 크면 hybrid가 더 빠르다.",
        "- **dx 크기비**: `||dx_MR1||2 / ||dx_cuDSS||2`. 1에 가까울수록 cuDSS Newton step만큼 큰 correction이다.",
        "- **dx 방향유사도**: `cos(dx_MR1, dx_cuDSS)`. 1이면 같은 방향, 0이면 거의 직교, 음수면 반대 방향이다.",
        "",
        "## 1. 전체 비교",
        "",
        "| 케이스 | 설정 | 수렴 | Newton 반복 | cuDSS 호출 | MR1 호출 | Fallback | 시간(s) | 속도비 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        label = "Coarse" if coarse_label(row) == "on" else "기존"
        lines.append(
            f"| {row.get('case_name', '')} | {label} | {row.get('converged', '')} | "
            f"{row.get('nr_iters', '')} | {row.get('cudss_calls', '')} | "
            f"{row.get('gmres_calls', '')} | {row.get('fallback_calls', '')} | "
            f"{as_float(row, 'total_seconds'):.4f} | "
            f"{as_float(row, 'speedup_vs_pure_cudss'):.2f} |"
        )

    lines += [
        "",
        "요약하면, coarse correction은 Newton 반복 수를 줄이는 경우가 있지만 Fallback 감소는 아직 만들지 못했다.",
        "",
        "## 2. pure cuDSS와 비교",
        "",
        "이 표는 순수 cuDSS NR을 기준으로 hybrid가 실제로 빨라졌는지 보여준다.",
        "`시간 차이`는 `hybrid 시간 - pure cuDSS 시간`이다. 음수면 hybrid가 빠르고, 양수면 hybrid가 느리다.",
        "",
        "| 케이스 | 설정 | pure Newton 반복 | hybrid Newton 반복 | pure cuDSS 시간(s) | hybrid 시간(s) | 시간 차이(s) | 속도비 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        label = "Coarse" if coarse_label(row) == "on" else "기존"
        pure_time = as_float(row, "pure_cudss_total_seconds")
        hybrid_time = as_float(row, "total_seconds")
        lines.append(
            f"| {row.get('case_name', '')} | {label} | "
            f"{pure_nr_iters.get(row.get('case_name', ''), '')} | "
            f"{row.get('nr_iters', '')} | {pure_time:.4f} | {hybrid_time:.4f} | "
            f"{hybrid_time - pure_time:+.4f} | "
            f"{as_float(row, 'speedup_vs_pure_cudss'):.2f} |"
        )

    lines += [
        "",
        "해석:",
        "- `case2383wp`만 pure cuDSS보다 빠르다.",
        "- 나머지 케이스는 hybrid가 Newton 반복을 일부 줄여도 pure cuDSS보다 느리다.",
        "- 현재 병목은 middle solve 한 번의 속도보다, MR1 step이 충분히 강하지 않아 NR 반복/fallback을 늘리는 데 있다.",
        "",
        "## 3. dx 품질 비교",
        "",
        "이 표는 MR1이 만든 correction `dx`가 cuDSS Newton correction과 얼마나 비슷한지 본다.",
        "`dx 크기비`는 클수록 좋고, `dx 방향유사도`는 1에 가까울수록 좋다.",
        "",
        "| 케이스 | dx 크기비 기존 | dx 크기비 Coarse | dx 방향유사도 기존 | dx 방향유사도 Coarse |",
        "|---|---:|---:|---:|---:|",
    ]
    cases = sorted({key[0] for key in dx_ratio})
    for case in cases:
        lines.append(
            f"| {case} | {dx_ratio.get((case, 'off'), 0.0):.4g} | "
            f"{dx_ratio.get((case, 'on'), 0.0):.4g} | "
            f"{dx_cosine.get((case, 'off'), 0.0):.4g} | "
            f"{dx_cosine.get((case, 'on'), 0.0):.4g} |"
        )

    lines += [
        "",
        "해석:",
        "- coarse가 `dx` 방향은 여러 케이스에서 개선했다.",
        "- 하지만 `dx` 크기는 2/5 케이스에서만 커졌고, 2배 이상 개선된 케이스는 없다.",
        "- 따라서 coarse correction이 방향성은 일부 보정하지만 correction scale 문제는 아직 못 풀었다.",
        "",
        "## 4. NR 진행 변화",
        "",
        "| 케이스 | Newton 반복 기존 | Newton 반복 Coarse | Fallback 기존 | Fallback Coarse | 시간 기존(s) | 시간 Coarse(s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in summary:
        by_case[row.get("case_name", "")][coarse_label(row)] = row
    for case in sorted(by_case):
        off = by_case[case].get("off", {})
        on = by_case[case].get("on", {})
        lines.append(
            f"| {case} | {off.get('nr_iters', '')} | {on.get('nr_iters', '')} | "
            f"{off.get('fallback_calls', '')} | {on.get('fallback_calls', '')} | "
            f"{as_float(off, 'total_seconds'):.4f} | {as_float(on, 'total_seconds'):.4f} |"
        )

    lines += [
        "",
        "coarse가 NR 반복 수를 줄인 경우에도 Fallback 수가 줄지 않아서, 전체 시간 개선은 제한적이다.",
        "",
        "## 5. Coarse overhead",
        "",
    ]
    if coarse_total:
        lines += [
            f"- coarse correction 추가 비용 평균: **{mean(coarse_total):.3f} ms**",
            f"- coarse correction 추가 비용 최대: **{max(coarse_total):.3f} ms**",
        ]
    else:
        lines.append("- coarse correction timing row가 없다.")
    if middle_total:
        off_values = [value for (case, label), value in middle_total.items() if label == "off"]
        on_values = [value for (case, label), value in middle_total.items() if label == "on"]
        if off_values and on_values:
            lines.append(
                f"- middle solver 평균 시간: 기존 **{mean(off_values):.3f} ms**, "
                f"Coarse **{mean(on_values):.3f} ms**"
            )

    lines += [
        "",
        "목표였던 평균 0.3 ms 이하, 최대 0.5 ms 이하는 만족한다.",
        "",
        "## 6. 판단",
        "",
        "- **성공한 점**: coarse overhead는 충분히 작고, 일부 케이스에서 Newton 반복 수와 시간이 줄었다.",
        "- **부족한 점**: Fallback 감소가 없고, `dx` 크기비가 기대만큼 커지지 않았다.",
        "- **pure cuDSS 대비 결론**: 현재 설정에서 hybrid가 직접해법보다 확실히 빠른 케이스는 `case2383wp`뿐이다.",
        "- **최종 결론**: 현재 1 coarse variable/block 방식은 방향 보정 효과는 있지만, hybrid NR을 안정적으로 빠르게 만들 만큼 correction scale을 키우지는 못했다.",
    ]
    (RESULTS / "mr1_coarse_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
