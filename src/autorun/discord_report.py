"""Render Discord report messages (PLAN §6.2) and archive them (§6.3).

The autorun harness does not POST to a webhook directly: the Claude Code agent
relays the rendered text to the Discord bot channel. Every rendered message is
also written to report/autorun/log/discord_sent/<cycle>_<type>.txt for audit.
"""

import json
import math
import pathlib

SENT_DIR = "report/autorun/log/discord_sent"


def _header(cycle, milestone, status, ts):
    return f"[mysolver] cycle {cycle} | {milestone}:{status} | {ts}"


def _save(cycle, kind, text, payload=None, sent_dir=SENT_DIR):
    d = pathlib.Path(sent_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cycle:04d}_{kind}.txt").write_text(text)
    if payload is not None:
        (d / f"{cycle:04d}_{kind}.json").write_text(json.dumps(payload, indent=2))
    return text


def change_report(cycle, milestone, status, ts, files, intent, build, unit, ref, nxt,
                  sent_dir=SENT_DIR):
    lines = [_header(cycle, milestone, status, ts), "\U0001F6E0 change_report", "  files:"]
    for f in files:
        lines.append(f"    - {f}")
    lines.append(f"  intent: {intent}")
    lines.append(f"  build:  {build}")
    lines.append(f"  unit:   {unit}")
    if ref:
        lines.append(f"  \U0001F517 ref: {ref}")
    lines.append(f"  next:   {nxt}")
    return _save(cycle, "change", "\n".join(lines), sent_dir=sent_dir)


def benchmark_report(cycle, milestone, status, ts, matrix_set, solvers, res, csv_path,
                     have_prev_best=False, sent_dir=SENT_DIR):
    tm = res["target_metrics"]
    worst_berr = max((m["berr"] for m in tm.values()
                      if not math.isnan(m["berr"])), default=float("nan"))
    worst_berr_case = max(tm.items(),
                          key=lambda kv: (kv[1]["berr"] if not math.isnan(kv[1]["berr"]) else -1),
                          default=(None, None))[0]
    worst_abs = max((m["absolute_error"] for m in tm.values()
                     if not math.isnan(m["absolute_error"])), default=float("nan"))

    lines = [_header(cycle, milestone, status, ts), "\U0001F4CA benchmark_report"]
    lines.append(f"  set:   {matrix_set}   ({len(res['matrices'])} matrices, {len(solvers)} solvers)")
    if res["accuracy_ok"]:
        lines.append(f"  accuracy:  ✅ berr ≤ {worst_berr:.2e} (worst {worst_berr_case}), "
                     f"absolute ≤ {worst_abs:.2e}")
    else:
        lines.append(f"  accuracy:  \U0001F6A8 {len(res['violations'])} VIOLATION(S): "
                     + ", ".join(f"{v['matrix']}(berr={v['berr']:.2e},ok={v['success']})"
                                 for v in res["violations"]))

    lines.append("  vs best:")
    if not have_prev_best:
        lines.append("    ⚪ baseline (M0 first run — no prior best recorded)")
    elif res["regressions"]:
        for r in res["regressions"]:
            lines.append(f"    \U0001F7E1 {r['matrix']} {r['metric']} {r['best']:.3g} → "
                         f"{r['now']:.3g} (+{r['delta_pct']}%, regression flag)")
    else:
        lines.append("    \U0001F7E2 no regression vs best")

    # vs cudss-gpu: report the worst factor-time ratio among comparable matrices.
    ratios = []
    for c in res["comparisons"]:
        m = c["matrix"]
        cud = c.get("cudss-gpu")
        if cud and m in tm and cud["factor_ms"] > 0 and not math.isnan(tm[m]["factor_ms"]):
            ratios.append((tm[m]["factor_ms"] / cud["factor_ms"], m, tm[m]["factor_ms"], cud["factor_ms"]))
    if ratios:
        r, m, mine, theirs = max(ratios)
        lines.append(f"  vs cudss-gpu ({m}): mysolver {mine:.2f}ms / cudss {theirs:.2f}ms ({r:.2f}x)")

    klu_ratios = []
    for c in res["comparisons"]:
        m = c["matrix"]
        klu = c.get("klu")
        if klu and m in tm and klu["factor_ms"] > 0 and not math.isnan(tm[m]["factor_ms"]):
            klu_ratios.append(tm[m]["factor_ms"] / klu["factor_ms"])
    if klu_ratios:
        avg = sum(klu_ratios) / len(klu_ratios)
        lines.append(f"  vs klu (factor, {matrix_set}): mysolver {avg:.2f}x KLU avg")

    lines.append(f"  \U0001F4CE {csv_path}")
    return _save(cycle, "benchmark", "\n".join(lines), payload=res, sent_dir=sent_dir)


def alert_report(cycle, kind, what, cause, action, nxt, sent_dir=SENT_DIR):
    lines = [f"[mysolver] \U0001F6A8 alert_report  cycle {cycle}",
             f"  type: {kind}",
             f"  what: {what}",
             f"  cause: {cause}",
             f"  action: {action}",
             f"  next plan: {nxt}"]
    return _save(cycle, "alert", "\n".join(lines), sent_dir=sent_dir)


def milestone_report(cycle, ts, frm, to, exit_lines, next_goals, active_set, sent_dir=SENT_DIR):
    lines = [f"[mysolver] \U0001F389 milestone_report  {frm} → {to}",
             f"  cycle {cycle} | {ts}",
             "  exit metrics passed:"]
    for e in exit_lines:
        lines.append(f"    - {e}")
    lines.append(f"  next milestone goals: {next_goals}")
    lines.append(f"  active_matrix_set: {active_set}")
    return _save(cycle, "milestone", "\n".join(lines), sent_dir=sent_dir)
