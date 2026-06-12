#!/usr/bin/env python3
"""Operator-breakdown pie charts for the 2026-06-12 lab meeting, from the deterministic cuPF
re-measurement (report 07). One pie per (config, case, B): the per-solve operator split.
Source: docs/05-reports/07-cupf-backend-comparison-2026-06-11/operator_ms_full.tsv (microseconds)."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver/docs")
DATA = ROOT / "05-reports" / "07-cupf-backend-comparison-2026-06-11" / "operator_ms_full.tsv"
OUT_ROOT = ROOT / "20260612_lab_meeting" / "figures" / "operator_pies"

# (label, tsv column(s) summed, colour)
OPERATORS = [
    ("factorize",        ["factorize_us"],                 "#285C8E"),
    ("triangular\nsolve",["tri_solve_us"],                 "#4DA3D9"),
    ("jacobian",         ["jacobian_us"],                  "#F2C94C"),
    ("ibus (SpMV)",      ["ibus_us"],                      "#8E44AD"),
    ("mismatch\n+norm",  ["mismatch_us", "mnorm_us"],      "#E74C3C"),
    ("voltage\nupdate",  ["voltage_update_us"],            "#F39C12"),
]
CASE_ORDER = {c: i for i, c in enumerate(
    ["case3012wp","case6468rte","case8387pegase","case13659pegase","case_ACTIVSg25k","case_SyntheticUSA"])}

def fmt_dur(us):
    if us >= 1e6: return f"{us/1e6:.3g} s"
    if us >= 1e3: return f"{us/1e3:.3g} ms"
    return f"{us:.3g} us"
def fmt_pct(p):
    return "<1%" if 0.0 < p < 0.5 else f"{p:.0f}%"

def draw(row):
    cfg, case, B = row["config"], row["case"], row["B"]
    vals = np.array([sum(float(row[c]) for c in cols) for _, cols, _ in OPERATORS])
    colors = [c for _, _, c in OPERATORS]
    labels = [l for l, _, _ in OPERATORS]
    pct = vals / vals.sum() * 100.0
    out_dir = OUT_ROOT / cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{case}_B{B}_operator_pie.png"
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":28,"font.weight":"bold",
        "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
    fig,(ax,lab_ax)=plt.subplots(1,2,figsize=(15.0,6.6),dpi=160,
        gridspec_kw={"width_ratios":[1.02,1.18],"wspace":0.02})
    ax.pie(vals,colors=colors,startangle=90,counterclock=False,radius=1.0,center=(0,0),
        wedgeprops={"edgecolor":"#F7F7F7","linewidth":2.4})
    ax.set(aspect="equal"); ax.set_axis_off(); lab_ax.set_axis_off()
    lab_ax.set_xlim(0,1.18); lab_ax.set_ylim(0,1.0)
    fig.suptitle(f"{case} · {cfg} · B={B}  (iters {row['iters']})  "
                 f"[solve_total {fmt_dur(float(row['solve_total_us']))}]",
                 fontsize=21, fontweight="bold", y=0.965)
    for y,l,p,v,c in zip(np.linspace(0.82,0.12,len(labels)),labels,pct,vals,colors):
        lab_ax.text(0.05,y,"■",ha="left",va="center",color=c,fontsize=32,fontweight="bold")
        lab_ax.text(0.17,y,l.replace("\n"," "),ha="left",va="center",color=c,fontsize=26,fontweight="bold")
        lab_ax.text(1.16,y,f"{fmt_pct(p)} ({fmt_dur(v)})",ha="right",va="center",color=c,fontsize=22,fontweight="bold")
    fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.02)
    fig.savefig(out,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
    return out

if __name__=="__main__":
    # Lab meeting scope: Mixed profile only (cuDSS vs custom fp32/tf32), B=1.
    CONFIGS = {"cudss-mixed", "custom-mixed-fp32", "custom-mixed-tf32"}
    rows=sorted(csv.DictReader(DATA.open(newline="",encoding="utf-8"),delimiter="\t"),
        key=lambda r:(r["config"],CASE_ORDER.get(r["case"],9),int(r["B"])))
    n=0
    for row in rows:
        if row["config"] not in CONFIGS or row["B"] != "1": continue
        draw(row); n+=1
    print(f"{n} pies -> {OUT_ROOT}")
