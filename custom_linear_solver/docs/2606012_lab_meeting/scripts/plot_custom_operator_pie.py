#!/usr/bin/env python3
"""Operator-breakdown pie charts for the custom solver (fp32/tf32/fp16), per case/precision/B.
Mirrors plot_cupf_operator_pie_demo.py but reads the custom ops_ms CSV and adds precision to
the filename + title (since each case/B now has three precisions)."""
from __future__ import annotations
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver/docs/2606012_lab_meeting")
DATA = ROOT / "data" / "cupf_custom_fp32_tf32_fp16_ops_ms.csv"
OUT_DIR = ROOT / "figures" / "cupf_custom_operator_pies"

OPERATORS = [
    ("factorize", "factorize_ms", "#285C8E"),
    ("triangular\nsolve", "triangular_solve_ms", "#4DA3D9"),
    ("mismatch\nnorm", "mnorm_ms", "#F39C12"),
    ("ibus", "ibus_ms", "#8E44AD"),
    ("jacobian", "jacobian_ms", "#F2C94C"),
    ("mismatch", "mismatch_ms", "#E74C3C"),
    ("voltage\nupdate", "voltage_update_ms", "#2ECC71"),
]

def prec_of(src):
    m = re.search(r'(fp32|tf32|fp16)', src); return m.group(1) if m else "?"
def format_duration(v):
    return f"{v/1000.0:.3g} s" if v>=1000 else (f"{v:.3g} ms" if v>=1 else f"{v*1000.0:.3g} us")
def format_percent(p):
    return "<1%" if 0.0<p<0.5 else f"{p:.0f}%"

def draw(row):
    case, batch, prec = row["case_name"], row["B"], prec_of(row["source"])
    labels = [i[0] for i in OPERATORS]
    values = np.array([float(row[i[1]]) for i in OPERATORS])
    colors = [i[2] for i in OPERATORS]
    pct = values/values.sum()*100.0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{case}_{prec}_B{batch}_operator_pie.png"
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":28,"font.weight":"bold",
        "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
    fig,(ax,label_ax)=plt.subplots(1,2,figsize=(15.0,8.6),dpi=180,
        gridspec_kw={"width_ratios":[1.02,1.18],"wspace":0.02})
    ax.pie(values,colors=colors,startangle=90,counterclock=False,labels=None,radius=1.0,
        center=(0.0,0.0),wedgeprops={"edgecolor":"#F7F7F7","linewidth":2.4})
    ax.set(aspect="equal"); ax.set_axis_off(); label_ax.set_axis_off()
    label_ax.set_xlim(0.0,1.18); label_ax.set_ylim(0.0,1.0)
    diverged = int(row.get("iterations","0") or 0) >= 30
    fig.suptitle(f"{case} · custom {prec}{' (DIVERGED)' if diverged else ''} · B={batch}  "
                 f"[solve_total {format_duration(float(row['solve_total_ms']))}]",
                 fontsize=22, fontweight="bold", y=0.99)
    for y,label,percent,value,color in zip(np.linspace(0.80,0.16,len(labels)),labels,pct,values,colors):
        label_ax.text(0.05,y,"■",ha="left",va="center",color=color,fontsize=34,fontweight="bold")
        label_ax.text(0.17,y,label.replace("\n"," "),ha="left",va="center",color=color,fontsize=28,fontweight="bold")
        label_ax.text(1.12,y,f"{format_percent(percent)} ({format_duration(value)})",ha="right",va="center",color=color,fontsize=24,fontweight="bold")
    fig.subplots_adjust(left=0.01,right=0.99,top=0.94,bottom=0.02)
    fig.savefig(out,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
    return out

if __name__=="__main__":
    rows=sorted(csv.DictReader(DATA.open(newline="",encoding="utf-8")),
        key=lambda r:(int(r["n_bus"]),{"fp32":0,"tf32":1,"fp16":2}.get(prec_of(r["source"]),9),int(r["B"])))
    n=0
    for row in rows: draw(row); n+=1
    print(f"{n} pies -> {OUT_DIR}")
