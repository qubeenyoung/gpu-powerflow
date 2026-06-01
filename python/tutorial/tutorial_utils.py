"""Shared helpers for the tutorial notebooks.

The functions here intentionally favor readability over framework cleverness:
the notebooks are for first-time readers, so each helper keeps the math and
data flow close to the Newton-Raphson power-flow equations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X
from pandapower.pypower.idx_bus import PD, QD
from pandapower.pypower.idx_gen import GEN_BUS, PG, QG, GEN_STATUS

from python.tests import matpower_data


DEFAULT_DATASET_ROOT = matpower_data.DEFAULT_DATASET_ROOT
TUTORIAL_LARGE_CASE = "case6468rte"
TUTORIAL_RUN_ROOT = Path(__file__).resolve().parent / "_runs"
BUS_COLORS = {
    "Slack": "#d62728",
    "PV": "#1f77b4",
    "PQ": "#2ca02c",
}
STAGE_COLORS = {
    "Linear solve": "#d62728",
    "Jacobian": "#ff7f0e",
    "Mismatch": "#1f77b4",
    "Ibus": "#17becf",
    "Mismatch norm": "#9467bd",
    "Voltage update": "#2ca02c",
    "Upload/download": "#8c564b",
    "Other": "#7f7f7f",
}


@dataclass
class NewtonTrace:
    voltage: np.ndarray
    rows: pd.DataFrame
    stage_totals_ms: pd.Series
    converged: bool
    iterations: int
    final_mismatch: float


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def tail(self, lines: int = 30) -> str:
        text = "\n".join(part for part in [self.stdout, self.stderr] if part)
        return "\n".join(text.splitlines()[-lines:])


@dataclass
class PowerFlowSnapshot:
    """The physical power-flow equation evaluated at one voltage vector."""

    voltage: np.ndarray
    ibus: np.ndarray
    s_calc: np.ndarray
    s_spec: np.ndarray
    mismatch_complex: np.ndarray
    mismatch_reduced: np.ndarray


@dataclass
class NewtonStepSnapshot:
    """One Newton-Raphson step, exposed as data the notebooks can inspect."""

    voltage: np.ndarray
    mismatch: np.ndarray
    jacobian: sp.csr_matrix
    dx: np.ndarray
    next_voltage: np.ndarray
    pvpq: np.ndarray


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def timestamp_run_name(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def tutorial_output_root() -> Path:
    TUTORIAL_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    return TUTORIAL_RUN_ROOT


def run_shell_command(
    command: list[str],
    *,
    timeout: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=repo_root(),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\nTimed out after {timeout} seconds.",
            elapsed_s=time.perf_counter() - start,
        )
    except OSError as exc:
        return CommandResult(
            command=command,
            returncode=127,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.perf_counter() - start,
        )
    return CommandResult(
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_s=time.perf_counter() - start,
    )


def command_summary(result: CommandResult, *, tail_lines: int = 24) -> str:
    status = "OK" if result.ok else f"FAILED ({result.returncode})"
    cmd = " ".join(result.command)
    tail = result.tail(tail_lines)
    if tail:
        return f"$ {cmd}\n[{status}] elapsed={result.elapsed_s:.1f}s\n{tail}"
    return f"$ {cmd}\n[{status}] elapsed={result.elapsed_s:.1f}s"


def build_eval(kind: str, *, jobs: int | None = None, timeout: int | None = None) -> CommandResult:
    command = ["bash", "benchmark/scripts/build_eval.bash", kind]
    if jobs is not None:
        command += ["--jobs", str(jobs)]
    return run_shell_command(command, timeout=timeout)


def run_tutorial_benchmark(
    *,
    variants: list[str],
    run_name: str | None = None,
    cases: list[str] | None = None,
    repeats: int = 1,
    warmup: int = 0,
    output_root: str | Path | None = None,
    timeout: int | None = None,
    skip_matlab: bool = False,
    skip_cupf: bool = False,
) -> tuple[Path, CommandResult]:
    output = Path(output_root) if output_root is not None else tutorial_output_root()
    output.mkdir(parents=True, exist_ok=True)
    name = run_name or timestamp_run_name("tutorial")
    command = [
        sys.executable,
        "-m",
        "python.tests.run_benchmark",
        "--output-root",
        str(output),
        "--run-name",
        name,
        "--cases",
        *(cases or [TUTORIAL_LARGE_CASE]),
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--variants",
        *variants,
    ]
    if skip_matlab:
        command.append("--skip-matlab")
    if skip_cupf:
        command.append("--skip-cupf")
    result = run_shell_command(command, timeout=timeout)
    return output / name, result


def benchmark_result_table(run_dir: str | Path) -> pd.DataFrame:
    """Return a compact table that separates successful runs from skips."""

    runs = load_tutorial_runs(run_dir)
    summary = summarize_runs(runs)
    skips = skipped_variants(run_dir)
    if skips.empty:
        return summary
    skip_rows = pd.DataFrame(
        {
            "variant": skips["variant"],
            "cases": 0,
            "successful_rows": 0,
            "converged_rows": 0,
            "initialize_ms": math.nan,
            "solve_ms": math.nan,
            "worst_residual": math.nan,
            "linear_solver": "",
            "jacobian": "",
            "entrypoint": "skipped: " + skips["reason"].astype(str),
        }
    )
    if summary.empty:
        return skip_rows
    return pd.concat([summary, skip_rows], ignore_index=True)


def load_tutorial_runs(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    frames = []
    for path in sorted(run_dir.glob("*/runs.csv")):
        frame = pd.read_csv(path)
        frame["variant_dir"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def skipped_variants(run_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(run_dir).glob("*/SKIPPED.txt")):
        rows.append({"variant": path.parent.name, "reason": path.read_text(encoding="utf-8").strip()})
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    data = runs.copy()
    for col in ["success", "warmup", "converged"]:
        if col in data:
            data[col] = data[col].astype(str).str.lower().isin(["1", "true", "yes"])
    if "warmup" in data:
        data = data[~data["warmup"]]
    rows = []
    for variant, group in data.groupby("variant", sort=False):
        ok = group[group["success"]]
        solve = pd.to_numeric(ok.get("solve_ms", pd.Series(dtype=float)), errors="coerce")
        init = pd.to_numeric(ok.get("initialize_ms", pd.Series(dtype=float)), errors="coerce")
        residual = pd.to_numeric(ok.get("output_mismatch", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "variant": variant,
                "cases": int(group["case_name"].nunique()) if "case_name" in group else 0,
                "successful_rows": int(len(ok)),
                "converged_rows": int(ok.get("converged", pd.Series(dtype=bool)).sum()) if len(ok) else 0,
                "initialize_ms": float(init.mean()) if len(init.dropna()) else math.nan,
                "solve_ms": float(solve.mean()) if len(solve.dropna()) else math.nan,
                "worst_residual": float(residual.max()) if len(residual.dropna()) else math.nan,
                "linear_solver": str(ok["linear_solver"].iloc[0]) if len(ok) and "linear_solver" in ok else "",
                "jacobian": str(ok["jacobian"].iloc[0]) if len(ok) and "jacobian" in ok else "",
                "entrypoint": str(ok["entrypoint"].iloc[0]) if len(ok) and "entrypoint" in ok else "",
            }
        )
    return pd.DataFrame(rows).sort_values("solve_ms", na_position="last")


def plot_run_solve_bars(summary: pd.DataFrame, ax: Any | None = None, title: str = "Live solve time") -> Any:
    ax = ax or plt.gca()
    if summary.empty:
        ax.text(0.5, 0.5, "No successful runs", ha="center", va="center")
        ax.axis("off")
        return ax
    data = summary.sort_values("solve_ms", ascending=True)
    ax.barh(data["variant"], data["solve_ms"], color="#1f77b4")
    ax.set_xlabel("solve_ms (mean of non-warmup repeats)")
    ax.set_title(title)
    for y, value in enumerate(data["solve_ms"]):
        if np.isfinite(value):
            ax.text(value, y, f" {value:.2f} ms", va="center", fontsize=8)
    return ax


def plot_init_solve_stack(summary: pd.DataFrame, ax: Any | None = None, title: str = "Initialize vs solve") -> Any:
    ax = ax or plt.gca()
    if summary.empty:
        ax.text(0.5, 0.5, "No successful runs", ha="center", va="center")
        ax.axis("off")
        return ax
    data = summary.sort_values("solve_ms", ascending=True)
    y = np.arange(len(data))
    init = pd.to_numeric(data["initialize_ms"], errors="coerce").fillna(0.0)
    solve = pd.to_numeric(data["solve_ms"], errors="coerce").fillna(0.0)
    ax.barh(y, init, color="#ff7f0e", label="initialize")
    ax.barh(y, solve, left=init, color="#1f77b4", label="solve")
    ax.set_yticks(y)
    ax.set_yticklabels(data["variant"])
    ax.set_xlabel("ms")
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def solver_catalog() -> pd.DataFrame:
    return solver_path_table()


def solver_path_table() -> pd.DataFrame:
    """Map tutorial benchmark names to the Newton step component they exercise."""

    return pd.DataFrame(
        [
            {
                "Path": "pandapower PYPOWER-derived NR",
                "Jacobian": "pandapower.pypower dSbus_dV sparse block assembly",
                "Linear solver": "scipy.sparse.linalg.spsolve; SuperLU here unless scikits.umfpack is installed",
                "Benchmark ID": "pypower-pandapower",
            },
            {
                "Path": "MATPOWER default",
                "Jacobian": "MATPOWER makeJac/dSbus_dV NR Jacobian",
                "Linear solver": "MATLAB default sparse solve",
                "Benchmark ID": "matpower-default",
            },
            {
                "Path": "MATPOWER LU5",
                "Jacobian": "MATPOWER makeJac/dSbus_dV NR Jacobian",
                "Linear solver": "MATPOWER pf.nr.lin_solver='LU5'",
                "Benchmark ID": "matpower-lu5",
            },
            {
                "Path": "cuPF CPU comparable path",
                "Jacobian": "Pandapower-like CPU Jacobian",
                "Linear solver": "SuiteSparse UMFPACK or KLU",
                "Benchmark ID": "cupf-cpu-*-pandapower-jac",
            },
            {
                "Path": "cuPF CPU optimized path",
                "Jacobian": "native fixed-pattern Jacobian fill",
                "Linear solver": "KLU with symbolic reuse",
                "Benchmark ID": "cupf-cpu-klu",
            },
            {
                "Path": "cuPF GPU",
                "Jacobian": "CUDA Edge, EdgeAtomic, or VertexWarp",
                "Linear solver": "cuDSS or custom",
                "Benchmark ID": "cupf-fp64-cudss-*",
            },
        ]
    )


def matlab_env_summary() -> pd.DataFrame:
    """Summarize MATLAB/MATPOWER environment without exposing secret values."""

    env_path = repo_root() / ".env"
    keys = [
        "MATLAB_BIN",
        "MATPOWER_HOME",
        "MATLAB_LICMODE",
        "MATLAB_USER_ID",
        "MATLAB_PASSWORD",
        "MATLAB_LICENSE_FILE",
        "MLM_LICENSE_FILE",
    ]
    file_values: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            file_values[key.strip()] = value.strip().strip('"').strip("'")
    rows = []
    for key in keys:
        source = []
        if key in os.environ and os.environ[key]:
            source.append("process env")
        if key in file_values and file_values[key]:
            source.append(".env")
        rows.append(
            {
                "key": key,
                "status": "set" if source else "unset",
                "source": ", ".join(source) if source else "",
                "secret": "yes" if key in {"MATLAB_USER_ID", "MATLAB_PASSWORD"} else "no",
            }
        )
    return pd.DataFrame(rows)


def resolve_case(name: str = "case9", dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> Path:
    return matpower_data.resolve_case_paths(dataset_root, [name])[0]


def load_case(name: str = "case9", dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> matpower_data.PreprocessedCase:
    return matpower_data.load_case(resolve_case(name, dataset_root))


def bus_types(case: matpower_data.PreprocessedCase) -> dict[int, str]:
    out = {int(i): "PQ" for i in range(case.ybus.shape[0])}
    for idx in case.pv:
        out[int(idx)] = "PV"
    for idx in case.ref:
        out[int(idx)] = "Slack"
    return out


def case_graph(case: matpower_data.PreprocessedCase) -> nx.Graph:
    graph = nx.Graph()
    types = bus_types(case)
    pd = case.bus[:, PD] if case.bus.shape[1] > PD else np.zeros(case.ybus.shape[0])
    qd = case.bus[:, QD] if case.bus.shape[1] > QD else np.zeros(case.ybus.shape[0])
    generation_p = np.zeros(case.ybus.shape[0])
    generation_q = np.zeros(case.ybus.shape[0])
    for gen in case.gen:
        if int(gen[GEN_STATUS]) > 0:
            bus = int(gen[GEN_BUS])
            generation_p[bus] += gen[PG]
            generation_q[bus] += gen[QG]
    for bus in range(case.ybus.shape[0]):
        graph.add_node(
            bus,
            label=f"{bus + 1}",
            bus_type=types.get(bus, "PQ"),
            pd=float(pd[bus]),
            qd=float(qd[bus]),
            pg=float(generation_p[bus]),
            qg=float(generation_q[bus]),
        )
    for idx, branch in enumerate(case.branch):
        f_bus = int(branch[F_BUS])
        t_bus = int(branch[T_BUS])
        graph.add_edge(
            f_bus,
            t_bus,
            index=idx,
            resistance=float(branch[BR_R]),
            reactance=float(branch[BR_X]),
        )
    return graph


def stable_layout(graph: nx.Graph) -> dict[int, np.ndarray]:
    try:
        return nx.kamada_kawai_layout(graph)
    except Exception:
        return nx.spring_layout(graph, seed=7)


def plot_case_graph(case: matpower_data.PreprocessedCase, ax: Any | None = None) -> Any:
    ax = ax or plt.gca()
    graph = case_graph(case)
    pos = stable_layout(graph)
    types = nx.get_node_attributes(graph, "bus_type")
    node_colors = [BUS_COLORS[types[node]] for node in graph.nodes]
    node_sizes = [
        500 + 2.5 * (graph.nodes[node]["pd"] + graph.nodes[node]["pg"])
        for node in graph.nodes
    ]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=1.8, alpha=0.55, edge_color="#555555")
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: str(node + 1) for node in graph.nodes},
        ax=ax,
        font_size=9,
        font_color="white",
        font_weight="bold",
    )
    for name, color in BUS_COLORS.items():
        ax.scatter([], [], c=color, s=90, label=name)
    ax.set_title(f"{case.case_name}: bus/branch topology")
    ax.legend(loc="upper right", frameon=False)
    ax.axis("off")
    return ax


def plot_power_balance(case: matpower_data.PreprocessedCase, ax: Any | None = None) -> Any:
    ax = ax or plt.gca()
    n_bus = case.ybus.shape[0]
    load_p = case.bus[:, PD] if case.bus.shape[1] > PD else np.zeros(n_bus)
    gen_p = np.zeros(n_bus)
    for gen in case.gen:
        if int(gen[GEN_STATUS]) > 0:
            gen_p[int(gen[GEN_BUS])] += gen[PG]
    x = np.arange(n_bus) + 1
    ax.bar(x - 0.18, gen_p, width=0.36, label="Generation P", color="#1f77b4")
    ax.bar(x + 0.18, load_p, width=0.36, label="Load P", color="#ff7f0e")
    ax.set_xlabel("Bus")
    ax.set_ylabel("MW")
    ax.set_title("Where power is injected and consumed")
    ax.legend(frameon=False)
    return ax


def plot_ybus(case: matpower_data.PreprocessedCase, ax: Any | None = None) -> Any:
    ax = ax or plt.gca()
    ax.spy(case.ybus, markersize=7, color="#1f77b4")
    ax.set_title(f"Ybus sparsity: {case.ybus.nnz} nonzeros")
    ax.set_xlabel("Column bus")
    ax.set_ylabel("Row bus")
    return ax


def plot_voltage_phasors(voltage: np.ndarray, ax: Any | None = None, title: str = "Voltage phasors") -> Any:
    ax = ax or plt.gca()
    voltage = np.asarray(voltage, dtype=np.complex128)
    origin = np.zeros(voltage.size)
    ax.quiver(
        origin,
        origin,
        voltage.real,
        voltage.imag,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#1f77b4",
        width=0.005,
    )
    for idx, value in enumerate(voltage):
        ax.text(value.real * 1.04, value.imag * 1.04, str(idx + 1), fontsize=8)
    lim = max(1.15, float(np.max(np.abs(voltage))) * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="#dddddd", linewidth=0.8)
    ax.axvline(0, color="#dddddd", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    return ax


def mismatch_vector(case: matpower_data.PreprocessedCase, voltage: np.ndarray) -> np.ndarray:
    return matpower_data.mismatch_vector(case.ybus, case.sbus, voltage, case.pv, case.pq)


def power_flow_snapshot(case: matpower_data.PreprocessedCase, voltage: np.ndarray) -> PowerFlowSnapshot:
    """Evaluate S_calc(V) and the reduced Newton mismatch at one voltage."""

    voltage = np.asarray(voltage, dtype=np.complex128)
    ibus = case.ybus @ voltage
    s_calc = voltage * np.conj(ibus)
    mismatch_complex = case.sbus - s_calc
    return PowerFlowSnapshot(
        voltage=voltage,
        ibus=np.asarray(ibus).reshape(-1),
        s_calc=np.asarray(s_calc).reshape(-1),
        s_spec=np.asarray(case.sbus).reshape(-1),
        mismatch_complex=np.asarray(mismatch_complex).reshape(-1),
        mismatch_reduced=mismatch_vector(case, voltage),
    )


def power_flow_bus_table(snapshot: PowerFlowSnapshot, limit: int = 12) -> pd.DataFrame:
    """Small bus-level table for connecting the equation to actual numbers."""

    n = min(limit, snapshot.voltage.size)
    rows = []
    for bus in range(n):
        rows.append(
            {
                "bus": bus + 1,
                "|V|": abs(snapshot.voltage[bus]),
                "Va_deg": math.degrees(math.atan2(snapshot.voltage[bus].imag, snapshot.voltage[bus].real)),
                "P_spec": snapshot.s_spec[bus].real,
                "Q_spec": snapshot.s_spec[bus].imag,
                "P_calc": snapshot.s_calc[bus].real,
                "Q_calc": snapshot.s_calc[bus].imag,
                "P_mis": snapshot.mismatch_complex[bus].real,
                "Q_mis": snapshot.mismatch_complex[bus].imag,
            }
        )
    return pd.DataFrame(rows)


def build_reduced_jacobian(case: matpower_data.PreprocessedCase, voltage: np.ndarray) -> sp.csr_matrix:
    pvpq = np.r_[case.pv, case.pq]
    dS_dVm, dS_dVa = dSbus_dV(case.ybus, voltage)
    j11 = dS_dVa[np.ix_(pvpq, pvpq)].real
    j12 = dS_dVm[np.ix_(pvpq, case.pq)].real
    j21 = dS_dVa[np.ix_(case.pq, pvpq)].imag
    j22 = dS_dVm[np.ix_(case.pq, case.pq)].imag
    return vstack([hstack([j11, j12]), hstack([j21, j22])], format="csr")


def jacobian_block_shapes(case: matpower_data.PreprocessedCase) -> pd.DataFrame:
    """Explain the reduced Jacobian dimensions before showing the sparse plot."""

    npv = len(case.pv)
    npq = len(case.pq)
    n_pvpq = npv + npq
    return pd.DataFrame(
        [
            {"block": "J11 = dP/dVa", "rows": n_pvpq, "cols": n_pvpq, "equations": "P at PV+PQ", "unknowns": "Va at PV+PQ"},
            {"block": "J12 = dP/dVm", "rows": n_pvpq, "cols": npq, "equations": "P at PV+PQ", "unknowns": "Vm at PQ"},
            {"block": "J21 = dQ/dVa", "rows": npq, "cols": n_pvpq, "equations": "Q at PQ", "unknowns": "Va at PV+PQ"},
            {"block": "J22 = dQ/dVm", "rows": npq, "cols": npq, "equations": "Q at PQ", "unknowns": "Vm at PQ"},
        ]
    )


def newton_step_snapshot(case: matpower_data.PreprocessedCase, voltage: np.ndarray) -> NewtonStepSnapshot:
    """Perform one transparent Newton update using the same reduced system as PYPOWER/MATPOWER."""

    voltage = np.asarray(voltage, dtype=np.complex128)
    pvpq = np.r_[case.pv, case.pq]
    jac = build_reduced_jacobian(case, voltage)
    mismatch = mismatch_vector(case, voltage)
    dx = -np.asarray(spsolve(jac, mismatch)).reshape(-1)
    va = np.angle(voltage).copy()
    vm = np.abs(voltage).copy()
    n_pvpq = len(pvpq)
    va[pvpq] += dx[:n_pvpq]
    vm[case.pq] += dx[n_pvpq:]
    next_voltage = vm * np.exp(1j * va)
    return NewtonStepSnapshot(voltage, mismatch, jac, dx, next_voltage, pvpq)


def newton_step_table(step: NewtonStepSnapshot, limit: int = 12) -> pd.DataFrame:
    """Show only the head of dx; the full vector is usually too long for a notebook."""

    n_angle = len(step.pvpq)
    rows = []
    for idx, value in enumerate(step.dx[: min(limit, step.dx.size)]):
        if idx < n_angle:
            name = f"Va[{int(step.pvpq[idx]) + 1}]"
        else:
            pq_idx = idx - n_angle
            name = f"Vm(PQ position {int(pq_idx) + 1})"
        rows.append({"unknown": name, "dx": value})
    return pd.DataFrame(rows)


def plot_jacobian_blocks(case: matpower_data.PreprocessedCase, voltage: np.ndarray, ax: Any | None = None) -> Any:
    ax = ax or plt.gca()
    jac = build_reduced_jacobian(case, voltage)
    ax.spy(jac, markersize=2, color="#d62728")
    npv = len(case.pv)
    npq = len(case.pq)
    n_pvpq = npv + npq
    ax.axhline(n_pvpq - 0.5, color="#333333", linewidth=0.9)
    ax.axvline(n_pvpq - 0.5, color="#333333", linewidth=0.9)
    ax.set_title(f"Reduced Jacobian sparsity ({jac.shape[0]} x {jac.shape[1]})")
    ax.set_xlabel("Unknowns: angle | magnitude")
    ax.set_ylabel("Equations: P | Q")
    ax.text(n_pvpq * 0.35, n_pvpq * 0.20, "J11", ha="center", va="center")
    ax.text(n_pvpq + max(npq, 1) * 0.45, n_pvpq * 0.20, "J12", ha="center", va="center")
    ax.text(n_pvpq * 0.35, n_pvpq + max(npq, 1) * 0.45, "J21", ha="center", va="center")
    ax.text(n_pvpq + max(npq, 1) * 0.45, n_pvpq + max(npq, 1) * 0.45, "J22", ha="center", va="center")
    return ax


def plot_jacobian_block_pattern(case: matpower_data.PreprocessedCase, voltage: np.ndarray, ax: Any | None = None) -> Any:
    """Alias with a name that states why this sparse plot exists."""

    return plot_jacobian_blocks(case, voltage, ax)


def run_newton_with_stage_timing(
    case: matpower_data.PreprocessedCase,
    tolerance: float = 1e-8,
    max_iter: int = 50,
) -> NewtonTrace:
    voltage = case.v0.astype(np.complex128, copy=True)
    va = np.angle(voltage)
    vm = np.abs(voltage)
    pvpq = np.r_[case.pv, case.pq]
    n_pvpq = len(pvpq)
    rows: list[dict[str, float | int | str]] = []
    converged = False
    final_mismatch = math.inf
    iterations = 0

    for iteration in range(1, max_iter + 1):
        t0 = time.perf_counter()
        f = mismatch_vector(case, voltage)
        mismatch_ms = (time.perf_counter() - t0) * 1000.0
        final_mismatch = float(np.linalg.norm(f, np.inf)) if f.size else 0.0
        rows.append(
            {
                "iteration": iteration,
                "stage": "Mismatch",
                "time_ms": mismatch_ms,
                "mismatch_norm": final_mismatch,
            }
        )
        if final_mismatch < tolerance:
            converged = True
            iterations = iteration - 1
            break

        t0 = time.perf_counter()
        jac = build_reduced_jacobian(case, voltage)
        rows.append(
            {
                "iteration": iteration,
                "stage": "Jacobian",
                "time_ms": (time.perf_counter() - t0) * 1000.0,
                "mismatch_norm": final_mismatch,
            }
        )

        t0 = time.perf_counter()
        dx = -np.asarray(spsolve(jac, f)).reshape(-1)
        rows.append(
            {
                "iteration": iteration,
                "stage": "Linear solve",
                "time_ms": (time.perf_counter() - t0) * 1000.0,
                "mismatch_norm": final_mismatch,
            }
        )

        t0 = time.perf_counter()
        va[pvpq] += dx[:n_pvpq]
        vm[case.pq] += dx[n_pvpq:]
        voltage = vm * np.exp(1j * va)
        vm = np.abs(voltage)
        va = np.angle(voltage)
        rows.append(
            {
                "iteration": iteration,
                "stage": "Voltage update",
                "time_ms": (time.perf_counter() - t0) * 1000.0,
                "mismatch_norm": final_mismatch,
            }
        )
        iterations = iteration

    df = pd.DataFrame(rows)
    totals = df.groupby("stage")["time_ms"].sum().sort_values(ascending=False)
    return NewtonTrace(voltage, df, totals, converged, iterations, final_mismatch)


def newton_trace(
    case: matpower_data.PreprocessedCase,
    tolerance: float = 1e-8,
    max_iter: int = 50,
) -> NewtonTrace:
    """Educational name for the staged Newton loop used in the notebooks."""

    return run_newton_with_stage_timing(case, tolerance=tolerance, max_iter=max_iter)


def plot_convergence(trace: NewtonTrace, ax: Any | None = None) -> Any:
    ax = ax or plt.gca()
    mismatch = (
        trace.rows[trace.rows["stage"] == "Mismatch"][["iteration", "mismatch_norm"]]
        .drop_duplicates()
        .sort_values("iteration")
    )
    ax.semilogy(mismatch["iteration"], mismatch["mismatch_norm"], marker="o", color="#1f77b4")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("max |F|")
    ax.set_title("Residual convergence")
    ax.grid(True, which="both", alpha=0.25)
    return ax


def plot_newton_convergence(trace: NewtonTrace, ax: Any | None = None) -> Any:
    """Alias used by the notebooks after the Newton loop has been introduced."""

    return plot_convergence(trace, ax)


def plot_stage_pie(stage_ms: pd.Series | dict[str, float], ax: Any | None = None, title: str = "") -> Any:
    ax = ax or plt.gca()
    series = pd.Series(stage_ms, dtype=float)
    series = series[series > 0].sort_values(ascending=False)
    colors = [STAGE_COLORS.get(name, STAGE_COLORS["Other"]) for name in series.index]
    ax.pie(
        series.values,
        labels=series.index,
        autopct=lambda pct: f"{pct:.0f}%" if pct >= 5 else "",
        startangle=90,
        counterclock=False,
        colors=colors,
        textprops={"fontsize": 9},
    )
    ax.set_title(title or f"Stage share ({series.sum():.1f} ms)")
    return ax


def plot_stage_bar(stage_ms: pd.Series | dict[str, float], ax: Any | None = None, title: str = "") -> Any:
    ax = ax or plt.gca()
    series = pd.Series(stage_ms, dtype=float).sort_values(ascending=True)
    colors = [STAGE_COLORS.get(name, STAGE_COLORS["Other"]) for name in series.index]
    ax.barh(series.index, series.values, color=colors)
    ax.set_xlabel("ms")
    ax.set_title(title)
    for y, value in enumerate(series.values):
        ax.text(value, y, f" {value:.2f}", va="center", fontsize=8)
    return ax


def plot_stage_timing(stage_ms: pd.Series | dict[str, float], ax: Any | None = None, title: str = "") -> Any:
    """Use a bar chart as the default bottleneck view; it preserves absolute time."""

    return plot_stage_bar(stage_ms, ax, title)


def plot_variant_timing(summary: pd.DataFrame, ax: Any | None = None, title: str = "Variant solve time") -> Any:
    """Compare solver variants using the same solve_ms column used by runs.csv."""

    return plot_run_solve_bars(summary, ax, title)


def load_benchmark_runs(result_root: str | Path | None = None) -> pd.DataFrame:
    if result_root is None:
        result_root = tutorial_output_root()
    result_root = Path(result_root)
    frames = []
    for path in sorted(result_root.glob("*/runs.csv")):
        frame = pd.read_csv(path)
        frame["variant_dir"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_benchmark(result_root: str | Path | None = None) -> pd.DataFrame:
    runs = load_benchmark_runs(result_root)
    if runs.empty:
        return runs
    runs = runs[(runs["success"].astype(str).isin(["1", "True", "true"])) & (runs["warmup"].astype(str).isin(["0", "False", "false"]))]
    rows = []
    for variant, group in runs.groupby("variant"):
        solve = pd.to_numeric(group["solve_ms"], errors="coerce")
        solve = solve[solve > 0]
        init = pd.to_numeric(group["initialize_ms"], errors="coerce")
        rows.append(
            {
                "variant": variant,
                "cases": group["case_name"].nunique(),
                "solve_geomean_ms": float(np.exp(np.log(solve).mean())) if len(solve) else math.nan,
                "initialize_mean_ms": float(init.mean()) if len(init) else math.nan,
                "converged": int(pd.to_numeric(group["converged"], errors="coerce").fillna(0).sum()),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("solve_geomean_ms")


def plot_variant_solve_bars(summary: pd.DataFrame, ax: Any | None = None) -> Any:
    ax = ax or plt.gca()
    data = summary.sort_values("solve_geomean_ms", ascending=True)
    ax.barh(data["variant"], data["solve_geomean_ms"], color="#1f77b4")
    ax.set_xlabel("Geomean solve time (ms, log scale)")
    ax.set_xscale("log")
    ax.set_title("Representative benchmark variants")
    for y, value in enumerate(data["solve_geomean_ms"]):
        ax.text(value, y, f" {value:.3g} ms", va="center", fontsize=8)
    return ax


def native_timing_stages(result_root: str | Path, variant: str, case_name: str) -> pd.Series:
    path = Path(result_root) / variant / "timing.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    timing = pd.read_csv(path)
    timing = timing[timing["case_name"] == case_name].copy()
    if timing.empty:
        return pd.Series(dtype=float)
    timing["ms"] = pd.to_numeric(timing["total_us"], errors="coerce") / 1000.0
    means = timing.groupby("timer_name")["ms"].mean()
    linear = sum(means.get(name, 0.0) for name in ["NR.iteration.prepare_rhs", "NR.iteration.factorize", "NR.iteration.solve"])
    stages = pd.Series(
        {
            "Linear solve": linear,
            "Jacobian": means.get("NR.iteration.jacobian", 0.0),
            "Ibus": means.get("NR.iteration.ibus", 0.0),
            "Mismatch": means.get("NR.iteration.mismatch", 0.0),
            "Mismatch norm": means.get("NR.iteration.mismatch_norm", 0.0),
            "Voltage update": means.get("NR.iteration.voltage_update", 0.0),
            "Upload/download": means.get("NR.solve.upload", 0.0) + means.get("NR.solve.download", 0.0),
        }
    )
    return stages[stages > 0].sort_values(ascending=False)


def acceleration_story_stages(cpu: pd.Series, gpu: pd.Series) -> pd.DataFrame:
    cpu = pd.Series(cpu, dtype=float)
    gpu = pd.Series(gpu, dtype=float)
    linear_gpu = cpu.copy()
    linear_gpu["Linear solve"] = gpu.get("Linear solve", linear_gpu.get("Linear solve", 0.0))
    jac_gpu = linear_gpu.copy()
    jac_gpu["Jacobian"] = gpu.get("Jacobian", jac_gpu.get("Jacobian", 0.0))
    full_gpu = gpu.copy()
    frames = []
    for label, series, kind in [
        ("Python/SciPy baseline", cpu, "measured"),
        ("+ GPU linear solve", linear_gpu, "component estimate"),
        ("+ GPU linear solve + Jacobian", jac_gpu, "component estimate"),
        ("cuPF full GPU path", full_gpu, "measured"),
    ]:
        row = {"step": label, "kind": kind, "total_ms": float(series.sum())}
        row.update(series.to_dict())
        frames.append(row)
    return pd.DataFrame(frames)


def kcc_vertex_edge_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Case": "ACTIVSg25k", "Edge atomic us": 17.765, "Vertex us": 35.775, "Edge/Vertex speedup": 2.014, "Atomic time share %": 28.2},
            {"Case": "Base Eastern", "Edge atomic us": 51.903, "Vertex us": 116.289, "Edge/Vertex speedup": 2.241, "Atomic time share %": 15.2},
            {"Case": "Base West", "Edge atomic us": 17.156, "Vertex us": 31.284, "Edge/Vertex speedup": 1.823, "Atomic time share %": 28.3},
            {"Case": "Memphis", "Edge atomic us": 7.422, "Vertex us": 7.939, "Edge/Vertex speedup": 1.070, "Atomic time share %": 0.0},
            {"Case": "Texas7K", "Edge atomic us": 9.712, "Vertex us": 15.469, "Edge/Vertex speedup": 1.593, "Atomic time share %": 11.8},
        ]
    )


@contextmanager
def prepend_sys_path(paths: Iterable[Path]):
    original = list(sys.path)
    for path in reversed([str(Path(p)) for p in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


def cupf_build_paths(kind: str = "cpu") -> list[Path]:
    root = repo_root()
    build = root / "cuPF" / "build" / f"eval-{kind}"
    return [build, root / "cuPF" / "python"]


def import_cupf_from_build(kind: str = "cpu") -> tuple[Any | None, str]:
    try:
        with prepend_sys_path(cupf_build_paths(kind)):
            return importlib.import_module("cupf"), f"imported from eval-{kind}"
    except Exception as exc:
        return None, f"cuPF import skipped: {type(exc).__name__}: {exc}"


def solve_case_with_cupf(
    case: matpower_data.PreprocessedCase,
    *,
    kind: str = "cpu",
    backend: str = "cpu",
    compute: str = "fp64",
    cpu_jacobian: str = "native",
    cpu_linear_solver: str = "klu",
    cuda_jacobian: str = "edge",
    cuda_linear_solver: str = "cudss",
    tolerance: float = 1e-8,
    max_iter: int = 50,
) -> tuple[Any | None, str]:
    cupf, message = import_cupf_from_build(kind)
    if cupf is None:
        return None, message
    options = cupf.NewtonOptions()
    options.backend = cupf.BackendKind.CUDA if backend == "cuda" else cupf.BackendKind.CPU
    if compute == "mixed":
        options.compute = cupf.ComputePolicy.Mixed
    elif compute == "fp32":
        options.compute = cupf.ComputePolicy.FP32
    else:
        options.compute = cupf.ComputePolicy.FP64
    if hasattr(cupf, "CpuJacobianKind"):
        options.cpu_jacobian = {
            "native": cupf.CpuJacobianKind.Native,
            "pandapower": cupf.CpuJacobianKind.Pandapower,
        }[cpu_jacobian]
    if hasattr(cupf, "CpuLinearSolverKind"):
        options.cpu_linear_solver = {
            "klu": cupf.CpuLinearSolverKind.KLU,
            "umfpack": cupf.CpuLinearSolverKind.UMFPACK,
        }[cpu_linear_solver]
    if hasattr(cupf, "CudaJacobianKind"):
        options.cuda_jacobian = {
            "edge": cupf.CudaJacobianKind.Edge,
            "edge_atomic": cupf.CudaJacobianKind.EdgeAtomic,
            "vertex_warp": cupf.CudaJacobianKind.VertexWarp,
        }[cuda_jacobian]
    if backend == "cuda" and cuda_linear_solver == "custom":
        options.cuda_linear_solver = cupf.CudaLinearSolverKind.Custom
    solver = cupf.NewtonSolver(options)
    solver.initialize(
        case.ybus.indptr,
        case.ybus.indices,
        case.ybus.data,
        case.ybus.shape[0],
        case.ybus.shape[1],
        case.pv,
        case.pq,
    )
    config = cupf.NRConfig()
    config.tolerance = tolerance
    config.max_iter = max_iter
    result = solver.solve(
        case.ybus.indptr,
        case.ybus.indices,
        case.ybus.data,
        case.ybus.shape[0],
        case.ybus.shape[1],
        case.sbus,
        case.v0,
        case.pv,
        case.pq,
        config,
    )
    return result, message


def print_environment_note() -> str:
    try:
        import scikits.umfpack  # type: ignore  # noqa: F401
    except Exception:
        umfpack = "not installed; SciPy spsolve uses SuperLU-style path here"
    else:
        umfpack = "installed; SciPy may use UMFPACK when requested"
    return f"pandapower.pypower.newtonpf calls scipy.sparse.linalg.spsolve; scikits.umfpack is {umfpack}."
