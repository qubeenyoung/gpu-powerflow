#!/usr/bin/env python3
import argparse
import csv
import math
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def value_as_us(value, unit):
    if value is None:
        return None
    unit = (unit or "").lower()
    if unit in {"ns", "nsecond"}:
        return value / 1000.0
    if unit in {"us", "usecond"}:
        return value
    if unit in {"ms", "msecond"}:
        return value * 1000.0
    if unit in {"s", "second"}:
        return value * 1000000.0
    return value


def read_timing(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_ncu_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = None
        rows = []
        for row in reader:
            if not row:
                continue
            if header is None:
                if row[0] == "ID":
                    header = row
                continue
            if len(row) < len(header):
                continue
            rows.append(dict(zip(header, row)))
    if header is None:
        raise RuntimeError(f"could not find NCU CSV header in {path}")
    return rows


def kernel_kind(kernel_name):
    if "fill_jacobian_edge(" in kernel_name:
        return "edge"
    if "fill_jacobian_vertex_warp(" in kernel_name:
        return "vertex_warp"
    if "fill_jacobian_vertex(" in kernel_name:
        return "vertex_thread"
    return "other"


def group_ncu(rows):
    groups = OrderedDict()
    for row in rows:
        group_id = row.get("ID", "")
        if not group_id.isdigit():
            continue
        group = groups.setdefault(group_id, {
            "id": int(group_id),
            "kernel": row.get("Kernel Name", ""),
            "kind": kernel_kind(row.get("Kernel Name", "")),
            "block_size": row.get("Block Size", ""),
            "grid_size": row.get("Grid Size", ""),
            "metrics": {},
            "units": {},
        })
        metric = row.get("Metric Name", "")
        value = parse_float(row.get("Metric Value", ""))
        group["metrics"][metric] = value
        group["units"][metric] = row.get("Metric Unit", "")
    return list(groups.values())


def metric(group, name):
    return group["metrics"].get(name)


def metric_us(group, name):
    return value_as_us(group["metrics"].get(name), group["units"].get(name, ""))


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def pct(numerator, denominator):
    value = ratio(numerator, denominator)
    if value is None:
        return None
    return 100.0 * value


def metric_sum(group, base, suffixes):
    total = 0.0
    seen = False
    for suffix in suffixes:
        value = metric(group, f"{base}{suffix}.sum")
        if value is not None:
            total += value
            seen = True
    return total if seen else None


def sass_bytes_per_sector(group, op):
    return metric(group, f"smsp__sass_average_data_bytes_per_sector_mem_global_op_{op}.ratio")


def weighted_average(items):
    weighted_sum = 0.0
    weight_sum = 0.0
    for value, weight in items:
        if value is None or weight is None or weight == 0.0:
            continue
        weighted_sum += value * weight
        weight_sum += weight
    if weight_sum == 0.0:
        return None
    return weighted_sum / weight_sum


def assign_cases(groups, timing_rows):
    cases = [row["case"] for row in timing_rows]
    if not cases:
        return []
    if len(groups) % len(cases) != 0:
        raise RuntimeError(
            f"NCU kernel count {len(groups)} is not divisible by timing case count {len(cases)}"
        )

    kernels_per_case = len(groups) // len(cases)
    records = []
    for index, group in enumerate(groups):
        case_index = index // kernels_per_case
        timing = timing_rows[case_index]
        gld_requests = metric(group, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
        gld_sectors = metric(group, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        write_suffixes = ("_op_st", "_op_atom", "_op_red")
        gst_sectors = metric(group, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
        gatom_sectors = metric(group, "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum")
        gred_sectors = metric(group, "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum")
        gwrite_requests = metric_sum(group, "l1tex__t_requests_pipe_lsu_mem_global", write_suffixes)
        gwrite_sectors = metric_sum(group, "l1tex__t_sectors_pipe_lsu_mem_global", write_suffixes)
        gld_bytes_per_sector = sass_bytes_per_sector(group, "ld")
        gwrite_bytes_per_sector = weighted_average((
            (sass_bytes_per_sector(group, "st"), gst_sectors),
            (sass_bytes_per_sector(group, "atom"), gatom_sectors),
            (sass_bytes_per_sector(group, "red"), gred_sectors),
        ))
        gld_spr_metric = metric(
            group,
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
        )
        records.append({
            "case": cases[case_index],
            "kernel": group["kind"],
            "n_bus": timing.get("n_bus", ""),
            "ybus_nnz": timing.get("ybus_nnz", ""),
            "jac_nnz": timing.get("jac_nnz", ""),
            "ncu_time_us": metric_us(group, "gpu__time_duration.sum"),
            "occ_pct": metric(group, "sm__warps_active.avg.pct_of_peak_sustained_active"),
            "lane_util_pct": metric(group, "smsp__thread_inst_executed_per_inst_executed.pct"),
            "avg_lanes": metric(group, "smsp__thread_inst_executed_per_inst_executed.ratio"),
            "thread_inst_sum": metric(group, "smsp__thread_inst_executed.sum"),
            "inst_sum": metric(group, "smsp__inst_executed.sum"),
            "thread_count": metric(group, "launch__thread_count"),
            "registers_per_thread": metric(group, "launch__registers_per_thread"),
            "gld_requests": gld_requests,
            "gld_sectors": gld_sectors,
            "gld_bytes_per_sector": gld_bytes_per_sector,
            "gld_sectors_per_request": (
                gld_spr_metric
                if gld_spr_metric is not None
                else ratio(gld_sectors, gld_requests)
            ),
            "gld_sector_util_pct": pct(gld_bytes_per_sector, 32.0),
            "gst_requests": metric(group, "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"),
            "gatom_requests": metric(group, "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"),
            "gred_requests": metric(group, "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum"),
            "gst_sectors": gst_sectors,
            "gatom_sectors": gatom_sectors,
            "gred_sectors": gred_sectors,
            "gwrite_requests": gwrite_requests,
            "gwrite_sectors": gwrite_sectors,
            "gwrite_bytes_per_sector": gwrite_bytes_per_sector,
            "gwrite_sectors_per_request": ratio(gwrite_sectors, gwrite_requests),
            "gwrite_sector_util_pct": pct(gwrite_bytes_per_sector, 32.0),
            "block_size": group["block_size"],
            "grid_size": group["grid_size"],
        })
    return records


def fmt(value, places=2):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        value = parse_float(value)
        if value is None:
            return ""
    return f"{value:.{places}f}"


def fmt_int(value):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        value = parse_float(value)
        if value is None:
            return ""
    return f"{int(round(value))}"


def geomean(values):
    filtered = [v for v in values if v is not None and v > 0.0]
    if not filtered:
        return None
    return math.exp(sum(math.log(v) for v in filtered) / len(filtered))


def mean(values):
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


BUS_SIZE_BINS = (
    ("<=100", 0, 100),
    ("101-999", 101, 999),
    ("1k-9,999", 1000, 9999),
    ("10k-49,999", 10000, 49999),
    (">=50k", 50000, None),
)


def bus_size_bin(n_bus):
    for label, lo, hi in BUS_SIZE_BINS:
        if n_bus >= lo and (hi is None or n_bus <= hi):
            return label
    return "other"


def build_bin_rows(timing_rows, by_case):
    bins = OrderedDict((label, []) for label, _, _ in BUS_SIZE_BINS)
    for row in timing_rows:
        n_bus = parse_float(row.get("n_bus", ""))
        if n_bus is None:
            continue
        label = bus_size_bin(int(n_bus))
        if label not in bins:
            bins[label] = []
        edge = by_case[row["case"]].get("edge", {})
        vertex = by_case[row["case"]].get("vertex_thread", {})
        edge_fill = parse_float(row.get("edge_fill_ms", ""))
        vertex_fill = parse_float(row.get("vertex_thread_fill_ms", ""))
        bins[label].append({
            "n_bus": n_bus,
            "edge_fill": edge_fill,
            "vertex_fill": vertex_fill,
            "vertex_over_edge": ratio(vertex_fill, edge_fill),
            "edge_lane": edge.get("lane_util_pct"),
            "vertex_lane": vertex.get("lane_util_pct"),
            "edge_occ": edge.get("occ_pct"),
            "vertex_occ": vertex.get("occ_pct"),
            "edge_ld_spr": edge.get("gld_sectors_per_request"),
            "vertex_ld_spr": vertex.get("gld_sectors_per_request"),
            "edge_wr_spr": edge.get("gwrite_sectors_per_request"),
            "vertex_wr_spr": vertex.get("gwrite_sectors_per_request"),
            "edge_ld_util": edge.get("gld_sector_util_pct"),
            "vertex_ld_util": vertex.get("gld_sector_util_pct"),
            "edge_wr_util": edge.get("gwrite_sector_util_pct"),
            "vertex_wr_util": vertex.get("gwrite_sector_util_pct"),
        })

    rows = []
    for label, values in bins.items():
        if not values:
            continue
        rows.append({
            "label": label,
            "count": len(values),
            "avg_n_bus": mean(v["n_bus"] for v in values),
            "edge_fill": mean(v["edge_fill"] for v in values),
            "vertex_fill": mean(v["vertex_fill"] for v in values),
            "vertex_over_edge": geomean(v["vertex_over_edge"] for v in values),
            "edge_lane": mean(v["edge_lane"] for v in values),
            "vertex_lane": mean(v["vertex_lane"] for v in values),
            "edge_occ": mean(v["edge_occ"] for v in values),
            "vertex_occ": mean(v["vertex_occ"] for v in values),
            "edge_ld_spr": mean(v["edge_ld_spr"] for v in values),
            "vertex_ld_spr": mean(v["vertex_ld_spr"] for v in values),
            "edge_wr_spr": mean(v["edge_wr_spr"] for v in values),
            "vertex_wr_spr": mean(v["vertex_wr_spr"] for v in values),
            "edge_ld_util": mean(v["edge_ld_util"] for v in values),
            "vertex_ld_util": mean(v["vertex_ld_util"] for v in values),
            "edge_wr_util": mean(v["edge_wr_util"] for v in values),
            "vertex_wr_util": mean(v["vertex_wr_util"] for v in values),
        })
    return rows


def write_summary_csv(path, records):
    fields = [
        "case",
        "kernel",
        "n_bus",
        "ybus_nnz",
        "jac_nnz",
        "ncu_time_us",
        "occ_pct",
        "lane_util_pct",
        "avg_lanes",
        "thread_count",
        "registers_per_thread",
        "gld_sectors_per_request",
        "gld_sector_util_pct",
        "gld_requests",
        "gld_sectors",
        "gld_bytes_per_sector",
        "gwrite_sectors_per_request",
        "gwrite_sector_util_pct",
        "gwrite_requests",
        "gwrite_sectors",
        "gwrite_bytes_per_sector",
        "gst_requests",
        "gatom_requests",
        "gred_requests",
        "gst_sectors",
        "gatom_sectors",
        "gred_sectors",
        "inst_sum",
        "thread_inst_sum",
        "block_size",
        "grid_size",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def write_summary_md(path, timing_rows, records, args):
    by_case = defaultdict(dict)
    for record in records:
        by_case[record["case"]][record["kernel"]] = record
    bin_rows = build_bin_rows(timing_rows, by_case)

    ratios = []
    for row in timing_rows:
        edge = parse_float(row.get("edge_fill_ms", ""))
        vertex = parse_float(row.get("vertex_thread_fill_ms", ""))
        if edge and vertex:
            ratios.append(vertex / edge)

    edge_lane = [r["lane_util_pct"] for r in records if r["kernel"] == "edge" and r["lane_util_pct"] is not None]
    vertex_lane = [r["lane_util_pct"] for r in records if r["kernel"] == "vertex_thread" and r["lane_util_pct"] is not None]
    edge_occ = [r["occ_pct"] for r in records if r["kernel"] == "edge" and r["occ_pct"] is not None]
    vertex_occ = [r["occ_pct"] for r in records if r["kernel"] == "vertex_thread" and r["occ_pct"] is not None]
    edge_gld_spr = [r["gld_sectors_per_request"] for r in records if r["kernel"] == "edge" and r["gld_sectors_per_request"] is not None]
    vertex_gld_spr = [r["gld_sectors_per_request"] for r in records if r["kernel"] == "vertex_thread" and r["gld_sectors_per_request"] is not None]
    edge_gwrite_spr = [r["gwrite_sectors_per_request"] for r in records if r["kernel"] == "edge" and r["gwrite_sectors_per_request"] is not None]
    vertex_gwrite_spr = [r["gwrite_sectors_per_request"] for r in records if r["kernel"] == "vertex_thread" and r["gwrite_sectors_per_request"] is not None]
    edge_gld_util = [r["gld_sector_util_pct"] for r in records if r["kernel"] == "edge" and r["gld_sector_util_pct"] is not None]
    vertex_gld_util = [r["gld_sector_util_pct"] for r in records if r["kernel"] == "vertex_thread" and r["gld_sector_util_pct"] is not None]
    edge_gwrite_util = [r["gwrite_sector_util_pct"] for r in records if r["kernel"] == "edge" and r["gwrite_sector_util_pct"] is not None]
    vertex_gwrite_util = [r["gwrite_sector_util_pct"] for r in records if r["kernel"] == "vertex_thread" and r["gwrite_sector_util_pct"] is not None]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(path, "w", newline="") as f:
        f.write("# Jacobian Assembly MATPOWER Lane And Coalescing Metrics\n\n")
        f.write("## Environment\n\n")
        f.write(f"- Generated: `{generated}`\n")
        f.write(f"- Data root: `{args.data_root}`\n")
        f.write(f"- Timing command mode: `{args.mode}`\n")
        f.write(f"- Timing iterations: `{args.timing_iters}`, warmup: `{args.timing_warmup}`\n")
        f.write(f"- NCU iterations: `{args.ncu_iters}`, warmup: `{args.ncu_warmup}`\n")
        f.write("- NCU metrics include lane utilization plus L1TEX global load/store/atomic/reduction request, sector, and byte counters.\n")
        f.write("\n## Definitions\n\n")
        f.write("- `fill ms` is the CUDA event average from the timing run.\n")
        f.write("- `occupancy % = sm__warps_active.avg.pct_of_peak_sustained_active`.\n")
        f.write("- `lane util % = smsp__thread_inst_executed_per_inst_executed.pct`.\n")
        f.write("- `avg lanes = smsp__thread_inst_executed_per_inst_executed.ratio`.\n")
        f.write("- `global load sectors/request = l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`; lower generally means better coalescing.\n")
        f.write("- `global write sectors/request` combines `st + atom + red` operations for edge/vertex comparability; lower generally means better coalescing.\n")
        f.write("- `sector util % = smsp__sass_average_data_bytes_per_sector / 32 * 100`; higher means each 32-byte sector carried more useful data bytes.\n")
        f.write("- `ncu us` is from the separate NCU run and is not the timing comparison source.\n")
        f.write("\n## Rollup\n\n")
        f.write(f"- Cases: `{len(timing_rows)}`\n")
        if ratios:
            f.write(f"- Geomean vertex/thread fill over edge fill: `{geomean(ratios):.3f}x`\n")
        if edge_lane:
            f.write(f"- Edge lane util avg: `{sum(edge_lane) / len(edge_lane):.2f}%`\n")
        if vertex_lane:
            f.write(f"- Vertex-thread lane util avg: `{sum(vertex_lane) / len(vertex_lane):.2f}%`\n")
        if edge_occ:
            f.write(f"- Edge occupancy avg: `{sum(edge_occ) / len(edge_occ):.2f}%`\n")
        if vertex_occ:
            f.write(f"- Vertex-thread occupancy avg: `{sum(vertex_occ) / len(vertex_occ):.2f}%`\n")
        if edge_gld_spr:
            f.write(f"- Edge global load sectors/request avg: `{sum(edge_gld_spr) / len(edge_gld_spr):.2f}`\n")
        if vertex_gld_spr:
            f.write(f"- Vertex-thread global load sectors/request avg: `{sum(vertex_gld_spr) / len(vertex_gld_spr):.2f}`\n")
        if edge_gwrite_spr:
            f.write(f"- Edge global write sectors/request avg: `{sum(edge_gwrite_spr) / len(edge_gwrite_spr):.2f}`\n")
        if vertex_gwrite_spr:
            f.write(f"- Vertex-thread global write sectors/request avg: `{sum(vertex_gwrite_spr) / len(vertex_gwrite_spr):.2f}`\n")
        if edge_gld_util:
            f.write(f"- Edge global load sector util avg: `{sum(edge_gld_util) / len(edge_gld_util):.2f}%`\n")
        if vertex_gld_util:
            f.write(f"- Vertex-thread global load sector util avg: `{sum(vertex_gld_util) / len(vertex_gld_util):.2f}%`\n")
        if edge_gwrite_util:
            f.write(f"- Edge global write sector util avg: `{sum(edge_gwrite_util) / len(edge_gwrite_util):.2f}%`\n")
        if vertex_gwrite_util:
            f.write(f"- Vertex-thread global write sector util avg: `{sum(vertex_gwrite_util) / len(vertex_gwrite_util):.2f}%`\n")

        f.write("\n## Bus Size Bins\n\n")
        f.write("| Bus bin | Cases | Avg n_bus | Edge fill ms avg | Vertex fill ms avg | V/E geomean | Edge lane % | Vertex lane % | Edge occ % | Vertex occ % |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in bin_rows:
            f.write(
                f"| {row['label']} | {row['count']} | {fmt(row['avg_n_bus'], 0)} | "
                f"{fmt(row['edge_fill'], 6)} | {fmt(row['vertex_fill'], 6)} | "
                f"{fmt(row['vertex_over_edge'], 2)}x | "
                f"{fmt(row['edge_lane'])} | {fmt(row['vertex_lane'])} | "
                f"{fmt(row['edge_occ'])} | {fmt(row['vertex_occ'])} |\n"
            )

        f.write("\n| Bus bin | Edge ld sec/req | Vertex ld sec/req | Edge wr sec/req | Vertex wr sec/req | Edge ld sector util % | Vertex ld sector util % | Edge wr sector util % | Vertex wr sector util % |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in bin_rows:
            f.write(
                f"| {row['label']} | "
                f"{fmt(row['edge_ld_spr'])} | {fmt(row['vertex_ld_spr'])} | "
                f"{fmt(row['edge_wr_spr'])} | {fmt(row['vertex_wr_spr'])} | "
                f"{fmt(row['edge_ld_util'])} | {fmt(row['vertex_ld_util'])} | "
                f"{fmt(row['edge_wr_util'])} | {fmt(row['vertex_wr_util'])} |\n"
            )

        f.write("\n## Case Metrics\n\n")
        f.write("| Case | n_bus | J nnz | Edge fill ms | Vertex fill ms | Vertex/Edge | Edge occ % | Vertex occ % | Edge lane util % | Vertex lane util % | Edge avg lanes | Vertex avg lanes | Edge ncu us | Vertex ncu us |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in timing_rows:
            case = row["case"]
            edge_t = parse_float(row.get("edge_fill_ms", ""))
            vertex_t = parse_float(row.get("vertex_thread_fill_ms", ""))
            ratio = vertex_t / edge_t if edge_t and vertex_t else None
            edge = by_case[case].get("edge", {})
            vertex = by_case[case].get("vertex_thread", {})
            f.write(
                f"| {case} | {row.get('n_bus', '')} | {row.get('jac_nnz', '')} | "
                f"{fmt(edge_t, 6)} | {fmt(vertex_t, 6)} | {fmt(ratio, 2)}x | "
                f"{fmt(edge.get('occ_pct'))} | {fmt(vertex.get('occ_pct'))} | "
                f"{fmt(edge.get('lane_util_pct'))} | {fmt(vertex.get('lane_util_pct'))} | "
                f"{fmt(edge.get('avg_lanes'))} | {fmt(vertex.get('avg_lanes'))} | "
                f"{fmt(edge.get('ncu_time_us'), 3)} | {fmt(vertex.get('ncu_time_us'), 3)} |\n"
            )

        f.write("\n## Coalescing Metrics\n\n")
        f.write("| Case | Edge ld sec/req | Vertex ld sec/req | Edge wr sec/req | Vertex wr sec/req | Edge ld sector util % | Vertex ld sector util % | Edge wr sector util % | Vertex wr sector util % |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in timing_rows:
            case = row["case"]
            edge = by_case[case].get("edge", {})
            vertex = by_case[case].get("vertex_thread", {})
            f.write(
                f"| {case} | "
                f"{fmt(edge.get('gld_sectors_per_request'))} | "
                f"{fmt(vertex.get('gld_sectors_per_request'))} | "
                f"{fmt(edge.get('gwrite_sectors_per_request'))} | "
                f"{fmt(vertex.get('gwrite_sectors_per_request'))} | "
                f"{fmt(edge.get('gld_sector_util_pct'))} | "
                f"{fmt(vertex.get('gld_sector_util_pct'))} | "
                f"{fmt(edge.get('gwrite_sector_util_pct'))} | "
                f"{fmt(vertex.get('gwrite_sector_util_pct'))} |\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", required=True)
    parser.add_argument("--ncu", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--timing-iters", default="")
    parser.add_argument("--timing-warmup", default="")
    parser.add_argument("--ncu-iters", default="")
    parser.add_argument("--ncu-warmup", default="")
    args = parser.parse_args()

    timing_rows = read_timing(args.timing)
    ncu_rows = read_ncu_rows(args.ncu)
    groups = [group for group in group_ncu(ncu_rows) if group["kind"] != "other"]
    records = assign_cases(groups, timing_rows)
    write_summary_csv(args.out_csv, records)
    write_summary_md(args.out_md, timing_rows, records, args)


if __name__ == "__main__":
    main()
