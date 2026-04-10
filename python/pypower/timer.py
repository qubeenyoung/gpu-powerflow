from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
import sys


@dataclass
class TimingEntry:
    tag: str
    op_name: str
    iter_idx: int
    elapsed_sec: float


class TimingLog:
    def __init__(self, enabled: bool, emit_log: bool = True, stream=None):
        self.enabled = enabled
        self.emit_log = emit_log
        self.stream = stream or sys.stdout
        self.entries: list[TimingEntry] = []

    def record(self, tag: str, op_name: str, iter_idx: int, elapsed_sec: float) -> None:
        if not self.enabled:
            return
        entry = TimingEntry(tag=tag, op_name=op_name, iter_idx=iter_idx, elapsed_sec=elapsed_sec)
        self.entries.append(entry)
        if self.emit_log:
            print(
                f"{tag} {op_name} {iter_idx} {elapsed_sec:.6f}",
                file=self.stream,
                flush=True,
            )

    def to_dicts(self) -> list[dict]:
        return [
            {
                "tag": entry.tag,
                "op_name": entry.op_name,
                "iter_idx": entry.iter_idx,
                "elapsed_sec": entry.elapsed_sec,
            }
            for entry in self.entries
        ]


class BlockTimer:
    def __init__(self, timing_log: TimingLog | None, tag: str, op_name: str, iter_idx: int):
        self.timing_log = timing_log
        self.tag = tag
        self.op_name = op_name
        self.iter_idx = iter_idx
        self.start: float | None = None

    def __enter__(self):
        if self.timing_log is not None and self.timing_log.enabled:
            self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if self.timing_log is not None and self.start is not None:
            self.timing_log.record(
                self.tag,
                self.op_name,
                self.iter_idx,
                perf_counter() - self.start,
            )
        return False


def summarize_entries(entries: list[TimingEntry]) -> dict[str, dict[str, float | int]]:
    buckets: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_sec": 0.0, "avg_sec": 0.0}
    )
    for entry in entries:
        key = (entry.tag, entry.op_name)
        bucket = buckets[key]
        bucket["count"] = int(bucket["count"]) + 1
        bucket["total_sec"] = float(bucket["total_sec"]) + entry.elapsed_sec

    summary: dict[str, dict[str, float | int]] = {}
    for (tag, op_name), bucket in buckets.items():
        count = int(bucket["count"])
        total = float(bucket["total_sec"])
        bucket["avg_sec"] = total / count if count else 0.0
        summary[f"{tag}.{op_name}"] = bucket
    return summary

