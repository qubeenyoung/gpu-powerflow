"""STATE.json read/write for the mysolver autorun loop (PLAN §3.1)."""

import datetime
import json
import pathlib

DEFAULT_STATE = {
    "current_milestone": "M0",
    "milestone_status": "in_progress",
    "cycle_count": 0,
    "last_cycle_at": None,
    "last_report_at": None,
    "active_matrix_set": "smoke",
    "best_metrics_per_case": {},
    "regression_alerts": [],
    "todo": [],
    "blocked_reason": None,
}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path):
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return dict(DEFAULT_STATE)
    data = json.loads(p.read_text())
    merged = dict(DEFAULT_STATE)
    merged.update(data)
    return merged


def save(path, state):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2) + "\n")
