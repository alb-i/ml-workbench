"""Scans RUNS_DIR and tails metrics.jsonl.

A run folder (written by the training code) contains meta.json, config.yaml, heartbeat and
metrics.jsonl, optionally images in samples/ (e.g. an augmented training batch). The workbench
only reads these files.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

# heartbeat is touched every 10 s; a run whose heartbeat is older than this has crashed
STALE_SECONDS = 30


def read_metrics(path: Path, offset: int = 0) -> tuple[list[dict], int]:
    """Complete JSON lines from byte `offset` on, and the offset after the last complete line."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
    except FileNotFoundError:
        return [], offset
    end = chunk.rfind(b"\n") + 1
    records = []
    for line in chunk[:end].splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records, offset + end


def sample_images(run_dir: Path) -> list[str]:
    """Names of the PNGs in samples/, sorted."""
    return sorted(p.name for p in (run_dir / "samples").glob("*.png") if p.is_file())


def read_meta(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "meta.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_config(run_dir: Path) -> dict:
    try:
        return yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    except FileNotFoundError:
        return {}


def heartbeat_time(run_dir: Path) -> float | None:
    try:
        return (run_dir / "heartbeat").stat().st_mtime
    except FileNotFoundError:
        return None


def run_status(run_dir: Path, meta: dict | None = None, now: float | None = None) -> str:
    """finished / crashed as recorded by the run; running only while the heartbeat is fresh."""
    meta = read_meta(run_dir) if meta is None else meta
    status = meta.get("status", "running")
    if status != "running":
        return status
    beat = heartbeat_time(run_dir)
    if beat is None or (now or time.time()) - beat > STALE_SECONDS:
        return "crashed"
    return "running"


def parse_time(s: str | None) -> float | None:
    try:
        return datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


def model_name(config: dict) -> str:
    model = config.get("model", "")
    return str(model.get("name", "") if isinstance(model, dict) else model)


@dataclass
class RunSummary:
    run_id: str
    status: str
    model: str
    dataset_version: str
    git_commit: str
    start_time: str
    duration_s: float | None
    best_val_acc: float | None
    last_step: int | None
    config: dict = field(repr=False)


def summarize(run_dir: Path, now: float | None = None) -> RunSummary:
    meta = read_meta(run_dir)
    config = read_config(run_dir)
    records, _ = read_metrics(run_dir / "metrics.jsonl")
    val_acc = [r["acc"] for r in records if r.get("split") == "val" and "acc" in r]
    steps = [r["step"] for r in records if "step" in r]
    start = parse_time(meta.get("start_time"))
    end = parse_time(meta.get("end_time")) or heartbeat_time(run_dir)
    return RunSummary(
        run_id=meta.get("run_id", run_dir.name),
        status=run_status(run_dir, meta, now),
        model=model_name(config),
        dataset_version=meta.get("dataset_version", ""),
        git_commit=meta.get("git_commit", ""),
        start_time=meta.get("start_time", ""),
        duration_s=end - start if start and end else None,
        best_val_acc=max(val_acc) if val_acc else None,
        last_step=max(steps) if steps else None,
        config=config,
    )


def list_runs(runs_dir: Path) -> list[RunSummary]:
    """All run folders (those with a meta.json), newest first."""
    if not runs_dir.is_dir():
        return []
    dirs = [d for d in runs_dir.iterdir() if (d / "meta.json").exists()]
    summaries = [summarize(d) for d in dirs]
    return sorted(summaries, key=lambda s: s.start_time, reverse=True)


def flatten(config: dict, prefix: str = "") -> dict:
    """{"optimizer": {"lr": 0.1}} -> {"optimizer.lr": 0.1}"""
    out = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def config_diff(configs: list[dict]) -> list[tuple[str, list, bool]]:
    """Rows (key, values per run, differs) over the union of all flattened keys."""
    flat = [flatten(c) for c in configs]
    keys = sorted(set().union(*flat)) if flat else []
    rows = []
    for key in keys:
        values = [f.get(key, "—") for f in flat]
        rows.append((key, values, any(v != values[0] for v in values)))
    return rows
