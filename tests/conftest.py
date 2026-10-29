import json
import os
import time

import pytest
import yaml


def write_run(root, run_id, status="running", heartbeat_age=0.0, records=(), config=None):
    """A run folder in the format of plan 6.3, as digits.runlog writes it."""
    d = root / run_id
    d.mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "dataset_version": "e3b0c442",
        "start_time": "2026-10-01T12:00:00+01:00",
        "host": "laptop",
        "status": status,
    }
    if status != "running":
        meta["end_time"] = "2026-10-01T12:02:30+01:00"
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "config.yaml").write_text(yaml.safe_dump(config or {"model": "mlp"}))
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (d / "heartbeat").write_text("")
    beat = time.time() - heartbeat_age
    os.utime(d / "heartbeat", (beat, beat))
    return d


RECORDS = [
    {"step": 100, "epoch": 0, "split": "train", "loss": 0.5, "acc": 0.85, "lr": 0.1, "ts": 1.0},
    {"step": 422, "epoch": 0, "split": "val", "loss": 0.3, "acc": 0.91, "ts": 2.0},
    {"step": 500, "epoch": 1, "split": "train", "loss": 0.2, "acc": 0.94, "lr": 0.1, "ts": 3.0},
    {"step": 844, "epoch": 1, "split": "val", "loss": 0.25, "acc": 0.93, "ts": 4.0},
]


@pytest.fixture
def runs_dir(tmp_path):
    write_run(tmp_path, "20261001-120000-mlp", "finished", 600, RECORDS)
    write_run(tmp_path, "20261001-130000-cnn", "running", 2, RECORDS[:2], {"model": "cnn"})
    write_run(tmp_path, "20261001-140000-dead", "running", 120, RECORDS[:1])
    return tmp_path
