import json

from conftest import RECORDS

from workbench import runs


def test_status_from_meta_and_heartbeat(runs_dir):
    status = {s.run_id: s.status for s in runs.list_runs(runs_dir)}
    assert status == {
        "20261001-120000-mlp": "finished",
        "20261001-130000-cnn": "running",
        "20261001-140000-dead": "crashed",  # meta says running, heartbeat stale
    }


def test_crash_recorded_by_run(tmp_path):
    from conftest import write_run

    d = write_run(tmp_path, "r", status="crashed")
    assert runs.run_status(d) == "crashed"


def test_summary(runs_dir):
    s = runs.summarize(runs_dir / "20261001-120000-mlp")
    assert s.model == "mlp" and s.dataset_version == "e3b0c442"
    assert s.best_val_acc == 0.93 and s.last_step == 844
    assert s.duration_s == 150
    assert [r.run_id for r in runs.list_runs(runs_dir)][0] == "20261001-140000-dead"


def test_list_runs_ignores_other_folders(runs_dir):
    (runs_dir / "not-a-run").mkdir()
    assert len(runs.list_runs(runs_dir)) == 3
    assert runs.list_runs(runs_dir / "missing") == []


def test_tail_metrics_incrementally(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(RECORDS[0]) + "\n" + '{"step": 2')  # partial last line
    records, offset = runs.read_metrics(path)
    assert records == [RECORDS[0]]
    with open(path, "a") as f:
        f.write(', "split": "train"}\n' + json.dumps(RECORDS[1]) + "\n")
    records, offset2 = runs.read_metrics(path, offset)
    assert records == [{"step": 2, "split": "train"}, RECORDS[1]]
    assert runs.read_metrics(path, offset2) == ([], offset2)
    assert runs.read_metrics(tmp_path / "missing.jsonl") == ([], 0)


def test_config_diff():
    a = {"model": "cnn", "optimizer": {"name": "adamw", "lr": 1e-3}, "epochs": 10}
    b = {"model": "cnn", "optimizer": {"name": "adamw", "lr": 3e-3}, "aug": True}
    rows = {key: (values, differs) for key, values, differs in runs.config_diff([a, b])}
    assert rows["model"] == (["cnn", "cnn"], False)
    assert rows["optimizer.lr"] == ([1e-3, 3e-3], True)
    assert rows["epochs"] == ([10, "—"], True)
    assert rows["aug"] == (["—", True], True)
