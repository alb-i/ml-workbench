import csv

import numpy as np
import pandas as pd
import pytest
from conftest import write_run
from fastapi.testclient import TestClient

from workbench import review
from workbench.app import create_app
from workbench.curation import CSV_COLUMNS, CurationStore
from workbench.datasets import ArrayAdapter

CV = "20261002-120000-cv-oof"


@pytest.fixture
def runs_dir(tmp_path):
    """label_issues.parquet: train candidates ranks 1-3 (train-00001 is no candidate), 1 test."""
    run = write_run(tmp_path / "runs", CV, "finished")
    pd.DataFrame({
        "id": ["train-00000", "train-00001", "train-00002", "train-00003", "test-0000"],
        "split": ["train", "train", "train", "train", "test"],
        "label": [4, 1, 7, 3, 5],
        "pred": [9, 1, 1, 3, 6],
        "self_confidence": [0.01, 0.95, 0.05, 0.4, 0.2],
        "candidate": [True, False, True, True, True],
        "rank": pd.array([1, None, 2, 3, 1], dtype="Int64"),
    }).to_parquet(run / "label_issues.parquet")  # fmt: skip
    return tmp_path / "runs"


@pytest.fixture
def client(runs_dir, tmp_path):
    ids = ["train-00000", "train-00001", "train-00002", "train-00003"]
    adapter = ArrayAdapter({
        "train": (np.array(ids), np.zeros((4, 4, 4), np.uint8), np.array([4, 1, 7, 3])),
        "test": (np.array(["test-0000"]), np.zeros((1, 4, 4), np.uint8), np.array([5])),
    })  # fmt: skip
    (tmp_path / "curation").mkdir()
    return TestClient(create_app(runs_dir, adapter=adapter, curation_dir=tmp_path / "curation"))


def test_store_validates_and_undoes(tmp_path):
    store = CurationStore(tmp_path)
    with pytest.raises(ValueError, match="action"):
        store.add("train-00000", "maybe", reviewer="immo", round=1)
    with pytest.raises(ValueError, match="new_label"):
        store.add("train-00000", "relabel", reviewer="immo", round=1)
    with pytest.raises(ValueError, match="use keep"):
        store.add("train-00000", "relabel", reviewer="immo", round=1, new_label=4, given_label=4)
    with pytest.raises(ValueError, match="takes no new_label"):
        store.add("train-00000", "drop", reviewer="immo", round=1, new_label=4)
    first = store.add("train-00000", "unsure", reviewer="immo", round=1)
    second = store.add("train-00000", "relabel", reviewer="immo", round=2, new_label=9)
    assert store.latest()["train-00000"]["id"] == second and store.max_round() == 2
    assert store.undo(second) and not store.undo(second)
    assert store.latest()["train-00000"]["id"] == first


def test_store_stats_measure_agreement_with_the_model(tmp_path):
    store = CurationStore(tmp_path)
    kw = {"reviewer": "immo", "round": 1}
    store.add("a", "relabel", new_label=9, given_label=4, model_pred=9, **kw)  # agrees, disputed
    store.add("b", "keep", given_label=7, model_pred=1, **kw)  # disagrees, disputed
    store.add("c", "keep", given_label=3, model_pred=3, **kw)  # agrees, undisputed
    store.add("d", "drop", given_label=5, model_pred=6, **kw)  # not judged
    s = store.stats()
    assert s["counts"] == {"relabel": 1, "keep": 2, "drop": 1, "unsure": 0}
    assert (s["agree"], s["judged"], s["agree_disputed"], s["disputed"]) == (2, 3, 1, 2)


def test_export_and_import_round_trip(tmp_path):
    store = CurationStore(tmp_path)
    store.add("train-07080", "relabel", new_label=1, reason='clearly a 1, labelled "7"',
              reviewer="immo", round=1, timestamp="2026-11-01T10:00:00")  # fmt: skip
    store.add("test-0001", "drop", reviewer="immo", round=1, timestamp="2026-11-01T10:01:00")
    assert store.export_csv() == 2
    with open(tmp_path / "decisions.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_COLUMNS
    assert rows[1] == ["train-07080", "relabel", "1", 'clearly a 1, labelled "7"', "immo",
                       "2026-11-01T10:00:00", "1"]  # fmt: skip
    assert rows[2][:3] == ["test-0001", "drop", ""]
    (tmp_path / "decisions.sqlite").unlink()
    again = CurationStore(tmp_path)  # empty database: imports the CSV
    assert again.count() == 2 and again.latest()["train-07080"]["new_label"] == 1
    assert CurationStore(tmp_path).count() == 2  # no second import into a filled database


def test_queue(runs_dir):
    candidates = review.load_candidates(runs_dir / CV, "train")
    assert candidates["id"].tolist() == ["train-00000", "train-00002", "train-00003"]
    latest = {"train-00000": {"action": "keep"}, "train-00002": {"action": "unsure"}}
    assert review.queue(candidates, latest)["id"].tolist() == ["train-00003"]
    assert review.queue(candidates, latest, unsure=True)["id"].tolist() == ["train-00002"]
    current, upcoming = review.current_and_next(review.queue(candidates, {}), after=1)
    assert current["id"] == "train-00002" and upcoming["id"] == "train-00003"
    assert review.review_runs(runs_dir) == [CV]


def decide(client, sample_id, action, **extra):
    data = {"run": CV, "split": "train", "sample_id": sample_id, "action": action,
            "round": 1, "reviewer": "immo", **extra}  # fmt: skip
    return client.post("/review/decide", data=data, follow_redirects=False)


def test_review_flow(client):
    html = client.get("/review").text
    assert "<h1>train-00000</h1>" in html and "0 of 3 candidates" in html
    assert 'src="/images/train-00000.png?scale=12"' in html
    assert 'rel="prefetch" href="/images/train-00002.png' in html
    r = decide(client, "train-00000", "relabel", new_label="9", reason="clearly a 9")
    assert r.status_code == 303 and r.headers["location"].startswith("/review?run=")
    html = client.get(r.headers["location"]).text
    assert "<h1>train-00002</h1>" in html and "1 of 3 candidates" in html
    # the digit of the given label means keep
    assert decide(client, "train-00002", "relabel", new_label="7").status_code == 303
    store = client.app.state.store
    assert store.latest()["train-00002"]["action"] == "keep"
    assert store.latest()["train-00000"]["model_pred"] == 9
    assert store.latest()["train-00000"]["source_run"] == CV
    # skip the last one: end of queue with one left
    html = client.get(f"/review?run={CV}&split=train&after=3").text
    assert "1 skipped candidates are left" in html
    decide(client, "train-00003", "unsure")
    assert "Nothing left" in client.get("/review").text
    assert "<h1>train-00003</h1>" in client.get("/review?unsure=true").text
    assert "<h1>test-0000</h1>" in client.get("/review?split=test").text


def test_review_rejects_bad_input(client):
    assert decide(client, "train-00001", "keep").status_code == 404  # not a candidate
    assert decide(client, "train-00000", "maybe").status_code == 400
    assert decide(client, "train-00000", "relabel", new_label="x").status_code == 400


def test_curation_page_undo_and_export(client, tmp_path):
    decide(client, "train-00000", "relabel", new_label="9")
    decide(client, "train-00003", "keep")
    html = client.get("/curation").text
    assert "2 / 2" in html and "1 / 1" in html  # agree with model: overall / disputed
    assert 'href="/runs/' + CV + '/data/train-00000"' in html
    assert "train-00003" not in client.get("/curation?action=relabel").text
    decision_id = client.app.state.store.latest()["train-00003"]["id"]
    r = client.post(f"/curation/{decision_id}/undo", data={"back": "/curation?action=keep"},
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303 and r.headers["location"] == "/curation?action=keep"
    assert client.post("/curation/999/undo").status_code == 404
    r = client.post("/curation/export", follow_redirects=False)
    assert r.headers["location"] == "/curation?exported=1"
    assert (tmp_path / "curation" / "decisions.csv").read_text().count("\n") == 2
    assert "wrote 1 decisions" in client.get("/curation?exported=1").text


def test_sample_page_shows_decisions(client, runs_dir):
    run = runs_dir / CV
    (run / "predictions").mkdir()
    probs = {f"p{k}": [0.9 if k == 9 else 0.1 / 9] for k in range(10)}
    frame = pd.DataFrame({"id": ["train-00000"], "label": [4], "pred": [9], **probs, "loss": [4.4]})
    frame.to_parquet(run / "predictions" / "train.parquet")
    decide(client, "train-00000", "relabel", new_label="9", reason="clearly a 9")
    html = client.get(f"/runs/{CV}/data/train-00000").text
    assert "relabel → 9" in html and "clearly a 9" in html


def test_without_curation_folder(runs_dir):
    client = TestClient(create_app(runs_dir))
    assert client.get("/review").status_code == 404
    assert client.get("/curation").status_code == 404
