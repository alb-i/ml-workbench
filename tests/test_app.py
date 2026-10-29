import asyncio
import json

import pytest
from conftest import RECORDS
from fastapi.testclient import TestClient

from workbench.app import create_app, event_stream
from workbench.gpu import gpu_stats


@pytest.fixture
def client(runs_dir):
    return TestClient(create_app(runs_dir, poll_seconds=0.01))


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"]), fields.get("id")))
    return events


def test_runs_page(client):
    html = client.get("/").text
    for run_id in ["20261001-120000-mlp", "20261001-130000-cnn", "20261001-140000-dead"]:
        assert f'href="/runs/{run_id}"' in html
    assert 'class="status crashed"' in html and "93.00 %" in html and "2:30" in html


def test_runs_table_keeps_selection(client):
    html = client.get("/partials/runs", params={"ids": ["20261001-130000-cnn"]}).text
    assert 'value="20261001-130000-cnn" checked' in html
    assert 'value="20261001-120000-mlp" >' in html


def test_live_page(client):
    html = client.get("/runs/20261001-130000-cnn").text
    assert "/stream" in html and 'class="status running"' in html


def test_unknown_run_is_404(client):
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/api/runs/..%2F..%2Fetc/metrics").status_code == 404


def test_metrics_api(client):
    assert client.get("/api/runs/20261001-120000-mlp/metrics").json() == RECORDS
    runs = client.get("/api/runs").json()
    assert {r["run_id"] for r in runs} >= {"20261001-120000-mlp"}


def test_stream_of_finished_run_ends(client):
    r = client.get("/api/runs/20261001-120000-mlp/stream")
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e[0] for e in events] == ["metrics", "status", "gpu", "end"]
    assert events[0][1] == RECORDS and events[1][1] == "finished"


def test_stream_resumes_from_last_event_id(client, runs_dir):
    first = parse_sse(client.get("/api/runs/20261001-120000-mlp/stream").text)[0]
    r = client.get("/api/runs/20261001-120000-mlp/stream", headers={"Last-Event-ID": first[2]})
    assert parse_sse(r.text)[0][1] == []


def test_stream_tails_running_run(runs_dir):
    run_dir = runs_dir / "20261001-130000-cnn"

    async def collect():
        stream = event_stream(run_dir, 0, lambda: asyncio.sleep(0, False), poll_seconds=0.01)
        first = await anext(stream)
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(RECORDS[2]) + "\n")
        async for event in stream:
            if event.startswith("event: metrics"):
                return first, event

    first, second = asyncio.run(collect())
    assert parse_sse(first)[0][1] == RECORDS[:2]
    assert parse_sse(second)[0][1] == [RECORDS[2]]


def test_compare(client):
    ids = ["20261001-120000-mlp", "20261001-130000-cnn"]
    html = client.get("/compare", params={"ids": ids}).text
    assert all(i in html for i in ids)
    assert 'class="differs"' in html and "<code>model</code>" in html
    assert client.get("/compare", params={"ids": ids[:1]}).status_code == 400


def test_gpu_stats_shape():
    for g in gpu_stats():
        assert set(g) >= {"util", "mem_used_gb", "mem_total_gb", "temp_c"}


def test_weight_delta_chart(client, runs_dir):
    from conftest import write_run

    records = [
        {"step": 100, "split": "train", "weight_delta_100": 9.0},
        {"step": 500, "split": "train", "weight_delta_100": 3.0, "weight_delta_500": 20.0},
    ]
    write_run(runs_dir, "wd", records=records)
    html = client.get("/runs/wd").text
    assert 'id="c-wd"' in html and "weightDeltaSeries(records)" in html
    html = client.get("/compare", params={"ids": ["wd", "20261001-120000-mlp"]}).text
    assert "weightDeltaSeries(records)" in html
    assert client.get("/api/runs/wd/metrics").json() == records
    js = client.get("/static/charts.js").text
    assert "function weightDeltaSeries" in js and "/^weight_delta_(\\d+)$/" in js


def test_line_search_chart(client, runs_dir):
    from conftest import write_run

    records = [
        {"step": 50, "split": "train", "loss": 1.0, "lr": 1.0, "ls_step": 0.5, "ls_factor": 0.5},
        {"step": 100, "split": "train", "loss": 0.8, "lr": 2.0, "ls_step": 0.5},  # older run
    ]
    write_run(runs_dir, "ls", records=records)
    html = client.get("/runs/ls").text
    assert 'id="ls-chart" hidden' in html and "shorteningSeries(records)" in html
    html = client.get("/compare", params={"ids": ["ls", "20261001-120000-mlp"]}).text
    assert 'id="c-ls"' in html and "shorteningSeries(records)" in html
    js = client.get("/static/charts.js").text
    assert "function shorteningSeries" in js and "r.ls_factor ?? r.ls_step / r.lr" in js
    assert "yLog ? 'logarithmic'" in js


def test_sample_images_on_live_page(client, runs_dir):
    (runs_dir / "20261001-130000-cnn" / "samples").mkdir()
    png = b"\x89PNG\r\n\x1a\nfake"
    (runs_dir / "20261001-130000-cnn" / "samples" / "augmented.png").write_bytes(png)
    (runs_dir / "20261001-130000-cnn" / "samples" / "notes.txt").write_text("not an image")
    html = client.get("/runs/20261001-130000-cnn").text
    assert 'src="/runs/20261001-130000-cnn/samples/augmented.png"' in html
    assert "notes.txt" not in html
    r = client.get("/runs/20261001-130000-cnn/samples/augmented.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content == png
    for bad in ["notes.txt", "missing.png", "..%2F..%2Fmeta.json"]:
        assert client.get(f"/runs/20261001-130000-cnn/samples/{bad}").status_code == 404
    assert "samples/" not in client.get("/runs/20261001-120000-mlp").text  # no samples/ folder
