import gzip
import io

import numpy as np
import pandas as pd
import pytest
from conftest import write_run
from fastapi.testclient import TestClient
from PIL import Image

from workbench import evaluation as ev
from workbench.app import create_app, load_adapter
from workbench.datasets import ArrayAdapter, MnistAdapter, png_bytes, read_idx

RUN = "20261002-120000-cnn"


def predictions(ids, labels, probs) -> pd.DataFrame:
    frame = pd.DataFrame({"id": ids, "label": labels, "pred": probs.argmax(1)})
    for k in range(probs.shape[1]):
        frame[f"p{k}"] = probs[:, k]
    frame["loss"] = -np.log(probs[np.arange(len(labels)), labels])
    return frame


def probs_for(preds, confidence, k=3):
    confidence = np.broadcast_to(np.asarray(confidence, dtype=float), len(preds))
    p = np.repeat(((1 - confidence) / (k - 1))[:, None], k, axis=1)
    p[np.arange(len(preds)), preds] = confidence
    return p


@pytest.fixture
def run_with_predictions(tmp_path):
    """6 test images, 3 classes: two errors (0 -> 1 and 2 -> 1); 4 val images, all right."""
    run = write_run(tmp_path, RUN, "finished")
    (run / "predictions").mkdir()
    labels, preds = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 1, 2])
    conf = np.array([0.9, 0.6, 0.95, 0.7, 0.55, 0.8])
    ids = np.array([f"test-{i:04d}" for i in range(6)])
    predictions(ids, labels, probs_for(preds, conf)).to_parquet(run / "predictions/test.parquet")
    val_labels = np.array([0, 1, 2, 2])
    val_ids = np.array([f"train-{i:05d}" for i in range(4)])
    val = predictions(val_ids, val_labels, probs_for(val_labels, np.full(4, 0.9)))
    val.to_parquet(run / "predictions/val.parquet")
    return tmp_path


@pytest.fixture
def adapter():
    images = np.arange(6 * 4 * 4, dtype=np.uint8).reshape(6, 4, 4)
    return ArrayAdapter({"test": (np.array([f"test-{i:04d}" for i in range(6)]), images,
                                  np.array([0, 0, 1, 1, 2, 2]))})  # fmt: skip


@pytest.fixture
def client(run_with_predictions, adapter):
    return TestClient(create_app(run_with_predictions, adapter=adapter, poll_seconds=0.01))


def write_idx(path, array):
    header = b"\x00\x00\x08" + bytes([array.ndim])
    header += b"".join(int(d).to_bytes(4, "big") for d in array.shape)
    path.write_bytes(gzip.compress(header + array.astype(np.uint8).tobytes()))


def test_mnist_adapter_reads_idx(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    for prefix, n in (("train", 3), ("t10k", 2)):
        write_idx(raw / f"{prefix}-images-idx3-ubyte.gz", rng.integers(0, 256, (n, 28, 28)))
        write_idx(raw / f"{prefix}-labels-idx1-ubyte.gz", np.arange(n))
    adapter = MnistAdapter(tmp_path)
    assert adapter.list_samples("test") == ["test-0000", "test-0001"]
    assert adapter.list_samples("train")[-1] == "train-00002"
    assert adapter.get_label("test-0001") == 1
    expected = read_idx(raw / "train-images-idx3-ubyte.gz")[1]
    assert (adapter.get_image("train-00001") == expected).all()
    with pytest.raises(KeyError):
        adapter.get_image("test-0002")
    assert isinstance(load_adapter(tmp_path), MnistAdapter) and load_adapter(None) is None
    truncated = b"\x00\x00\x08\x01" + (5).to_bytes(4, "big") + b"\x01"  # 5 labels announced
    (raw / "t10k-labels-idx1-ubyte.gz").write_bytes(gzip.compress(truncated))
    assert load_adapter(tmp_path) is None  # truncated file: no adapter instead of a crash


def test_png_bytes_scales_with_sharp_pixels():
    image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    with Image.open(io.BytesIO(png_bytes(image, scale=3))) as png:
        pixels = np.asarray(png)
    assert pixels.shape == (6, 6) and pixels[:3, :3].max() == 0 and pixels[:3, 3:].min() == 255


def test_evaluation_numbers(run_with_predictions):
    frame = ev.load_predictions(run_with_predictions / RUN, "test")
    r = ev.evaluation(frame)
    assert r["n"] == 6 and r["errors"] == 2 and r["acc"] == pytest.approx(4 / 6)
    assert r["confusion"] == [[1, 1, 0], [0, 2, 0], [0, 1, 1]] and r["max_off_diagonal"] == 1
    # class 1: precision 2/4, recall 2/2
    assert r["classes"][1]["precision"] == 0.5 and r["classes"][1]["recall"] == 1.0
    # 15 bins: one image per bin except none; ECE = mean |correct - confidence|
    correct = np.array([1, 0, 1, 1, 0, 1])
    conf = np.array([0.9, 0.6, 0.95, 0.7, 0.55, 0.8])
    assert r["reliability"]["ece"] == pytest.approx(np.abs(correct - conf).mean())
    assert ev.prediction_splits(run_with_predictions / RUN) == ["test", "val"]
    assert ev.top_k(frame, 1, 2) == [(1, pytest.approx(0.6)), (0, pytest.approx(0.2))]


def test_filter_samples(run_with_predictions):
    frame = ev.load_predictions(run_with_predictions / RUN, "test")
    # losses of the errors: -ln 0.2 (test-0001) > -ln 0.225 (test-0004)
    assert ev.filter_samples(frame, errors=True)["id"].tolist() == ["test-0001", "test-0004"]
    ones = ev.filter_samples(frame, label=1, pred=1, sort="id")
    assert ones["id"].tolist() == ["test-0002", "test-0003"]
    unsure = ev.filter_samples(frame, max_confidence=0.6, sort="confidence")
    assert unsure["id"].tolist() == ["test-0004", "test-0001"]
    assert len(ev.filter_samples(frame, min_loss=1.0)) == 2  # the two errors


def test_image_route(client):
    r = client.get("/images/test-0002.png?scale=2")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(r.content)) as png:
        assert png.size == (8, 8)
    assert client.get("/images/test-9999.png").status_code == 404
    assert client.get("/images/test-0002.png?scale=99").status_code == 422
    no_adapter = TestClient(create_app(client.app.state.runs_dir))
    assert no_adapter.get("/images/test-0002.png").status_code == 404


def test_evaluation_page(client):
    html = client.get(f"/runs/{RUN}/evaluation").text
    assert "66.67 %" in html and 'id="c-rel"' in html
    assert f'href="/runs/{RUN}/data?split=test&label=0&pred=1"' in html  # error cell links
    assert 'class="num off"' in html and "?split=val" in html
    assert "100.00 %" in client.get(f"/runs/{RUN}/evaluation?split=val").text
    assert client.get(f"/runs/{RUN}/evaluation?split=train").status_code == 404
    assert client.get("/runs/20261001-120000-nope/evaluation").status_code == 404


def test_live_page_links_to_evaluation(client, run_with_predictions):
    assert f'href="/runs/{RUN}/evaluation"' in client.get(f"/runs/{RUN}").text
    write_run(run_with_predictions, "no-preds")
    assert "/evaluation" not in client.get("/runs/no-preds").text


def test_data_page_filters_and_pages(client, monkeypatch):
    query = "split=test&label=&pred=1&errors=true&min_loss=&max_confidence=&sort=loss"  # the form
    html = client.get(f"/runs/{RUN}/data?{query}").text
    assert "2 images" in html and html.index("test-0001") < html.index("test-0004")
    assert 'src="/images/test-0004.png?scale=3"' in html and "test-0002" not in html
    assert client.get(f"/runs/{RUN}/data?label=x").status_code == 400
    assert client.get(f"/runs/{RUN}/data?sort=bogus").status_code == 400
    monkeypatch.setattr("workbench.app.PAGE_SIZE", 4)
    page2 = client.get(f"/runs/{RUN}/data?sort=id&page=2").text
    assert "page 2 of 2" in page2 and "test-0004" in page2 and "test-0000" not in page2
    assert "page=1" in page2 and "sort=id" in page2


def test_sample_page(client):
    html = client.get(f"/runs/{RUN}/data/test-0001").text
    assert "original label (dataset)" in html and "0.6000" in html
    assert 'src="/images/test-0001.png?scale=10"' in html
    assert html.index("<th>1</th>") < html.index("<th>0</th>")  # top-3 sorted by probability
    assert client.get(f"/runs/{RUN}/data/train-00002").status_code == 200  # found in val
    assert client.get(f"/runs/{RUN}/data/train-00002?split=test").status_code == 404
    assert client.get(f"/runs/{RUN}/data/test-9999").status_code == 404
