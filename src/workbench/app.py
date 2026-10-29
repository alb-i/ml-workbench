"""FastAPI app, binds 127.0.0.1:8000.

Pages (v1): runs table, live run (curves + GPU stats via SSE, images from samples/),
compare (2-5 runs). v2: evaluation of a run's predictions (clickable confusion matrix, per-class
precision/recall/F1, reliability diagram), data explorer (grid of samples with filters) and
sample detail; images come from the dataset adapter (/images/<id>.png). v3: review queue
(candidates of a run's label_issues.parquet, keyboard decisions) and curation (decisions in
SQLite, undo, export to decisions.csv); the only pages that write, and only to the curation
folder.
"""

import argparse
import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from workbench import evaluation as ev
from workbench import review, runs
from workbench.curation import ACTIONS, CurationStore
from workbench.datasets import DatasetAdapter, MnistAdapter, png_bytes
from workbench.gpu import gpu_stats

PACKAGE_DIR = Path(__file__).parent
REPO_DIR = PACKAGE_DIR.parents[1]
HOST = "127.0.0.1"
MAX_COMPARE = 5
DEFAULT_CURATION = REPO_DIR.parent / "mnist-showcase" / "curation"
PAGE_SIZE = 96
RunIds = Annotated[list[str] | None, Query()]


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def sse(event: str, data, event_id: int | None = None) -> str:
    head = f"event: {event}\n" + (f"id: {event_id}\n" if event_id is not None else "")
    return head + f"data: {json.dumps(data)}\n\n"


async def event_stream(
    run_dir: Path,
    offset: int,
    is_disconnected: Callable[[], Awaitable[bool]],
    poll_seconds: float = 1.0,
    gpu_seconds: float = 2.0,
) -> AsyncIterator[str]:
    """New metrics lines (id = byte offset, so a reconnect resumes), status changes, GPU stats.

    Ends with an "end" event once the run is finished or crashed.
    """
    status, last_gpu, first = None, 0.0, True
    while True:
        records, offset = runs.read_metrics(run_dir / "metrics.jsonl", offset)
        if records or first:
            yield sse("metrics", records, offset)
            first = False
        new_status = runs.run_status(run_dir)
        if new_status != status:
            status = new_status
            yield sse("status", status)
        if time.time() - last_gpu >= gpu_seconds:
            last_gpu = time.time()
            yield sse("gpu", {"ts": round(last_gpu, 3), "gpus": gpu_stats()})
        if status != "running":
            yield sse("end", status)
            return
        if await is_disconnected():
            return
        await asyncio.sleep(poll_seconds)


def load_adapter(data_dir: Path | None) -> DatasetAdapter | None:
    """MNIST from <data_dir>/raw if it is there; None otherwise (pages then show no images)."""
    if data_dir is None or not (Path(data_dir) / "raw").is_dir():
        return None
    try:
        return MnistAdapter(data_dir)
    except (OSError, ValueError):
        return None


def optional(value: str, kind: type):
    """Empty form field -> None, else kind(value); 400 if it does not parse."""
    if value.strip() == "":
        return None
    try:
        return kind(value)
    except ValueError:
        raise HTTPException(400, f"invalid value {value!r}") from None


def create_app(
    runs_dir: Path,
    data_dir: Path | None = None,
    poll_seconds: float = 1.0,
    adapter: DatasetAdapter | None = None,
    curation_dir: Path | None = None,
    reviewer: str = "immo",
) -> FastAPI:
    app = FastAPI(title="ml-workbench")
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    templates.env.filters["duration"] = format_duration
    app.state.runs_dir = runs_dir
    app.state.reviewer = reviewer
    app.state.data_dir = data_dir
    app.state.adapter = adapter if adapter is not None else load_adapter(data_dir)
    app.state.store = CurationStore(curation_dir) if curation_dir is not None else None

    def store() -> CurationStore:
        if app.state.store is None:
            raise HTTPException(404, "no curation folder (start with --curation)")
        return app.state.store

    def redirect(path: str, **params) -> RedirectResponse:
        query = urlencode({k: v for k, v in params.items() if v not in (None, "")})
        return RedirectResponse(f"{path}?{query}" if query else path, status_code=303)

    @lru_cache(maxsize=4096)
    def image_png(sample_id: str, scale: int) -> bytes:
        return png_bytes(app.state.adapter.get_image(sample_id), scale)

    def predictions(path: Path, split: str | None) -> tuple[str, list[str], pd.DataFrame]:
        splits = ev.prediction_splits(path)
        if not splits:
            raise HTTPException(404, f"no predictions in {path.name}")
        split = split or splits[0]
        if split not in splits:
            raise HTTPException(404, f"no predictions for split {split}")
        return split, splits, ev.load_predictions(path, split)

    def run_dir(run_id: str) -> Path:
        path = runs_dir / run_id
        if path.parent.resolve() != runs_dir.resolve() or not (path / "meta.json").exists():
            raise HTTPException(404, f"no run {run_id}")
        return path

    @app.get("/", response_class=HTMLResponse)
    def runs_page(request: Request):
        return templates.TemplateResponse(
            request, "runs.html", {"runs": runs.list_runs(runs_dir), "selected": []}
        )

    @app.get("/partials/runs", response_class=HTMLResponse)
    def runs_table(request: Request, ids: RunIds = None):
        return templates.TemplateResponse(
            request, "_runs_table.html", {"runs": runs.list_runs(runs_dir), "selected": ids or []}
        )

    @app.get("/partials/gpu", response_class=HTMLResponse)
    def gpu_badge(request: Request):
        return templates.TemplateResponse(request, "_gpu.html", {"gpus": gpu_stats()})

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def live_page(request: Request, run_id: str):
        path = run_dir(run_id)
        context = {
            "run": runs.summarize(path),
            "samples": runs.sample_images(path),
            "has_predictions": bool(ev.prediction_splits(path)),
        }
        return templates.TemplateResponse(request, "live.html", context)

    @app.get("/images/{sample_id}.png")
    def image(sample_id: str, scale: Annotated[int, Query(ge=1, le=16)] = 3):
        if app.state.adapter is None:
            raise HTTPException(404, "no dataset adapter (start with --data)")
        try:
            content = image_png(sample_id, scale)
        except KeyError:
            raise HTTPException(404, f"no sample {sample_id}") from None
        return Response(content, media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})  # fmt: skip

    @app.get("/runs/{run_id}/evaluation", response_class=HTMLResponse)
    def evaluation_page(request: Request, run_id: str, split: str | None = None):
        path = run_dir(run_id)
        split, splits, frame = predictions(path, split)
        context = {"run": runs.summarize(path), "split": split, "splits": splits,
                   "ev": ev.evaluation(frame)}  # fmt: skip
        return templates.TemplateResponse(request, "evaluation.html", context)

    @app.get("/runs/{run_id}/data", response_class=HTMLResponse)
    def data_page(
        request: Request,
        run_id: str,
        split: str | None = None,
        label: str = "",
        pred: str = "",
        errors: bool = False,
        min_loss: str = "",
        max_confidence: str = "",
        sort: str = "loss",
        page: Annotated[int, Query(ge=1)] = 1,
    ):
        path = run_dir(run_id)
        split, splits, frame = predictions(path, split)
        if sort not in ev.SORTS:
            raise HTTPException(400, f"sort must be one of {list(ev.SORTS)}")
        filters = {
            "label": optional(label, int),
            "pred": optional(pred, int),
            "errors": errors,
            "min_loss": optional(min_loss, float),
            "max_confidence": optional(max_confidence, float),
        }
        rows = ev.filter_samples(frame, **filters, sort=sort)
        pages = max(1, -(-len(rows) // PAGE_SIZE))
        shown = rows.iloc[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        query = {k: v for k, v in request.query_params.items() if k != "page"}
        context = {
            "run": runs.summarize(path),
            "split": split,
            "splits": splits,
            "filters": filters,
            "sort": sort,
            "sorts": list(ev.SORTS),
            "items": shown[["id", "label", "pred", "confidence", "loss"]].to_dict("records"),
            "total": len(rows),
            "page": page,
            "pages": pages,
            "page_query": urlencode(query),
            "classes": range(len(ev.prob_columns(frame))),
            "images": app.state.adapter is not None,
        }
        return templates.TemplateResponse(request, "data.html", context)

    @app.get("/runs/{run_id}/data/{sample_id}", response_class=HTMLResponse)
    def sample_page(request: Request, run_id: str, sample_id: str, split: str | None = None):
        path = run_dir(run_id)
        splits = ev.prediction_splits(path)
        for candidate in [split] if split else splits:
            if candidate not in splits:
                break
            frame = ev.load_predictions(path, candidate)
            hits = (frame["id"] == sample_id).to_numpy().nonzero()[0]
            if len(hits):
                row = int(hits[0])
                original = None
                if app.state.adapter is not None:
                    try:
                        original = app.state.adapter.get_label(sample_id)
                    except KeyError:
                        pass
                context = {
                    "run": runs.summarize(path),
                    "split": candidate,
                    "sample": frame.iloc[row][["id", "label", "pred", "confidence", "loss"]],
                    "top": ev.top_k(frame, row),
                    "original_label": original,
                    "images": app.state.adapter is not None,
                    "history": (
                        app.state.store.decisions(sample_id=sample_id) if app.state.store else None
                    ),
                }
                return templates.TemplateResponse(request, "sample.html", context)
        raise HTTPException(404, f"no sample {sample_id} in the predictions of {run_id}")

    @app.get("/runs/{run_id}/samples/{name}")
    def sample_image(run_id: str, name: str):
        path = run_dir(run_id)
        if name not in runs.sample_images(path):  # only listed files, no paths
            raise HTTPException(404, f"no sample {name}")
        return FileResponse(path / "samples" / name, media_type="image/png")

    @app.get("/compare", response_class=HTMLResponse)
    def compare_page(request: Request, ids: RunIds = None):
        ids = ids or []
        if not 2 <= len(ids) <= MAX_COMPARE:
            raise HTTPException(400, f"select 2-{MAX_COMPARE} runs to compare")
        summaries = [runs.summarize(run_dir(i)) for i in ids]
        diff = runs.config_diff([s.config for s in summaries])
        return templates.TemplateResponse(
            request, "compare.html", {"runs": summaries, "diff": diff}
        )

    @app.get("/review", response_class=HTMLResponse)
    def review_page(
        request: Request,
        run: str = "",
        split: str = "train",
        after: Annotated[int, Query(ge=0)] = 0,
        unsure: bool = False,
        round: Annotated[int | None, Query(ge=1)] = None,
        reviewer: str = "",
    ):
        curation = store()
        choices = review.review_runs(runs_dir)
        if not choices:
            raise HTTPException(404, "no run with label_issues.parquet")
        run = run or choices[0]
        candidates = review.load_candidates(run_dir(run), split)
        latest = curation.latest()
        pending = review.queue(candidates, latest, unsure)
        current, upcoming = review.current_and_next(pending, after)
        splits = sorted(review.load_candidates(run_dir(run))["split"].unique())
        context = {
            "run": run,
            "runs": choices,
            "split": split,
            "splits": splits,
            "after": after,
            "unsure": unsure,
            "round": round or curation.max_round(),
            "reviewer": reviewer or app.state.reviewer,
            "total": len(candidates),
            "reviewed": int(candidates["id"].isin(list(latest)).sum()),
            "pending": len(pending),
            "current": current,
            "upcoming": upcoming,
            "original_label": None,
            "images": app.state.adapter is not None,
        }
        if current is not None and app.state.adapter is not None:
            try:
                context["original_label"] = app.state.adapter.get_label(current["id"])
            except KeyError:
                pass
        return templates.TemplateResponse(request, "review.html", context)

    @app.post("/review/decide")
    def review_decide(
        run: Annotated[str, Form()],
        split: Annotated[str, Form()],
        sample_id: Annotated[str, Form()],
        action: Annotated[str, Form()],
        round: Annotated[int, Form(ge=1)],
        reviewer: Annotated[str, Form(min_length=1)],
        new_label: Annotated[str, Form()] = "",
        reason: Annotated[str, Form()] = "",
        after: Annotated[int, Form(ge=0)] = 0,
        unsure: Annotated[bool, Form()] = False,
    ):
        curation = store()
        candidates = review.load_candidates(run_dir(run), split)
        match = candidates[candidates["id"] == sample_id]
        if match.empty:
            raise HTTPException(404, f"{sample_id} is not a candidate of {run} ({split})")
        row = match.iloc[0]
        label = optional(new_label, int)
        if action == "relabel" and label == int(row["label"]):
            action, label = "keep", None  # the digit of the given label confirms it
        try:
            curation.add(sample_id, action, reviewer=reviewer, round=round, new_label=label,
                         reason=reason, given_label=int(row["label"]),
                         model_pred=int(row["pred"]), source_run=run)  # fmt: skip
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return redirect("/review", run=run, split=split, after=after, unsure=unsure or None,
                        round=round, reviewer=reviewer)  # fmt: skip

    @app.get("/curation", response_class=HTMLResponse)
    def curation_page(request: Request, action: str = "", round: str = "", exported: str = ""):
        curation = store()
        if action and action not in ACTIONS:
            raise HTTPException(400, f"action must be one of {ACTIONS}")
        rows = curation.decisions(action or None, optional(round, int))
        context = {
            "rows": rows,
            "stats": curation.stats(),
            "action": action,
            "round": round,
            "actions": ACTIONS,
            "exported": exported,
            "csv_path": curation.csv_path,
            "images": app.state.adapter is not None,
        }
        return templates.TemplateResponse(request, "curation.html", context)

    @app.post("/curation/{decision_id}/undo")
    def curation_undo(decision_id: int, back: Annotated[str, Form()] = ""):
        if not store().undo(decision_id):
            raise HTTPException(404, f"no decision {decision_id}")
        return RedirectResponse(back if back.startswith("/") else "/curation", status_code=303)

    @app.post("/curation/export")
    def curation_export():
        return redirect("/curation", exported=store().export_csv())

    @app.get("/api/runs")
    def api_runs():
        return [
            {k: v for k, v in vars(s).items() if k != "config"} for s in runs.list_runs(runs_dir)
        ]

    @app.get("/api/runs/{run_id}/metrics")
    def api_metrics(run_id: str):
        records, _ = runs.read_metrics(run_dir(run_id) / "metrics.jsonl")
        return records

    @app.get("/api/runs/{run_id}/stream")
    async def api_stream(request: Request, run_id: str):
        path = run_dir(run_id)
        try:
            offset = int(request.headers.get("last-event-id", 0))
        except ValueError:
            offset = 0
        return StreamingResponse(
            event_stream(path, offset, request.is_disconnected, poll_seconds),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/gpu")
    def api_gpu():
        return gpu_stats()

    return app


def main() -> None:
    load_dotenv(REPO_DIR / ".env")
    parser = argparse.ArgumentParser(description="ml-workbench (binds 127.0.0.1 only)")
    parser.add_argument("--runs", type=Path, default=Path(os.environ["RUNS_DIR"]) / "mnist")
    parser.add_argument("--data", type=Path, default=Path(os.environ["DATA_DIR"]) / "mnist")
    parser.add_argument(
        "--curation",
        type=Path,
        default=DEFAULT_CURATION,
        help="folder with decisions.csv (decisions.sqlite is kept next to it)",
    )
    parser.add_argument("--reviewer", default="immo")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    curation = args.curation if args.curation.is_dir() else None
    if curation is None:
        print(f"no curation folder {args.curation}: review and curation pages are off")
    app = create_app(args.runs, args.data, curation_dir=curation, reviewer=args.reviewer)
    uvicorn.run(app, host=HOST, port=args.port)
