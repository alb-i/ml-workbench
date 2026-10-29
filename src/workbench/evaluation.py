"""Evaluation of a run's predictions/<split>.parquet (columns id, label, pred, p0..p<K-1>, loss).

Confusion matrix, per-class precision/recall/F1, reliability diagram and ECE (equal-width
confidence bins), top-k, and the filters of the data explorer. Frames are cached per file and
modification time.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

N_BINS = 15


def prediction_splits(run_dir: Path) -> list[str]:
    """Splits with a predictions file, test first (then val, train, others)."""
    order = {"test": 0, "val": 1, "train": 2}
    names = [p.stem for p in (run_dir / "predictions").glob("*.parquet")]
    return sorted(names, key=lambda s: (order.get(s, 3), s))


def prob_columns(frame: pd.DataFrame) -> list[str]:
    """p0, p1, ... in class order."""
    cols = [c for c in frame.columns if c[:1] == "p" and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


@lru_cache(maxsize=16)
def _read(path: Path, mtime: float) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    probs = probabilities(frame)
    frame["confidence"] = probs.max(axis=1)
    frame["p_label"] = probs[np.arange(len(frame)), frame["label"].to_numpy()]
    return frame


def load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    """The predictions plus confidence (max probability) and p_label (probability of the label)."""
    path = run_dir / "predictions" / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read(path, path.stat().st_mtime)


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame[prob_columns(frame)].to_numpy(np.float64)


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, k: int) -> np.ndarray:
    """(k, k) counts; rows are labels, columns predictions."""
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


def per_class(cm: np.ndarray) -> list[dict]:
    """Per class: support, precision, recall, F1 (0 where undefined)."""
    tp = np.diag(cm).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.nan_to_num(tp / cm.sum(axis=0))
        recall = np.nan_to_num(tp / cm.sum(axis=1))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return [{"cls": k, "support": int(cm[k].sum()), "precision": float(precision[k]),
             "recall": float(recall[k]), "f1": float(f1[k])} for k in range(len(cm))]  # fmt: skip


def reliability(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Equal-width bins over [0, 1] (right-closed, the first includes 0): per non-empty bin the
    mean confidence, accuracy and count; ece = sum count / N * |accuracy - confidence|."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    count = np.bincount(bins, minlength=n_bins)
    used = count > 0
    conf = np.bincount(bins, confidence, n_bins)[used] / count[used]
    acc = np.bincount(bins, correct, n_bins)[used] / count[used]
    ece = float((count[used] * np.abs(acc - conf)).sum() / max(len(confidence), 1))
    return {"confidence": conf.tolist(), "accuracy": acc.tolist(),
            "count": count[used].tolist(), "ece": ece}  # fmt: skip


def evaluation(frame: pd.DataFrame) -> dict:
    labels, preds = frame["label"].to_numpy(), frame["pred"].to_numpy()
    k = len(prob_columns(frame))
    cm = confusion_matrix(labels, preds, k)
    classes = per_class(cm)
    correct = labels == preds
    return {
        "n": len(frame),
        "errors": int((~correct).sum()),
        "acc": float(correct.mean()) if len(frame) else 0.0,
        "macro_f1": float(np.mean([c["f1"] for c in classes])),
        "loss": float(frame["loss"].mean()) if len(frame) else 0.0,
        "confusion": cm.tolist(),
        "max_off_diagonal": int((cm - np.diag(np.diag(cm))).max()) if k else 0,
        "classes": classes,
        "reliability": reliability(frame["confidence"].to_numpy(), correct),
    }


def top_k(frame: pd.DataFrame, row: int, k: int = 3) -> list[tuple[int, float]]:
    """The k most probable classes of one row with their probabilities."""
    p = probabilities(frame.iloc[[row]])[0]
    order = np.argsort(-p, kind="stable")[:k]
    return [(int(c), float(p[c])) for c in order]


SORTS = {
    "loss": ("loss", False),  # highest loss first
    "confidence": ("confidence", True),  # least confident first
    "p_label": ("p_label", True),  # lowest probability of the label first
    "id": ("id", True),
}


def filter_samples(
    frame: pd.DataFrame,
    label: int | None = None,
    pred: int | None = None,
    errors: bool = False,
    min_loss: float | None = None,
    max_confidence: float | None = None,
    sort: str = "loss",
) -> pd.DataFrame:
    """Rows matching all given filters, sorted (see SORTS)."""
    mask = np.ones(len(frame), dtype=bool)
    if label is not None:
        mask &= frame["label"].to_numpy() == label
    if pred is not None:
        mask &= frame["pred"].to_numpy() == pred
    if errors:
        mask &= frame["label"].to_numpy() != frame["pred"].to_numpy()
    if min_loss is not None:
        mask &= frame["loss"].to_numpy() >= min_loss
    if max_confidence is not None:
        mask &= frame["confidence"].to_numpy() <= max_confidence
    column, ascending = SORTS[sort]
    return frame[mask].sort_values(column, ascending=ascending, kind="stable")
