"""Review queue (plan 6.2 v3, 10.3): candidates from a run's label_issues.parquet (D21, D22).

label_issues.parquet has one row per image with at least id, split, label, pred, candidate and
rank (1 = most suspicious candidate). The queue is the candidates of one split in rank order
without the samples that already have a decision; with unsure=True it is the samples whose
latest decision is "unsure" (second look).
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

ISSUES_FILE = "label_issues.parquet"


def review_runs(runs_dir: Path) -> list[str]:
    """Run ids with candidates, newest first."""
    if not runs_dir.is_dir():
        return []
    return sorted((d.name for d in runs_dir.iterdir() if (d / ISSUES_FILE).is_file()), reverse=True)


@lru_cache(maxsize=8)
def _candidates(path: Path, mtime: float) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[frame["candidate"]].copy()
    frame["rank"] = frame["rank"].astype(int)
    return frame.sort_values("rank", kind="stable").reset_index(drop=True)


def load_candidates(run_dir: Path, split: str | None = None) -> pd.DataFrame:
    """Candidates (all splits, or one) sorted by rank."""
    path = run_dir / ISSUES_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = _candidates(path, path.stat().st_mtime)
    return frame if split is None else frame[frame["split"] == split].reset_index(drop=True)


def queue(candidates: pd.DataFrame, latest: dict[str, dict], unsure: bool = False) -> pd.DataFrame:
    """Candidates still to review: undecided ones, or (unsure=True) those last marked unsure."""
    if unsure:
        keep = [latest.get(i, {}).get("action") == "unsure" for i in candidates["id"]]
    else:
        keep = [i not in latest for i in candidates["id"]]
    return candidates[keep]


def current_and_next(pending: pd.DataFrame, after: int) -> tuple[dict | None, dict | None]:
    """The first pending candidate with rank > after, and the one after it (for prefetching)."""
    rest = pending[pending["rank"] > after].head(2).to_dict("records")
    return (rest[0] if rest else None), (rest[1] if len(rest) > 1 else None)
