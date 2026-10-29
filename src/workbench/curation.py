"""SQLite store for curation decisions, CSV export/import (plan 6.1, 6.4, D23).

<dir>/decisions.sqlite holds every decision; "Export" overwrites <dir>/decisions.csv in the
format of plan 6.4 (sample_id, action, new_label, reason, reviewer, timestamp, round; one row per
decision, oldest first; a later row for the same sample wins). An empty database imports an
existing decisions.csv, so both stay consistent. Besides the CSV columns the database keeps the
given label, the model prediction and the run the candidate came from, to measure how often the
reviewer agrees with the model (confirmation bias, plan 10.3).
"""

import csv
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

ACTIONS = ("relabel", "keep", "drop", "unsure")
CSV_COLUMNS = ["sample_id", "action", "new_label", "reason", "reviewer", "timestamp", "round"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('relabel', 'keep', 'drop', 'unsure')),
    new_label INTEGER,
    reason TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    round INTEGER NOT NULL,
    given_label INTEGER,
    model_pred INTEGER,
    source_run TEXT
)
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class CurationStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.csv_path = self.dir / "decisions.csv"
        self._lock = threading.Lock()  # FastAPI runs sync routes in a thread pool
        self._db = sqlite3.connect(self.dir / "decisions.sqlite", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.execute(SCHEMA)
        if self.count() == 0 and self.csv_path.is_file():
            self.import_csv()

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

    def add(self, sample_id: str, action: str, *, reviewer: str, round: int,
            new_label: int | None = None, reason: str = "", given_label: int | None = None,
            model_pred: int | None = None, source_run: str | None = None,
            timestamp: str | None = None) -> int:  # fmt: skip
        """New decision; returns its id. relabel needs a new label different from the given one;
        the other actions take none."""
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
        if action == "relabel":
            if new_label is None:
                raise ValueError("relabel needs new_label")
            if given_label is not None and new_label == given_label:
                raise ValueError("new_label equals the given label (use keep)")
        elif new_label is not None:
            raise ValueError(f"{action} takes no new_label")
        row = (sample_id, action, new_label, reason.strip(), reviewer, timestamp or now(), round,
               given_label, model_pred, source_run)  # fmt: skip
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO decisions (sample_id, action, new_label, reason, reviewer, timestamp,"
                " round, given_label, model_pred, source_run) VALUES (?,?,?,?,?,?,?,?,?,?)",
                row,
            )
            return cursor.lastrowid

    def undo(self, decision_id: int) -> bool:
        """Deletes a decision; False if there was none with this id."""
        with self._lock, self._db:
            cursor = self._db.execute("DELETE FROM decisions WHERE id = ?", (decision_id,))
            return cursor.rowcount > 0

    def decisions(self, action: str | None = None, round: int | None = None,
                  sample_id: str | None = None) -> list[dict]:  # fmt: skip
        """Decisions newest first, optionally filtered."""
        where, args = [], []
        for column, value in (("action", action), ("round", round), ("sample_id", sample_id)):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(value)
        sql = "SELECT * FROM decisions" + (" WHERE " + " AND ".join(where) if where else "")
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY id DESC", args).fetchall()
        return [dict(r) for r in rows]

    def latest(self) -> dict[str, dict]:
        """The latest decision per sample."""
        out = {}
        for row in reversed(self.decisions()):  # oldest first, later ones win
            out[row["sample_id"]] = row
        return out

    def max_round(self) -> int:
        with self._lock:
            value = self._db.execute("SELECT MAX(round) FROM decisions").fetchone()[0]
        return value or 1

    def stats(self) -> dict:
        """Counts per action (latest decision per sample) and the agreement with the model: among
        the keep/relabel decisions with a known model prediction, the share whose resulting label
        is the model's prediction; separately for the cases where the model disagreed with the
        given label."""
        latest = list(self.latest().values())
        counts = {a: sum(r["action"] == a for r in latest) for a in ACTIONS}
        judged = [r for r in latest if r["action"] in ("keep", "relabel")
                  and r["model_pred"] is not None and r["given_label"] is not None]  # fmt: skip

        def final(r):
            return r["new_label"] if r["action"] == "relabel" else r["given_label"]

        disputed = [r for r in judged if r["model_pred"] != r["given_label"]]
        agree = sum(final(r) == r["model_pred"] for r in judged)
        agree_disputed = sum(final(r) == r["model_pred"] for r in disputed)
        return {
            "samples": len(latest),
            "decisions": self.count(),
            "counts": counts,
            "judged": len(judged),
            "agree": agree,
            "disputed": len(disputed),
            "agree_disputed": agree_disputed,
        }

    def export_csv(self) -> int:
        """Writes all decisions (oldest first) to decisions.csv; returns the number of rows."""
        rows = list(reversed(self.decisions()))
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(CSV_COLUMNS)
            for r in rows:
                new_label = "" if r["new_label"] is None else r["new_label"]
                writer.writerow([r["sample_id"], r["action"], new_label, r["reason"], r["reviewer"],
                                 r["timestamp"], r["round"]])  # fmt: skip
        tmp.replace(self.csv_path)
        return len(rows)

    def import_csv(self) -> int:
        """Adds the rows of decisions.csv (without model information); returns their number."""
        with open(self.csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            new_label = int(r["new_label"]) if r["new_label"] else None
            self.add(r["sample_id"], r["action"], reviewer=r["reviewer"], round=int(r["round"]),
                     new_label=new_label, reason=r["reason"], timestamp=r["timestamp"])  # fmt: skip
        return len(rows)
