"""Incremental disk checkpoint + progress for P2 vendor LLM baselines."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_scored_keys(scores_path: Path, key_field: str = "instance_audit_key") -> set[str]:
    if not scores_path.is_file():
        return set()
    done: set[str] = set()
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        k = str(row.get(key_field) or "")
        if k:
            done.add(k)
    return done


def append_score_line(scores_path: Path, record: dict[str, Any]) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_progress(progress_path: Path, payload: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _now()
    progress_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def eta_line(done: int, total: int, t0: float) -> str:
    if done <= 0 or total <= 0:
        return "ETA: —"
    elapsed = time.perf_counter() - t0
    per = elapsed / done
    remain = per * (total - done)
    return f"ETA ~{remain / 3600:.1f}h ({remain / 60:.0f} min) @ {per:.1f}s/row"
