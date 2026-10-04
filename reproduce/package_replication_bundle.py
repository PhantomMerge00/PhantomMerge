#!/usr/bin/env python3
"""Tarball helper for offline code sync (no large data)."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproduce"

INCLUDE_PATHS = ["ccer", "scripts", "third_party/jacobian-lens", "third_party/RARR", "reproduce"]

PROTOCOL_GLOBS = [
    "results/split_manifest.json",
    "results/cohort_manifest.json",
    "results/model_manifest.json",
    "results/agr/split_manifest_agr.json",
    "results/line_k/dev_thresholds.json",
    "results/line_a/v3/cohort_manifest.json",
    "results/agr/DETECTOR_COMPARISON_SUMMARY.json",
]

EXCLUDE_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", "instance_activations", "node_modules"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_files() -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        rp = p.resolve()
        if not rp.is_file() or rp in seen:
            return
        if any(part in EXCLUDE_DIR_NAMES for part in rp.parts):
            return
        seen.add(rp)
        files.append(rp)

    for rel in INCLUDE_PATHS:
        p = ROOT / rel
        if p.is_file():
            add(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    add(f)

    for glob_pat in PROTOCOL_GLOBS:
        for f in ROOT.glob(glob_pat):
            if f.is_file():
                add(f)

    sync = REPRO / "DATA_SYNC_MANIFEST.json"
    if sync.is_file():
        add(sync)

    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPRO / "dist")
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tarball = out_dir / f"phantom_merge_code_{ts}.tar.gz"
    manifest_path = out_dir / f"phantom_merge_code_{ts}.manifest.json"

    files = _collect_files()
    manifest_entries = []
    with tarfile.open(tarball, "w:gz") as tar:
        for f in files:
            arcname = f.relative_to(ROOT).as_posix()
            tar.add(f, arcname=arcname)
            manifest_entries.append(
                {"path": arcname, "bytes": f.stat().st_size, "sha256": _sha256_file(f)}
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "tarball": str(tarball),
        "n_files": len(manifest_entries),
        "files": manifest_entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {tarball}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
