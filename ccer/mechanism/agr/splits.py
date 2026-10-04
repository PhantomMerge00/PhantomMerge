"""Load AGR four-way splits."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from ccer.paths import AGR_SPLIT_MANIFEST


@lru_cache(maxsize=1)
def load_agr_manifest() -> dict[str, Any]:
    if not AGR_SPLIT_MANIFEST.is_file():
        from ccer.audit.agr_split_manifest import write_agr_split_manifest

        return write_agr_split_manifest()
    return json.loads(AGR_SPLIT_MANIFEST.read_text(encoding="utf-8"))


def instance_agr_split(instance_audit_key: str) -> str:
    return str(load_agr_manifest()["instance_split"].get(instance_audit_key) or "test")


def filter_by_agr_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    mp = load_agr_manifest()["instance_split"]
    return [r for r in rows if mp.get(str(r.get("instance_audit_key") or "")) == split]
