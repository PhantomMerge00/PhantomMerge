"""Model manifest aggregation (§5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ccer.io_utils import load_json, write_json
from ccer.paths import MODEL_MANIFEST_JSON, SHOPPING_METADATA

ROOT = Path("${PHANTOM_MERGE_ROOT}")
HF_SNAPSHOT = (
    ROOT
    / "runtime/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
)
DEFAULT_MODEL = ROOT / "runtime/.cache/huggingface/hub/Qwen3-32B"


def _resolve_weights_path() -> str:
    if DEFAULT_MODEL.is_dir() and (DEFAULT_MODEL / "config.json").is_file():
        return str(DEFAULT_MODEL)
    if HF_SNAPSHOT.is_dir():
        return str(HF_SNAPSHOT)
    return str(DEFAULT_MODEL)


def build_model_manifest() -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if SHOPPING_METADATA.is_file():
        meta = load_json(SHOPPING_METADATA)

    missing: list[str] = []
    if not meta.get("model_revision_or_commit"):
        missing.append("model_revision_or_commit")
    if not meta.get("vllm_version"):
        missing.append("vllm_version")
    if not meta.get("prompt_token_ids_logged"):
        missing.append("original_prompt_token_ids")

    manifest_id = "qwen3-32b_shopping_v1"
    entry = {
        "model_manifest_id": manifest_id,
        "model_id": meta.get("model_id") or "Qwen/Qwen3-32B",
        "weights_path": _resolve_weights_path(),
        "model_revision_or_commit": meta.get("model_revision_or_commit"),
        "tokenizer": "Qwen3Tokenizer",
        "chat_template": "qwen3",
        "thinking_enabled": False,
        "precision": meta.get("environment", {}).get("dtype") or "bfloat16",
        "decoding": meta.get("decoding") or {
            "temperature": 0,
            "top_p": 1.0,
            "seed": 42,
            "do_sample": False,
        },
        "system_prompt_file": meta.get("system_prompt_file"),
        "protocol_id": meta.get("protocol_id"),
        "vllm": {
            "base_url": "http://127.0.0.1:8003/v1",
            "enable_thinking": False,
            "vllm_version": meta.get("vllm_version"),
        },
        "limitations": [
            "original_prompt_token_ids never logged — exact_token_equality not_available",
            "greedy replay may diverge if chat template backend differs from original vLLM run",
        ],
        "missing_fields": missing,
        "exact_replay_claim": False if missing else False,
    }
    out = {
        "schema_version": "ccer_model_manifest_v1",
        "default_manifest_id": manifest_id,
        "manifests": {manifest_id: entry},
        "source_metadata_files": [str(SHOPPING_METADATA)],
    }
    write_json(MODEL_MANIFEST_JSON, out)
    return out


if __name__ == "__main__":
    print(json.dumps(build_model_manifest(), indent=2))
