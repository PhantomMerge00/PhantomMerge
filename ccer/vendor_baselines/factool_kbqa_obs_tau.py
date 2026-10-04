"""FacTool knowledge_qa_pipeline._verification + Obs_τ (no search tools)."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("${PHANTOM_MERGE_ROOT}")
VENDOR = ROOT / "third_party/factool"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))

from ccer.mechanism.bind_surprise import try_auroc  # noqa: E402
from ccer.vendor_baselines.factool_vllm_chat import (  # noqa: E402
    VllmFactoolChat,
    assert_pm_vllm_model_on_server,
)
from ccer.vendor_baselines.obs_tau_common import (  # noqa: E402
    fair_obs_tau_corpus,
    iter_detection_rows_with_traj,
)
from ccer.vendor_baselines.p2_checkpoint import (  # noqa: E402
    append_score_line,
    eta_line,
    load_scored_keys,
    write_progress,
)
import yaml  # noqa: E402


class ObsTauKbqaPipeline:
    """Vendor agreement_verification.yaml + same message layout as knowledge_qa_pipeline._verification."""

    def __init__(self) -> None:
        self.chat = VllmFactoolChat()
        prompts_path = ROOT / "third_party/factool/factool/utils/prompts"
        with open(prompts_path / "agreement_verification.yaml", "r", encoding="utf-8") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        self.verification_prompt = data["knowledge_qa"]

    async def _verification(self, claims, evidences):
        messages_list = [
            [
                {"role": "system", "content": self.verification_prompt["system"]},
                {
                    "role": "user",
                    "content": self.verification_prompt["user"].format(
                        claim=claim["claim"], evidence=str(evidence)
                    ),
                },
            ]
            for claim, evidence in zip(claims, evidences)
        ]
        return await self.chat.async_run(messages_list, dict)


def _pm_risk_from_verification(ver: dict[str, Any] | None) -> float:
    if not ver:
        return 0.5
    fact = ver.get("factuality")
    if fact is True:
        return 0.0
    if fact is False:
        return 1.0
    return 0.5


async def _verify_batch(pipe: ObsTauKbqaPipeline, batch: list[tuple[str, str]]) -> list[float]:
    claims = [{"claim": c} for c, _ in batch]
    evidences = [[e] for _, e in batch]
    verifications = await pipe._verification(claims, evidences)
    return [_pm_risk_from_verification(v) for v in verifications]


def run_factool_kbqa_obs_tau(
    *,
    out_dir: Path | None = None,
    batch_size: int = 16,
    limit: int | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or (ROOT / "results/rq2/p2_vendor/factool_kbqa")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "llm_cache.jsonl"

    pm_url, pm_model = assert_pm_vllm_model_on_server()
    pipe = ObsTauKbqaPipeline()
    rows_meta: list[dict[str, Any]] = []
    pending: list[tuple[str, str, dict[str, Any]]] = []

    for row, traj, anchor_pid, _rollout in iter_detection_rows_with_traj():
        if limit is not None and len(pending) >= limit:
            break
        claim = str(row.get("response_quote") or "")
        corpus = fair_obs_tau_corpus(traj, anchor_pid)
        pending.append((claim, corpus, row))

    scores_path = out_dir / "scores.jsonl"
    progress_path = out_dir / "progress.json"
    done_keys = load_scored_keys(scores_path)
    todo = [(c, e, row) for c, e, row in pending if str(row["instance_audit_key"]) not in done_keys]
    total = len(pending)

    async def _run_all() -> None:
        import time
        from ccer.mechanism.line_l_llm_common import load_disk_cache

        load_disk_cache(cache_path)
        t0 = time.perf_counter()
        n_new = 0
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            batch_scores = await _verify_batch(pipe, [(c, e) for c, e, _ in chunk])
            for (_, _, row), sc in zip(chunk, batch_scores):
                rec = {
                    "instance_audit_key": row["instance_audit_key"],
                    "split": row["split"],
                    "y": row["y"],
                    "p_factool_kbqa": sc,
                }
                append_score_line(scores_path, rec)
                n_new += 1
            done = total - len(todo) + min(i + batch_size, len(todo))
            write_progress(
                progress_path,
                {
                    "arm": "factool_kbqa",
                    "done": done,
                    "total": total,
                    "n_new_this_run": n_new,
                    "eta": eta_line(n_new, len(todo), t0) if n_new else "resume",
                    "pm_vllm": {"base_url": pm_url, "model": pm_model},
                },
            )
            print(f"[factool_kbqa] {done}/{total} {eta_line(n_new, len(todo), t0)}", flush=True)

    asyncio.run(_run_all())
    records: list[dict[str, Any]] = []
    scores: list[float] = []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records.append(rec)
        scores.append(float(rec.get("p_factool_kbqa", 0.5)))
        rows_meta.append(rec)

    test_idx = [i for i, r in enumerate(records) if r["split"] == "test"]
    scored = [{"y_pm": records[i]["y"], "p_score": scores[i]} for i in test_idx]
    payload = {
        "schema": "factool_kbqa_obs_tau_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vendor": "third_party/factool knowledge_qa_pipeline._verification",
        "adaptation": "Obs_τ corpus replaces search; VllmFactoolChat (CCER_PM_VLLM_BASE_URL)",
        "pm_vllm": {"base_url": pm_url, "model": pm_model},
        "vllm_available": True,
        "test_auroc": try_auroc(scored, "p_score"),
        "n_test": len(test_idx),
        "n_pm_test": sum(records[i]["y"] for i in test_idx),
    }
    payload["n_rows"] = len(records)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
