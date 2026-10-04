"""SelfCheckNLI (vendor) with Obs_τ evidence (single- or multi-sample)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("${PHANTOM_MERGE_ROOT}")
VENDOR = ROOT / "third_party/selfcheckgpt"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))

from ccer.mechanism.bind_surprise import try_auroc  # noqa: E402
from ccer.vendor_baselines.obs_tau_common import (  # noqa: E402
    iter_detection_rows_with_traj,
    multi_sample_passages,
)


def _load_nli(device: str):
    import torch
    from selfcheckgpt.utils import NLIConfig
    from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

    nli_model = NLIConfig.nli_model
    tok = DebertaV2Tokenizer.from_pretrained(nli_model)
    model = DebertaV2ForSequenceClassification.from_pretrained(nli_model)
    model.eval()
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(dev)
    return tok, model, dev


def _mean_contradiction(tok, model, dev, sentence: str, passages: list[str]) -> float:
    import torch

    if not sentence or not passages:
        return 0.5
    probs: list[float] = []
    for passage in passages:
        if not passage:
            continue
        enc = tok(
            sentence,
            passage,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)
            probs.append(float(p[0][1].item()))
    if not probs:
        return 0.5
    return float(np.mean(probs))


def run_selfcheck_nli_obs_tau(
    *,
    device: str = "cuda",
    out_dir: Path | None = None,
    multi_sample_k: int = 1,
) -> dict[str, Any]:
    sub = "selfcheck_nli_multi" if multi_sample_k > 1 else "selfcheck_nli"
    out_dir = out_dir or (ROOT / "results/rq2/p2_vendor" / sub)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model, dev = _load_nli(device)
    scores: list[float] = []
    records: list[dict[str, Any]] = []
    score_key = "p_selfcheck_nli"

    for row, traj, anchor_pid, rollout in iter_detection_rows_with_traj():
        claim = str(row.get("response_quote") or "")
        if multi_sample_k > 1:
            passages = multi_sample_passages(traj, anchor_pid, rollout, k=multi_sample_k)
        else:
            passages = multi_sample_passages(traj, anchor_pid, rollout, k=1)
        sc = _mean_contradiction(tok, model, dev, claim, passages)
        scores.append(sc)
        records.append(
            {
                "instance_audit_key": row["instance_audit_key"],
                "split": row["split"],
                "y": row["y"],
                score_key: sc,
                "n_passages": len([p for p in passages if p]),
            }
        )

    scores_arr = np.array(scores, dtype=float)
    test_idx = [i for i, r in enumerate(records) if r["split"] == "test"]
    scored = [{"y_pm": records[i]["y"], "p_score": scores_arr[i]} for i in test_idx]
    adaptation = (
        f"K={multi_sample_k} rollout product excerpts + anchor corpus; mean P(contradiction) per SelfCheck-NLI"
        if multi_sample_k > 1
        else "single anchor corpus as one pseudo-sample"
    )
    payload = {
        "schema": "selfcheck_nli_obs_tau_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vendor": "third_party/selfcheckgpt SelfCheck-NLI scoring (vendor formula: mean over samples)",
        "adaptation": adaptation,
        "multi_sample_k": multi_sample_k,
        "test_auroc": try_auroc(scored, "p_score"),
        "n_test": len(test_idx),
        "n_pm_test": sum(records[i]["y"] for i in test_idx),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return payload
