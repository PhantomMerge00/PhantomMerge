"""Slot-level DeBERTa-MNLI entailment for rewrite acceptance (cached, shadow-capable)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

_DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
_DEFAULT_TAU = 0.5
_CACHE_ENV = "REWRITE_SLOT_NLI_CACHE"
_SHADOW_ENV = "REWRITE_SLOT_NLI_SHADOW"
_DEVICE_ENV = "REWRITE_SLOT_NLI_DEVICE"
_TAU_ENV = "REWRITE_SLOT_NLI_TAU"

_checker_lock = threading.Lock()
_checker_instance: SlotEntailChecker | None = None


def _default_cache_path() -> Path:
    custom = os.environ.get(_CACHE_ENV, "").strip()
    if custom:
        return Path(custom)
    return Path(__file__).resolve().parents[3] / "results/rq3/cache/slot_nli_rewrite.jsonl"


def nli_shadow_mode() -> bool:
    return os.environ.get(_SHADOW_ENV, "").strip() in ("1", "true", "yes")


def nli_entail_threshold() -> float:
    try:
        return float(os.environ.get(_TAU_ENV, str(_DEFAULT_TAU)))
    except ValueError:
        return _DEFAULT_TAU


class SlotEntailChecker:
    """Lazy DeBERTa-MNLI checker with optional JSONL cache."""

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        cache_path: Path | None = None,
        entail_threshold: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = (device or os.environ.get(_DEVICE_ENV) or "cpu").strip()
        self.cache_path = cache_path or _default_cache_path()
        self.entail_threshold = (
            entail_threshold if entail_threshold is not None else nli_entail_threshold()
        )
        self._tokenizer = None
        self._model = None
        self._id2label: dict[int, str] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_loaded = False

    def _cache_key(self, premise: str, hypothesis: str) -> str:
        blob = f"{premise}\n---\n{hypothesis}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if self._cache_loaded:
            return
        self._cache_loaded = True
        if not self.cache_path.is_file():
            return
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = str(rec.get("key") or "")
                if key:
                    self._cache[key] = rec
        except (OSError, json.JSONDecodeError):
            pass

    def _append_cache(self, key: str, record: dict[str, Any]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**record, "key": key}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self._model = self._model.to(self.device)
        else:
            self.device = "cpu"
            self._model = self._model.to("cpu")
        self._id2label = {
            int(k): str(v).lower() for k, v in (self._model.config.id2label or {}).items()
        }

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        """P(entailment) for premise/hypothesis pair."""
        prem = str(premise or "").strip()
        hyp = str(hypothesis or "").strip()
        if not prem or not hyp:
            return 0.0
        self._load_cache()
        key = self._cache_key(prem, hyp)
        if key in self._cache:
            return float(self._cache[key].get("p_entail") or 0.0)

        self._ensure_model()
        import torch

        enc = self._tokenizer(
            [prem],
            [hyp],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
        entail_idx = next(
            (i for i, lab in self._id2label.items() if lab == "entailment"),
            0,
        )
        p_entail = float(probs[entail_idx].item())
        rec = {"p_entail": p_entail, "premise_len": len(prem), "hypothesis": hyp[:200]}
        self._cache[key] = rec
        self._append_cache(key, rec)
        return p_entail

    def slot_entails(self, premise: str, hypothesis: str) -> bool:
        return self.entail_prob(premise, hypothesis) >= self.entail_threshold


def get_slot_entail_checker() -> SlotEntailChecker:
    global _checker_instance
    with _checker_lock:
        if _checker_instance is None:
            _checker_instance = SlotEntailChecker()
        return _checker_instance


def set_slot_entail_checker(checker: SlotEntailChecker | None) -> None:
    """Test hook: inject mock checker."""
    global _checker_instance
    with _checker_lock:
        _checker_instance = checker


def slot_entails(premise: str, hypothesis: str, *, checker: SlotEntailChecker | None = None) -> bool:
    chk = checker or get_slot_entail_checker()
    return chk.slot_entails(premise, hypothesis)
