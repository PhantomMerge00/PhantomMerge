"""AGR signal extraction: z1 (probe logit) and z2 (J-slot log mass)."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ccer.mechanism.activation_store import get_vector, load_activation_npz
from ccer.mechanism.bind_surprise import probe_logit
from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3
from ccer.mechanism.line_a_instance_activations import instance_npz_path
from ccer.mechanism.supervised_probe import normalize_quote_for_bow, _bow_vectorizer
from ccer.mechanism.agr.anchor_mass_cache import load_anchor_mass_cache_index
from ccer.mechanism.agr.splits import load_agr_manifest
from ccer.paths import PRISM_AUDIT_JSONL

LINE_A_LAYER = 49
LINE_A_POSITION = "claim_onset"
LOG_MASS_FLOOR = -12.0
EPS = 1e-9

ASF_VERDICT = "anchored_hallucination"
PM_TYPES = ("cross_object_merge", "constraint_projection", ASF_VERDICT)


@dataclass
class AgrProbeModel:
    scaler: StandardScaler
    clf: LogisticRegression
    layer: int = LINE_A_LAYER
    position: str = LINE_A_POSITION

    def logit(self, h: np.ndarray) -> float:
        x = self.scaler.transform(h.reshape(1, -1))
        p = float(self.clf.predict_proba(x)[0, 1])
        return probe_logit(p, eps=EPS)

    def prob(self, h: np.ndarray) -> float:
        x = self.scaler.transform(h.reshape(1, -1))
        return float(self.clf.predict_proba(x)[0, 1])

    @property
    def w_p(self) -> np.ndarray:
        return np.asarray(self.clf.coef_[0], dtype=np.float64)

    @property
    def b_p(self) -> float:
        return float(self.clf.intercept_[0])


def _load_hidden(iak: str, tid: str) -> np.ndarray | None:
    path = instance_npz_path(iak, tid)
    if not path.is_file():
        return None
    loaded = load_activation_npz(path)
    return get_vector(loaded, position=LINE_A_POSITION, layer=LINE_A_LAYER)


def train_agr_probe(*, train_split: str = "D_p") -> AgrProbeModel:
    ds = build_probe_dataset_v3(require_activation=True)
    mp = load_agr_manifest()["instance_split"]
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for row in ds.instances:
        if mp.get(str(row["instance_audit_key"])) != train_split:
            continue
        h = _load_hidden(str(row["instance_audit_key"]), str(row["trajectory_id"]))
        if h is None:
            continue
        X_list.append(h)
        y_list.append(int(row["y"]))
    if not X_list:
        raise RuntimeError(f"No activations for AGR split {train_split}")
    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xs, y)
    return AgrProbeModel(scaler=scaler, clf=clf)


def train_agr_bow(*, train_split: str = "D_p") -> tuple[Any, LogisticRegression]:
    ds = build_probe_dataset_v3(require_activation=True)
    mp = load_agr_manifest()["instance_split"]
    texts: list[str] = []
    y_list: list[int] = []
    for row in ds.instances:
        if mp.get(str(row["instance_audit_key"])) != train_split:
            continue
        texts.append(normalize_quote_for_bow(str(row.get("response_quote") or "")))
        y_list.append(int(row["y"]))
    vec = _bow_vectorizer()
    X = vec.fit_transform(texts)
    y = np.asarray(y_list, dtype=int)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X, y)
    return vec, clf


def _prism_index() -> dict[str, dict[str, Any]]:
    if not PRISM_AUDIT_JSONL.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in PRISM_AUDIT_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[str(r["instance_audit_key"])] = r
    return out


def value_token_count(claim_value: str, *, tokenizer: Any | None = None) -> int:
    text = str(claim_value or "").strip()
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(re.findall(r"\S+", text))


def apply_z2_variant(
    rows: list[dict[str, Any]],
    variant: str = "log_mass",
    *,
    z2_source: str = "lexicon",
) -> str:
    """Select primary z2 column. Preserves ``z2_log_mass`` (lexicon) for legacy ablations."""
    src = "anchor-value" if z2_source == "anchor_value" else "slot-lexicon"
    if variant == "odds":
        for r in rows:
            r["z2"] = float(r["z2_odds"])
        return f"z2_odds (Eq.2 log-odds on {src} mass)"
    if variant != "log_mass":
        raise ValueError(f"unknown z2 variant: {variant}")
    return f"log mass on {src} (primary z2 column)"


def build_agr_signal_rows(
    probe: AgrProbeModel,
    *,
    tokenizer: Any | None = None,
    z2_source: str = "anchor_value",
) -> list[dict[str, Any]]:
    ds = build_probe_dataset_v3(require_activation=True)
    mp = load_agr_manifest()["instance_split"]
    prism = _prism_index()
    anchor_cache = load_anchor_mass_cache_index() if z2_source == "anchor_value" else {}
    rows: list[dict[str, Any]] = []
    for row in ds.instances:
        iak = str(row["instance_audit_key"])
        tid = str(row["trajectory_id"])
        agr_sp = mp.get(iak, "test")
        h = _load_hidden(iak, tid)
        if h is None:
            continue
        p_pm = probe.prob(h)
        z1 = probe.logit(h)
        pr = prism.get(iak) or {}

        # Legacy slot-lexicon mass (internal; z_fun uses anchor-value readout in paper)
        log_lex = float(pr.get("log_slot_mass") if pr.get("log_slot_mass") is not None else float("-inf"))
        if not math.isfinite(log_lex):
            log_lex = LOG_MASS_FLOOR
        log_lex = max(log_lex, LOG_MASS_FLOOR)
        lex_mass = float(math.exp(log_lex)) if log_lex > LOG_MASS_FLOOR else 0.0
        lex_mass = min(max(lex_mass, 0.0), 1.0)

        if z2_source == "anchor_value":
            ac = anchor_cache.get(iak) or {}
            z2_available = bool(ac.get("z2_available"))
            anchor_value = ac.get("anchor_value")
            anchor_provenance = ac.get("anchor_provenance")
            if z2_available:
                raw_log = ac.get("log_anchor_value_mass")
                log_sm = float(raw_log) if raw_log is not None and math.isfinite(float(raw_log)) else float("-inf")
                slot_mass = float(ac.get("anchor_value_mass") or 0.0)
                slot_mass = min(max(slot_mass, 0.0), 1.0)
                z2 = log_sm
                z2_odds = math.log((1.0 - slot_mass + EPS) / (slot_mass + EPS))
            else:
                log_sm = float("nan")
                slot_mass = float("nan")
                z2 = float("nan")
                z2_odds = float("nan")
        else:
            z2_available = True
            log_sm = log_lex
            slot_mass = lex_mass
            anchor_value = None
            anchor_provenance = None
            z2 = log_sm
            z2_odds = math.log((1.0 - slot_mass + EPS) / (slot_mass + EPS))
        claim_value = str(pr.get("claim_value") or "")
        n_tok = value_token_count(claim_value, tokenizer=tokenizer)
        rows.append(
            {
                **row,
                "agr_split": agr_sp,
                "y_pm": int(row["y"]),
                "p_pm_agr": p_pm,
                "z1": z1,
                "z2": z2,
                "z2_odds": z2_odds,
                "z2_log_mass": log_lex,
                "z2_available": z2_available if z2_source == "anchor_value" else True,
                "z2_anchor_log_mass": log_sm if z2_source == "anchor_value" else None,
                "z2_anchor_odds": z2_odds if z2_source == "anchor_value" else None,
                "slot_mass": slot_mass,
                "log_slot_mass_raw": float(pr.get("log_slot_mass") or float("-nan")),
                "anchor_value": anchor_value,
                "anchor_provenance": anchor_provenance,
                "value_token_count": n_tok,
                "value_span_group": "single_token" if n_tok <= 1 else "multi_token",
            }
        )
    return rows
