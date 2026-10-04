"""TF-IDF ablation: strip template/surface form, test semantic core only."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ccer.audit.tau3_pooled_tfidf_control import (
    EVAL_DOMAINS,
    TextRow,
    _fit_eval_tfidf,
    _pool_pooled_shop_air_tel,
    _pool_shopping_train,
    _shopping_text,
    _tau3_text,
    _template_overlap,
)
from ccer.audit.tau3_quote_template_split import quote_fingerprint
from ccer.io_utils import write_json
from ccer.mechanism.supervised_probe import normalize_quote_for_bow
from ccer.mechanism.tau3.domain import get_tau3_domain
from ccer.paths import TAU3_DIR

_JSON_LIKE = re.compile(r"^\s*[\{\[]")
_JSON_KEY = re.compile(r'"[A-Za-z_][A-Za-z0-9_]*"\s*:\s*')
_JSON_STR = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
_TOOL_WRAPPER = re.compile(r"^(Error:|Success:|OK:)\s*", re.I)


def _quote_structure(q: str) -> str:
    s = str(q or "").strip()
    if _JSON_LIKE.match(s):
        return "json_like"
    if s.startswith("###") or re.match(r"^\d+\.\s*\*\*", s):
        return "markdown_list"
    return "prose"


def _flatten_json_values(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_json_values(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_flatten_json_values(v))
    elif obj is not None:
        out.append(str(obj))
    return out


def text_raw(r: TextRow) -> str:
    return r.text


def text_bow_norm(r: TextRow) -> str:
    return normalize_quote_for_bow(r.text)


def text_template_masked(r: TextRow) -> str:
    return quote_fingerprint(r.text)


def text_json_values_only(r: TextRow) -> str:
    q = r.text.strip()
    try:
        obj = json.loads(q)
        vals = _flatten_json_values(obj)
        return " ".join(vals).lower()
    except json.JSONDecodeError:
        # fallback: drop keys, keep quoted string literals
        stripped = _JSON_KEY.sub(" ", q)
        parts = _JSON_STR.findall(stripped)
        if parts:
            return " ".join(parts).lower()
        return quote_fingerprint(q)


def text_claim_value(r: TextRow) -> str:
    v = str(r.claim_value or "").strip()
    if not v:
        return text_template_masked(r)
    return normalize_quote_for_bow(v)


def text_slot_value(r: TextRow) -> str:
    slot = str(r.slot_norm or "").replace("_", " ").strip()
    val = normalize_quote_for_bow(r.claim_value or r.text)
    if slot and slot not in ("evidence_echo", "unknown"):
        return f"{slot} {val}".strip()
    return val


def text_prose_only(r: TextRow) -> str:
    if _quote_structure(r.text) != "prose":
        return ""
    return normalize_quote_for_bow(r.text)


def text_no_json_shell(r: TextRow) -> str:
    """Remove JSON punctuation + common tool wrapper tokens."""
    q = text_json_values_only(r) if _quote_structure(r.text) == "json_like" else text_bow_norm(r)
    q = _TOOL_WRAPPER.sub("", q)
    q = re.sub(r"[\{\}\[\]\":,]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


TEXT_MODES: dict[str, Callable[[TextRow], str]] = {
    "raw_quote": text_raw,
    "bow_normalized": text_bow_norm,
    "template_masked": text_template_masked,
    "json_values_only": text_json_values_only,
    "claim_value_only": text_claim_value,
    "slot_plus_value": text_slot_value,
    "prose_only": text_prose_only,
    "no_json_shell": text_no_json_shell,
}


@dataclass(frozen=True)
class RichTextRow(TextRow):
    claim_value: str = ""
    slot_norm: str = ""


def _shopping_rich(*, splits: set[str]) -> list[RichTextRow]:
    from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3

    ds = build_probe_dataset_v3(require_activation=True)
    out: list[RichTextRow] = []
    for row in ds.instances:
        sp = str(row.get("split") or "")
        if sp not in splits:
            continue
        text = str(row.get("response_quote") or "").strip()
        if not text:
            continue
        out.append(
            RichTextRow(
                text=text,
                y=int(row["y"]),
                domain="shopping",
                split=sp,
                verdict=str(row.get("gold_verdict") or ""),
                instance_audit_key=str(row["instance_audit_key"]),
                claim_value=str(row.get("claim_value") or row.get("response_quote") or ""),
                slot_norm="",
            )
        )
    return out


def _tau3_rich(cfg, *, inst_splits: set[str]) -> list[RichTextRow]:
    from ccer.mechanism.tau3.agr_slot_eval import _load_split_maps
    from ccer.mechanism.tau3.cohort import tau3_instance_rows

    _, inst_split = _load_split_maps(cfg)
    out: list[RichTextRow] = []
    for row in tau3_instance_rows(require_activation=True, cfg=cfg):
        sp = inst_split.get(row.instance_audit_key, "unassigned")
        if sp not in inst_splits:
            continue
        text = str(row.response_quote or "").strip()
        if not text:
            continue
        out.append(
            RichTextRow(
                text=text,
                y=row.y,
                domain=cfg.domain,
                split=sp,
                verdict=row.gold_verdict,
                instance_audit_key=row.instance_audit_key,
                claim_value=str(row.claim_value or ""),
                slot_norm=str(row.slot_norm or ""),
            )
        )
    return out


def _pool_pooled_rich() -> list[RichTextRow]:
    return (
        _shopping_rich(splits={"train"})
        + _tau3_rich(get_tau3_domain("airline"), inst_splits={"D_p"})
        + _tau3_rich(get_tau3_domain("telecom"), inst_splits={"D_p"})
    )


def _test_rich(domain: str) -> list[RichTextRow]:
    if domain == "shopping":
        return _shopping_rich(splits={"test"})
    return _tau3_rich(get_tau3_domain(domain), inst_splits={"test"})


def _apply_mode(rows: list[RichTextRow], mode: str) -> list[TextRow]:
    fn = TEXT_MODES[mode]
    out: list[TextRow] = []
    for r in rows:
        t = fn(r).strip()
        if not t:
            continue
        out.append(
            TextRow(
                text=t,
                y=r.y,
                domain=r.domain,
                split=r.split,
                verdict=r.verdict,
                instance_audit_key=r.instance_audit_key,
            )
        )
    return out


def _fit_eval_mode(train: list[RichTextRow], test: list[RichTextRow], mode: str) -> dict[str, Any]:
    tr = _apply_mode(train, mode)
    te = _apply_mode(test, mode)
    res = _fit_eval_tfidf(tr, te)
    res["n_train_after_filter"] = len(tr)
    res["n_test_after_filter"] = len(te)
    res["empty_text_dropped_test"] = len(test) - len(te)
    if te:
        res["template_overlap"] = _template_overlap(te)
    return res


def run_ablation() -> dict[str, Any]:
    train_pool = _pool_pooled_rich()
    out: dict[str, Any] = {
        "schema": "tau3_tfidf_core_content_ablation_v1",
        "train_pool": "pooled_shop_air_tel",
        "n_train_raw": len(train_pool),
        "modes": list(TEXT_MODES.keys()),
        "by_domain": {},
    }

    sweep_path = TAU3_DIR / "pooled_probe_sweep.json"
    l49 = {}
    if sweep_path.is_file():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        l49 = sweep["strategies"]["shop_plus_airline_plus_telecom"]["eval"]
    out["l49_pooled_probe_reference"] = {
        d: {"probe_auroc": l49[d]["probe_auroc"], "bind_auroc": l49[d]["bind_surprise_auroc"]}
        for d in EVAL_DOMAINS
        if d in l49
    }

    for dom in EVAL_DOMAINS:
        test_pool = _test_rich(dom)
        dom_block: dict[str, Any] = {"n_test_raw": len(test_pool), "modes": {}}
        for mode in TEXT_MODES:
            dom_block["modes"][mode] = _fit_eval_mode(train_pool, test_pool, mode)
        out["by_domain"][dom] = dom_block

    # narrative: does stripping drop telecom TF-IDF below L49 gap?
    tel_raw = out["by_domain"]["telecom"]["modes"]["raw_quote"].get("test_auroc")
    tel_core = out["by_domain"]["telecom"]["modes"]["claim_value_only"].get("test_auroc")
    tel_jsonv = out["by_domain"]["telecom"]["modes"]["json_values_only"].get("test_auroc")
    l49_tel = out["l49_pooled_probe_reference"].get("telecom", {}).get("probe_auroc")
    out["telecom_strip_summary"] = {
        "raw_tfidf": tel_raw,
        "json_values_only": tel_jsonv,
        "claim_value_only": tel_core,
        "l49_pooled_probe": l49_tel,
        "gap_l49_minus_claim_value_tfidf": None
        if l49_tel is None or tel_core is None
        else round(l49_tel - tel_core, 4),
    }
    if tel_core is not None and tel_raw is not None and tel_raw - tel_core >= 0.2:
        out["verdict"] = "template_stripping_helps_but_check_claim_value_coverage"
    elif tel_core is not None and l49_tel is not None and l49_tel - tel_core >= 0.15:
        out["verdict"] = "core_content_tfidf_still_below_L49"
    else:
        out["verdict"] = "stripping_insufficient_telecom_still_confounded"
    return out


def write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# TF-IDF Core-Content Ablation (strip templates)",
        "",
        "> Pooled train (shop+air+tel D_p); fixed test. Compare text modes vs L49 pooled probe.",
        "",
        f"**Verdict**: `{payload.get('verdict')}`",
        "",
    ]
    l49 = payload.get("l49_pooled_probe_reference") or {}
    for dom in EVAL_DOMAINS:
        lines += [f"## {dom}", "", "| Mode | TF-IDF AUROC | n_test (after filter) |", "|------|-------------:|----------------------:|"]
        for mode, res in payload["by_domain"][dom]["modes"].items():
            au = res.get("test_auroc")
            n = res.get("n_test_after_filter", res.get("n_test"))
            au_s = f"{au:.3f}" if au is not None else "—"
            lines.append(f"| {mode} | {au_s} | {n} |")
        ref = l49.get(dom, {})
        lines.append(f"| **L49 pooled probe (ref)** | **{ref.get('probe_auroc', '—')}** | — |")
        lines.append("")

    ts = payload.get("telecom_strip_summary") or {}
    lines += [
        "## Telecom strip summary",
        "",
        f"- raw TF-IDF: {ts.get('raw_tfidf')}",
        f"- json values only: {ts.get('json_values_only')}",
        f"- claim_value only: {ts.get('claim_value_only')}",
        f"- L49 pooled probe: {ts.get('l49_pooled_probe')}",
        f"- L49 − claim_value TF-IDF: {ts.get('gap_l49_minus_claim_value_tfidf')}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    payload = run_ablation()
    write_json(TAU3_DIR / "tfidf_core_content_ablation.json", payload)
    write_report(payload, TAU3_DIR / "TAU3_TFIDF_CORE_CONTENT_ABLATION_REPORT.md")
    print(json.dumps({"verdict": payload["verdict"], "telecom": payload["telecom_strip_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
