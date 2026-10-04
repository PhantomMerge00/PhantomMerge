"""Step 1/2: AGR(slot) zero-shot and in-domain cross-domain evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.audit.tau3_agr_split_manifest import write_tau3_agr_split_manifest
from ccer.io_utils import write_json
from ccer.mechanism.tau3.agr_slot_eval import run_eval
from ccer.mechanism.tau3.domain import get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import export_shopping_frozen_probe


def _write_report(results: dict[str, Any], cfg_domain: str) -> None:
    cfg = get_tau3_domain(cfg_domain)
    zs = results.get("zeroshot") or {}
    idm = results.get("indomain") or {}
    delta = results.get("delta_indomain_minus_zeroshot") or {}
    zs_det = zs.get("detection") or {}
    idm_det = idm.get("detection") or {}
    zs_gate = zs.get("gating") or {}
    idm_gate = idm.get("gating") or {}

    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    domain_title = cfg.domain.capitalize()
    zs_label = zs.get("effective_method_label") or f"{domain_title} zero-shot"
    idm_label = idm.get("effective_method_label") or f"{domain_title} in-domain"
    probe_only = (zs.get("effective_method") == "probe_only") or (idm.get("effective_method") == "probe_only")
    lines = [
        f"# Tau3 {domain_title} Cross-Domain Report",
        "",
        "> **RQ2 门禁**：floor_rate≥99% 时下列分数为 **probe-only**（J-lens 零贡献），"
        "不可作为 AGR(slot) 融合有效性证据。详见 `results/tau3/TAU3_CROSS_DOMAIN_AUDIT_REPORT.md`。",
        "",
        "## 对比表（test claims）",
        "",
        "| Method | AUROC | F1 | PM-rate (gated) | CB-retention |",
        "|--------|-------|-----|-----------------|--------------|",
        "| Shopping AGR(slot) ref | 0.962 | 0.891 | 0.053 | 0.628 |",
        f"| {zs_label} | {_fmt(zs.get('auroc'))} | {_fmt(zs_det.get('f1'))} | "
        f"{_fmt(zs_gate.get('gated_pm_rate'))} | {_fmt(zs_gate.get('cb_retention_rate'))} |",
        f"| {idm_label} | {_fmt(idm.get('auroc'))} | {_fmt(idm_det.get('f1'))} | "
        f"{_fmt(idm_gate.get('gated_pm_rate'))} | {_fmt(idm_gate.get('cb_retention_rate'))} |",
        f"| Δ (in-domain − zero-shot) | {_fmt(delta.get('auroc'))} | {_fmt(delta.get('f1'))} | "
        f"{_fmt(delta.get('pm_rate_reduction_delta'))} | {_fmt(delta.get('cb_retention_rate'))} |",
        "",
        "> Shopping PM-rate/CB-retention 来自封稿 `prism/mitigation_summary.json` aggressive gating。",
        "",
        "## 样本量",
        "",
        f"- {domain_title} test claims: {zs.get('n_claims')} (PM {zs.get('n_pm')}, clean {zs.get('n_clean')})",
        f"- 全量激活覆盖: {zs.get('n_all_claims_with_activation')} claims",
        "",
        "## 诚实边界",
        "",
        f"- effective_method: zero-shot=`{zs.get('effective_method')}`, in-domain=`{idm.get('effective_method')}`",
        f"- J-lens slot floor 触发率: zero-shot {zs.get('floor_rate', 0):.1%}, "
        f"in-domain {idm.get('floor_rate', 0):.1%}",
    ]
    if probe_only:
        lines.append(
            "- floor=100% → agr_slot_score = probe_logit + 常数；**AGR 融合项未参与**，表中 in-domain/零样本行为 probe-only"
        )
    lines += [
        f"- {cfg.domain} slot lexicon 与 shopping 词表不对齐",
        "- 零样本 AUROC<0.5 提示探针方向反转（见 audit 报告 sign-flip 检查）",
        "",
        "## Split 泄漏检查",
        "",
        f"- {json.dumps(zs.get('split_leak_check') or idm.get('split_leak_check'), ensure_ascii=False)}",
        "",
        "## 阈值来源",
        "",
        f"- Zero-shot τ_B: shopping prism 冻结 ({zs.get('tau_b')})",
        f"- In-domain τ_B: D_f F1 拟合 ({idm.get('tau_b')})",
    ]
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="telecom", choices=["telecom", "airline"])
    ap.add_argument("--mode", choices=["zeroshot", "indomain", "both"], default="both")
    args = ap.parse_args()

    cfg = get_tau3_domain(args.domain)
    write_tau3_agr_split_manifest(domain=args.domain)
    export_shopping_frozen_probe()

    results = run_eval(args.mode, domain=args.domain)
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    if "zeroshot" in results:
        write_json(cfg.zeroshot_summary, results["zeroshot"])
    if "indomain" in results:
        write_json(cfg.indomain_summary, results["indomain"])
    if args.mode == "both":
        _write_report(results, args.domain)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
