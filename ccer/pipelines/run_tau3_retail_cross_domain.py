"""Retail tau3: normalize → adjudication → zeroshot eval (test-only cohort)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.audit.tau3_bind_surprise_reval import run_domain as run_bind_surprise_reval
from ccer.audit.tau3_retail_split_manifest import build_retail_zeroshot_test_manifest
from ccer.io_utils import write_json
from ccer.mechanism.tau3.agr_slot_eval import run_zeroshot_eval
from ccer.mechanism.tau3.domain import get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import export_shopping_frozen_probe
from ccer.mechanism.tau3.quote_align import build_adjudication
from ccer.normalize.retail_qwen_1k import run_normalize_retail_qwen_1k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-normalize", action="store_true")
    ap.add_argument("--skip-adjudication", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    cfg = get_tau3_domain("retail")
    stats: dict = {}

    if not args.eval_only:
        if not args.skip_normalize:
            stats["normalize"] = run_normalize_retail_qwen_1k()
        if not args.skip_adjudication:
            stats["adjudication"] = build_adjudication(cfg)

    manifest = build_retail_zeroshot_test_manifest(cfg)
    write_json(cfg.split_manifest, manifest)
    stats["split"] = {
        "mode": manifest["mode"],
        "n_instances": manifest["n_instances_total"],
        "indomain_eligible": manifest["indomain_eligible"],
    }

    export_shopping_frozen_probe()
    zs = run_zeroshot_eval(cfg)
    zs["indomain_eligible"] = False
    zs["eval_mode"] = "zeroshot_test_only"
    write_json(cfg.zeroshot_summary, zs)

    bs = run_bind_surprise_reval(cfg)
    write_json(cfg.report.parent / "bind_surprise_reval.json", bs)

    report_lines = [
        "# Tau3 Retail Cross-Domain Report (zeroshot test-only)",
        "",
        f"> PM 轨迹仅 **31/1000 (3.1%)**，42 PM instances — **不作 in-domain 重训**，全量作 shopping 迁移测试集。",
        "",
        "## Zero-shot（shopping frozen probe → retail 全量 test）",
        "",
        f"| 分数 | 纯探针 p_pm | BindSurprise |",
        f"|------|-------------|--------------|",
        f"| AUROC | {zs.get('probe_auroc')} | {bs['original_split_test']['zeroshot']['bind_surprise_auroc']} |",
        f"| floor_rate | {zs.get('floor_rate')} | {bs['bind_surprise_health']['floor_rate']} |",
        f"| effective_method | probe-only | {bs['bind_surprise_health']['effective_method']} |",
        "",
        f"- n_claims: {zs.get('n_claims')} (PM {zs.get('n_pm')}, clean {zs.get('n_clean')})",
        f"- pack_source: base_114=114, boost_886=886",
        "",
        "## 与 Shopping 主域对比",
        "",
        "- Shopping BindSurprise AUROC ≈ 0.962（同域）",
        f"- Retail zero-shot probe AUROC = {zs.get('probe_auroc')}",
        f"- Retail zero-shot BindSurprise AUROC = {bs['original_split_test']['zeroshot']['bind_surprise_auroc']}",
        "",
        "## 门禁",
        "",
        "- `indomain_eligible=false` — PM 太少，禁止报告 in-domain AUROC",
        "- 仅作跨域迁移 **diagnostic**，不进 RQ2 主证据",
        "",
    ]
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    stats["zeroshot"] = {
        "probe_auroc": zs.get("auroc"),
        "bind_surprise_auroc": bs["original_split_test"]["zeroshot"]["bind_surprise_auroc"],
        "n_pm": zs.get("n_pm"),
    }
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
