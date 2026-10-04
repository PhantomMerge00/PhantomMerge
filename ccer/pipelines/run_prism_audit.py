"""PRISM Phase 1: claim-level audit (probe + J-lens slot alignment)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jlens
from jlens.lens import JacobianLens

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.line_k_claim_filter import load_frozen_dev_thresholds
from ccer.mechanism.bind_surprise import export_pr_curve
from ccer.mechanism.prism_audit import (
    render_case_studies_md,
    render_diagnosis_report,
    run_prism_audit,
    select_case_studies,
    summarize_prism_audit,
)
from ccer.replay.hf_forward import load_hf_model

PRISM_DIR = ROOT / "artifacts" / "ccer" / "prism"
LENS_PATH = ROOT / "artifacts" / "ccer" / "jspace" / "J_l_qwen32b_L49.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-path", default=str(LENS_PATH))
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--tau", type=float, default=None)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    PRISM_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = load_frozen_dev_thresholds()
    tau = float(args.tau if args.tau is not None else thresholds["line_a_probe_aggressive"])

    lens = JacobianLens.load(args.lens_path)
    model, tokenizer, _ = load_hf_model(device_map=args.device_map)
    jlens_model = jlens.from_hf(model, tokenizer)

    audit_rows, summary = run_prism_audit(
        lens=lens,
        jlens_model=jlens_model,
        tokenizer=tokenizer,
        tau=tau,
        topk=args.topk,
    )
    test_rows = [r for r in audit_rows if r.get("split") == "test"]
    test_summary = (
        summarize_prism_audit(
            test_rows,
            tau=tau,
            tau_b=summary.get("tau_b"),
            tau_fit=summary.get("tau_b_fit"),
            tau_recall_fit=summary.get("tau_b_recall_matched_fit"),
        )
        if test_rows
        else None
    )

    pr_curve = export_pr_curve(test_rows) if test_rows else None

    slim_rows = []
    for r in audit_rows:
        slim = {k: v for k, v in r.items() if k != "jlens_topk"}
        slim_rows.append(slim)

    write_jsonl(PRISM_DIR / "prism_audit.jsonl", slim_rows)
    write_json(PRISM_DIR / "prism_summary.json", {**summary, "test_split": test_summary, "pr_curve": pr_curve})
    if pr_curve:
        write_json(PRISM_DIR / "pr_curve_test.json", pr_curve)
    cases = select_case_studies(audit_rows)
    (PRISM_DIR / "audit_case_studies.md").write_text(
        render_case_studies_md(cases), encoding="utf-8"
    )
    (PRISM_DIR / "PRISM_DIAGNOSIS_REPORT.md").write_text(
        render_diagnosis_report(summary, test_summary=test_summary),
        encoding="utf-8",
    )
    print(json.dumps({**summary, "test_split": test_summary}, indent=2))


if __name__ == "__main__":
    main()
