"""Track M: J-lens readout + dual dissociation at claim_onset L49."""
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
from ccer.mechanism.jspace_readout import run_track_m_readout
from ccer.replay.hf_forward import load_hf_model
from ccer.replay.live_position import LINE_D_CLAIM_LAYER_DEFAULT

TRACK_M_DIR = ROOT / "artifacts" / "ccer" / "jspace" / "track_m"
LENS_PATH = ROOT / "artifacts" / "ccer" / "jspace" / "J_l_qwen32b_L49.pt"


def _build_case_studies(summary: dict) -> str:
    details = summary.get("pair_details") or []
    successes = [
        r
        for r in details
        if (r.get("pm_scores") or {}).get("owner_axis") == "distractor"
        and (r.get("clean_scores") or {}).get("owner_axis") == "anchor"
    ]
    failures = [
        r
        for r in details
        if (r.get("pm_scores") or {}).get("owner_axis") != "distractor"
        or (r.get("clean_scores") or {}).get("owner_axis") != "anchor"
        or not (r.get("pm_scores") or {}).get("value_readable")
    ]
    lines = [
        "# Track M Case Studies",
        "",
        f"Eligible pairs: {summary.get('n_pairs', 0)}",
        f"PM distractor-owner rate: {summary.get('pm_distractor_owner_rate', 0):.1%}",
        f"Clean anchor-owner rate: {summary.get('clean_anchor_owner_rate', 0):.1%}",
        "",
        "## Success cases (PM→distractor, clean→anchor)",
        "",
    ]
    for r in successes[:5]:
        lines.append(f"### {r['pm_trajectory_id']}")
        lines.append(f"- slot: {r.get('slot_norm')} claim={r.get('claim_value')}")
        lines.append(f"- anchor_value={r.get('anchor_value')} distractor_value={r.get('distractor_value')}")
        pm_top = (r.get("pm_jlens_topk") or [])[:5]
        cl_top = (r.get("clean_jlens_topk") or [])[:5]
        lines.append(f"- PM J-lens top-5: {[t['token'] for t in pm_top]}")
        lines.append(f"- Clean J-lens top-5: {[t['token'] for t in cl_top]}")
        lines.append("")
    lines.append("## Failure / ambiguous cases (required)")
    lines.append("")
    for r in failures[:5]:
        lines.append(f"### {r['pm_trajectory_id']}")
        pm_s = r.get("pm_scores") or {}
        cl_s = r.get("clean_scores") or {}
        lines.append(
            f"- PM owner={pm_s.get('owner_axis')} readable={pm_s.get('value_readable')}; "
            f"clean owner={cl_s.get('owner_axis')}"
        )
        pm_top = (r.get("pm_jlens_topk") or [])[:5]
        lines.append(f"- PM J-lens top-5: {[t['token'] for t in pm_top]}")
        lines.append("")
    return "\n".join(lines)


def _build_report(summary: dict, install_path: Path) -> str:
    n = int(summary.get("n_pairs") or 0)
    pm_d = float(summary.get("pm_distractor_owner_rate") or 0)
    cl_a = float(summary.get("clean_anchor_owner_rate") or 0)
    return "\n".join(
        [
            "# Track M Report — J-lens Readout + Dual Dissociation",
            "",
            "## Setup",
            f"- Layer: L{LINE_D_CLAIM_LAYER_DEFAULT} @ claim_onset",
            f"- Lens: `{LENS_PATH}`",
            f"- Install manifest: `{install_path}`",
            f"- Eligible pairs: {n}",
            "",
            "## Aggregate",
            f"- PM owner→distractor rate: **{pm_d:.1%}** CI {summary.get('pm_distractor_owner_ci95')}",
            f"- Clean owner→anchor rate: **{cl_a:.1%}** CI {summary.get('clean_anchor_owner_ci95')}",
            f"- PM value-readable rate: {summary.get('pm_value_readable_rate', 0):.1%}",
            "",
            "## Interpretation",
            "Dual dissociation: value content readable (J-lens surfaces concrete slot values) "
            "while owner axis may point to distractor on PM vs anchor on clean.",
            "",
            "## Official repo gap",
            "anthropics/jacobian-lens provides fit/apply/transport only; readout uses transport+unembed.",
            "",
            "See `case_studies.md` for success and failure examples.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-path", default=str(LENS_PATH))
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    TRACK_M_DIR.mkdir(parents=True, exist_ok=True)

    lens = JacobianLens.load(args.lens_path)
    # Readout only needs unembed; CPU avoids contending with J_l fitting on GPU.
    model, tokenizer, _ = load_hf_model(device_map=args.device_map)
    jlens_model = jlens.from_hf(model, tokenizer)

    summary = run_track_m_readout(
        lens,
        jlens_model,
        tokenizer,
        layer=LINE_D_CLAIM_LAYER_DEFAULT,
        topk=args.topk,
    )
    write_json(TRACK_M_DIR / "readout_summary.json", summary)
    write_jsonl(TRACK_M_DIR / "readout_pair_details.jsonl", summary.get("pair_details") or [])
    case_md = _build_case_studies(summary)
    (TRACK_M_DIR / "case_studies.md").write_text(case_md, encoding="utf-8")
    report = _build_report(summary, ROOT / "artifacts" / "ccer" / "jspace" / "INSTALL_MANIFEST.json")
    (TRACK_M_DIR / "TASK_M_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "pair_details"}, indent=2))


if __name__ == "__main__":
    main()
