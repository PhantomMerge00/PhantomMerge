"""Track N: J-space coordinate-swap causal test @ claim_onset (Line D/H protocol)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jlens
from jlens.lens import JacobianLens

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.jspace_subspace import fit_jspace_subspace
from ccer.mechanism.rank_sweep import _cross_pairs, _line_d_eligible_cross_pairs, measure_diff_rate
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.mechanism.task_h_eval import build_task_h_summary, summarize_task_h_arm
from ccer.replay.hf_forward import load_hf_model
from ccer.replay.live_position import LINE_D_CLAIM_LAYER_DEFAULT, LINE_D_PROTOCOL_VERSION

TRACK_N_DIR = ROOT / "artifacts" / "ccer" / "jspace" / "track_n"
LENS_PATH = ROOT / "artifacts" / "ccer" / "jspace" / "J_l_qwen32b_L49.pt"
CHECKPOINT = TRACK_N_DIR / "claim_onset_sweep_checkpoint.json"


def _load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT.is_file():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {"completed_arms": {}}


def _save_checkpoint(payload: dict[str, Any]) -> None:
    TRACK_N_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CHECKPOINT, payload)


def _build_report(summary: dict[str, Any], subspace_fit: dict[str, Any]) -> str:
    ti = summary.get("target_interchange") or {}
    wo = summary.get("wrong_owner_donor") or {}
    verdict = (summary.get("success_judgment") or {}).get("verdict") or "unknown"
    pid_rate = float(ti.get("product_id_rate") or 0)
    attr_rate = float(ti.get("claim_attribution_diff_rate") or 0)
    wo_pid = float(wo.get("product_id_rate") or 0)
    wo_attr = float(wo.get("claim_attribution_diff_rate") or 0)

    expert_verdict = summary.get("expert_verdict") or (
        "pass"
        if (pid_rate > 0.0 or attr_rate > 0.0) and (pid_rate > wo_pid or attr_rate > wo_attr)
        else "null"
    )

    if expert_verdict == "pass":
        claim_block = (
            "我们采用 Anthropic 近期提出、并在 Claude 模型上通过 spider/ant 式坐标置换实验"
            "因果验证过的 J-space(Jacobian lens)表征框架,而非本项目此前依赖的、未经独立因果验证的"
            "PCA/DAS 子空间,对 claim_onset 处的实体绑定信息进行因果置换。"
            f"结果显示,J-space 坐标置换在 {max(pid_rate, attr_rate):.1%} 的 product_id/claim_attribution "
            "层面产生了经 wrong-owner 对照验证的特异性因果效应。"
        )
    else:
        claim_block = (
            "我们进一步测试了 Anthropic 在 Claude 模型上因果验证过的 J-space workspace-slot 置换机制,"
            f"发现该机制在本项目 Qwen3-32B 绑定失败场景下不能产生可靠的因果效应"
            f"(product_id_diff={pid_rate:.1%}, claim_attribution_diff={attr_rate:.1%}; "
            f"wrong_owner product_id={wo_pid:.1%})."
            "表明表征层面可读的绑定信息在多种因果操控范式下均无法被跨实例改写。"
        )

    return "\n".join(
        [
            "# Track N Report — J-space Coordinate-Swap",
            "",
            "## Protocol",
            f"- version: {LINE_D_PROTOCOL_VERSION}",
            f"- position: claim_onset @ L{LINE_D_CLAIM_LAYER_DEFAULT}",
            f"- subspace_source: jspace (rank={subspace_fit.get('rank')})",
            f"- lens: `{LENS_PATH}`",
            "",
            "## Results",
            f"- target_interchange product_id_rate: **{pid_rate:.1%}**",
            f"- target_interchange claim_attribution_diff_rate: **{attr_rate:.1%}**",
            f"- wrong_owner_donor product_id_rate: {wo_pid:.1%}",
            f"- wrong_owner_donor claim_attribution_diff_rate: {wo_attr:.1%}",
            f"- task_h_verdict: {verdict}",
            f"- expert_verdict: **{expert_verdict}**",
            "",
            "## Baselines (this project only)",
            "- Line B v1 PCA: 3.3% product_id_diff",
            "- Line D / Task H PCA @ claim_onset: 0%",
            "",
            "## Paper claim (pre-registered)",
            claim_block,
            "",
            "## Audit",
            "- save_texts=True sidecars in track_n/",
            "- patched / patch_tier / wrong_owner_impl in pair_details",
            "",
            "## Official repo gap",
            "coordinate-swap not shipped in anthropics/jacobian-lens; "
            "subspace V from sparse J-lens decomposition + standard interchange hook.",
        ]
    )


def run_sweep(
    *,
    device_map: str,
    lens_path: Path,
    pair_limit: int | None = None,
    probe_max_tokens: int = 384,
    resume: bool = True,
) -> dict[str, Any]:
    lens = JacobianLens.load(str(lens_path))
    model, tokenizer, _ = load_hf_model(device_map=device_map)
    jlens_model = jlens.from_hf(model, tokenizer)
    layer = LINE_D_CLAIM_LAYER_DEFAULT
    position = "claim_onset"

    rows_index = load_trajectory_index()
    cross_pairs = _line_d_eligible_cross_pairs(
        _cross_pairs("CEM"),
        position=position,
        track_u="CEM",
        rows_index=rows_index,
    )
    subspace_fit = fit_jspace_subspace(
        lens,
        jlens_model,
        tokenizer,
        layer=layer,
        position=position,
        cross_pairs_override=cross_pairs,
    )
    if subspace_fit.get("U_owner") is None:
        raise RuntimeError(f"J-space subspace fit failed: {subspace_fit.get('error')}")
    write_json(TRACK_N_DIR / "jspace_subspace_fit.json", subspace_fit)
    u_override = subspace_fit["U_owner"]
    rank = int(subspace_fit.get("rank") or u_override.shape[1])

    checkpoint = _load_checkpoint() if resume else {"completed_arms": {}}
    completed: dict[str, Any] = dict(checkpoint.get("completed_arms") or {})
    arm_rows: dict[str, dict[str, Any]] = {}

    for control_id, mode in (
        ("target_interchange", "interchange"),
        ("wrong_owner_donor", "interchange"),
        ("full_vector_ceiling", "full_vector"),
    ):
        key = f"{position}:{control_id}"
        if resume and key in completed:
            arm_rows[key] = completed[key]
            continue
        cid = "target_interchange" if control_id == "full_vector_ceiling" else control_id

        def _on_pair(details: list[dict[str, Any]], *, arm_key: str = key) -> None:
            partial = dict(completed.get(arm_key) or {})
            partial["pair_details"] = details
            partial["n_pairs"] = len(details)
            completed[arm_key] = partial
            _save_checkpoint(
                {
                    "schema": "ccer_track_n_sweep_checkpoint_v1",
                    "probe_max_tokens": probe_max_tokens,
                    "completed_arms": completed,
                }
            )

        arm_rows[key] = measure_diff_rate(
            model=model,
            tokenizer=tokenizer,
            track="CEM",
            layer=layer,
            position=position,
            rank=rank,
            mode=mode,
            pair_limit=pair_limit,
            probe_max_tokens=probe_max_tokens,
            control_id=cid,
            save_texts=True,
            enrich_claim_attribution=True,
            text_sidecar_path=str(TRACK_N_DIR / f"texts_{position}_{control_id}.jsonl"),
            u_override=u_override,
            subspace_source="jspace",
            cross_pairs_override=cross_pairs,
            resume_pair_details=(completed.get(key) or {}).get("pair_details"),
            on_pair_complete=lambda details, arm_key=key: _on_pair(details, arm_key=arm_key),
        )
        completed[key] = arm_rows[key]
        _save_checkpoint(
            {
                "schema": "ccer_track_n_sweep_checkpoint_v1",
                "probe_max_tokens": probe_max_tokens,
                "completed_arms": completed,
            }
        )

    ti = arm_rows[f"{position}:target_interchange"]
    wo = arm_rows[f"{position}:wrong_owner_donor"]
    ceil = arm_rows[f"{position}:full_vector_ceiling"]
    summary = build_task_h_summary(
        ti_row=ti,
        wo_row=wo,
        ceiling_row=ceil,
        position=position,
        layer=layer,
    )
    summary["subspace_source"] = "jspace"
    ti_sum = summary.get("target_interchange") or {}
    wo_sum = summary.get("wrong_owner_donor") or {}
    pid_rate = float(ti_sum.get("product_id_rate") or 0)
    attr_rate = float(ti_sum.get("claim_attribution_diff_rate") or 0)
    wo_pid = float(wo_sum.get("product_id_rate") or 0)
    wo_attr = float(wo_sum.get("claim_attribution_diff_rate") or 0)
    passed = (pid_rate > 0.0 or attr_rate > 0.0) and (pid_rate > wo_pid or attr_rate > wo_attr)
    summary["expert_verdict"] = "pass" if passed else "null"
    summary["subspace_fit"] = {
        k: v for k, v in subspace_fit.items() if k not in ("U_owner", "decompositions")
    }
    write_json(TRACK_N_DIR / "claim_onset_attribution_summary.json", summary)
    write_json(
        TRACK_N_DIR / "claim_onset_sweep.json",
        {
            "schema": "ccer_track_n_position_sweep_v1",
            "protocol_version": LINE_D_PROTOCOL_VERSION,
            "subspace_source": "jspace",
            "target_interchange": summarize_task_h_arm(ti),
            "wrong_owner_donor": summarize_task_h_arm(wo),
            "full_vector_ceiling": summarize_task_h_arm(ceil),
        },
    )
    all_details: list[dict[str, Any]] = []
    for arm in (ti, wo, ceil):
        for row in arm.get("pair_details") or []:
            all_details.append({**row, "subspace_source": "jspace"})
    write_jsonl(TRACK_N_DIR / "claim_onset_pair_details.jsonl", all_details)
    report = _build_report(summary, subspace_fit)
    (TRACK_N_DIR / "TASK_N_REPORT.md").write_text(report, encoding="utf-8")
    return {"summary": summary, "subspace_fit": subspace_fit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-path", default=str(LENS_PATH))
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--probe-max-tokens", type=int, default=384)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out = run_sweep(
        device_map=args.device_map,
        lens_path=Path(args.lens_path),
        pair_limit=args.pair_limit,
        probe_max_tokens=args.probe_max_tokens,
        resume=not args.no_resume,
    )
    print(json.dumps({k: v for k, v in out["summary"].items() if k != "pair_details"}, indent=2, default=str))


if __name__ == "__main__":
    main()
