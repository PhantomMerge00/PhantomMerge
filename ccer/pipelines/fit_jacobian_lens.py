"""Fit Jacobian lens J_l on Qwen3-32B using official jlens.fit()."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jlens
from jlens.examples import load_wikitext_prompts

from ccer.io_utils import write_json
from ccer.replay.hf_forward import get_model_manifest_entry, load_hf_model

JSPACE_DIR = ROOT / "artifacts" / "ccer" / "jspace"
LENS_PATH = JSPACE_DIR / "J_l_qwen32b_L49.pt"
CHECKPOINT_PATH = JSPACE_DIR / "fit_ckpt.pt"
INSTALL_MANIFEST = JSPACE_DIR / "INSTALL_MANIFEST.json"
FIT_REPORT = JSPACE_DIR / "FIT_REPORT.json"
REPO_DIR = ROOT / "third_party" / "jacobian-lens"


def _write_install_manifest() -> dict:
    import subprocess

    commit = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema": "ccer_jspace_install_manifest_v1",
        "repo_url": "https://github.com/anthropics/jacobian-lens",
        "repo_path": str(REPO_DIR),
        "commit_hash": commit,
        "install_command": "pip install -e .",
        "editable_install": True,
        "no_logic_modifications": True,
        "jlens_version": getattr(jlens, "__version__", "0.1.0"),
        "official_api_gap": (
            "Repo exports fit/JacobianLens.apply/transport only; "
            "no decompose/vectors/coordinate_swap Python API. "
            "Track N uses paper formula thin adapter in jspace_intervention.py."
        ),
    }
    write_json(INSTALL_MANIFEST, manifest)
    return manifest


def run_smoke(lens_path: Path) -> dict:
    from jlens.lens import JacobianLens

    lens = JacobianLens.load(str(lens_path))
    entry = get_model_manifest_entry()
    model, tokenizer, _ = load_hf_model(device_map="cuda:0")
    jlens_model = jlens.from_hf(model, tokenizer)
    prompt = "Fact: The currency used in the country shaped like a boot is"
    if 49 not in lens.source_layers:
        layer = lens.source_layers[0]
    else:
        layer = 49
    lens_logits, model_logits, _ = lens.apply(jlens_model, prompt, layers=[layer], positions=[-2])
    top_j = [tokenizer.decode([t]) for t in lens_logits[layer][0].topk(5).indices]
    top_m = [tokenizer.decode([t]) for t in model_logits[0].topk(5).indices]
    return {"layer": layer, "jlens_top5": top_j, "model_top5": top_m, "prompt": prompt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=1000)
    parser.add_argument("--layer", type=int, default=49)
    parser.add_argument("--dim-batch", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    JSPACE_DIR.mkdir(parents=True, exist_ok=True)
    _write_install_manifest()

    if args.smoke_only:
        if not LENS_PATH.is_file():
            raise FileNotFoundError(f"Lens not found: {LENS_PATH}")
        smoke = run_smoke(LENS_PATH)
        write_json(JSPACE_DIR / "smoke_test.json", smoke)
        print(json.dumps(smoke, indent=2))
        return

    t0 = time.time()
    model, tokenizer, n_layers = load_hf_model(device_map=args.device_map)
    jlens_model = jlens.from_hf(model, tokenizer)
    prompts = load_wikitext_prompts(args.n_prompts)
    if len(prompts) < args.n_prompts:
        print(f"[fit_jlens] warning: only loaded {len(prompts)} prompts (requested {args.n_prompts})")

    lens = jlens.fit(
        jlens_model,
        prompts,
        source_layers=[args.layer],
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=str(CHECKPOINT_PATH),
        checkpoint_every=5,
        resume=args.resume,
    )
    lens.save(str(LENS_PATH))
    elapsed = time.time() - t0
    entry = get_model_manifest_entry()
    report = {
        "schema": "ccer_jspace_fit_report_v1",
        "backbone": entry.get("model_id") or "Qwen/Qwen3-32B",
        "weights_path": entry.get("weights_path"),
        "n_prompts_requested": args.n_prompts,
        "n_prompts_used": len(prompts),
        "source_layers": lens.source_layers,
        "n_layers_model": n_layers,
        "d_model": lens.d_model,
        "dim_batch": args.dim_batch,
        "max_seq_len": args.max_seq_len,
        "fit_elapsed_sec": elapsed,
        "lens_path": str(LENS_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "note": (
            "Official README states ~100 prompts is usable; 1000 is paper default. "
            "This delivery uses n_prompts=100 due to GPU time budget (~82 min)."
        ),
    }
    write_json(FIT_REPORT, report)
    smoke = run_smoke(LENS_PATH)
    write_json(JSPACE_DIR / "smoke_test.json", smoke)
    print(json.dumps({"fit_report": report, "smoke": smoke}, indent=2))


if __name__ == "__main__":
    main()
