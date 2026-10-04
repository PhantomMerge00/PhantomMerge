"""FActScore vendor AtomicFactGenerator + FactScorer._get_score with Obs_τ retrieval and vLLM LM."""

from __future__ import annotations

import json
import os
import string
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("${PHANTOM_MERGE_ROOT}")
VENDOR = ROOT / "third_party/FActScore"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))

from ccer.mechanism.bind_surprise import try_auroc  # noqa: E402
from ccer.mechanism.line_l_llm_common import llm_chat  # noqa: E402
from ccer.vendor_baselines.obs_tau_common import (  # noqa: E402
    fair_obs_tau_corpus,
    iter_detection_rows_with_traj,
)
from ccer.vendor_baselines.p2_checkpoint import (  # noqa: E402
    append_score_line,
    eta_line,
    load_scored_keys,
    write_progress,
)


def _bm25_okapi():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError(
            "pip install -r ccer/vendor_baselines/requirements-p2.txt  # rank_bm25"
        ) from e
    return BM25Okapi
from factscore.atomic_facts import AtomicFactGenerator  # noqa: E402
from factscore.factscorer import FactScorer  # noqa: E402
from factscore.lm import LM  # noqa: E402


def _factscore_vllm_base_url() -> str:
    """Same stack as mitigation / Line-L+ (REGISTRY: Qwen3-8B @8012)."""
    return (
        os.environ.get("LINE_L_PLUS_VLLM_BASE_URL")
        or os.environ.get("LINE_L_VLLM_BASE_URL")
        or os.environ.get("CCER_MITIGATION_VLLM_BASE_URL")
        or "http://127.0.0.1:8012/v1"
    )


def _factscore_vllm_model() -> str:
    return (
        os.environ.get("LINE_L_PLUS_LLM_MODEL")
        or os.environ.get("CCER_MITIGATION_LLM_MODEL")
        or "Qwen3-8B"
    )


class VllmCompletionLM(LM):
    """Drop-in for factscore OpenAIModel.generate (completion-style prompts)."""

    def __init__(self, cache_file: Path, *, base_url: str | None = None, model: str | None = None):
        super().__init__(str(cache_file))
        self._base_url = base_url
        self._model = model

    def load_model(self) -> None:
        return None

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        text, meta = llm_chat(
            [{"role": "user", "content": prompt}],
            cache_key=prompt[-3000:],
            base_url=self._base_url,
            model=self._model,
            max_tokens=max_output_length,
        )
        return (text or ""), meta


class ObsTauCorpusDB:
    def __init__(self, passages: list[dict[str, str]]):
        self._passages = passages

    def get_text_from_title(self, topic: str) -> list[dict[str, str]]:
        return self._passages


class ObsTauRetrieval:
    """BM25 over Obs_τ chunks (same interface as factscore.retrieval.Retrieval.get_passages)."""

    def __init__(self, corpus: str, topic: str = "shopping_anchor"):
        chunks = []
        step = 1200
        text = corpus or ""
        for i in range(0, max(len(text), 1), step):
            part = text[i : i + step]
            if part.strip():
                chunks.append({"title": topic, "text": part})
        if not chunks:
            chunks = [{"title": topic, "text": ""}]
        self.db = ObsTauCorpusDB(chunks)
        self.cache: dict[str, list] = {}
        self.embed_cache: dict[str, BM25Okapi] = {}
        self.retrieval_type = "bm25"
        self.add_n = 0
        self.add_n_embed = 0

    def get_passages(self, topic: str, question: str, k: int):
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query
        if cache_key in self.cache:
            return self.cache[cache_key]
        passages = self.db.get_text_from_title(topic)
        bm25 = _bm25_okapi()([p["text"].split() for p in passages])
        scores = bm25.get_scores(retrieval_query.split())
        indices = np.argsort(-scores)[:k]
        self.cache[cache_key] = [passages[i] for i in indices]
        self.add_n += 1
        return self.cache[cache_key]

    def save_cache(self) -> None:
        return None


def _ensure_nltk_data() -> None:
    """NLTK 3.9+ needs punkt_tab for sent_tokenize (vendor atomic_facts)."""
    import nltk

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}/english/")
        except LookupError:
            nltk.download(pkg, quiet=True)


def _atoms_for_claim(af_gen: AtomicFactGenerator, claim: str) -> list[str]:
    pairs, _ = af_gen.get_atomic_facts_from_paragraph([claim.strip()])
    atoms: list[str] = []
    for _sent, alist in pairs:
        for a in alist or []:
            a = str(a).strip()
            if a:
                atoms.append(a)
    if not atoms and claim.strip():
        atoms = [claim.strip()]
    return atoms


def _unsupported_rate(fs: FactScorer, topic: str, generation: str, atoms: list[str], retrieval: ObsTauRetrieval) -> float:
    fs.retrieval = {"obs_tau": retrieval}
    decisions = fs._get_score(topic, generation, atoms, "obs_tau")
    if not decisions:
        return 0.5
    supported = sum(1 for d in decisions if d.get("is_supported"))
    return 1.0 - (supported / len(decisions))


def run_factscore_obs_tau(
    *,
    out_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or (ROOT / "results/rq2/p2_vendor/factscore")
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_nltk_data()
    cache_lm = out_dir / "factscore_vllm.pkl"
    demon_dir = VENDOR / "factscore/demos"

    base_url = _factscore_vllm_base_url()
    model = _factscore_vllm_model()

    lm = VllmCompletionLM(cache_lm, base_url=base_url, model=model)
    fs = FactScorer(model_name="retrieval+ChatGPT", openai_key="/dev/null")
    fs.lm = lm

    af_gen = AtomicFactGenerator(
        key_path="/dev/null",
        demon_dir=str(demon_dir),
        gpt3_cache_file=str(cache_lm),
    )
    af_gen.openai_lm = lm

    scores_path = out_dir / "scores.jsonl"
    progress_path = out_dir / "progress.json"
    done_keys = load_scored_keys(scores_path)
    records: list[dict[str, Any]] = []
    scores: list[float] = []
    if done_keys and scores_path.is_file():
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
                scores.append(float(json.loads(line).get("p_factscore", 0.5)))

    import time

    t0 = time.perf_counter()
    total_target = 1244 if limit is None else limit
    n_new = 0
    for row, traj, anchor_pid, _ in iter_detection_rows_with_traj():
        iak = str(row["instance_audit_key"])
        if iak in done_keys:
            continue
        if limit is not None and len(done_keys) + n_new >= limit:
            break
        claim = str(row.get("response_quote") or "")
        corpus = fair_obs_tau_corpus(traj, anchor_pid)
        topic = str(anchor_pid or row["trajectory_id"])
        atoms = _atoms_for_claim(af_gen, claim) if claim else []
        retrieval = ObsTauRetrieval(corpus, topic=topic)
        sc = _unsupported_rate(fs, topic, claim, atoms, retrieval) if claim else 0.5
        rec = {
            "instance_audit_key": iak,
            "split": row["split"],
            "y": row["y"],
            "p_factscore": sc,
            "n_atoms": len(atoms),
        }
        append_score_line(scores_path, rec)
        done_keys.add(iak)
        records.append(rec)
        scores.append(sc)
        n_new += 1
        done = len(done_keys)
        write_progress(
            progress_path,
            {
                "arm": "factscore",
                "done": done,
                "total": total_target,
                "n_new_this_run": n_new,
                "eta": eta_line(n_new, max(1, total_target - (done - n_new)), t0) if n_new else "resume",
                "llm_backend": {"base_url": base_url, "model": model},
            },
        )
        if n_new % 10 == 0:
            print(f"[factscore] {done}/{total_target} {eta_line(n_new, total_target - (done - n_new), t0)}", flush=True)
        if n_new % 50 == 0:
            lm.save_cache()

    lm.save_cache()
    records = []
    scores = []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records.append(rec)
        scores.append(float(rec.get("p_factscore", 0.5)))
    test_idx = [i for i, r in enumerate(records) if r["split"] == "test"]
    scored = [{"y_pm": records[i]["y"], "p_score": scores[i]} for i in test_idx]
    payload = {
        "schema": "factscore_obs_tau_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vendor": "third_party/FActScore AtomicFactGenerator + FactScorer._get_score",
        "adaptation": "enwiki replaced by Obs_τ BM25 chunks; OpenAI LM replaced by vLLM (LINE_L_VLLM_BASE_URL)",
        "llm_backend": {"base_url": base_url, "model": model},
        "test_auroc": try_auroc(scored, "p_score"),
        "n_test": len(test_idx),
        "n_pm_test": sum(records[i]["y"] for i in test_idx),
        "n_rows": len(records),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_progress(
        out_dir / "progress.json",
        {"arm": "factscore", "status": "complete", "done": len(records), "total": len(records)},
    )
    return payload
