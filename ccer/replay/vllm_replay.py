"""vLLM generation wrapper for CCER replay."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from ccer.paths import DEFAULT_VLLM_BASE_URL
from ccer.replay.answer_utils import normalize_final_synthesis_text, prepare_final_synthesis_messages


def _answer_excerpt(text: str) -> str:
    m = re.search(r"<response>(.+?)</response>", text, re.DOTALL | re.I)
    return ((m.group(1).strip() if m else text.strip()))[:200]


def vllm_generate(
    messages: list[dict[str, str]],
    *,
    base_url: str = DEFAULT_VLLM_BASE_URL,
    model: str | None = None,
    max_tokens: int = 2048,
    seed: int = 42,
    temperature: float = 0.0,
    final_synthesis: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Generate from messages.

    final_synthesis=True (default for CCER replay): disable tool calls; stop at </response>.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    t0 = time.perf_counter()
    req_messages = prepare_final_synthesis_messages(messages) if final_synthesis else messages
    model_id = model
    if not model_id:
        models_url = base_url.rstrip("/") + "/models"
        with urllib.request.urlopen(
            urllib.request.Request(models_url, method="GET"), timeout=10
        ) as resp:
            data = json.loads(resp.read().decode())
        model_id = str((data.get("data") or [{}])[0].get("id") or "Qwen3-32B")
    body: dict[str, Any] = {
        "model": model_id,
        "messages": req_messages,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "extra_body": {"enable_thinking": False, "do_sample": False},
    }
    if final_synthesis:
        body["tool_choice"] = "none"
        body["stop"] = ["</response>"]
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            pass
        raise urllib.error.HTTPError(
            exc.url, exc.code, f"{exc.reason}; body={detail}", exc.headers, exc.fp
        ) from exc
    elapsed = time.perf_counter() - t0
    text = str(data["choices"][0]["message"]["content"] or "")
    if final_synthesis:
        text = normalize_final_synthesis_text(text)
    usage = data.get("usage") or {}
    meta = {
        "latency_sec": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "backend": "vllm_openai_api",
        "base_url": base_url,
        "final_synthesis": final_synthesis,
        "answer_excerpt": _answer_excerpt(text),
    }
    return text, meta


def check_vllm_available(base_url: str = DEFAULT_VLLM_BASE_URL) -> bool:
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False
