"""Local vLLM chat shim matching FacTool OpenAIChat.async_run contract."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Any

from ccer.mechanism.line_l_llm_common import llm_chat

ROOT = Path("${PHANTOM_MERGE_ROOT}")
_OPENAI_WRAPPER = ROOT / "third_party/factool/factool/utils/openai_wrapper.py"


def _load_openai_chat_class():
    spec = importlib.util.spec_from_file_location("factool_openai_wrapper", _OPENAI_WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.OpenAIChat


OpenAIChat = _load_openai_chat_class()


def pm_vllm_base_url() -> str:
    # Do NOT fall back to LINE_L_PLUS @8012 (8B only) — Qwen3-32B there returns HTTP 404.
    return os.environ.get("CCER_PM_VLLM_BASE_URL") or "http://127.0.0.1:8003/v1"


def pm_vllm_model() -> str:
    return os.environ.get("CCER_PM_LLM_MODEL") or "Qwen3-32B"


def resolve_pm_vllm_model_name(model_name: str | None) -> str:
    """FacTool clone uses placeholder names like local-vllm — never send those to vLLM."""
    if not model_name or model_name in ("local-vllm", "gpt-3.5-turbo", "gpt-4"):
        return pm_vllm_model()
    return model_name


def assert_pm_vllm_model_on_server() -> tuple[str, str]:
    """Raise RuntimeError if CCER_PM model id is not served at CCER_PM base URL."""
    import json
    import urllib.request

    base = pm_vllm_base_url().rstrip("/")
    want = pm_vllm_model()
    with urllib.request.urlopen(f"{base}/models", timeout=15) as resp:
        data = json.loads(resp.read().decode())
    ids = [str(m.get("id") or "") for m in (data.get("data") or [])]
    if want in ids:
        return base, want
    if ids:
        hint = (
            f"CCER_PM_LLM_MODEL={want} 不在 {base} 上（当前: {ids}）。"
            f" FacTool PM 请 export CCER_PM_VLLM_BASE_URL=http://127.0.0.1:8003/v1"
        )
        raise RuntimeError(hint)
    raise RuntimeError(f"no models at {base}/models")


class VllmFactoolChat(OpenAIChat):
    """Use vendor _type_check / _boolean_fix; generation via llm_chat (CCER_PM_* → vLLM)."""

    def __init__(self, model_name: str | None = None, **kwargs: Any):
        resolved = resolve_pm_vllm_model_name(model_name)
        super().__init__(model_name=resolved, **kwargs)
        self._base_url = pm_vllm_base_url()
        self._model = resolved

    async def async_run(self, messages_list, expected_type):
        sem = asyncio.Semaphore(int(os.environ.get("CCER_FACTOOL_ASYNC_WORKERS", "8")))

        async def _one(messages):
            async with sem:
                key = str(messages)[-2000:]
                text, _meta = await asyncio.to_thread(
                    llm_chat,
                    messages,
                    cache_key=key,
                    cache_path=Path(
                        os.environ.get(
                            "CCER_FACTOOL_LLM_CACHE",
                            str(ROOT / "results/rq2/p2_vendor/factool_kbqa/llm_cache.jsonl"),
                        )
                    ),
                    base_url=self._base_url,
                    model=self._model,
                    max_tokens=min(int(self.config.get("max_tokens") or 1024), 1024),
                )
                if not text:
                    return None
                fixed = self._boolean_fix(text)
                parsed = self._type_check(fixed, expected_type)
                if parsed is not None:
                    return parsed
                frag = self.extract_dict_from_string(fixed)
                if frag:
                    try:
                        out = ast.literal_eval(frag)
                        if isinstance(out, expected_type):
                            return out
                    except (SyntaxError, ValueError):
                        pass
                return None

        return await asyncio.gather(*[_one(m) for m in messages_list])
