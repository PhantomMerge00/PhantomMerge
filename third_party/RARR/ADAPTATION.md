# RARR integration adaptations (CCER)

**Pinned vendor**: `third_party/RARR` @ `51a1a10fe5bada837a368f98cb55288ac5168c9e` (see `PINNED_SHA`)

## What we use from vendor

| Vendor module | CCER usage |
|---------------|------------|
| `prompts/rarr_prompts.py` | **Yes** — QGEN, AGREEMENT_GATE, EDITOR prompt strings |
| `prompts/hallucination_prompts.py` | No (PM task uses claim-local gate, not passage hallucination suite) |
| `utils/agreement_gate.py` | **No** — OpenAI API + Google Search coupling; replaced by shim parsers |
| `utils/editor.py` | **No** — OpenAI `run_rarr_editor`; replaced by `rarr_vllm_shim` + `parse_editor_response` |
| `utils/question_generation.py` | **No** — inlined via `rarr_prompts` + vLLM |
| `utils/evidence_selection.py` | **No** — Obs_τ retriever replaces web evidence ranking |
| `utils/search.py` | **No** — Google Search not reproducible in CCER |

## Registered replacements (Integration, not Method)

1. **Retrieval**: Google Search → **Obs_τ** (`ccer/mechanism/obs_tau_retriever.py`, `retrieve_for_rarr`).
2. **LLM backend**: OpenAI API → **vLLM** Qwen3-8B @8012 (`ccer/mechanism/rarr_vllm_shim.py`, `completion_create`).
3. **Task shape**: passage-level RARR → **claim-local** single-span rewrite on Shopping agent trajectories.
4. **Detection gate**: vendor agreement gate AUROC baseline uses cache + `parse_agreement_gate` only (prompt-faithful); not full RARR loop as detection.

## Code entry

- Mitigation: `ccer/mechanism/baseline_rarr_official.py`
- Detection (gate only): `results/prism` matrix row `rarr_agreement_gate_any_open`

## External wording

- Allowed: **vendor-prompt RARR**, **Obs_τ-adapted RARR loop**
- Forbidden: **we reproduce RARR**, **official RARR end-to-end**, **as-published web RARR**

## Future utils wiring (P1 audit)

- `utils/agreement_gate.parse_api_response` is API-format-specific; CCER uses `parse_agreement_gate` in shim with same prompt text.
- Full `run_editor_sequential.py` pipeline not imported — would reintroduce search + OpenAI dependencies.
