"""Tau3 cross-domain AGR(slot) domain configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ccer.paths import (
    AIRLINE_GOLD_EXPORT,
    NORMALIZED_AIRLINE_QWEN_1K,
    NORMALIZED_RETAIL_QWEN_1K,
    NORMALIZED_TELECOM,
    RETAIL_GOLD_EXPORT,
    TAU3_ADJUDICATION,
    TAU3_ADJUDICATION_AIRLINE,
    TAU3_ADJUDICATION_RETAIL,
    TAU3_AGR_SPLIT_MANIFEST_AIRLINE,
    TAU3_AGR_SPLIT_MANIFEST_RETAIL,
    TAU3_AGR_SPLIT_MANIFEST_TELECOM,
    TAU3_AIRLINE_ACTIVATIONS,
    TAU3_AIRLINE_DIR,
    TAU3_CROSS_DOMAIN_REPORT_AIRLINE,
    TAU3_CROSS_DOMAIN_REPORT_RETAIL,
    TAU3_CROSS_DOMAIN_REPORT,
    TAU3_DIR,
    TAU3_INDOMAIN_SUMMARY,
    TAU3_INDOMAIN_SUMMARY_AIRLINE,
    TAU3_INSTANCE_ACTIVATIONS,
    TAU3_RETAIL_ACTIVATIONS,
    TAU3_RETAIL_DIR,
    TAU3_SLOT_READOUT,
    TAU3_SLOT_READOUT_AIRLINE,
    TAU3_SLOT_READOUT_RETAIL,
    TAU3_ZEROSHOT_SUMMARY,
    TAU3_ZEROSHOT_SUMMARY_AIRLINE,
    TAU3_ZEROSHOT_SUMMARY_RETAIL,
    TELECOM_GOLD_EXPORT,
)

SUPPORTED_DOMAINS = frozenset({"telecom", "airline", "retail"})


@dataclass(frozen=True)
class Tau3DomainConfig:
    domain: str
    gold_export: Path
    normalized: Path
    adjudication: Path
    activations_dir: Path
    slot_readout: Path
    split_manifest: Path
    zeroshot_summary: Path
    indomain_summary: Path
    report: Path
    clean_iak_prefix: str
    quote_align_stats: Path
    quote_align_failures: Path
    line_a_dir: Path

    @property
    def adjudication_schema(self) -> str:
        return f"tau3_{self.domain}_adjudication_v1"


def get_tau3_domain(domain: str = "telecom") -> Tau3DomainConfig:
    d = str(domain).strip().lower()
    if d not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported tau3 domain: {domain!r} (supported: {sorted(SUPPORTED_DOMAINS)})")
    if d == "telecom":
        return Tau3DomainConfig(
            domain="telecom",
            gold_export=TELECOM_GOLD_EXPORT,
            normalized=NORMALIZED_TELECOM,
            adjudication=TAU3_ADJUDICATION,
            activations_dir=TAU3_INSTANCE_ACTIVATIONS,
            slot_readout=TAU3_SLOT_READOUT,
            split_manifest=TAU3_AGR_SPLIT_MANIFEST_TELECOM,
            zeroshot_summary=TAU3_ZEROSHOT_SUMMARY,
            indomain_summary=TAU3_INDOMAIN_SUMMARY,
            report=TAU3_CROSS_DOMAIN_REPORT,
            clean_iak_prefix="telecom_clean",
            quote_align_stats=TAU3_DIR / "quote_align_stats_telecom.json",
            quote_align_failures=TAU3_DIR / "quote_align_failures_telecom.jsonl",
            line_a_dir=TAU3_DIR / "line_a",
        )
    if d == "airline":
        return Tau3DomainConfig(
            domain="airline",
            gold_export=AIRLINE_GOLD_EXPORT,
            normalized=NORMALIZED_AIRLINE_QWEN_1K,
            adjudication=TAU3_ADJUDICATION_AIRLINE,
            activations_dir=TAU3_AIRLINE_ACTIVATIONS,
            slot_readout=TAU3_SLOT_READOUT_AIRLINE,
            split_manifest=TAU3_AGR_SPLIT_MANIFEST_AIRLINE,
            zeroshot_summary=TAU3_ZEROSHOT_SUMMARY_AIRLINE,
            indomain_summary=TAU3_INDOMAIN_SUMMARY_AIRLINE,
            report=TAU3_CROSS_DOMAIN_REPORT_AIRLINE,
            clean_iak_prefix="airline_clean",
            quote_align_stats=TAU3_AIRLINE_DIR / "quote_align_stats.json",
            quote_align_failures=TAU3_AIRLINE_DIR / "quote_align_failures.jsonl",
            line_a_dir=TAU3_AIRLINE_DIR / "line_a",
        )
    return Tau3DomainConfig(
        domain="retail",
        gold_export=RETAIL_GOLD_EXPORT,
        normalized=NORMALIZED_RETAIL_QWEN_1K,
        adjudication=TAU3_ADJUDICATION_RETAIL,
        activations_dir=TAU3_RETAIL_ACTIVATIONS,
        slot_readout=TAU3_SLOT_READOUT_RETAIL,
        split_manifest=TAU3_AGR_SPLIT_MANIFEST_RETAIL,
        zeroshot_summary=TAU3_ZEROSHOT_SUMMARY_RETAIL,
        indomain_summary=TAU3_RETAIL_DIR / "agr_slot_indomain_summary.json",
        report=TAU3_CROSS_DOMAIN_REPORT_RETAIL,
        clean_iak_prefix="retail_clean",
        quote_align_stats=TAU3_RETAIL_DIR / "quote_align_stats.json",
        quote_align_failures=TAU3_RETAIL_DIR / "quote_align_failures.jsonl",
        line_a_dir=TAU3_RETAIL_DIR / "line_a",
    )
