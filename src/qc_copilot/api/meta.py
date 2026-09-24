"""Read-only description of what the service knows and how it has been measured.

Everything here is derived from files on disk at request time: the policy documents and
generated catalog for the knowledge summary, and the committed evaluation reports for the
numbers. Nothing is hardcoded, so the page can never claim a metric that was not run.
"""

from __future__ import annotations

import json
from pathlib import Path

from qc_copilot import __version__
from qc_copilot.config import Settings
from qc_copilot.ingest.chunker import split_sections
from qc_copilot.ingest.loader import load_policy_documents
from qc_copilot.llm.providers import parse_chain
from qc_copilot.models import KnowledgeSummary, MetaResponse

REPORTS = {
    "subset_30": ("baseline_30.json", "after_30.json"),
    "full_60": ("baseline.json", "after.json"),
}

HEADLINE_KEYS = (
    "tool_call_success_rate",
    "citation_coverage",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "hallucination_rate",
)


def knowledge_summary(settings: Settings, chunk_count: int | None = None) -> KnowledgeSummary:
    policies = [
        {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "sections": [heading for heading, _ in split_sections(doc.text) if heading],
        }
        for doc in load_policy_documents(settings.policies_dir)
    ]
    catalog = settings.catalog_dir
    stores = json.loads((catalog / "dark_stores.json").read_text(encoding="utf-8"))
    products = json.loads((catalog / "products.json").read_text(encoding="utf-8"))
    return KnowledgeSummary(
        policies=policies,
        dark_stores=[
            {"store_id": s["store_id"], "name": s["name"], "city": s["city"]} for s in stores
        ],
        categories=sorted({p["category"] for p in products}),
        brands=sorted({p["brand"] for p in products}),
        product_count=len(products),
        chunk_count=chunk_count,
    )


def _headline(report: dict) -> dict[str, object]:
    kb = report.get("knowledge_base", {})
    ragas = kb.get("ragas", {})
    return {
        "label": report.get("label"),
        "mode": report.get("config", {}).get("mode"),
        "rows": report.get("dataset", {}).get("rows"),
        "tool_rows": report.get("dataset", {}).get("tool_rows"),
        "tool_call_success_rate": report.get("tool_rows", {}).get("tool_call_success_rate"),
        "citation_coverage": kb.get("citations", {}).get("citation_coverage"),
        "hallucination_rate": kb.get("hallucination", {}).get("hallucination_rate"),
        "faithfulness": ragas.get("faithfulness"),
        "answer_relevancy": ragas.get("answer_relevancy"),
        "context_precision": ragas.get("context_precision"),
        "context_recall": ragas.get("context_recall"),
        "p95_latency_ms": report.get("latency", {}).get("p95_ms"),
        "created_at": report.get("created_at"),
    }


def eval_summary(results_dir: Path) -> dict[str, object]:
    """Baseline and after headlines for every report pair that exists on disk."""
    summary: dict[str, object] = {}
    for name, (baseline_file, after_file) in REPORTS.items():
        pair: dict[str, object] = {}
        for role, filename in (("baseline", baseline_file), ("after", after_file)):
            path = results_dir / filename
            if path.exists():
                pair[role] = _headline(json.loads(path.read_text(encoding="utf-8")))
        if pair:
            summary[name] = pair
    return summary


def build_meta(settings: Settings, chunk_count: int | None, tools: list[str]) -> MetaResponse:
    keys = settings.api_keys()
    return MetaResponse(
        service="quickcommerce-copilot",
        version=__version__,
        description=(
            "Agentic RAG over a synthetic quick-commerce catalog and its operating policies. "
            "Retrieves policy and product passages, calls live inventory tools over MCP, "
            "reflects on its draft, and refuses when the evidence is missing."
        ),
        links={
            "docs": "/docs",
            "health": "/health",
            "ask": "POST /ask",
            "source": "https://github.com/Jharwal77/quickcommerce-copilot-ai",
            "readme": "https://github.com/Jharwal77/quickcommerce-copilot-ai#readme",
        },
        mode=settings.agent_mode,
        tools=tools,
        answer_chain=[ref.key for ref in parse_chain(settings.answer_chain, keys)],
        reflect_chain=[ref.key for ref in parse_chain(settings.reflect_chain, keys)],
        tool_call_cap=settings.tool_call_cap,
        knowledge=knowledge_summary(settings, chunk_count),
        eval=eval_summary(settings.eval_results_dir),
    )

