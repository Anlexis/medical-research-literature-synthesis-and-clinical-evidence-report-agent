"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic BaseModel. LangGraph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption. Extend AgentState with agent-specific fields only. Do NOT add
# credentials, secrets, or Pydantic models.
#
# HCR-C2-045 — Medical Research Literature Synthesis & Clinical Evidence Report
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner domain
# workflow (BaseGraph). The fields below cover both layers.
#
# Security note: no API keys, tokens, or patient-identifying data in State.
# Credentials are accessed through the invocation context only.

from typing import Any, Dict, List, Optional

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Flat TypedDict for the medical literature synthesis pipeline.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Outer layer — set by PreProcessNode / HcrC2045GraphNode.merge_output
    # ------------------------------------------------------------------

    # Validated invocation payload produced by PreProcessNode.
    # Keys: {"query": str, "date_from": str|None, "date_to": str|None,
    #         "sources": list[str], "max_results": int,
    #         "literature_records": list[dict], "record_source": str}
    # Raw user_input is NOT persisted beyond PreProcessNode.
    validated_input: Optional[Dict[str, Any]]

    # Final clinical evidence report string (aliased from evidence_report).
    # Written by HcrC2045GraphNode.merge_output() from the inner graph output.
    # PostProcessNode reads state["result"] for the output gate.
    result: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer — domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # Runtime settings forwarded from config/config.yaml into the inner graph
    # (JSON string, so State stays flat and checkpoint-serializable).
    # Keys: {"timeout_s": float, "evidence": {...}}
    runtime_config: Optional[str]

    # LiteratureSearchNode output.
    # Literature records under appraisal. Each record:
    #   {"title": str, "authors": list[str], "abstract": str,
    #    "source": str, "year": int, "doi": str}
    literature_items: Optional[List[Dict[str, Any]]]

    # Provenance of literature_items: "caller" when the caller supplied the
    # records, "baseline" when the built-in sample corpus was used. Rendered in
    # the report so a reader can never mistake sample data for a real search.
    record_source: Optional[str]

    # EvidenceExtractNode output.
    # Structured evidence extracted from the literature records. Each item:
    #   {"claim": str, "evidence_level": str, "source_id": str,
    #    "year": int, "population": str, "outcome": str}
    evidence_items: Optional[List[Dict[str, Any]]]

    # ClinicalSynthesisNode output.
    # Keys: {"summary": str, "key_findings": list[str],
    #        "evidence_strength": str, "limitations": list[str],
    #        "recommendation": str, "evidence_count": int}
    synthesis_findings: Optional[Dict[str, Any]]

    # ReportGenerateNode output — full clinical evidence report in markdown.
    # Scanned by the output gate before it is surfaced to the caller.
    evidence_report: Optional[str]

    # ReportGenerateNode output — metadata for the audit trail.
    # Keys: {"query": str, "sources": list[str], "generated_at": str,
    #        "evidence_count": int, "report_version": str,
    #        "record_source": str}
    report_metadata: Optional[Dict[str, Any]]

    # Names of caller-supplied fields whose content was redacted before it was
    # rendered (field names only — never the redacted values).
    redaction_flags: Optional[List[str]]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history is inherited from AgentState; listed here for clarity.
