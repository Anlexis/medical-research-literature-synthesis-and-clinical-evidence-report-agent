"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#
# HCR-C2-045 — ReportGenerateNode
# Inner domain node 4: assemble the clinical evidence report in markdown.
# Registered as "report_generate" in DomainWorkflowGraph.
#
# Input:  state["synthesis_findings"], state["literature_items"],
#         state["evidence_items"], state["runtime_config"], the validated request
# Output: state["evidence_report"], state["result"], state["report_metadata"]
#
# Assembly is deterministic: the report is a rendering of the appraised
# evidence, not a generated narrative. Every value it renders reaches it
# already validated and sanitized, and the generation timestamp is the real
# time of the run — a report that reads as current while carrying a fixed date
# is a clinical-documentation defect, not a cosmetic one.

import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# Audit trail — free-function form only.
from shared.utils.audit_logger import emit_trace_event
from src.nodes.literature_search_node import load_runtime_settings
from src.schemas.validation import sanitize_render_text

logger = logging.getLogger(__name__)

# Report format version, recorded in the metadata for the audit trail.
_REPORT_VERSION = "1.1.0"

# References rendered when config/config.yaml declares no limit.
_DEFAULT_MAX_REFERENCES = 20
_MIN_MAX_REFERENCES = 1
_MAX_MAX_REFERENCES = 200

_SAMPLE_DATA_NOTICE = (
    "> **Sample data.** No literature records were supplied with this request, so the "
    "report was produced from the built-in sample corpus. It illustrates the report "
    "format and must not be read as a search of the published literature."
)


def resolve_max_references(settings: Dict[str, Any]) -> int:
    """Return the reference-list cap, falling back to the documented default."""
    evidence = settings.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    raw = evidence.get("max_references")
    if isinstance(raw, bool) or raw is None:
        return _DEFAULT_MAX_REFERENCES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_REFERENCES
    if not (_MIN_MAX_REFERENCES <= value <= _MAX_MAX_REFERENCES):
        return _DEFAULT_MAX_REFERENCES
    return value


def _request_from(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the validated request mapping, however this node was driven.

    Inside the inner graph the request arrives as user_input, because the outer
    graph hands the validated contract across the graph boundary there.
    """
    validated = state.get("validated_input")
    if isinstance(validated, dict) and validated:
        return validated
    user_input = state.get("user_input")
    if isinstance(user_input, dict) and user_input:
        return user_input
    return {}


def _format_key_findings(key_findings: List[str]) -> str:
    """Render the key findings as a numbered list."""
    if not key_findings:
        return "_No key findings identified._\n"
    return "\n".join(f"{i + 1}. {finding}" for i, finding in enumerate(key_findings)) + "\n"


def _format_limitations(limitations: List[str]) -> str:
    """Render the limitations as a bulleted list."""
    if not limitations:
        return "_No major limitations identified._\n"
    return "\n".join(f"- {limitation}" for limitation in limitations) + "\n"


def _format_references(literature_items: List[Dict[str, Any]], max_references: int) -> str:
    """Render the reference list.

    Every rendered value is passed through the render sanitizer again here.
    The caller's fields were already cleaned at the input gate; repeating it at
    the render step means the report cannot be restructured by a value that
    entered the pipeline through any other route.
    """
    if not literature_items:
        return "_No references available._\n"
    lines: List[str] = []
    for i, record in enumerate(literature_items[:max_references]):
        raw_authors = record.get("authors") or []
        authors = [sanitize_render_text(a, limit=80) for a in raw_authors[:3]]
        author_str = ", ".join(a for a in authors if a) or "Author not stated"
        if len(raw_authors) > 3:
            author_str += " et al."
        title = sanitize_render_text(record.get("title", ""), limit=300) or "Title not stated"
        source = sanitize_render_text(record.get("source", "unknown"), limit=40).upper()
        year = record.get("year", "n.d.")
        doi = sanitize_render_text(record.get("doi", ""), limit=120)
        line = f"{i + 1}. {author_str}. {title}. *{source}* ({year})."
        if doi:
            line += f" DOI: {doi}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _assemble_report(
    *,
    query: str,
    date_from: Optional[str],
    date_to: Optional[str],
    sources: List[str],
    synthesis_findings: Dict[str, Any],
    literature_items: List[Dict[str, Any]],
    record_source: str,
    generated_at: str,
    max_references: int,
) -> str:
    """Assemble the clinical evidence report in markdown."""
    summary = str(synthesis_findings.get("summary", ""))
    key_findings = list(synthesis_findings.get("key_findings", []))
    evidence_strength = str(synthesis_findings.get("evidence_strength", "N/A"))
    limitations = list(synthesis_findings.get("limitations", []))
    recommendation = str(synthesis_findings.get("recommendation", ""))
    evidence_count = int(synthesis_findings.get("evidence_count", 0))

    date_range = ""
    if date_from or date_to:
        date_range = f"**Date Range:** {date_from or 'earliest'} — {date_to or 'present'}  \n"

    sources_str = ", ".join(sanitize_render_text(s, limit=40).upper() for s in sources)
    provenance = "Caller-supplied literature records" if record_source == "caller" else "Built-in sample corpus"
    notice = "" if record_source == "caller" else f"{_SAMPLE_DATA_NOTICE}\n\n"

    return (
        f"# Clinical Evidence Report\n\n"
        f"{notice}"
        f"**Research Query:** {sanitize_render_text(query, limit=2000)}  \n"
        f"**Sources Searched:** {sources_str}  \n"
        f"{date_range}"
        f"**Evidence Base:** {provenance}  \n"
        f"**Evidence Count:** {evidence_count}  \n"
        f"**Overall Evidence Strength:** {evidence_strength}  \n"
        f"**Generated:** {generated_at}  \n"
        f"**Report Version:** {_REPORT_VERSION}  \n\n"
        f"---\n\n"
        f"## 1. Executive Summary\n\n"
        f"{summary}\n\n"
        f"---\n\n"
        f"## 2. Key Findings\n\n"
        f"{_format_key_findings(key_findings)}\n"
        f"---\n\n"
        f"## 3. Evidence Strength Assessment\n\n"
        f"**Overall Grade:** {evidence_strength}\n\n"
        f"Strength is graded by counting the higher-quality study designs "
        f"(systematic reviews, controlled trials and prospective cohorts) in the "
        f"evidence set against the thresholds this deployment declares. "
        f"{evidence_count} evidence item(s) were appraised from "
        f"{len(literature_items)} literature record(s).\n\n"
        f"---\n\n"
        f"## 4. Limitations\n\n"
        f"{_format_limitations(limitations)}\n"
        f"---\n\n"
        f"## 5. Clinical Recommendation\n\n"
        f"{recommendation}\n\n"
        f"---\n\n"
        f"## 6. References\n\n"
        f"{_format_references(literature_items, max_references)}\n"
        f"---\n\n"
        f"*This report summarises published research. It is decision support for a "
        f"qualified clinician and is not a diagnosis or a treatment decision.*\n"
    )


class ReportGenerateNode(FunctionNode):
    """Assemble the clinical evidence report in markdown.

    Inner domain node 4. Reads the appraised findings and the literature set and
    renders a structured report with a fixed section layout, which the output
    gate then verifies before the report can leave the graph.

    Input state keys:
        synthesis_findings: appraised findings from ClinicalSynthesisNode
        literature_items:   the records under appraisal
        record_source:      "caller" or "baseline"
        runtime_config:     forwarded settings, JSON string

    Output state keys (partial dict):
        evidence_report: the report in markdown
        result:          the same string, which the outer output gate scans
        report_metadata: {query, sources, generated_at, evidence_count,
                          report_version, record_source}
        status:          AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:       (error path only) messages
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        synthesis_findings: Optional[Dict[str, Any]] = state.get("synthesis_findings")
        if not synthesis_findings or not isinstance(synthesis_findings, dict):
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "ReportGenerateNode: synthesis_findings is missing — " "ClinicalSynthesisNode must run first"
                ],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("ReportGenerateNode: synthesis_findings is missing — ClinicalSynthesisNode must run first"),
            }

        literature_items: List[Dict[str, Any]] = state.get("literature_items") or []
        evidence_items: List[Dict[str, Any]] = state.get("evidence_items") or []
        record_source: str = str(state.get("record_source") or "baseline")

        request = _request_from(state)

        query: str = str(request.get("query", "research query"))
        date_from: Optional[str] = request.get("date_from")
        date_to: Optional[str] = request.get("date_to")
        sources: List[str] = list(request.get("sources") or ["clinical-trials", "j-stage", "pubmed"])
        evidence_count = int(synthesis_findings.get("evidence_count", len(evidence_items)))

        settings = load_runtime_settings(state)
        max_references = resolve_max_references(settings)
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        try:
            report: str = _assemble_report(
                query=query,
                date_from=date_from,
                date_to=date_to,
                sources=sources,
                synthesis_findings=synthesis_findings,
                literature_items=literature_items,
                record_source=record_source,
                generated_at=generated_at,
                max_references=max_references,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ReportGenerateNode: report assembly failed: %s", exc)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["ReportGenerateNode: the report could not be assembled"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("ReportGenerateNode: the report could not be assembled"),
            }

        report_metadata: Dict[str, Any] = {
            "query": query,
            "sources": sources,
            "generated_at": generated_at,
            "evidence_count": evidence_count,
            "report_version": _REPORT_VERSION,
            "record_source": record_source,
        }

        logger.info(
            "ReportGenerateNode: report assembled — chars=%d evidence_count=%d source=%s",
            len(report),
            evidence_count,
            record_source,
        )

        emit_trace_event(
            "report_generate_complete",
            {
                "report_length_chars": len(report),
                "evidence_count": evidence_count,
                "literature_count": len(literature_items),
                "record_source": record_source,
                "max_references": max_references,
                "report_version": _REPORT_VERSION,
            },
            state,
        )

        return {
            "evidence_report": report,
            "result": report,
            "report_metadata": report_metadata,
            "status": AgentStatus.SUCCESS.value,
        }
