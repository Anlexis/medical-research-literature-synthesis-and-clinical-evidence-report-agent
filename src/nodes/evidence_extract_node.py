"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#
# HCR-C2-045 — EvidenceExtractNode
# Inner domain graph node 2: extract structured clinical evidence from literature records.
# Registered as "evidence_extract" in DomainWorkflowGraph.
#
# Input:  state["literature_items"] — list of literature records
# Output: state["evidence_items"] — list of structured evidence items
#
# Extraction is deterministic and rule-based: structured fields are read
# directly and the abstract is matched against a fixed outcome vocabulary, so
# the same record always yields the same evidence item. Wire a real extraction
# service in _extract_evidence_from_record when one is available.

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# Audit trail — free-function form only.
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# Evidence level assigned by source type.
_SOURCE_EVIDENCE_LEVEL: Dict[str, str] = {
    "pubmed": "Level I — Systematic Review / RCT",
    "j-stage": "Level II — Prospective Cohort",
    "clinical-trials": "Level I — RCT",
}

# Outcome keywords extracted deterministically from abstract text.
_OUTCOME_KEYWORDS: List[str] = [
    "efficacy",
    "safety",
    "adverse",
    "benefit",
    "mortality",
    "morbidity",
    "remission",
    "response rate",
    "hazard ratio",
    "NNT",
    "p<",
    "95% CI",
]


def _extract_evidence_from_record(
    record: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """Extract a structured evidence item from a single literature record.

    Deterministic extraction from the record's structured fields and its
    abstract text.
    """
    abstract: str = str(record.get("abstract", ""))
    source: str = str(record.get("source", "unknown"))
    year: int = int(record.get("year") or 0)
    title: str = str(record.get("title", ""))
    doi: str = str(record.get("doi", ""))

    # Extract outcome evidence from abstract (deterministic keyword match).
    found_outcomes = [kw for kw in _OUTCOME_KEYWORDS if kw.lower() in abstract.lower()]
    outcome_summary = ", ".join(found_outcomes[:3]) if found_outcomes else "clinical outcome reported"

    # Assign evidence level by source type.
    evidence_level = _SOURCE_EVIDENCE_LEVEL.get(source, "Level III — Observational")

    # Derive a study population estimate from abstract keywords.
    population = "adult patients"
    abstract_lower = abstract.lower()
    if "children" in abstract_lower or "paediatric" in abstract_lower:
        population = "paediatric patients"
    elif "elderly" in abstract_lower or "geriatric" in abstract_lower:
        population = "elderly patients (≥65 years)"

    # Extract key claim from the title (first 100 chars as the claim).
    claim = title[:100] if len(title) > 100 else title

    return {
        "claim": claim,
        "evidence_level": evidence_level,
        "source_id": doi or f"{source}-{index:03d}",
        "year": year,
        "population": population,
        "outcome": outcome_summary,
        "abstract_excerpt": abstract[:300],
    }


class EvidenceExtractNode(FunctionNode):
    """Extract structured clinical evidence from literature records.

    Inner domain graph node 2 (registered as "evidence_extract" in DomainWorkflowGraph).
    Performs deterministic rule-based extraction from the structured
    literature fields and the abstract text.

    Input state keys:
        literature_items: list of literature records from LiteratureSearchNode

    Output state keys (partial dict):
        evidence_items: list of structured evidence [{claim, evidence_level, source_id,
                        year, population, outcome, abstract_excerpt}]
        status:         AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:      (on error) list of error messages
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        literature_items: Optional[List[Dict[str, Any]]] = state.get("literature_items")

        if literature_items is None:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "EvidenceExtractNode: literature_items is missing — " "LiteratureSearchNode must run first"
                ],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("EvidenceExtractNode: literature_items is missing — LiteratureSearchNode must run first"),
            }

        if not isinstance(literature_items, list):
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    f"EvidenceExtractNode: literature_items must be a list, " f"got {type(literature_items).__name__}"
                ],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (f"EvidenceExtractNode: literature_items must be a list, got {type(literature_items).__name__}"),
            }

        if not literature_items:
            logger.warning("EvidenceExtractNode: no literature_items to extract evidence from")
            # Return empty evidence list — not an error (graceful degradation).
            emit_trace_event(
                "evidence_extract_complete",
                {"evidence_count": 0, "literature_count": 0, "empty": True},
                state,
            )
            return {
                "evidence_items": [],
                "status": AgentStatus.SUCCESS.value,
            }

        logger.info(
            "EvidenceExtractNode: extracting evidence from %d literature records",
            len(literature_items),
        )

        evidence_items: List[Dict[str, Any]] = []
        for idx, record in enumerate(literature_items):
            if not isinstance(record, dict):
                logger.warning("EvidenceExtractNode: skipping non-dict record at index %d", idx)
                continue
            try:
                item = _extract_evidence_from_record(record, idx)
                evidence_items.append(item)
            except Exception as exc:  # noqa: BLE001
                logger.warning("EvidenceExtractNode: failed to extract from record %d: %s", idx, exc)
                # Skip malformed records; do not abort the whole run.

        logger.info(
            "EvidenceExtractNode: extracted %d evidence items from %d records",
            len(evidence_items),
            len(literature_items),
        )

        # Count by evidence level for audit.
        level_counts: Dict[str, int] = {}
        for item in evidence_items:
            lvl = str(item.get("evidence_level", "unknown"))
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

        # Audit trail: record the extraction outcome.
        emit_trace_event(
            "evidence_extract_complete",
            {
                "evidence_count": len(evidence_items),
                "literature_count": len(literature_items),
                "evidence_level_distribution": level_counts,
            },
            state,
        )

        return {
            "evidence_items": evidence_items,
            "status": AgentStatus.SUCCESS.value,
        }
