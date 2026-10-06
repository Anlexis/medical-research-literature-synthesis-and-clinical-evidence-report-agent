"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#
# HCR-C2-045 — ClinicalSynthesisNode
# Inner domain node 3: appraise the extracted evidence and synthesise findings.
# Registered as "clinical_synthesis" in DomainWorkflowGraph.
#
# Input:  state["evidence_items"], state["runtime_config"]
# Output: state["synthesis_findings"]
#
# Appraisal is deterministic and rule-based: the strength grade is a count of
# higher-quality study designs against the thresholds declared in
# config/config.yaml. Wire a real appraisal service in _compute_evidence_strength
# when one is available; nothing else in the pipeline needs to change.

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# Audit trail — free-function form only.
from shared.utils.audit_logger import emit_trace_event
from src.nodes.literature_search_node import load_runtime_settings
from src.services.source_disclosure import source_label

logger = logging.getLogger(__name__)

# Defaults used when config/config.yaml declares no evidence settings.
_DEFAULT_STRONG_MIN = 3
_DEFAULT_MODERATE_MIN = 1

# Bounds the declared thresholds are held to; a value outside them falls back
# to the default rather than being honoured.
_MIN_THRESHOLD = 1
_MAX_THRESHOLD = 100

# Findings rendered under "Key Findings".
_MAX_KEY_FINDINGS = 5


def resolve_thresholds(settings: Dict[str, Any]) -> Dict[str, int]:
    """Return the appraisal thresholds, falling back to the documented defaults."""
    evidence = settings.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "strong": _bounded_int(evidence.get("strong_evidence_min"), _DEFAULT_STRONG_MIN),
        "moderate": _bounded_int(evidence.get("moderate_evidence_min"), _DEFAULT_MODERATE_MIN),
    }


def _bounded_int(raw: Any, default: int) -> int:
    """Return *raw* as a whole number inside the supported bounds, else *default*."""
    if isinstance(raw, bool) or raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if not (_MIN_THRESHOLD <= value <= _MAX_THRESHOLD):
        return default
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


def _higher_quality_count(evidence_items: List[Dict[str, Any]]) -> int:
    """Count evidence graded as a systematic review, trial or prospective cohort."""
    return sum(
        1
        for item in evidence_items
        if "Level I" in str(item.get("evidence_level", "")) or "Level II" in str(item.get("evidence_level", ""))
    )


def _compute_evidence_strength(evidence_items: List[Dict[str, Any]], thresholds: Dict[str, int]) -> str:
    """Derive the overall strength grade: STRONG / MODERATE / WEAK / INSUFFICIENT."""
    count = _higher_quality_count(evidence_items)
    if count >= thresholds["strong"]:
        return "STRONG (Grade A)"
    if count >= thresholds["moderate"]:
        return "MODERATE (Grade B)"
    if evidence_items:
        return "WEAK (Grade C)"
    return "INSUFFICIENT"


def _build_key_findings(evidence_items: List[Dict[str, Any]]) -> List[str]:
    """Return the claims from the highest-quality, most recent evidence."""
    sorted_items = sorted(
        evidence_items,
        key=lambda x: (
            0
            if "Level I" in str(x.get("evidence_level", ""))
            else 1
            if "Level II" in str(x.get("evidence_level", ""))
            else 2,
            -int(x.get("year") or 0),
        ),
    )

    findings: List[str] = []
    for item in sorted_items[:_MAX_KEY_FINDINGS]:
        claim = str(item.get("claim", "")).strip()
        year = item.get("year", "")
        level = str(item.get("evidence_level", "")).split("—")[-1].strip()
        if claim:
            findings.append(f"{claim} ({level}, {year})")

    return findings or ["No key findings could be extracted from the evidence set"]


def _build_limitations(evidence_items: List[Dict[str, Any]]) -> List[str]:
    """Identify limitations of the evidence set from its composition."""
    limitations: List[str] = []
    total = len(evidence_items)

    if total < 3:
        limitations.append("Limited evidence base: fewer than 3 independent studies identified")

    observational = sum(
        1
        for item in evidence_items
        if "Level III" in str(item.get("evidence_level", "")) or "Observational" in str(item.get("evidence_level", ""))
    )
    if total and observational > total / 2:
        limitations.append("Majority of evidence from observational studies; controlled trial data limited")

    populations = {str(item.get("population", "")) for item in evidence_items}
    if len(populations) > 2:
        limitations.append("Heterogeneous study populations; generalisability may be limited")

    years = [int(item.get("year") or 0) for item in evidence_items if item.get("year")]
    if years and min(years) < 2015:
        limitations.append("Some evidence predates 2015; the current standard of care may differ")

    return limitations or ["No major limitations identified in the available evidence"]


def _build_recommendation(evidence_strength: str, query: str) -> str:
    """Return the recommendation matching the strength grade."""
    topic = query[:80]
    if evidence_strength.startswith("STRONG"):
        return (
            f"The evidence reviewed supports '{topic}'. Adoption may be considered on the "
            "strength of the higher-quality studies identified. Monitor for the adverse "
            "events described in the safety evidence."
        )
    if evidence_strength.startswith("MODERATE"):
        return (
            f"The evidence reviewed moderately supports '{topic}'. Clinical judgement is "
            "required; further controlled trials would be needed to confirm the benefit."
        )
    if evidence_strength.startswith("WEAK"):
        return (
            f"The evidence reviewed for '{topic}' is weak. Use with caution; further "
            "high-quality research is required before broad adoption."
        )
    return (
        f"The evidence reviewed is insufficient to support a recommendation for '{topic}'. "
        "The evidence search should be widened."
    )


class ClinicalSynthesisNode(FunctionNode):
    """Appraise the extracted evidence and synthesise structured findings.

    Inner domain node 3. Appraisal is deterministic and reproducible: the same
    evidence set and the same thresholds always produce the same grade.

    Input state keys:
        evidence_items: structured evidence from EvidenceExtractNode
        user_input:     validated request (the query is used for context)
        runtime_config: forwarded settings, JSON string

    Output state keys (partial dict):
        synthesis_findings: {summary, key_findings, evidence_strength,
                             limitations, recommendation, evidence_count}
        status:             AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:          (error path only) messages
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        evidence_items: Optional[List[Dict[str, Any]]] = state.get("evidence_items")

        if evidence_items is None:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["ClinicalSynthesisNode: evidence_items is missing — EvidenceExtractNode must run first"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("ClinicalSynthesisNode: evidence_items is missing — EvidenceExtractNode must run first"),
            }
        if not isinstance(evidence_items, list):
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["ClinicalSynthesisNode: evidence_items must be a list"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("ClinicalSynthesisNode: evidence_items must be a list"),
            }

        request = _request_from(state)
        query: str = str(request.get("query", "the research topic"))

        thresholds = resolve_thresholds(load_runtime_settings(state))
        evidence_count = len(evidence_items)

        evidence_strength = _compute_evidence_strength(evidence_items, thresholds)
        key_findings = _build_key_findings(evidence_items)
        limitations = _build_limitations(evidence_items)
        recommendation = _build_recommendation(evidence_strength, query)

        summary = (
            f"Synthesis of {evidence_count} evidence item(s) for the research query "
            f"'{query[:100]}'. Overall evidence strength: {evidence_strength}. "
            f"{len(key_findings)} key finding(s) identified. "
            f"{len(limitations)} limitation(s) noted."
        )

        # Say where the answer came from. This agent answers from a corpus defined inside
        # its own module; a reader seeing a citation has no way to tell that from a live query
        # against the system of record, and the review round rated that confusion its most
        # serious finding. Added before the gate below so it passes the same checks the answer
        # does.
        summary = summary + source_label(state)
        synthesis_findings: Dict[str, Any] = {
            "summary": summary,
            "key_findings": key_findings,
            "evidence_strength": evidence_strength,
            "limitations": limitations,
            "recommendation": recommendation,
            "evidence_count": evidence_count,
        }

        logger.info(
            "ClinicalSynthesisNode: appraisal complete — strength=%s findings=%d",
            evidence_strength,
            len(key_findings),
        )

        emit_trace_event(
            "clinical_synthesis_complete",
            {
                "evidence_count": evidence_count,
                "evidence_strength": evidence_strength,
                "key_findings_count": len(key_findings),
                "limitations_count": len(limitations),
                "higher_quality_count": _higher_quality_count(evidence_items),
                "thresholds": thresholds,
            },
            state,
        )

        return {
            "synthesis_findings": synthesis_findings,
            "status": AgentStatus.SUCCESS.value,
        }
