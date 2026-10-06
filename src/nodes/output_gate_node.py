"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#
# HCR-C2-045 — OutputGateNode
# Inner domain node 5 (last inner node): the inner output gate.
# Registered as "output_gate" in DomainWorkflowGraph.
#
# It enforces this template's stated output invariant on the assembled report:
#
#   1. the report is present and non-empty
#   2. its metadata is populated
#   3. it carries every section the format promises
#   4. it carries no credential-shaped value
#   5. it carries no direct personal identifier
#
# The report renders population-level research findings. There is no monetary
# aggregate anywhere in the format, so a rounding grid does not apply; the
# invariant enforced instead is the one this report actually makes — a complete,
# identifier-free clinical summary.
#
# On a violation the gate returns an error AND CLEARS every output-bearing
# field. Returning an error without clearing is not containment: the graph's
# output builder falls back to the raw report field even on an error status, so
# a gate that only flags a violation still ships the content it rejected.

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# Audit trail — free-function form only.
from shared.utils.audit_logger import emit_trace_event
from src.schemas.validation import contains_credentials, find_identifiers

logger = logging.getLogger(__name__)

# Sections the report format promises. A clinical evidence report that reaches
# a reader without its recommendation or its summary is incomplete, not merely
# untidy, so a missing section blocks rather than warns.
_REQUIRED_SECTIONS: List[str] = [
    "## 1. Executive Summary",
    "## 2. Key Findings",
    "## 3. Evidence Strength Assessment",
    "## 4. Limitations",
    "## 5. Clinical Recommendation",
    "## 6. References",
]


def _missing_sections(content: str) -> List[str]:
    """Return the required sections absent from *content*."""
    return [section for section in _REQUIRED_SECTIONS if section not in content]


class OutputGateNode(FunctionNode):
    """Verify the assembled report before it leaves the inner graph.

    Inner domain node 5. The outer PostProcessNode applies the backbone's own
    output gate afterwards; this one is what keeps a rejected report from
    crossing the graph boundary in the first place.

    Input state keys:
        evidence_report: the assembled report from ReportGenerateNode
        report_metadata: report metadata; must be populated

    Output state keys (partial dict):
        evidence_report: the report, unchanged when it passes
        result:          cleared on a violation, alongside evidence_report
        report_metadata: cleared on a violation
        status:          AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:       (error path only) the reason, naming no content
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        evidence_report: str = state.get("evidence_report") or ""
        report_metadata: Any = state.get("report_metadata") or {}

        reason = self._violation(evidence_report, report_metadata)
        if reason is not None:
            logger.error("OutputGateNode: report withheld (%s)", reason)
            emit_trace_event(
                "output_gate_complete",
                {"passed": False, "reason": reason, "report_length": len(evidence_report)},
                state,
            )
            return self._withhold(reason)

        report_length = len(evidence_report)
        logger.info("OutputGateNode: report released — %d chars", report_length)

        emit_trace_event(
            "output_gate_complete",
            {
                "passed": True,
                "report_length": report_length,
                "metadata_keys": (sorted(report_metadata.keys()) if isinstance(report_metadata, dict) else []),
            },
            state,
        )

        return {
            "evidence_report": evidence_report,
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _violation(evidence_report: str, report_metadata: Any) -> Optional[str]:
        """Return the reason the report must be withheld, or None if it passes."""
        if not evidence_report or not evidence_report.strip():
            return "empty_report"
        if not report_metadata:
            return "missing_metadata"
        missing = _missing_sections(evidence_report)
        if missing:
            return f"incomplete_report:{len(missing)}_section(s)_missing"
        credential = contains_credentials(evidence_report)
        if credential:
            return f"credential_pattern:{credential}"
        identifiers = find_identifiers(evidence_report)
        if identifiers:
            return f"personal_identifier:{','.join(identifiers)}"
        return None

    @staticmethod
    def _withhold(reason: str) -> Dict[str, Any]:
        """Return the containment result: an error with every output field cleared.

        Every output-bearing field is cleared together. Clearing only the field
        the violation was found in leaves the others to be read as the answer.
        """
        return {
            "evidence_report": "",
            "result": "",
            "report_metadata": {},
            "status": AgentStatus.ERROR.value,
            "error_log": [f"OutputGateNode: the report was withheld ({reason})"],
        }
