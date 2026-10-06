"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#
# HCR-C2-045 — PostProcessNode
# Outer backbone post_process slot — the mandatory output gate.
#
# It reads state["result"] (the report the main node merged out of the inner
# graph) and applies the output invariant one last time, at the boundary the
# caller actually reads:
#
#   1. the report is present and non-empty
#   2. its metadata is populated
#   3. it carries no credential-shaped value
#   4. it carries no direct personal identifier
#
# Detection delegates to the framework's own credential detector rather than to
# a local pattern list. A local list narrower than the framework's would let a
# value through that the framework then raises on inside this same gate — and
# that raise returns a bare error partial which DISCARDS the clearing below, so
# a detector gap is a containment bypass, not just a missed finding.
#
# On a violation the gate returns an error AND clears every output-bearing
# field: the graph's output builder falls back to the raw result field even on
# an error status, so flagging without clearing still ships the content.

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# Audit trail — free-function form only.
from shared.utils.audit_logger import emit_trace_event
from src.schemas.validation import contains_credentials, find_identifiers
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

logger = logging.getLogger(__name__)


class PostProcessNode(FunctionNode):
    """Scan the clinical evidence report before the caller receives it.

    Outer backbone post_process slot.

    Input state keys:
        result:          the report merged out of the inner graph
        report_metadata: metadata; must be populated

    Output state keys (partial dict):
        result:          the report, unchanged when it passes; cleared otherwise
        evidence_report: cleared alongside result on a violation
        report_metadata: cleared on a violation
        status:          AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:       (error path only) the reason, naming no content
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Domain checks run inline in execute(). They are NOT installed as an
        # _extra_security_gate_output method: the framework auto-wraps that hook
        # and passes the next node's state as None, which raises on the full
        # invoke path.
        result: str = state.get("result") or ""
        report_metadata: Any = state.get("report_metadata") or {}

        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=result,
            domain="HCR MedicalResearchLiteratureSynthesisAgent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if _review and isinstance(result, str) and not self._violation(result + _review, report_metadata):
            result = result + _review

        reason = self._violation(result, report_metadata)
        if reason is not None:
            logger.error("PostProcessNode: report withheld (%s)", reason)
            emit_trace_event(
                "post_process_complete",
                {"passed": False, "reason": reason, "result_length": len(result)},
                state,
            )
            return self._withhold(reason)

        evidence_count: int = int(report_metadata.get("evidence_count", 0)) if isinstance(report_metadata, dict) else 0

        logger.info(
            "PostProcessNode: report released — %d chars, evidence_count=%d",
            len(result),
            evidence_count,
        )

        emit_trace_event(
            "post_process_complete",
            {
                "passed": True,
                "result_length": len(result),
                "evidence_count": evidence_count,
                "report_metadata_keys": (sorted(report_metadata.keys()) if isinstance(report_metadata, dict) else []),
            },
            state,
        )

        return {
            "result": result,
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _violation(result: str, report_metadata: Any) -> Optional[str]:
        """Return the reason the report must be withheld, or None if it passes.

        An empty result is a violation here, not a graceful degradation: the
        pipeline has no path that legitimately produces no report, so an empty
        one means an earlier step failed silently and the caller must be told.
        """
        if not result or not result.strip():
            return "empty_report"
        if not report_metadata:
            return "missing_metadata"
        credential = contains_credentials(result)
        if credential:
            return f"credential_pattern:{credential}"
        identifiers: List[str] = find_identifiers(result)
        if identifiers:
            return f"personal_identifier:{','.join(identifiers)}"
        return None

    @staticmethod
    def _withhold(reason: str) -> Dict[str, Any]:
        """Return the containment result: an error with every output field cleared."""
        return {
            "result": "",
            "evidence_report": "",
            "report_metadata": {},
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PostProcessNode: the report was withheld ({reason})"],
        }
