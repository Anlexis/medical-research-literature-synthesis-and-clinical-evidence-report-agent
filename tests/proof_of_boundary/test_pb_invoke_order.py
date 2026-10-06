"""AgentCore Platform v1.0"""

# Backbone invoke-order verification.
#
# A full Graph().invoke() by a verified external caller must:
#   1. return SUCCESS for a valid request, and
#   2. run the backbone nodes in the mandated order:
#        InitializeNode -> PreProcessNode -> HcrC2045GraphNode ->
#        PostProcessNode -> FinalizeNode
#
# The caller trust level is VERIFIED_EXTERNAL rather than INTERNAL on purpose:
# an internal context admits every trust level and would mask a node whose own
# requirement is set too high. A real external caller arrives verified.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

# The outer graph class (also exported as Graph for backward-compat callers).
from src.graph.graph import Graph, HcrC2045GraphNode

# Domain backbone slot classes (always assertable — part of this repo).
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.post_process_node import PostProcessNode

# Framework backbone boundary nodes.
from framework.nodes.defaults.finalize_node import FinalizeNode
from framework.nodes.defaults.initialize_node import InitializeNode

# _MAIN_SLOT_NODE: class assigned to the `main` slot in MedicalResearchLiteratureSynthesisAgent.
_MAIN_SLOT_NODE = HcrC2045GraphNode

# A plain query string that passes the input gate and yields SUCCESS through
# the whole pipeline. A plain string is read as the `query` field, with the
# default date, source and result-cap values.
_VALID_PAYLOAD = "clinical trial outcomes for cardiovascular disease treatment"


class TestPbInvokeOrder:
    """Full-invoke backbone-order verification with a verified external caller."""

    @pytest.fixture(autouse=True)
    def _patch_emit(self, monkeypatch):
        """Silence the domain audit calls in every node module."""
        for mod_path in [
            "src.nodes.pre_process_node",
            "src.nodes.literature_search_node",
            "src.nodes.evidence_extract_node",
            "src.nodes.clinical_synthesis_node",
            "src.nodes.report_generate_node",
            "src.nodes.output_gate_node",
            "src.nodes.post_process_node",
        ]:
            monkeypatch.setattr(mod_path + ".emit_trace_event", lambda *a, **k: None)

    def test_backbone_invoke_success_verified_external(self):
        """Full invoke with VERIFIED_EXTERNAL — must succeed and populate output."""
        agent = Graph()
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)

        result = agent.invoke(user_input=_VALID_PAYLOAD, ctx=ctx)

        assert result["status"] == AgentStatus.SUCCESS.value, (
            f"Expected AgentStatus.SUCCESS; got {result['status']!r}. " f"error_log: {result.get('error_log')}"
        )
        assert result.get("output"), "result['output'] must be non-empty on SUCCESS"

    def test_backbone_node_history_order(self):
        """Full invoke must record backbone nodes in the mandated order."""
        agent = Graph()
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)

        result = agent.invoke(user_input=_VALID_PAYLOAD, ctx=ctx)

        assert result["status"] == AgentStatus.SUCCESS.value, (
            f"Pipeline must yield SUCCESS for the node_history assert to be reliable. "
            f"Got {result['status']!r}. error_log: {result.get('error_log')}"
        )

        history = result.get("node_history", [])
        assert len(history) == 5, (
            f"Expected exactly 5 backbone nodes in node_history, got {len(history)}: "
            f"{[getattr(n, '__name__', n) for n in history]}"
        )

        # The framework stores class name strings (not class objects) in node_history.
        def _name(entry) -> str:
            """Return the class name string regardless of whether entry is a str or type."""
            return entry if isinstance(entry, str) else getattr(entry, "__name__", str(entry))

        # Domain backbone slots are always assertable from this repo.
        assert _name(history[1]) == PreProcessNode.__name__, (
            f"node_history[1] must be '{PreProcessNode.__name__}' (input gate), " f"got '{_name(history[1])}'"
        )
        assert _name(history[2]) == _MAIN_SLOT_NODE.__name__, (
            f"node_history[2] must be '{_MAIN_SLOT_NODE.__name__}' (main slot), " f"got '{_name(history[2])}'"
        )
        assert _name(history[3]) == PostProcessNode.__name__, (
            f"node_history[3] must be '{PostProcessNode.__name__}' (output gate), " f"got '{_name(history[3])}'"
        )

        assert (
            _name(history[0]) == InitializeNode.__name__
        ), f"node_history[0] must be '{InitializeNode.__name__}', got '{_name(history[0])}'"
        assert (
            _name(history[4]) == FinalizeNode.__name__
        ), f"node_history[4] must be '{FinalizeNode.__name__}', got '{_name(history[4])}'"

    def test_invoke_returns_correlation_id(self):
        """invoke() must return a correlation_id or trace_id (framework contract)."""
        agent = Graph()
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        result = agent.invoke(user_input=_VALID_PAYLOAD, ctx=ctx)
        assert result.get("correlation_id") or result.get(
            "trace_id"
        ), "invoke() must return a correlation_id or trace_id"
