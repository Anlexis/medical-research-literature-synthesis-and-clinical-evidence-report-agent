"""Boundary tests for the output gate, at full Graph().invoke() level.

The output invariant this template states is a complete, identifier-free
clinical summary carrying no credential-shaped value. There is no monetary
aggregate anywhere in the report format, so no rounding grid applies; what is
enforced instead is the invariant the report actually makes.

Containment is what these tests are really about. The graph's output builder
falls back to the raw report field even on an error status, so a gate that
merely flags a violation still ships the content it rejected — the assertions
below check the ENVELOPE the caller receives, not the gate's return value.
"""

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from src.graph.graph import Graph
from src.nodes import report_generate_node

_LEAKED_CREDENTIAL = "Bearer abcdefghijklmnop0123456789"
_LEAKED_IDENTIFIER = "MRN 4471902"


@pytest.fixture(autouse=True)
def _patch_emit(monkeypatch):
    """Silence the domain audit calls in every node module."""
    for module in [
        "src.nodes.pre_process_node",
        "src.nodes.literature_search_node",
        "src.nodes.evidence_extract_node",
        "src.nodes.clinical_synthesis_node",
        "src.nodes.report_generate_node",
        "src.nodes.output_gate_node",
        "src.nodes.post_process_node",
    ]:
        monkeypatch.setattr(module + ".emit_trace_event", lambda *a, **k: None)


@pytest.fixture()
def leaking_assembler(monkeypatch):
    """Force the assembled report to carry content the gate must withhold.

    A caller cannot inject this: the input gate refuses credential-shaped and
    identifier-bearing caller text. The injection point here stands in for a
    compromised or faulty upstream, which is the case the output gate exists
    for.
    """

    def _install(injected: str):
        original = report_generate_node._assemble_report

        def leaky(**kwargs):
            return original(**kwargs).replace(
                "## 1. Executive Summary",
                f"## 1. Executive Summary\n\n{injected}\n",
            )

        monkeypatch.setattr(report_generate_node, "_assemble_report", leaky)

    return _install


def _invoke(payload="containment probe"):
    agent = Graph()
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    return agent.invoke(payload, ctx=ctx)


def _envelope(result) -> str:
    """The whole response as the caller would see it, for leak assertions."""
    return json.dumps(result, default=str)


class TestOutputGateAtInvokeLevel:
    """A clean run is released; a violating one is contained."""

    def test_a_clean_pipeline_run_is_released(self):
        result = _invoke("clinical outcomes of early mobilisation")
        assert result["status"] == AgentStatus.SUCCESS.value, result.get("error_log")
        assert "## 5. Clinical Recommendation" in result["output"]

    @pytest.mark.parametrize("injected", [_LEAKED_CREDENTIAL, _LEAKED_IDENTIFIER])
    def test_a_violating_report_is_never_released_to_the_caller(self, leaking_assembler, injected):
        leaking_assembler(injected)
        result = _invoke()
        assert result["status"] != AgentStatus.SUCCESS
        assert not result.get("output")

    @pytest.mark.parametrize("injected", [_LEAKED_CREDENTIAL, _LEAKED_IDENTIFIER])
    def test_the_error_envelope_carries_no_released_text(self, leaking_assembler, injected):
        leaking_assembler(injected)
        envelope = _envelope(_invoke())
        assert injected not in envelope
        assert "## 5. Clinical Recommendation" not in envelope

    @pytest.mark.parametrize("injected", [_LEAKED_CREDENTIAL, _LEAKED_IDENTIFIER])
    def test_the_error_envelope_carries_no_traceback_and_no_source_paths(self, leaking_assembler, injected):
        """An escaping exception would put the formatted traceback in the log.

        Source paths and internal frames are as much a leak as the report text,
        and they are what the caller sees if the failure is simply re-raised.
        """
        leaking_assembler(injected)
        envelope = _envelope(_invoke())
        assert "Traceback" not in envelope
        assert 'File "' not in envelope
        assert "src/nodes/" not in envelope
        assert "site-packages" not in envelope

    def test_an_empty_report_is_contained_rather_than_returned_as_success(self, monkeypatch):
        monkeypatch.setattr(report_generate_node, "_assemble_report", lambda **kwargs: "   ")
        result = _invoke()
        assert result["status"] != AgentStatus.SUCCESS
        assert not result.get("output")

    def test_a_report_missing_a_required_section_is_contained(self, monkeypatch):
        original = report_generate_node._assemble_report
        monkeypatch.setattr(
            report_generate_node,
            "_assemble_report",
            lambda **kwargs: original(**kwargs).replace("## 5. Clinical Recommendation", "## 5. Other"),
        )
        result = _invoke()
        assert result["status"] != AgentStatus.SUCCESS
        assert not result.get("output")


class TestOutputInvariantScope:
    """The invariant is stated once and enforced for every representation."""

    def test_the_report_format_renders_no_monetary_aggregate(self):
        """Why no rounding grid applies to this template.

        The report carries counts, grades, study years and identifiers — never
        a monetary amount — so the grid the fleet applies to financial reports
        has nothing to act on here, and the identifier-and-credential invariant
        is enforced in its place.
        """
        report = _invoke("clinical outcomes of early mobilisation")["output"]
        for marker in ["JPY", "USD", "EUR", "¥", "$", "€", "£", "円"]:
            assert marker not in report
