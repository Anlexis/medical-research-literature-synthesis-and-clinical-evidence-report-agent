"""A value declared in config/config.yaml must reach the code that reads it.

This is asserted through a full invoke rather than against the readers. The
failure this guards against is silent by construction: the settings mapping
forwarded into the inner graph can be empty, every reader then falls back to
its default, and the whole suite stays green while nothing declared has any
effect. Only an end-to-end run where CHANGING the declared value changes the
observable output can tell the two apart.
"""

import json
import pathlib

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from src.graph.graph import Graph, load_runtime_config

_CONFIG_PATH = pathlib.Path(__file__).parents[2] / "config" / "config.yaml"

_RECORDS = [
    {
        "title": f"Study number {index}",
        "abstract": "Controlled trial reporting efficacy and safety outcomes.",
        "authors": ["Author One"],
        "source": "pubmed",
        "year": 2024,
        "doi": f"10.1234/rec-{index:03d}",
    }
    for index in range(6)
]


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


def _report(config, records=_RECORDS):
    agent = Graph(config=config)
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    result = agent.invoke(
        json.dumps({"query": "configuration probe"}),
        ctx=ctx,
        input_context={"literature_records": records},
    )
    assert result["status"] == AgentStatus.SUCCESS.value, result.get("error_log")
    return result["output"]


class TestTheShippedConfigFile:
    """The file the repository ships is loadable and carries what it claims."""

    def test_the_shipped_file_parses_into_a_mapping(self):
        config = load_runtime_config(_CONFIG_PATH)
        assert isinstance(config, dict) and config

    def test_every_declared_key_has_a_reader(self):
        """A declaration nothing reads is a setting that silently does nothing."""
        config = load_runtime_config(_CONFIG_PATH)
        assert set(config) == {"max_retry", "timeout_s", "evidence"}
        assert set(config["evidence"]) == {
            "max_references",
            "strong_evidence_min",
            "moderate_evidence_min",
        }

    def test_a_missing_file_degrades_to_defaults_rather_than_failing_start_up(self):
        assert load_runtime_config(pathlib.Path("/nonexistent/config.yaml")) == {}

    def test_the_agent_defaults_to_the_shipped_file(self):
        """A directly-constructed agent must behave like a platform-constructed one."""
        assert Graph().config == load_runtime_config(_CONFIG_PATH)


class TestSettingsReachTheInnerGraph:
    """Changing a declared value changes the report."""

    def test_the_reference_cap_reaches_the_report(self):
        capped = _report({"evidence": {"max_references": 2}})
        uncapped = _report({"evidence": {"max_references": 20}})
        assert capped.count("DOI: 10.1234/rec-") == 2
        assert uncapped.count("DOI: 10.1234/rec-") == 6

    def test_the_strength_thresholds_reach_the_appraisal(self):
        strict = _report({"evidence": {"strong_evidence_min": 20}}, records=_RECORDS[:2])
        lenient = _report({"evidence": {"strong_evidence_min": 1}}, records=_RECORDS[:2])
        assert "**Overall Evidence Strength:** MODERATE (Grade B)" in strict
        assert "**Overall Evidence Strength:** STRONG (Grade A)" in lenient

    def test_the_search_budget_reaches_the_search_node(self):
        """A source answering later than the declared budget fails the run."""
        import time

        from src.services import service

        original = service.CallerSuppliedSource.search

        def slow(self, query, **kwargs):
            time.sleep(0.02)
            return original(self, query, **kwargs)

        service.CallerSuppliedSource.search = slow
        try:
            agent = Graph(config={"timeout_s": 0.001})
            result = agent.invoke(
                json.dumps({"query": "budget probe"}),
                ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
                input_context={"literature_records": _RECORDS},
            )
            assert result["status"] != AgentStatus.SUCCESS
            assert not result.get("output")

            generous = Graph(config={"timeout_s": 30})
            ok = generous.invoke(
                json.dumps({"query": "budget probe"}),
                ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
                input_context={"literature_records": _RECORDS},
            )
            assert ok["status"] == AgentStatus.SUCCESS.value, ok.get("error_log")
        finally:
            service.CallerSuppliedSource.search = original

    def test_an_empty_config_still_produces_a_report_on_the_documented_defaults(self):
        assert "## 5. Clinical Recommendation" in _report({})
