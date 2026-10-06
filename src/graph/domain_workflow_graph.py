"""AgentCore Platform v1.0"""

# HCR-C2-045 — DomainWorkflowGraph (inner BaseGraph)
#
# The inner graph of the Cat 2 two-layer architecture. It encapsulates the
# medical literature synthesis workflow:
#
#   START → literature_search → evidence_extract → clinical_synthesis
#         → report_generate → output_gate → END
#
# Built by HcrC2045GraphNode.get_subgraph() (graph.py). get_output() shapes the
# dict consumed by merge_output() there; the two are designed together.
#
# Config (forwarded from the outer graph under config["configurable"]):
#   timeout_s  per-search budget for the literature source
#   evidence   appraisal settings: max_references, strong_evidence_min,
#              moderate_evidence_min
# Both are injected into State by _extra_initial_state() as a JSON string, so
# the no-argument nodes can read them and State stays flat.

import json
from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus
from src.nodes.clinical_synthesis_node import ClinicalSynthesisNode
from src.nodes.evidence_extract_node import EvidenceExtractNode
from src.nodes.literature_search_node import LiteratureSearchNode
from src.nodes.output_gate_node import OutputGateNode
from src.nodes.report_generate_node import ReportGenerateNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for the literature synthesis pipeline.

    Inherits BaseGraph directly for a fully custom node topology.

    Pipeline (linear):
        START
          → literature_search   retrieve the literature under appraisal
          → evidence_extract    extract structured evidence
          → clinical_synthesis  appraise and synthesise
          → report_generate     assemble the evidence report
          → output_gate         the inner output gate
        END

    All nodes are FunctionNode subclasses admitting anonymous callers; the
    caller-facing trust decision is taken once, at the outer input gate.
    initialize / finalize are outer backbone concerns and are not registered.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "hcr_c2_045_literature_synthesis_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across the inner and outer graph."""
        return State

    # ── Config ────────────────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate the inner graph config before compilation.

        Nothing is mandatory: every forwarded setting has a documented default
        in the node that reads it, and an unusable value is surfaced there as a
        graceful error rather than as a start-up crash.
        """
        pass

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the inner state with the forwarded runtime settings.

        The framework does not forward outer state into a subgraph invocation,
        so this hook is the channel that carries configuration across the graph
        boundary. It is serialised to JSON to keep State flat and
        checkpoint-safe.
        """
        configurable = self.config.get("configurable") or {}
        settings: Dict[str, Any] = {}
        if "timeout_s" in configurable:
            settings["timeout_s"] = configurable["timeout_s"]
        evidence = configurable.get("evidence")
        if isinstance(evidence, dict) and evidence:
            settings["evidence"] = dict(evidence)
        if not settings:
            return {}
        return {"runtime_config": json.dumps(settings, ensure_ascii=False, default=str)}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register the five inner domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract. Do NOT
        register initialize or finalize; those are outer backbone concerns.
        Every key registered here is referenced in add_edges(), and all nodes
        are constructed with no arguments.
        """
        self._nodes["literature_search"] = LiteratureSearchNode()
        self._nodes["evidence_extract"] = EvidenceExtractNode()
        self._nodes["clinical_synthesis"] = ClinicalSynthesisNode()
        self._nodes["report_generate"] = ReportGenerateNode()
        self._nodes["output_gate"] = OutputGateNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear literature synthesis topology.

        Each step passes its partial-dict output into the shared State. The
        topology is intentionally linear: add_conditional_edges() is not used,
        so route() below is never reached at runtime.
        """
        self._sg.add_edge(START, "literature_search")
        self._sg.add_edge("literature_search", "evidence_extract")
        self._sg.add_edge("evidence_extract", "clinical_synthesis")
        self._sg.add_edge("clinical_synthesis", "report_generate")
        self._sg.add_edge("report_generate", "output_gate")
        self._sg.add_edge("output_gate", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: State) -> str:
        """Required by the BaseGraph contract; the topology above is linear.

        The annotation is this graph's OWN State, not the shared base state,
        and that is load-bearing rather than cosmetic: a routing callable's
        annotation IS its input schema, and any field the annotation does not
        declare is projected away before the callable sees it. Annotating a
        route with the base state makes every domain field read as absent, so
        the branch it guards can never be taken while unit tests that call the
        method directly still pass.

        Both branches return END so that no unexpected re-entry loop is
        possible if the topology ever gains a conditional edge.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return END

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: State) -> Dict[str, Any]:
        """Shape the dict returned to the outer graph as sub_result.

        Received by HcrC2045GraphNode.merge_output() in graph.py. The two are
        designed together:

            this get_output() emits: evidence_report, result, report_metadata,
                                     record_source, redaction_flags, status,
                                     trace_id, correlation_id, node_history
            merge_output() reads:    evidence_report, result, report_metadata,
                                     record_source, redaction_flags

        Both report fields are read straight from state so that whatever the
        output gate cleared stays cleared: reconstructing one from the other
        here would reinstate content the gate withheld.
        """
        return {
            "evidence_report": state.get("evidence_report"),
            "result": state.get("result"),
            "report_metadata": state.get("report_metadata", {}),
            "record_source": state.get("record_source", ""),
            "redaction_flags": state.get("redaction_flags", []),
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
