"""AgentCore Platform v1.0"""

# HCR-C2-045 — outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Outer backbone (fixed — add_edges() is NOT overridden):
#
#   START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                           ↓ (retry, bounded by max_retry)
#                                        pre_process
#
# The `main` slot is a GraphNode subclass (HcrC2045GraphNode) that delegates the
# whole domain workflow to DomainWorkflowGraph (inner BaseGraph), so domain
# complexity is encapsulated and the backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 ← outer graph (this file)
#   src/graph/domain_workflow_graph.py ← inner graph (multi-step topology)
#
# Runtime settings live in config/config.yaml; config/agent.yaml is the static
# registration manifest and carries no runtime values. The agent is constructed
# with that file's contents, and the same mapping is forwarded to the inner
# graph — see load_runtime_config() and HcrC2045GraphNode._parent_config().

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

import yaml

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.trust_level import TrustLevel
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

if TYPE_CHECKING:  # the inner graph imports this package at runtime
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

logger = logging.getLogger(__name__)

# src/graph/graph.py -> parents[2] = repository root.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_runtime_config(path: Path = _RUNTIME_CONFIG_PATH) -> Dict[str, Any]:
    """Load config/config.yaml — the agent's runtime parameters.

    This is the mapping the platform passes to the graph constructor. The
    standalone entry point loads it the same way, so a value declared once
    takes effect on every path; a config file read on only one of them is a
    setting that silently does nothing on the other.

    A missing or unreadable file degrades to {} — the framework and the
    appraisal nodes all carry documented defaults — rather than failing
    start-up on a file a deployment may legitimately not have mounted.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


class HcrC2045GraphNode(GraphNode):
    """GraphNode assigned to the `main` slot; wraps DomainWorkflowGraph.

    Called by the backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()      — build DomainWorkflowGraph with the forwarded config
      extract_input()     — hand the validated request to the inner graph
      merge_output()      — map inner results into the outer state delta
      on_subgraph_error() — convert an inner failure into a cleared envelope
    """

    # S-1 declared on the wrapper too: the CI gate only AST-scans FunctionNode
    # subclasses, so a GraphNode main slot passes the pipeline without one and is
    # flagged at review. Same level the nodes in this repo already declare.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    # "handle": route inner failures through on_subgraph_error() below instead
    # of re-raising them. Re-raising is not the safer option here: the framework
    # turns an escaping exception into an error result whose log carries the
    # formatted traceback — source paths and internal frames included — and it
    # writes no output fields, so whatever the main slot had already merged
    # stays readable in the response. Handling the failure lets this node clear
    # the output fields itself and return a reason the caller can act on.
    error_strategy: ClassVar[str] = "handle"

    # False: human-review interrupts are handled inside the inner graph only.
    propagate_hitl: ClassVar[bool] = False

    def __init__(self) -> None:
        # Nodes take no constructor arguments, so the owning graph assigns the
        # runtime config after construction.
        super().__init__()
        self.parent_config: Dict[str, Any] = {}

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Build the inner domain workflow graph with the forwarded config.

        Imported inside the method to avoid a circular import at module load.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def _parent_config(self) -> Dict[str, Any]:
        """Forward runtime settings to the inner graph under "configurable".

        Carries the per-search budget and the appraisal settings from
        config/config.yaml. Returning an empty mapping here is the failure mode
        that makes every declared setting dead while the suite stays green, so
        the keys are copied explicitly and the inner graph asserts nothing about
        their presence.
        """
        runtime = self.parent_config if isinstance(self.parent_config, dict) else {}
        configurable: Dict[str, Any] = {}
        if "timeout_s" in runtime:
            configurable["timeout_s"] = runtime["timeout_s"]
        evidence = runtime.get("evidence")
        if isinstance(evidence, dict):
            configurable["evidence"] = dict(evidence)
        return {"configurable": configurable}

    def extract_input(self, state: AgentState) -> Dict[str, Any]:
        """Hand the validated request to the inner graph as its user_input.

        PreProcessNode writes the whole validated contract — query, filters and
        any caller-supplied literature records — to validated_input. The inner
        graph starts from a fresh state, so passing the mapping itself is what
        carries the caller's data across the graph boundary: the framework does
        not forward outer state fields into a subgraph invocation.

        Falls back to parsing user_input when this node is driven directly
        without a preceding pre_process step.
        """
        import json as _json

        validated = state.get("validated_input")
        if isinstance(validated, dict) and validated:
            return validated

        user_input = state.get("user_input", "")
        if isinstance(user_input, dict):
            return user_input
        try:
            parsed = _json.loads(user_input) if user_input else {}
        except (ValueError, TypeError):
            return {"query": str(user_input)}
        return parsed if isinstance(parsed, dict) else {"query": str(parsed)}

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's output into the outer state delta.

        Returns ONLY changed keys, and mirrors the inner graph's own
        containment decision: when the inner output gate withheld the report,
        every output-bearing key it cleared arrives cleared here too. Reading
        the raw report field independently of the gated one would reinstate the
        content the gate had just withheld.
        """
        return {
            "evidence_report": sub_result.get("evidence_report") or "",
            "result": sub_result.get("result") or "",
            "report_metadata": sub_result.get("report_metadata") or {},
            "record_source": sub_result.get("record_source") or "",
            "redaction_flags": sub_result.get("redaction_flags") or [],
        }

    def on_subgraph_error(self, state: AgentState, error: Exception) -> Dict[str, Any]:
        """Contain an inner-graph failure instead of letting it escape.

        Clears every output-bearing field and reports a fixed reason. The
        exception is logged for the operator; it is never placed in the state,
        because its text can carry the very content the inner gate withheld as
        well as internal source paths.
        """
        logger.error("Inner workflow failed: %s", type(error).__name__, exc_info=error)
        return {
            "evidence_report": "",
            "result": "",
            "report_metadata": {},
            "status": AgentStatus.ERROR.value,
            "error_log": ["The literature synthesis workflow could not complete; no report was produced."],
        }


class MedicalResearchLiteratureSynthesisAgent(AgentBaseGraph):
    """Outer graph for HCR-C2-045 (Cat 2).

    Inherits AgentBaseGraph directly. Domain logic is encapsulated in
    HcrC2045GraphNode (main slot), which delegates to DomainWorkflowGraph.

    Backbone (fixed):
        START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() is the only override:
      - super().register_nodes() fills initialize and finalize
      - pre_process:  PreProcessNode (input validation, VERIFIED_EXTERNAL)
      - main:         HcrC2045GraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate, ANONYMOUS)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Default to the repository's runtime config so a directly-constructed
        # agent behaves like a platform-constructed one. The framework reads
        # max_retry / hitl / memory_enabled straight off this mapping.
        super().__init__(config if config is not None else load_runtime_config())

    @property
    def name(self) -> str:
        """Agent identifier used at registration.

        Matches the name field in config/agent.yaml and the entry point's
        secret-provisioning call.
        """
        return "MedicalResearchLiteratureSynthesisAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all five backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default initialize node (schema version, session id, trust
        level) and finalize node (response metadata, total time).
        """
        super().register_nodes()  # fills: initialize, finalize

        main_node = HcrC2045GraphNode()
        main_node.parent_config = self.config

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = main_node
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Module-level alias so callers can import `from src.graph.graph import Graph`.
Graph = MedicalResearchLiteratureSynthesisAgent
