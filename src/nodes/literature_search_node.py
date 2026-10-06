"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#
# HCR-C2-045 — LiteratureSearchNode
# Inner domain node 1: assemble the literature set the appraisal runs on.
# Registered as "literature_search" in DomainWorkflowGraph.
#
# Input:  state["user_input"]  — the validated request handed across the graph
#                                boundary by the outer graph's extract_input()
#         state["runtime_config"] — forwarded settings (JSON string)
# Output: state["literature_items"], state["record_source"]
#
# Where the records come from:
#   caller    the caller supplied literature_records on the structured channel;
#             they were validated and sanitized at the input gate
#   baseline  no records were supplied, so the sample corpus is used and the
#             report says so — sample data is never presented as a real search

import json
import logging
import time
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# Audit trail — free-function form only.
from shared.utils.audit_logger import emit_trace_event
from src.services.service import (
    BaselineLiteratureSource,
    CallerSuppliedSource,
    LiteratureSource,
)

logger = logging.getLogger(__name__)

# Per-search budget used when config/config.yaml declares none.
_DEFAULT_TIMEOUT_S = 30.0

# Bounds the declared budget is held to. A deployment cannot disable the budget
# by declaring zero, nor make it meaningless by declaring an hour.
_MIN_TIMEOUT_S = 0.001
_MAX_TIMEOUT_S = 300.0

_ALL_SOURCES: List[str] = ["clinical-trials", "j-stage", "pubmed"]


def load_runtime_settings(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the runtime settings the inner graph seeded into state.

    An absent or unparseable value degrades to {}; every reader of this mapping
    carries its own documented default.
    """
    raw = state.get("runtime_config")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def resolve_timeout_s(settings: Dict[str, Any]) -> float:
    """Return the per-search budget in seconds.

    A declared value outside the supported bounds falls back to the default
    rather than being honoured, so a mis-declared budget cannot disable the
    check it exists to perform.
    """
    raw = settings.get("timeout_s")
    if raw is None:
        return _DEFAULT_TIMEOUT_S
    try:
        budget = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S
    if not (_MIN_TIMEOUT_S <= budget <= _MAX_TIMEOUT_S):
        logger.warning(
            "LiteratureSearchNode: declared timeout_s is outside the supported " "range; using the %.1fs default",
            _DEFAULT_TIMEOUT_S,
        )
        return _DEFAULT_TIMEOUT_S
    return budget


class LiteratureSearchNode(FunctionNode):
    """Assemble the literature records the clinical appraisal runs on.

    Inner domain node 1. Uses the caller's own records when they supplied any,
    and the sample corpus otherwise. The chosen provenance travels with the
    results so every downstream step — and the report itself — can state it.

    Input state keys:
        user_input:     validated request (query, date_from, date_to, sources,
                        max_results, literature_records)
        runtime_config: forwarded settings, JSON string

    Output state keys (partial dict):
        literature_items: the records under appraisal
        record_source:    "caller" or "baseline"
        result:           progress marker
        status:           AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:        (error path only) messages naming the failed field
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        request = self._request_from(state)
        if request is None:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["LiteratureSearchNode: the validated request is missing — " "the input gate did not run"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("LiteratureSearchNode: the validated request is missing — the input gate did not run"),
            }

        query = str(request.get("query", "")).strip()
        if not query:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["LiteratureSearchNode: 'query' is missing from the request"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("LiteratureSearchNode: 'query' is missing from the request"),
            }

        sources: List[str] = list(request.get("sources") or _ALL_SOURCES)
        max_results = int(request.get("max_results") or 20)
        date_from: Optional[str] = request.get("date_from")
        date_to: Optional[str] = request.get("date_to")

        settings = load_runtime_settings(state)
        budget_s = resolve_timeout_s(settings)
        source = self._source_for(request)

        started = time.monotonic()
        try:
            literature_items = source.search(
                query,
                sources=sources,
                date_from=date_from,
                date_to=date_to,
                max_results=max_results,
            )
        except Exception as exc:  # noqa: BLE001 — any source failure is the same outcome
            logger.error("LiteratureSearchNode: the literature source failed: %s", exc)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["LiteratureSearchNode: the literature source could not be read"],
                "literature_items": [],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("LiteratureSearchNode: the literature source could not be read"),
            }
        elapsed_s = time.monotonic() - started

        # A source that answers late is reported, never silently accepted: a
        # partial or stale result set would still render as a complete report.
        if elapsed_s > budget_s:
            logger.error(
                "LiteratureSearchNode: search exceeded the %.3fs budget (%.3fs)",
                budget_s,
                elapsed_s,
            )
            emit_trace_event(
                "literature_search_complete",
                {
                    "record_count": 0,
                    "sources": sources,
                    "record_source": source.name,
                    "within_budget": False,
                    "budget_s": budget_s,
                },
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"LiteratureSearchNode: the literature source exceeded its " f"{budget_s:g}s budget"],
                "literature_items": [],
                "record_source": source.name,
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (f"LiteratureSearchNode: the literature source exceeded its {budget_s:g}s budget"),
            }

        record_count = len(literature_items)
        logger.info(
            "LiteratureSearchNode: %d record(s) from the %s set, sources=%s",
            record_count,
            source.name,
            sources,
        )

        emit_trace_event(
            "literature_search_complete",
            {
                "record_count": record_count,
                "sources": sources,
                "record_source": source.name,
                "query_length": len(query),
                "has_date_filter": bool(date_from or date_to),
                "within_budget": True,
                "budget_s": budget_s,
            },
            state,
        )

        return {
            "literature_items": literature_items,
            "record_source": source.name,
            "result": (
                f"Assembled {record_count} literature record(s) from the "
                f"{source.name} set across {len(sources)} source(s)"
            ),
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _request_from(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the validated request mapping, or None when it is absent.

        Inside the inner graph the request arrives as user_input, because the
        outer graph's extract_input() hands the validated contract across the
        graph boundary. validated_input is checked first so the node also works
        when it is driven directly against an outer-shaped state.
        """
        validated = state.get("validated_input")
        if isinstance(validated, dict) and validated:
            return validated
        user_input = state.get("user_input")
        if isinstance(user_input, dict) and user_input:
            return user_input
        return None

    @staticmethod
    def _source_for(request: Dict[str, Any]) -> LiteratureSource:
        """Choose the literature source for this request.

        Register an adapter for a live index here; the budget above and the
        appraisal pipeline downstream both stay unchanged.
        """
        records = request.get("literature_records")
        if isinstance(records, list) and records:
            return CallerSuppliedSource(records)
        return BaselineLiteratureSource()
