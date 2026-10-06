"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never the full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Read the structured caller channel via state.get("input_context", {})
#
# HCR-C2-045 — PreProcessNode
# Outer backbone pre_process slot — the input validation gate.
#
# It owns the whole caller contract:
#
#   user_input      the search request: a plain query string, or a JSON object
#                   carrying query / date_from / date_to / sources / max_results
#   input_context   the structured channel: literature_records, the caller's own
#                   search results, appraised instead of the sample corpus
#
# The template enforces its own guarantees here rather than relying on the
# framework's input gate: that gate does not scan the structured channel at all,
# and a validator that only checks the fields it knows about does not remove the
# ones it does not — so every accepted field is rebuilt from validated parts and
# nothing else is carried forward.

import json
import logging
from datetime import date, datetime
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# Audit trail — the bare free-function form only.
from shared.utils.audit_logger import emit_trace_event
from src.schemas.validation import (
    CallerInputError,
    contains_credentials,
    parse_finite_number,
    parse_inert_id,
    redact_identifiers,
    sanitize_render_text,
    screen_text_both_ways,
)

logger = logging.getLogger(__name__)

# Allowed literature source identifiers.
_ALLOWED_SOURCES: FrozenSet[str] = frozenset({"j-stage", "pubmed", "clinical-trials"})

# Field bounds for the search request.
_MAX_QUERY_CHARS = 2000
_MIN_RESULTS = 1
_MAX_RESULTS = 100
_DEFAULT_RESULTS = 20

# Field bounds for caller-supplied literature records.
_MAX_RECORDS = 50
_MAX_TITLE_CHARS = 300
_MAX_ABSTRACT_CHARS = 4000
_MAX_AUTHORS = 20
_MAX_AUTHOR_CHARS = 80
_MIN_YEAR = 1800
_MAX_YEAR = 2200

# Keys accepted on the structured caller channel. Anything else is ignored, and
# ignoring is not the same as dropping — the adapter is what removes unknown
# keys before the request reaches the graph.
_CONTEXT_RECORDS_KEY = "literature_records"


class PreProcessNode(FunctionNode):
    """Validate and parse the literature synthesis request.

    Outer backbone pre_process slot. Rejects malformed or hostile requests
    before the inner workflow graph runs.

    Input state keys:
        user_input:     JSON string or plain query string. Recognised keys:
            query       (required) — research query text, max 2000 chars
            date_from   (optional) — ISO date YYYY-MM-DD, earliest publication
            date_to     (optional) — ISO date YYYY-MM-DD, not in the future
            sources     (optional) — source identifiers; defaults to all allowed
            max_results (optional) — whole number 1-100; defaults to 20
        input_context:  structured caller channel. Recognised key:
            literature_records (optional) — up to 50 records, each
            {title, abstract, authors, source, year, doi}

    Output state keys (partial dict):
        validated_input: validated request, including any accepted records
        redaction_flags: caller field names whose content was redacted
        status:          AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:       (error path only) messages naming the failed field
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Domain checks run inline in execute(). They are NOT installed as an
        # _extra_security_gate_input method: the framework auto-wraps that hook
        # and passes the next node's state as None, which raises on the full
        # invoke path.
        payload, error = self._parse_request(state.get("user_input", ""))
        if error is not None:
            return self._reject(error)

        try:
            query = self._validate_query(payload.get("query"))
            date_from, date_to = self._validate_dates(payload)
            max_results = int(
                parse_finite_number(
                    payload.get("max_results", _DEFAULT_RESULTS),
                    field="max_results",
                    minimum=_MIN_RESULTS,
                    maximum=_MAX_RESULTS,
                    integer=True,
                )
            )
            sources = self._validate_sources(payload.get("sources"))
            records, redaction_flags = self._validate_records(state.get("input_context") or {})
        except CallerInputError as exc:
            return self._reject(str(exc))

        validated_payload: Dict[str, Any] = {
            "query": query,
            "date_from": date_from,
            "date_to": date_to,
            "sources": sources,
            "max_results": max_results,
            "literature_records": records,
            "record_source": "caller" if records else "baseline",
        }

        logger.info(
            "PreProcessNode: accepted request query_chars=%d sources=%s max_results=%d records=%d",
            len(query),
            sources,
            max_results,
            len(records),
        )

        emit_trace_event(
            "pre_process_validated",
            {
                "query_length": len(query),
                "sources": sources,
                "max_results": max_results,
                "has_date_range": bool(date_from or date_to),
                "caller_record_count": len(records),
                "redaction_categories": redaction_flags,
            },
            state,
        )

        return {
            "validated_input": validated_payload,
            "redaction_flags": redaction_flags,
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------
    # Request parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_request(user_input: Any) -> Tuple[Dict[str, Any], Optional[str]]:
        """Return (payload, error). A plain string is treated as the query."""
        if isinstance(user_input, dict):
            return dict(user_input), None
        if not user_input:
            return {}, "'user_input' is empty or missing"
        stripped = str(user_input).strip()
        if not stripped:
            return {}, "'user_input' is blank"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"query": stripped}, None
        if isinstance(parsed, dict):
            return parsed, None
        return {"query": stripped}, None

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(raw: Any) -> str:
        """Return the validated research query, or raise CallerInputError."""
        query = str(raw or "").strip()
        if not query:
            raise CallerInputError("'query' is required and must be non-empty")
        if len(query) > _MAX_QUERY_CHARS:
            raise CallerInputError(f"'query' must be at most {_MAX_QUERY_CHARS} characters")
        hit = screen_text_both_ways(query)
        if hit:
            raise CallerInputError(f"'query' was refused by the input screen ({hit})")
        if contains_credentials(query):
            raise CallerInputError("'query' carries a credential-shaped value")
        return query

    @staticmethod
    def _validate_dates(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Return (date_from, date_to) as ISO strings, or raise."""
        date_from = PreProcessNode._parse_iso_date(payload.get("date_from"), field="date_from")
        date_to = PreProcessNode._parse_iso_date(payload.get("date_to"), field="date_to")
        if date_to is not None and date_to > date.today().isoformat():
            raise CallerInputError("'date_to' must not be in the future")
        if date_from and date_to and date_from > date_to:
            raise CallerInputError("'date_from' must not be later than 'date_to'")
        return date_from, date_to

    @staticmethod
    def _parse_iso_date(raw: Any, *, field: str) -> Optional[str]:
        """Return an ISO date string for *raw*, None when absent, else raise."""
        if raw is None:
            return None
        try:
            return datetime.strptime(str(raw).strip(), "%Y-%m-%d").date().isoformat()
        except (TypeError, ValueError):
            raise CallerInputError(f"'{field}' must be an ISO date in YYYY-MM-DD form") from None

    @staticmethod
    def _validate_sources(raw: Any) -> List[str]:
        """Return the requested source identifiers, or raise."""
        if raw is None:
            return sorted(_ALLOWED_SOURCES)
        if not isinstance(raw, list):
            raise CallerInputError("'sources' must be a list of source identifiers")
        if not raw:
            return sorted(_ALLOWED_SOURCES)
        if len(raw) > len(_ALLOWED_SOURCES):
            raise CallerInputError(f"'sources' must name at most {len(_ALLOWED_SOURCES)} sources")
        selected: List[str] = []
        for index, item in enumerate(raw):
            identifier = parse_inert_id(item, field=f"sources[{index}]")
            if identifier not in _ALLOWED_SOURCES:
                raise CallerInputError(
                    f"'sources[{index}]' is not a supported source; " f"supported: {sorted(_ALLOWED_SOURCES)}"
                )
            if identifier not in selected:
                selected.append(identifier)
        return selected

    # ------------------------------------------------------------------
    # Structured caller channel
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_records(input_context: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Return (records, redaction_categories) for the caller's records.

        An absent or empty channel is not an error — the pipeline degrades to
        the sample corpus and says so in the report.
        """
        if not isinstance(input_context, dict):
            raise CallerInputError(f"'{_CONTEXT_RECORDS_KEY}' channel must be a mapping")
        raw_records = input_context.get(_CONTEXT_RECORDS_KEY)
        if raw_records is None:
            return [], []
        if not isinstance(raw_records, list):
            raise CallerInputError(f"'{_CONTEXT_RECORDS_KEY}' must be a list of records")
        if len(raw_records) > _MAX_RECORDS:
            raise CallerInputError(f"'{_CONTEXT_RECORDS_KEY}' must hold at most {_MAX_RECORDS} records")

        records: List[Dict[str, Any]] = []
        redactions: List[str] = []
        for index, raw in enumerate(raw_records):
            record, flags = PreProcessNode._validate_record(raw, index=index)
            records.append(record)
            for flag in flags:
                if flag not in redactions:
                    redactions.append(flag)
        return records, redactions

    @staticmethod
    def _validate_record(raw: Any, *, index: int) -> Tuple[Dict[str, Any], List[str]]:
        """Validate one literature record; return (record, redaction flags)."""
        where = f"{_CONTEXT_RECORDS_KEY}[{index}]"
        if not isinstance(raw, dict):
            raise CallerInputError(f"'{where}' must be an object")

        flags: List[str] = []
        title = PreProcessNode._clean_prose(
            raw.get("title"),
            field=f"{where}.title",
            limit=_MAX_TITLE_CHARS,
            required=True,
            flags=flags,
        )
        abstract = PreProcessNode._clean_prose(
            raw.get("abstract"),
            field=f"{where}.abstract",
            limit=_MAX_ABSTRACT_CHARS,
            required=False,
            flags=flags,
        )

        raw_authors = raw.get("authors")
        if raw_authors is None:
            authors: List[str] = []
        elif not isinstance(raw_authors, list):
            raise CallerInputError(f"'{where}.authors' must be a list")
        elif len(raw_authors) > _MAX_AUTHORS:
            raise CallerInputError(f"'{where}.authors' must hold at most {_MAX_AUTHORS} names")
        else:
            authors = [
                PreProcessNode._clean_prose(
                    author,
                    field=f"{where}.authors[{position}]",
                    limit=_MAX_AUTHOR_CHARS,
                    required=True,
                    flags=flags,
                )
                for position, author in enumerate(raw_authors)
            ]

        source = parse_inert_id(raw.get("source"), field=f"{where}.source")
        if source not in _ALLOWED_SOURCES:
            raise CallerInputError(f"'{where}.source' is not a supported source; supported: {sorted(_ALLOWED_SOURCES)}")

        year = int(
            parse_finite_number(
                raw.get("year"),
                field=f"{where}.year",
                minimum=_MIN_YEAR,
                maximum=_MAX_YEAR,
                integer=True,
            )
        )

        doi_raw = raw.get("doi")
        doi = "" if doi_raw is None else parse_inert_id(doi_raw, field=f"{where}.doi")

        return (
            {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "source": source,
                "year": year,
                "doi": doi,
            },
            flags,
        )

    @staticmethod
    def _clean_prose(
        raw: Any,
        *,
        field: str,
        limit: int,
        required: bool,
        flags: List[str],
    ) -> str:
        """Screen, redact and sanitize one caller free-text field.

        Order matters. The identifier scan runs on the ORIGINAL string, before
        any rewriting, so no earlier transformation can split an identifier
        apart and hide it from the scan.
        """
        if raw is None or raw == "":
            if required:
                raise CallerInputError(f"'{field}' is required")
            return ""
        if not isinstance(raw, str):
            raise CallerInputError(f"'{field}' must be a string")
        if len(raw) > limit:
            raise CallerInputError(f"'{field}' must be at most {limit} characters")

        hit = screen_text_both_ways(raw)
        if hit:
            raise CallerInputError(f"'{field}' was refused by the input screen ({hit})")
        if contains_credentials(raw):
            raise CallerInputError(f"'{field}' carries a credential-shaped value")

        redacted, categories = redact_identifiers(raw)
        for category in categories:
            flag = f"{field}:{category}"
            if flag not in flags:
                flags.append(flag)

        cleaned = sanitize_render_text(redacted, limit=limit)
        if required and not cleaned:
            raise CallerInputError(f"'{field}' is required")
        return cleaned

    # ------------------------------------------------------------------

    @staticmethod
    def _reject(message: str) -> Dict[str, Any]:
        """Build the rejection result. Names the field, never the value."""
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {message}"],
        }
