"""AgentCore Platform v1.0"""

# Service layer: the seam between the pipeline and a literature source.
# Adapters live here; business logic, routing and credentials do not.
#
# LiteratureSearchNode calls a LiteratureSource. Two are shipped:
#
#   BaselineLiteratureSource   a small fixed sample corpus, used when the
#                              caller supplies no records of their own. It is
#                              labelled as sample data everywhere it surfaces
#                              so no reader can mistake it for a real search.
#   CallerSuppliedSource       returns the records the caller passed on the
#                              structured input channel, already validated and
#                              sanitized by the pre-processing node.
#
# To search a live index, implement `search()` against it and register the
# adapter in LiteratureSearchNode._source_for(); the node's per-search budget
# and the appraisal pipeline downstream of it stay unchanged.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

# The sample corpus. It carries no real study data: every entry is generic and
# is rendered under an explicit sample-data notice.
_BASELINE_CORPUS: List[Dict[str, Any]] = [
    {
        "title": "Systematic review of clinical outcomes",
        "authors": ["Sample Author A", "Sample Author B", "Sample Author C"],
        "abstract": (
            "Sample record. A systematic review pooling randomised controlled trials "
            "and reporting efficacy and safety outcomes with a hazard ratio and a "
            "95% CI. Provided so the pipeline has a shape to work on when no "
            "literature records are supplied."
        ),
        "source": "pubmed",
        "year": 2024,
        "doi": "10.0000/sample-pubmed-001",
    },
    {
        "title": "Clinical guideline update",
        "authors": ["Sample Author D", "Sample Author E"],
        "abstract": (
            "Sample record. A guideline update synthesising national and international "
            "evidence and issuing graded recommendations. Provided as sample data only."
        ),
        "source": "j-stage",
        "year": 2023,
        "doi": "10.0000/sample-jstage-001",
    },
    {
        "title": "Multicentre phase III randomised controlled trial",
        "authors": ["Sample Author F", "Sample Author G", "Sample Author H", "Sample Author I"],
        "abstract": (
            "Sample record. A multicentre phase III randomised controlled trial in adult "
            "patients reporting a primary efficacy endpoint. Provided as sample data only."
        ),
        "source": "clinical-trials",
        "year": 2023,
        "doi": "10.0000/sample-ct-001",
    },
    {
        "title": "Retrospective observational cohort",
        "authors": ["Sample Author J", "Sample Author K"],
        "abstract": (
            "Sample record. A retrospective cohort reporting real-world outcomes and a "
            "number needed to treat. Provided as sample data only."
        ),
        "source": "j-stage",
        "year": 2022,
        "doi": "10.0000/sample-jstage-002",
    },
    {
        "title": "Safety profile review of adverse events",
        "authors": ["Sample Author L", "Sample Author M", "Sample Author N"],
        "abstract": (
            "Sample record. A pooled safety analysis reporting adverse event rates "
            "against control and a monitoring recommendation. Provided as sample data only."
        ),
        "source": "pubmed",
        "year": 2022,
        "doi": "10.0000/sample-pubmed-002",
    },
]


class LiteratureSource(Protocol):
    """Port a literature source must implement to be usable by the pipeline."""

    name: str

    def search(
        self,
        query: str,
        *,
        sources: List[str],
        date_from: Optional[str],
        date_to: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Return literature records matching the search parameters."""
        ...


def _filter_records(
    records: List[Dict[str, Any]],
    *,
    sources: List[str],
    date_from: Optional[str],
    date_to: Optional[str],
    max_results: int,
) -> List[Dict[str, Any]]:
    """Apply the source and publication-year filters, then the result cap."""
    allowed = set(sources)
    selected = [r for r in records if str(r.get("source", "")) in allowed]
    if date_from:
        year_from = _leading_year(date_from)
        if year_from is not None:
            selected = [r for r in selected if int(r.get("year") or 0) >= year_from]
    if date_to:
        year_to = _leading_year(date_to)
        if year_to is not None:
            selected = [r for r in selected if int(r.get("year") or 0) <= year_to]
    return selected[:max_results]


def _leading_year(iso_date: str) -> Optional[int]:
    """Return the year of an ISO date string, or None when it has none."""
    try:
        return int(str(iso_date)[:4])
    except (TypeError, ValueError):
        return None


class BaselineLiteratureSource:
    """Fixed sample corpus used when the caller supplies no records."""

    name = "baseline"

    def search(
        self,
        query: str,
        *,
        sources: List[str],
        date_from: Optional[str],
        date_to: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Return sample records matching the source and date filters."""
        return [
            dict(record)
            for record in _filter_records(
                _BASELINE_CORPUS,
                sources=sources,
                date_from=date_from,
                date_to=date_to,
                max_results=max_results,
            )
        ]


class CallerSuppliedSource:
    """Returns the validated records the caller supplied with the request."""

    name = "caller"

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = [dict(record) for record in records]

    def search(
        self,
        query: str,
        *,
        sources: List[str],
        date_from: Optional[str],
        date_to: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Return the caller's records under the same filters as any source."""
        return _filter_records(
            self._records,
            sources=sources,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
        )
