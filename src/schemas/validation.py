"""AgentCore Platform v1.0"""

# HCR-C2-045 — shared caller-input validation and text-screening helpers.
#
# Every value a caller can influence passes through this module before the
# pipeline computes on it or renders it into a clinical evidence report:
#
#   parse_finite_number()  numbers  — finite, bounded, fail closed
#   parse_inert_id()       ids      — locked to an inert identifier alphabet
#   screen_text()          prose    — control-token and directive screening,
#                                     run on the raw string AND on the
#                                     re-assembled string
#   sanitize_render_text() prose    — neutralises markup before rendering
#   redact_identifiers()   prose    — removes direct personal identifiers
#   contains_credentials() any      — the framework's own credential detector
#
# Rejection messages name the FIELD, never the value: an evidence report and
# its error log are both caller-visible surfaces, and echoing a rejected value
# would put the very content that was refused back into the output.

import math
import re
import unicodedata
from typing import Any, List, Optional, Tuple

from framework.security.credential_detector import detect_credentials

# Longest caller string accepted for any single free-text field.
MAX_TEXT_CHARS = 4000

# Inert identifier alphabet for caller strings that are rendered verbatim.
_INERT_ID_RE = re.compile(r"^[a-z0-9_.:/-]{1,64}$")

# Chat-template control tokens. These are a CLASS, not a list of phrases: a
# payload that opens a template turn is an instruction-channel attack whatever
# words follow it, and a phrase-based screen alone does not see it.
_CONTROL_TOKEN_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("chat_control_token", re.compile(r"<\|[^|>]{0,64}\|>")),
    ("chat_control_token", re.compile(r"\[/?INST\]", re.IGNORECASE)),
    ("chat_control_token", re.compile(r"<</?SYS>>", re.IGNORECASE)),
]

# Directive patterns. Each one is anchored on an imperative construction so
# that ordinary clinical prose discussing the same words is not refused —
# a screen that blocks legitimate research text is a fail-closed defect in a
# tool whose whole job is to read research text.
_DIRECTIVE_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+|your\s+)*"
            r"(?:previous|prior|above|preceding|earlier|foregoing)\s+"
            r"(?:instruction|prompt|rule|direction|guideline)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_disclosure",
        re.compile(
            r"\b(?:reveal|disclose|print|output|repeat|show)\s+(?:me\s+)?"
            r"(?:your|the)\s+(?:system|initial|original|hidden)\s+"
            r"(?:prompt|instruction|message)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\b(?:you\s+are\s+now|act\s+as|behave\s+as|pretend\s+to\s+be)\s+"
            r"(?:a|an|the)\s+(?:system|admin(?:istrator)?|developer|root|"
            r"unrestricted|jailbroken|dan)\b",
            re.IGNORECASE,
        ),
    ),
    ("script_markup", re.compile(r"<\s*/?\s*(?:script|iframe|object|embed)\b", re.IGNORECASE)),
    ("script_markup", re.compile(r"javascript\s*:", re.IGNORECASE)),
    (
        "script_markup",
        re.compile(
            r"\bon(?:error|load|click|focus|blur|mouseover|submit|change)\s*=",
            re.IGNORECASE,
        ),
    ),
]

# Direct personal identifiers. A clinical evidence report is a research
# artefact: it summarises populations, never individuals, so any of these
# reaching the render step is a leak regardless of who put it there.
_IDENTIFIER_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("email", re.compile(r"\b[\w.%+-]+@[\w-]+\.[A-Za-z]{2,}\b")),
    ("national_id", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("national_id", re.compile(r"(?<!\d)\d{12}(?!\d)")),
    ("phone", re.compile(r"(?<![\d-])0\d{1,4}-\d{1,4}-\d{4}(?![\d-])")),
    ("phone", re.compile(r"\+\d{1,3}[\s-]\d{1,4}[\s-]\d{2,4}[\s-]\d{3,4}\b")),
    (
        "record_number",
        re.compile(
            r"\b(?:MRN|patient\s*(?:id|no\.?|number)|chart\s*(?:no\.?|number)|"
            r"medical\s*record\s*(?:no\.?|number))\s*[:#=]?\s*[A-Za-z0-9-]{4,}",
            re.IGNORECASE,
        ),
    ),
    (
        "date_of_birth",
        re.compile(
            r"\b(?:DOB|date\s+of\s+birth)\s*[:=]?\s*\d{2,4}[-/]\d{1,2}[-/]\d{1,4}",
            re.IGNORECASE,
        ),
    ),
]

_REDACTION_MARK = "[REDACTED]"

# Characters that would let caller text restructure the markdown report.
_MARKUP_CHARS = str.maketrans({"#": " ", "|": " ", "`": " ", "*": " ", "_": " ", "<": " ", ">": " "})

# Screening view: the same characters DELETED rather than spaced, so a
# directive split by markup is re-assembled before the screen reads it.
_SPLICE_CHARS = str.maketrans({c: None for c in "#|`*_<>~"})


class CallerInputError(ValueError):
    """Raised when a caller-supplied value fails validation.

    The message names the field and the bound it failed; it never carries the
    rejected value.
    """


def parse_finite_number(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
    integer: bool = False,
) -> float:
    """Return *value* as a finite number inside [minimum, maximum], or raise.

    Rejects booleans, non-numeric strings, and NaN / ±Infinity. The last of
    those is the reason this helper exists: ``float("NaN")`` and
    ``float("Infinity")`` both parse without error and arrive intact through
    raw JSON, and every comparison against NaN is False — so a plain range
    check silently ACCEPTS them and the pipeline computes on a value that can
    never be in range. Fail closed instead.
    """
    if isinstance(value, bool):
        raise CallerInputError(f"'{field}' must be a number, not a boolean")
    if isinstance(value, (int, float)):
        candidate = float(value)
    elif isinstance(value, str):
        try:
            candidate = float(value.strip())
        except (TypeError, ValueError):
            raise CallerInputError(f"'{field}' must be a number") from None
    else:
        raise CallerInputError(f"'{field}' must be a number")

    if not math.isfinite(candidate):
        raise CallerInputError(f"'{field}' must be a finite number")
    if not (minimum <= candidate <= maximum):
        raise CallerInputError(f"'{field}' must be between {minimum} and {maximum}")
    if integer:
        if candidate != int(candidate):
            raise CallerInputError(f"'{field}' must be a whole number")
        return float(int(candidate))
    return candidate


def parse_inert_id(value: Any, *, field: str) -> str:
    """Return *value* as an inert identifier, or raise.

    Caller strings that are rendered into the report verbatim are restricted to
    an alphabet with no markup and no whitespace, so a caller cannot use an
    identifier slot as an output-injection channel.
    """
    if not isinstance(value, str):
        raise CallerInputError(f"'{field}' must be a string identifier")
    candidate = value.strip().lower()
    if not _INERT_ID_RE.match(candidate):
        raise CallerInputError(f"'{field}' must be 1-64 characters of a-z, 0-9 and the separators . : / _ -")
    return candidate


def screen_text(text: str) -> Optional[str]:
    """Return the name of the first screening category *text* trips, else None.

    Screening runs over the RAW string and, separately, over the sanitized
    string. Both passes are needed and neither is redundant:

    * raw only — the sanitizer strips markup, so a control token would be
      removed silently and its directive residue forwarded as plain text,
      turning a detectable attack into an undetectable one;
    * sanitized only — a directive split by markup (``ig<b>nore all previous
      instructions``) is invisible until the sanitizer reassembles it.
    """
    if not isinstance(text, str) or not text:
        return None
    for name, pattern in _CONTROL_TOKEN_PATTERNS + _DIRECTIVE_PATTERNS:
        if pattern.search(text):
            return name
    return None


def despliced(value: Any) -> str:
    """Return *value* with markup and zero-width characters REMOVED, not spaced.

    This is the second screening view. The render sanitizer replaces markup
    characters with spaces, which is right for rendering and wrong for
    screening: it leaves ``ig<b>nore all previous instructions`` reading as
    three harmless words. Removing them instead re-assembles the directive so
    the screen sees what a downstream consumer would.
    """
    text = value if isinstance(value, str) else str(value)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\t\n")
    text = re.sub(r"<[^<>]{0,40}>", "", text)
    return text.translate(_SPLICE_CHARS)


def screen_text_both_ways(text: str) -> Optional[str]:
    """Screen *text* raw, then screen its re-assembled form. First hit wins.

    Neither pass is redundant. The raw pass catches control tokens before any
    transformation could strip them silently and forward the directive residue
    as ordinary prose. The re-assembled pass catches a directive that markup
    had split apart and that a consumer would put back together.
    """
    hit = screen_text(text)
    if hit:
        return hit
    return screen_text(despliced(text)[:MAX_TEXT_CHARS])


def sanitize_render_text(value: Any, *, limit: int = MAX_TEXT_CHARS) -> str:
    """Return *value* as plain text that cannot restructure the report.

    Removes Unicode control/format characters, neutralises the markdown
    characters that carry structure, collapses whitespace runs, and truncates
    to *limit*. This is a rendering safeguard, not a security screen —
    ``screen_text_both_ways`` is what refuses hostile content.
    """
    text = value if isinstance(value, str) else str(value)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\t\n")
    text = text.translate(_MARKUP_CHARS)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def redact_identifiers(text: str) -> Tuple[str, List[str]]:
    """Replace direct personal identifiers in *text*; return (text, categories).

    Runs BEFORE any other rewriting of the string, so no earlier transformation
    can break an identifier apart and hide it from this scan.
    """
    if not isinstance(text, str) or not text:
        return ("" if not isinstance(text, str) else text, [])
    found: List[str] = []
    redacted = text
    for name, pattern in _IDENTIFIER_PATTERNS:
        if pattern.search(redacted):
            if name not in found:
                found.append(name)
            redacted = pattern.sub(_REDACTION_MARK, redacted)
    return redacted, found


def find_identifiers(text: str) -> List[str]:
    """Return the identifier categories present in *text* without altering it."""
    if not isinstance(text, str) or not text:
        return []
    return sorted({name for name, pattern in _IDENTIFIER_PATTERNS if pattern.search(text)})


def contains_credentials(text: str) -> Optional[str]:
    """Return the framework's credential type for *text*, or None.

    Deliberately delegates to the framework's own detector rather than keeping
    a local pattern list. A local list that is narrower than the framework's
    lets a value through that the framework then raises on inside the output
    gate — and that raise discards whatever the gate had cleared, turning a
    detector gap into a containment bypass.
    """
    if not isinstance(text, str) or not text:
        return None
    findings = detect_credentials(text)
    return str(findings[0]["type"]) if findings else None
