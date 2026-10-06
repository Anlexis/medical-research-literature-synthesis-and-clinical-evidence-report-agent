# Template Design Specification — HCR-C2-045

## Position in the framework

| Item | Value |
|---|---|
| Agent class | `MedicalResearchLiteratureSynthesisAgent` (`src/graph/graph.py`) |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Composition | two-layer nested workflow: fixed outer backbone + custom inner graph |
| State | flat `TypedDict` (`src/schemas/state.py`) — no models, no arbitrary objects |

## Architecture

### Outer backbone (fixed — `add_edges()` is not overridden)

| Node | Responsibility | Reads | Writes | Class |
|------|---------------|-------|--------|-------|
| initialize | session id, schema version, trust level | — | session_id, schema_version | framework default |
| pre_process | input gate: validate the request and the caller's records | user_input, input_context | validated_input, redaction_flags | `PreProcessNode` (VERIFIED_EXTERNAL) |
| main | delegate to the inner workflow | validated_input | result, evidence_report, report_metadata, record_source | `HcrC2045GraphNode` |
| post_process | output gate: scan the report before the caller sees it | result, report_metadata | result (or cleared) | `PostProcessNode` (ANONYMOUS) |
| finalize | response metadata, total time | — | response_metadata | framework default |

### Inner workflow (`DomainWorkflowGraph`, `src/graph/domain_workflow_graph.py`)

| Node | Responsibility | Reads | Writes |
|------|---------------|-------|--------|
| literature_search | assemble the record set; enforce the per-search budget | validated request, runtime_config | literature_items, record_source |
| evidence_extract | extract structured evidence from each record | literature_items | evidence_items |
| clinical_synthesis | grade strength, derive findings and limitations | evidence_items, runtime_config | synthesis_findings |
| report_generate | render the report | synthesis_findings, literature_items, runtime_config | evidence_report, result, report_metadata |
| output_gate | verify the report before it leaves the inner graph | evidence_report, report_metadata | evidence_report (or cleared) |

All inner nodes admit anonymous callers. The caller-facing trust decision is taken once, at the
outer input gate; repeating it on inner nodes would only make the pipeline fail in the middle of
a run that was already admitted.

### Data flow

```
START
  → initialize
  → pre_process              validate the request and the caller's records
  → main [HcrC2045GraphNode]
      └── DomainWorkflowGraph:
            START → literature_search → evidence_extract → clinical_synthesis
                  → report_generate → output_gate → END
  → post_process             scan the report at the caller boundary
  → finalize
END
```

### How data crosses the graph boundary

The framework does not forward outer state into a subgraph invocation: `GraphNode` calls
`subgraph.invoke(user_input, session_id=..., ctx=...)` and nothing else travels. Two things
therefore have to be handed across explicitly, and both are proven end to end rather than at node
level, because a channel can be correct inside one node and dead at the boundary:

| What | How |
|---|---|
| the caller's validated request, including their literature records | `HcrC2045GraphNode.extract_input()` returns the whole validated mapping, which becomes the inner graph's `user_input` |
| runtime settings from `config/config.yaml` | `HcrC2045GraphNode._parent_config()` → `DomainWorkflowGraph._extra_initial_state()` → `state["runtime_config"]` (JSON string, so State stays flat) |

`config/config.yaml` is loaded by `load_runtime_config()` and passed to the graph constructor by
both the platform and the standalone entry point. A file read on only one of those paths is a
setting that silently does nothing on the other, so both read it the same way.

## State

| Field | Type | Purpose | Set by |
|-------|------|---------|--------|
| validated_input | `Dict[str, Any]` | the validated request: query, date range, sources, result cap, caller records | pre_process |
| runtime_config | `str` (JSON) | forwarded settings: search budget, appraisal settings | inner graph seed |
| literature_items | `List[Dict[str, Any]]` | the records under appraisal | literature_search |
| record_source | `str` | `caller` or `baseline` — rendered in the report | literature_search |
| evidence_items | `List[Dict[str, Any]]` | structured evidence per record | evidence_extract |
| synthesis_findings | `Dict[str, Any]` | summary, key findings, strength, limitations, recommendation | clinical_synthesis |
| evidence_report | `str` | the report in markdown | report_generate |
| result | `str` | the same string at the caller boundary | report_generate / merge_output |
| report_metadata | `Dict[str, Any]` | query, sources, generation time, counts, provenance | report_generate |
| redaction_flags | `List[str]` | caller field names whose content was redacted (names only) | pre_process |

**State constraints:** flat `TypedDict` only, JSON-serialisable values, no models or arbitrary
objects (checkpoints are serialised), and no credentials or personal data.

## The caller contract

`POST /invoke` accepts the request as `input` and, optionally, records on the structured
`input_context` channel.

| Field | Bound |
|---|---|
| `query` | non-empty, ≤ 2000 characters, screened |
| `date_from` / `date_to` | ISO `YYYY-MM-DD`; `date_to` not in the future; `date_from` ≤ `date_to` |
| `sources` | subset of `j-stage`, `pubmed`, `clinical-trials`; each an inert identifier |
| `max_results` | finite whole number, 1–100 |
| `literature_records` | ≤ 50 records |
| `…[].title` | required, ≤ 300 characters, screened, redacted, sanitised |
| `…[].abstract` | optional, ≤ 4000 characters, screened, redacted, sanitised |
| `…[].authors` | ≤ 20 names, each ≤ 80 characters |
| `…[].source` | one of the supported source identifiers |
| `…[].year` | finite whole number, 1800–2200 |
| `…[].doi` | inert identifier, ≤ 64 characters |

The entry point additionally caps the whole channel at 256 KB, **drops keys the pipeline does not
read**, and refuses a credential-shaped value with a 400 naming the field. The drop is not
tidiness: validators ignore keys they do not know, and ignoring is not stripping — an undeclared
key survives into the first node's result, where every value is scanned for credential patterns,
so a credential under a key nothing reads still fails the whole request with an error the caller
cannot act on.

Absent records are not an error. The pipeline falls back to the built-in sample corpus and the
report says so, at the top, in its own section.

## Security design

### Input

- Every caller-controlled **number** goes through a finite-and-bounded parser. NaN and the
  infinities parse as floats and compare `False` against any bound, so a plain range check
  accepts them; the parser rejects them explicitly and fails closed.
- Every caller-controlled **identifier** is locked to an inert alphabet with no markup and no
  whitespace, so an identifier slot cannot be used as an output-injection channel.
- Every caller-controlled **string** is screened twice: once raw, so a chat-template control
  token (`<|…|>`, `[INST]`, `<<SYS>>`) is caught before any transformation could strip it and
  forward its residue as ordinary prose; and once over a re-assembled view with markup removed,
  so a directive split by markup (`ig<b>nore all previous instructions`) is caught after a
  consumer would have put it back together.
- The directive patterns are anchored on imperative constructions. A screen that refuses
  *"system prompt engineering for clinical decision support"* or *"act as a screening instrument
  for delirium"* is a defect in a tool whose job is to read research prose, and both directions
  are tested.
- Rejections name the field and never the value. An error log is a caller-visible surface, and
  echoing a rejected value would put the refused content back into the output.

### Personal identifiers

The framework's own masking covers plain-text request fields; it does not reach the structured
channel. Record text is therefore scanned by this template, on the **original string, before any
other rewriting** — e-mail addresses, national identifiers, phone numbers, record numbers and
dates of birth are replaced, and the field names (never the values) are recorded in
`redaction_flags`. Sanitising first would space an identifier apart and hide it from the scan.

### Output

The report renders population-level research findings. There is no monetary aggregate anywhere in
its format, so the rounding grid the fleet applies to financial reports has nothing to act on
here. The invariant enforced instead is the one this report actually makes:

1. the report is present and non-empty;
2. its metadata is populated;
3. it carries every section the format promises;
4. it carries no credential-shaped value;
5. it carries no direct personal identifier.

Two gates enforce it: `OutputGateNode` inside the inner graph, and `PostProcessNode` at the
caller boundary. Both **clear every output-bearing field** on a violation rather than merely
flagging it — the graph's output builder falls back to the raw result field even on an error
status, so a gate that only reports a violation still ships the content it rejected.

Credential detection calls the framework's own detector rather than a local pattern list. A
narrower local list would let a value through that the framework then raises on inside the same
gate, and that raise returns a bare error partial which discards the clearing — a detector gap is
a containment bypass, not just a missed finding.

Inner-graph failures are handled rather than re-raised (`error_strategy = "handle"`). An escaping
exception is turned by the framework into an error result whose log carries the formatted
traceback — source paths and internal frames included — and writes no output fields, so anything
already merged stays readable. Handling the failure lets the node clear the output fields itself
and return a reason the caller can act on.

### Audit trail

Every `execute()` emits one domain event: `pre_process_validated`, `literature_search_complete`,
`evidence_extract_complete`, `clinical_synthesis_complete`, `report_generate_complete`,
`output_gate_complete`, `post_process_complete`. Payloads carry counts, flags and lengths — never
request text and never a rejected value.

### Gate behaviour by node type

- `FunctionNode` subclass — the framework's own input/output gates always run; extend them only
  through `_extra_security_gate_input()` / `_extra_security_gate_output()`. This template does
  its domain checks inline in `execute()` instead: the framework auto-wraps those hooks and
  passes the following node's state as `None`, which raises on the full invoke path.
- `GraphNode` — deliberate no-op at the subgraph boundary; the child graph's own gate has run.

## Configuration

`config/agent.yaml` is the static registration manifest and carries no runtime values.
`config/config.yaml` carries the runtime settings, every one of which has a reader:

| Key | Read by | Effect |
|---|---|---|
| `max_retry` | framework backbone routing | retry ceiling |
| `timeout_s` | `LiteratureSearchNode` | per-search budget; a source that answers later fails the run rather than producing a report that looks complete |
| `evidence.max_references` | `ReportGenerateNode` | references rendered |
| `evidence.strong_evidence_min` | `ClinicalSynthesisNode` | higher-quality studies required for STRONG |
| `evidence.moderate_evidence_min` | `ClinicalSynthesisNode` | higher-quality studies required for MODERATE |

Each reader holds its value to a supported range and falls back to a documented default outside
it, so a mis-declared setting cannot disable the behaviour it configures.

## Design decisions

| Decision | Alternative | Chosen | Rationale |
|---|---|---|---|
| Base class | `AutonomousBaseGraph` | `AgentBaseGraph` | fixed pipeline; no autonomous loop, no self-directed termination |
| Composition | flat backbone slots | nested inner graph | five domain steps; the backbone stays untouched |
| Inner graph parent | `AgentBaseGraph` | `BaseGraph` | fully custom topology, no forced five-node backbone inside |
| Evidence extraction | model-backed | deterministic rules | reproducibility: the same records and thresholds must always produce the same grade |
| Missing caller records | refuse the request | sample corpus, labelled | the format is demonstrable without data, and labelling is what keeps sample data from being read as a search |
| Inner failure handling | propagate | handle | a propagated exception ships a traceback and clears nothing |
| Output invariant | rounding grid | completeness + identifiers + credentials | no monetary aggregate exists in this report format |

## Import isolation

- No platform SDK import anywhere in `src/`.
- Imports are limited to `framework.*` and `shared.*`.
