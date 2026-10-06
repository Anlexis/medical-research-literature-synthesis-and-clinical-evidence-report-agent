# Test Specification — HCR-C2-045 MedicalResearchLiteratureSynthesisAgent

## Strategy

| Item | Value |
|---|---|
| Template ID | HCR-C2-045 |
| Category | Cat 2 nested (`AgentBaseGraph` + inner `DomainWorkflowGraph`) |
| Test types | unit, boundary (full-invoke and HTTP entry point) |
| Runner | pytest |

Two rules shape the whole suite.

**Behaviour, never wording.** A test that asserts a specific refusal message passes only against
the build that produced that message; it says nothing about whether the request was actually
refused. Assertions are on status, on what was withheld, and on what the caller can see.

**Both directions, every time.** A screen that refuses a hostile payload and also refuses real
clinical prose is not a working screen; it is a fail-closed defect in a tool whose job is to read
research text. Every screening test has a companion that pushes ordinary abstracts, statistics and
identifiers through the same path and asserts they are untouched.

### Layout

```
tests/
├── unit/
│   ├── test_agent.py                          every node, driven through execute() directly
│   ├── test_validation.py                     the shared validation and screening helpers
│   └── test_framework_compliance_tc06_tc07.py the framework's non-bypassable gates
└── proof_of_boundary/
    ├── test_import_isolation.py               no platform SDK imports in src/
    ├── test_state_safety.py                   State stays serialisable and credential-free
    ├── test_pb_invoke_order.py                backbone order on a full invoke
    ├── test_pb_invoke_asgi.py                 the real HTTP entry point
    ├── test_pb7_hitl_interrupt_propagation.py human-review propagation (conditional)
    ├── test_boundary_input_gate.py            the input gate at full-invoke level
    ├── test_boundary_output_gate.py           the output gate and containment
    └── test_config_propagation.py             declared settings reach their readers
```

---

## Framework compliance

| ID | Test | Expected |
|----|------|----------|
| TC-01 | State is a flat `TypedDict` with serialisable fields | no models, no arbitrary objects |
| TC-02 | The framework's input/output gates cannot be overridden | `TypeError` at class definition |
| TC-03 | No credential-shaped field names or values in State | scan clean |
| TC-04 | No lifecycle events re-emitted inside `execute()` | the framework owns those |
| TC-05 | `required_trust_level` declared on every node | present on all seven |
| TC-06 | Inner nodes and post_process admit anonymous callers | ANONYMOUS |
| TC-07 | pre_process requires a verified external caller | VERIFIED_EXTERNAL |
| TC-08 | One domain audit event per `execute()` | emitted on every success path |
| TC-09 | No platform SDK imports in `src/` | AST scan clean |
| TC-10 | Build dependencies exactly pinned | pinned |
| TC-11 | Class name agrees across graph, manifest and entry point | all three agree |

---

## Boundary tests

| ID | Boundary | File | Expected |
|----|----------|------|----------|
| PB-4 | Import isolation | `test_import_isolation.py` | no platform SDK imports |
| PB-5 | State serialisation | `test_state_safety.py` | fields stay serialisable primitives |
| PB-6 | Backbone order | `test_pb_invoke_order.py` | SUCCESS, and the five backbone nodes in order |
| PB-7 | Human-review propagation | `test_pb7_hitl_interrupt_propagation.py` | skipped: this template declares no human-review step |
| B-IN | Input gate | `test_boundary_input_gate.py` | rejected requests produce no report anywhere in the response; ordinary clinical queries succeed |
| B-OUT | Output gate | `test_boundary_output_gate.py` | a violating report never reaches the caller, and the error envelope carries no released text, no traceback and no source paths |
| B-CFG | Configuration | `test_config_propagation.py` | changing a declared value changes the report |
| B-HTTP | Entry point | `test_pb_invoke_asgi.py` | auth, size cap, credential refusal, and the caller channel reaching the pipeline |

---

## Unit tests: the input gate (`PreProcessNode`)

Driven through `execute()` directly, with no framework wrapper in front. That is the only way to
show the **template** refuses a payload rather than the framework doing it — a guarantee that
holds only where the framework gate is active is not a guarantee this template owns.

| Area | Cases |
|---|---|
| request shape | plain string read as the query; JSON object supplies the filters; empty and blank refused |
| query | over-length refused; screened; credential-shaped refused |
| dates | non-ISO refused; future `date_to` refused; inverted range refused |
| result cap | `NaN`, `Infinity`, `-Infinity`, `0`, `101`, `2.5`, `True`, non-numeric — all refused |
| sources | unsupported identifier refused; non-list refused; over-length list refused |
| screening | control tokens, directives and markup refused; clinical prose accepted |
| records | valid records accepted and marked as caller data; absent channel degrades to the sample corpus |
| record bounds | bad source, non-finite or out-of-range year, over-length title or abstract, too many authors, too many records, non-inert DOI — all refused |
| nested content | control token, credential and markup-spliced directive inside a record field, all refused, naming the field |
| identifiers | record numbers, e-mail addresses and national identifiers redacted and flagged |
| messages | the field is named; the rejected value never appears |

## Unit tests: the validation helpers (`test_validation.py`)

| Area | Cases |
|---|---|
| finite numbers | non-finite matrix per field; booleans; out-of-range; non-numeric types; fractional where whole required |
| inert identifiers | accepted alphabet; whitespace, markup and over-length refused |
| screening | control tokens as a class; directives; markup; the markup-spliced case; the raw-pass case |
| false positives | eight real clinical sentences, including statistics and trial reporting, must pass |
| identifier redaction | eight identifier forms found and removed; six ordinary statistics forms untouched |
| render sanitiser | structural markdown neutralised; control characters collapsed; length capped |
| credential detection | the template's detector and the framework's agree exactly |

## Unit tests: the domain nodes

| Node | Cases |
|---|---|
| `LiteratureSearchNode` | sample corpus when no records supplied; caller records when supplied; result cap honoured; missing request is an error; a source answering later than the budget fails the run; an unusable budget falls back to the default |
| `EvidenceExtractNode` | evidence extracted per record; empty set yields an empty result; missing or malformed input is an error |
| `ClinicalSynthesisNode` | findings carry every promised key; grade follows the declared thresholds; a changed threshold changes the grade; an unusable threshold falls back |
| `ReportGenerateNode` | every required section present; `result` mirrors the report; the timestamp is the time of the run; sample data is labelled and caller data is not; the reference cap is read from settings; markup in a rendered value cannot fabricate a section |

## Unit tests: the output gates

Both gates are tested for the same nine violation forms — empty, blank, missing metadata, missing
section, credential-shaped value, and three identifier forms — and each asserts that **every**
output-bearing field comes back cleared, not just the one the violation was found in.

## Containment

The containment tests are only worth having if they can fail. Each guard was removed in turn and
the suite re-run:

| Mutation | Result |
|---|---|
| inner gate returns an error but does not clear | 9 unit failures |
| outer gate returns an error but does not clear | 8 unit failures |
| both reverted to the pre-fix form (propagate, stub string, no clearing) | 17 unit failures |
| inner gate removed from the topology | 1 boundary failure — the completeness check is uniquely the inner gate's |
| inner gate removed **and** outer gate returns a stub instead of clearing | 3 boundary failures, including the report reaching the caller |

The last row is the one that matters: with the inner gate present, it masks the outer gate at
full-invoke level, so the outer clearing is provable at node level and provable end to end only
once the layer above it is out of the way. Both layers are kept — the inner one is the completeness
check and the last point before the report crosses the graph boundary; the outer one is the
backbone's mandatory gate at the caller boundary — and the table above records which mutation
proves which.
