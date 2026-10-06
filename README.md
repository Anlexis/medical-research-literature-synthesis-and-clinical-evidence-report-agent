# Medical Literature Evidence Report Agent

AI agent for synthesizing medical research literature into clinical evidence reports, built with Agentic Star.

> **Category**: Cat 2 (a multi-step domain workflow behind one endpoint)
> **Industry**: Healthcare
> **Template ID**: HCR-C2-045

## Overview

Turns a research question and a set of literature records into a structured clinical evidence
report: it appraises each record, grades the overall strength of the evidence, states the
limitations, and renders a referenced markdown report with a fixed section layout.

Records can be supplied by the caller — the results of your own PubMed, J-Stage or trial-registry
search — or left out, in which case the agent runs on a small built-in sample corpus and labels
the report as sample data so it can never be mistaken for a real search.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

> **Decision support, not a clinical decision.** The report summarises published research for a
> qualified clinician to read. It is not a diagnosis, a treatment recommendation, or a substitute
> for clinical judgement.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Calling it

`POST /invoke` takes the research request as `input` and, optionally, your own literature records
on the structured `input_context` channel:

```json
{
  "input": "{\"query\": \"early mobilisation after cardiac surgery\", \"max_results\": 10}",
  "input_context": {
    "literature_records": [
      {
        "title": "Randomised trial of early mobilisation after cardiac surgery",
        "abstract": "Controlled trial reporting efficacy and safety outcomes.",
        "authors": ["Author One", "Author Two"],
        "source": "pubmed",
        "year": 2025,
        "doi": "10.1234/example-001"
      }
    ]
  }
}
```

`input` also accepts a plain question, which is read as the query with default filters. Every
field above is validated against explicit bounds; anything outside them is refused with a message
naming the field. Direct personal identifiers found in record text are removed before the record
is used, and the finished report is checked again for identifiers and credential-shaped values
before it is released.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit and boundary tests
config/       agent manifest and runtime settings
docs/         design and operational documentation
```

See `docs/` for the design and the test specification.

## Customising

1. Adjust `config/config.yaml` for your own appraisal thresholds and search budget.
2. Implement a `LiteratureSource` in `src/services/service.py` against your own index and
   register it in `LiteratureSearchNode._source_for()`; the sample corpus is only a default.
3. Review the node implementations under `src/nodes/` for domain-specific logic — the evidence
   grading and limitation heuristics are deliberately simple and are the first thing most
   deployments will want to replace.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
