"""End-to-end through the real HTTP entry point.

The other boundary tests drive the compiled graph directly. This one goes
through the ASGI application a deployment actually serves — bearer auth, the
request model, the structured caller channel and the JSON response — because
that is the surface a caller reaches, and a channel can be wired correctly
inside the graph and still be dead at the entry point.
"""

import importlib
import json
import pathlib
import secrets
import warnings

import pytest

try:
    with warnings.catch_warnings():
        # The test client warns about the host environment's HTTP library
        # version. That is a property of the machine running the suite, not of
        # this template, and it should not show up as noise in its test run.
        warnings.simplefilter("ignore")
        from starlette.testclient import TestClient

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only when the framework wheel is absent
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(_IMPORT_ERROR is not None, reason=f"test client unavailable: {_IMPORT_ERROR}")

_TOKEN = "boundary-asgi-token"
_PAYLOAD_PATH = pathlib.Path(__file__).parents[2] / "deploy" / "invoke_payload.json"

_RECORD = {
    "title": "Distinctive caller-supplied trial title",
    "abstract": "Controlled trial reporting efficacy and safety outcomes.",
    "authors": ["Author One"],
    "source": "pubmed",
    "year": 2025,
    "doi": "10.1234/caller-001",
}


@pytest.fixture()
def client(monkeypatch):
    """A server whose entry-point auth is armed, imported under that setting."""
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)
    import src.api.server as server

    importlib.reload(server)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(server.app) as test_client:
            yield test_client
    monkeypatch.delenv("INVOKE_AUTH_TOKEN", raising=False)
    importlib.reload(server)


def _auth() -> dict:
    return {"Authorization": f"Bearer {_TOKEN}"}


class TestEntryPointAuth:
    """The entry point is the authentication boundary for direct callers."""

    def test_health_needs_no_credential(self, client):
        assert client.get("/health").json()["status"] == "ok"

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": "Bearer wrong-token"},
            {"Authorization": "not-even-a-scheme"},
            {"Authorization": "Bearer "},
        ],
    )
    def test_an_unauthenticated_caller_is_refused(self, client, headers):
        response = client.post("/invoke", json={"input": "evidence review"}, headers=headers)
        assert response.status_code == 401

    def test_the_401_body_does_not_say_which_way_the_token_was_wrong(self, client):
        absent = client.post("/invoke", json={"input": "x"}, headers={}).json()
        wrong = client.post("/invoke", json={"input": "x"}, headers={"Authorization": "Bearer nope"}).json()
        assert absent == wrong

    def test_a_non_ascii_credential_is_compared_without_raising(self):
        """Why the entry point compares BYTES rather than strings.

        Headers decode as latin-1, so a credential carrying non-ASCII
        characters arrives as a str that compare_digest refuses outright,
        raising TypeError — which would reach the caller as a 500 instead of
        the generic 401. The HTTP client used here rejects such a header before
        it ever reaches the server, so the comparison is exercised directly.
        """
        supplied = "Bearer \u00e9\u00e9\u00e9"
        with pytest.raises(TypeError):
            secrets.compare_digest(supplied, "Bearer x")
        assert secrets.compare_digest(supplied.encode(), b"Bearer x") is False


class TestInvokeThroughTheEntryPoint:
    """The caller-facing behaviour of POST /invoke."""

    def test_the_shipped_smoke_payload_produces_a_report(self, client):
        """The payload the deployment check posts must work against this build."""
        payload = json.loads(_PAYLOAD_PATH.read_text(encoding="utf-8"))
        response = client.post("/invoke", json=payload, headers=_auth())
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success", body
        assert "## 5. Clinical Recommendation" in body["output"]

    def test_caller_records_reach_the_pipeline_through_the_entry_point(self, client):
        response = client.post(
            "/invoke",
            json={
                "input": json.dumps({"query": "early mobilisation after cardiac surgery"}),
                "input_context": {"literature_records": [_RECORD]},
            },
            headers=_auth(),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success", body
        assert "Distinctive caller-supplied trial title" in body["output"]
        assert "Caller-supplied literature records" in body["output"]

    def test_a_rejected_request_returns_no_report(self, client):
        response = client.post(
            "/invoke",
            json={"input": json.dumps({"query": "oncology review", "max_results": "NaN"})},
            headers=_auth(),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] != "success"
        assert not body.get("output")


class TestStructuredChannelBounds:
    """The entry point holds the structured channel to its declared bounds."""

    def test_an_oversized_channel_is_refused_by_size_not_by_content(self, client):
        oversized = {"literature_records": [{"abstract": "x" * (300 * 1024)}]}
        response = client.post(
            "/invoke",
            json={"input": "evidence review", "input_context": oversized},
            headers=_auth(),
        )
        assert response.status_code == 413
        assert "byte limit" in response.json()["detail"]

    def test_a_credential_shaped_value_is_refused_readably_and_names_the_field(self, client):
        """The request cannot succeed either way, so refuse it in a way the caller can act on.

        A credential-shaped string anywhere on this channel is returned verbatim
        by the framework's first node and scanned there, so the run fails at
        node one with an error the caller cannot interpret. Refusing here turns
        that into a named, actionable 400. It is 400 rather than 422 because the
        validation layer owns 422 and answers there with a list of error
        objects, which would make client handling ambiguous.
        """
        response = client.post(
            "/invoke",
            json={
                "input": "evidence review",
                "input_context": {
                    "literature_records": [{"abstract": "Authorization: Bearer abcdefghijklmnop0123456789"}]
                },
            },
            headers=_auth(),
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "input_context.literature_records" in detail
        assert "abcdefghijklmnop" not in detail

    def test_ordinary_domain_text_on_the_same_field_still_passes(self, client):
        """The refusal above must not also block real literature records."""
        response = client.post(
            "/invoke",
            json={"input": "evidence review", "input_context": {"literature_records": [_RECORD]}},
            headers=_auth(),
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_an_undeclared_key_is_dropped_before_the_graph_sees_it(self, client):
        """Validators ignore undeclared keys; ignoring is not stripping.

        An undeclared key survives into the state the framework's first node
        returns, where every value is scanned — so a credential-shaped value
        under a key nothing reads still fails the whole request. Dropping the
        key at the adapter is what makes that impossible.
        """
        response = client.post(
            "/invoke",
            json={
                "input": "evidence review",
                "input_context": {"note": "Bearer abcdefghijklmnop0123456789"},
            },
            headers=_auth(),
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_the_refused_set_matches_the_framework_detector_exactly(self):
        """Anti-drift: this adapter must not be narrower or wider than the framework.

        Scanning per field is exactly equivalent to scanning the mapping — the
        detector's result for a mapping is the union over its values — and that
        identity is what lets the refusal name the field without changing which
        payloads are blocked.
        """
        from framework.security.credential_detector import detect_credentials_in_value
        from src.api import server

        candidates = [
            {"literature_records": [{"abstract": "Bearer abcdefghijklmnop0123456789"}]},
            {"literature_records": [{"abstract": "sk-abcdefghijklmnopqrstuvwxyz0123"}]},
            {"literature_records": [{"abstract": "ordinary clinical abstract"}]},
            {"literature_records": []},
            {},
        ]
        for candidate in candidates:
            refused = server._credential_field(candidate) is not None
            assert refused == bool(detect_credentials_in_value(candidate))
