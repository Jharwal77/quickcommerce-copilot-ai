import io

import httpx

from qc_copilot.llm.preflight import check_provider, run_preflight


def transport_returning(status_code: int, body: str = "") -> httpx.MockTransport:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(status_code, text=body)

    transport = httpx.MockTransport(handler)
    transport.seen = seen  # type: ignore[attr-defined]
    return transport


def test_valid_key_lists_models():
    transport = transport_returning(200, '{"data": []}')
    check = check_provider("groq", "k", transport)
    assert check.status == "ok" and check.usable
    request = transport.seen["request"]  # type: ignore[attr-defined]
    assert request.url.path.endswith("/models")
    assert request.headers["Authorization"] == "Bearer k"


def test_gemini_bad_key_is_a_400_with_api_key_text():
    body = '[{"error": {"code": 400, "message": "Please pass a valid API key"}}]'
    check = check_provider("gemini", "bad", transport_returning(400, body))
    assert check.status == "rejected"
    assert "valid API key" in check.detail
    assert "bad" not in check.detail


def test_401_is_rejected_and_429_is_usable():
    assert (
        check_provider("groq", "k", transport_returning(401, "Invalid API Key")).status
        == "rejected"
    )
    limited = check_provider("groq", "k", transport_returning(429, "slow down"))
    assert limited.status == "rate_limited" and limited.usable


def test_openrouter_uses_the_key_endpoint():
    transport = transport_returning(200, "{}")
    check_provider("openrouter", "k", transport)
    assert transport.seen["request"].url.path.endswith("/auth/key")  # type: ignore[attr-defined]


def test_network_failure_is_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    check = check_provider("groq", "k", httpx.MockTransport(handler))
    assert check.status == "unreachable" and not check.usable


def test_missing_key_is_reported_without_a_request():
    assert check_provider("hf", "").status == "missing"


def test_run_preflight_fails_only_on_required_unusable_providers():
    out = io.StringIO()
    keys = {"groq": "k", "gemini": "k", "openrouter": "", "hf": "", "github": ""}
    rejected = transport_returning(401, "nope")
    assert run_preflight(keys, ["groq"], rejected, out) == 1
    assert run_preflight(keys, [], rejected, io.StringIO()) == 0
    assert run_preflight(keys, ["groq", "gemini"], transport_returning(200), io.StringIO()) == 0
    lines = out.getvalue().splitlines()
    assert len(lines) == 2 and all("rejected" in line for line in lines)


def test_required_but_unconfigured_provider_blocks():
    keys = {"groq": "", "gemini": "", "openrouter": "", "hf": "", "github": ""}
    out = io.StringIO()
    assert run_preflight(keys, ["gemini"], transport_returning(200), out) == 1
    assert "gemini" in out.getvalue() and "missing" in out.getvalue()
