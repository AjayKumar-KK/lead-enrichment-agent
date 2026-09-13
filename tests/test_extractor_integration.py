"""End-to-end test of the Groq client against a local fake API.

Everything else in this suite uses stubs. This module exercises the real HTTP
path - auth header, request body, ``response_format``, usage parsing, markdown
fence stripping, the validation-retry loop and the error branches - by standing up
a throwaway HTTP server that speaks the OpenAI-compatible shape Groq uses.

It needs no API key, no internet and no network permissions beyond localhost, so
it runs identically on a laptop and in CI.
"""

import asyncio
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.config import Settings
from src.extractor import GroqExtractor, build_user_prompt

VALID_PAYLOAD = {
    "company_overview": "Acme builds a managed streaming database. Engineering teams use it to move data in real time.",
    "target_audience": "Backend engineers building real-time data applications.",
    "contact_emails": ["hello@acme.io", "sales@acme.io"],
    "team_members": [
        {"name": "Priya Raghavan", "title": "CEO", "linkedin_url": "https://linkedin.com/in/priya"}
    ],
    "self_reported_confidence": 0.85,
}

PAGES = {
    "https://acme.io": "Acme Data Systems builds a managed streaming database for engineering teams.",
    "https://acme.io/team": "Priya Raghavan, Chief Executive Officer. Daniel Okoye, CTO.",
}

HARVESTED = {
    "emails": ["hello@acme.io", "sales@acme.io"],
    "linkedin_profiles": ["https://linkedin.com/in/priya"],
    "linkedin_companies": [],
}


class FakeGroq(BaseHTTPRequestHandler):
    """Serves a scripted sequence of responses so retry paths can be driven."""

    script: list = []
    calls: list = []

    def log_message(self, *args):  # silence the default stderr logging
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeGroq.calls.append({"body": body, "auth": self.headers.get("authorization", "")})

        index = min(len(FakeGroq.calls) - 1, len(FakeGroq.script) - 1)
        status, content = FakeGroq.script[index]

        if status != 200:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "scripted failure"}}).encode())
            return

        payload = {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(script):
    """Start the fake API on an ephemeral port and return (settings, shutdown)."""
    FakeGroq.script = script
    FakeGroq.calls = []
    server = HTTPServer(("127.0.0.1", 0), FakeGroq)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    settings = Settings(
        groq_api_key="test-key-123",
        groq_base_url=f"http://127.0.0.1:{port}",
        model="llama-3.3-70b-versatile",
        llm_max_attempts=3,
        llm_timeout=10.0,
    )
    return settings, server.shutdown


def extract(settings):
    async def go():
        async with GroqExtractor(settings) as extractor:
            return await extractor.extract("acme.io", PAGES, HARVESTED)

    return asyncio.run(go())


# --------------------------------------------------------------------------- #


def test_successful_extraction_over_real_http():
    settings, shutdown = serve([(200, json.dumps(VALID_PAYLOAD))])
    try:
        result, usage, errors, attempts = extract(settings)
    finally:
        shutdown()

    assert result is not None
    assert result.company_overview.startswith("Acme builds a managed streaming database")
    assert result.contact_emails == ["hello@acme.io", "sales@acme.io"]
    assert len(result.team_members) == 1
    assert attempts == 1
    assert errors == []
    assert usage.total_tokens == 1200
    assert usage.llm_calls == 1


def test_request_is_well_formed():
    settings, shutdown = serve([(200, json.dumps(VALID_PAYLOAD))])
    try:
        extract(settings)
    finally:
        shutdown()

    call = FakeGroq.calls[0]
    assert call["auth"] == "Bearer test-key-123"
    assert call["body"]["model"] == "llama-3.3-70b-versatile"
    assert call["body"]["response_format"] == {"type": "json_object"}
    assert call["body"]["temperature"] <= 0.2
    # The generated JSON Schema must actually reach the model.
    assert "company_overview" in call["body"]["messages"][-1]["content"]


def test_cost_is_calculated_from_reported_usage():
    settings, shutdown = serve([(200, json.dumps(VALID_PAYLOAD))])
    try:
        _, usage, _, _ = extract(settings)
    finally:
        shutdown()

    in_price, out_price = settings.price_for()
    expected = 1000 / 1_000_000 * in_price + 200 / 1_000_000 * out_price
    assert abs(usage.estimated_cost_usd - expected) < 1e-9


def test_markdown_fences_are_stripped():
    fenced = "```json\n" + json.dumps(VALID_PAYLOAD) + "\n```"
    settings, shutdown = serve([(200, fenced)])
    try:
        result, _, _, attempts = extract(settings)
    finally:
        shutdown()

    assert result is not None
    assert attempts == 1


def test_invalid_json_triggers_a_corrective_retry():
    settings, shutdown = serve([
        (200, "here you go: {not json at all"),
        (200, json.dumps(VALID_PAYLOAD)),
    ])
    try:
        result, usage, errors, attempts = extract(settings)
    finally:
        shutdown()

    assert result is not None, "the second attempt should succeed"
    assert attempts == 2
    assert any("invalid json" in e for e in errors)
    assert usage.llm_calls == 2
    assert usage.total_tokens == 2400, "both attempts must be billed"
    # The failed response and a correction instruction were appended to the thread.
    assert len(FakeGroq.calls[1]["body"]["messages"]) > len(FakeGroq.calls[0]["body"]["messages"])


def test_schema_violation_triggers_a_corrective_retry():
    broken = dict(VALID_PAYLOAD, company_overview="N/A", self_reported_confidence=5)
    settings, shutdown = serve([
        (200, json.dumps(broken)),
        (200, json.dumps(VALID_PAYLOAD)),
    ])
    try:
        result, _, errors, attempts = extract(settings)
    finally:
        shutdown()

    assert result is not None
    assert attempts == 2
    assert any("schema validation failed" in e for e in errors)
    # The specific validation errors must be fed back, not a generic retry.
    correction = FakeGroq.calls[1]["body"]["messages"][-1]["content"]
    assert "company_overview" in correction or "self_reported_confidence" in correction


def test_gives_up_cleanly_after_max_attempts():
    settings, shutdown = serve([(200, "still not json")] * 5)
    try:
        result, usage, errors, attempts = extract(settings)
    finally:
        shutdown()

    assert result is None, "no data is better than invented data"
    assert attempts == settings.llm_max_attempts
    assert len(errors) >= settings.llm_max_attempts
    assert usage.total_tokens > 0, "failed attempts still cost money and must be reported"


def test_bad_api_key_surfaces_a_clear_message():
    settings, shutdown = serve([(401, "")])
    try:
        _, _, errors, _ = extract(settings)
    finally:
        shutdown()

    assert any("401" in e or "key" in e.lower() for e in errors)


def test_unknown_model_surfaces_a_clear_message():
    settings, shutdown = serve([(404, "")])
    try:
        _, _, errors, _ = extract(settings)
    finally:
        shutdown()

    assert any("not found" in e.lower() or "list-models" in e for e in errors)


def test_missing_api_key_is_rejected_before_any_request():
    try:
        GroqExtractor(Settings(groq_api_key=""))
    except ValueError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("an empty API key must raise")


# --------------------------------------------------------------------------- #
# Token budget
# --------------------------------------------------------------------------- #


def test_prompt_respects_the_character_budget():
    settings = Settings(groq_api_key="x", max_chars_per_page=500, max_chars_per_domain=1200)
    huge = {f"https://acme.io/p{i}": "word " * 5000 for i in range(10)}
    prompt = build_user_prompt("acme.io", huge, HARVESTED, settings)

    content = prompt.split("WEBSITE CONTENT:")[1]
    # Page headers add a little overhead; the content itself must stay in budget.
    assert len(content) < settings.max_chars_per_domain * 1.5


def test_verified_lists_are_always_included():
    settings = Settings(groq_api_key="x")
    prompt = build_user_prompt("acme.io", PAGES, HARVESTED, settings)
    assert "VERIFIED EMAILS" in prompt
    assert "hello@acme.io" in prompt
    assert "VERIFIED LINKEDIN PROFILES" in prompt


def test_empty_harvest_still_produces_a_valid_prompt():
    settings = Settings(groq_api_key="x")
    empty = {"emails": [], "linkedin_profiles": [], "linkedin_companies": []}
    prompt = build_user_prompt("acme.io", PAGES, empty, settings)
    assert "(none found)" in prompt


def test_model_override_is_honoured():
    settings, shutdown = serve([(200, json.dumps(VALID_PAYLOAD))])
    settings = replace(settings, model="llama-3.1-8b-instant")
    try:
        extract(settings)
    finally:
        shutdown()
    assert FakeGroq.calls[0]["body"]["model"] == "llama-3.1-8b-instant"
