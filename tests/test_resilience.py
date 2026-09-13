"""Resilience: the run must survive anything a website can do to it.

These tests drive the pipeline with fake fetchers and extractors that fail in
every way a real run can fail - dead homepages, exploding HTTP clients, broken
sub-pages, a model that never returns valid JSON - and assert that the process
always produces a well-formed record instead of a traceback.
"""

import asyncio
import json
from pathlib import Path

from src.config import Settings
from src.fetcher import FetchResult, Fetcher
from src.models import LLMExtraction, UsageRecord
from src.pipeline import enrich_domain, load_domains, run_pipeline

SETTINGS = Settings(groq_api_key="test-key", max_pages_per_domain=3, politeness_delay=0.0)

GOOD_HTML = """
<html><body>
<h1>Acme Data</h1>
<p>Acme Data Systems builds a managed streaming database used by engineering teams
   at more than four thousand companies to move data in real time.</p>
<p>Contact <a href="mailto:hello@acme.io">hello@acme.io</a> or
   <a href="https://linkedin.com/in/priya">Priya Raghavan, CEO</a>.</p>
<a href="/about">About</a><a href="/team">Team</a><a href="/contact">Contact</a>
</body></html>
"""

VALID_EXTRACTION = LLMExtraction(
    company_overview="Acme builds a managed streaming database. Engineering teams use it for real-time data.",
    target_audience="Backend engineers building real-time data applications.",
    contact_emails=["hello@acme.io"],
    team_members=[],
    self_reported_confidence=0.7,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeClient:
    """An HTTP client that always fails - discovery must cope."""

    def __init__(self, exc=RuntimeError("network down")):
        self.exc = exc

    async def get(self, *args, **kwargs):
        raise self.exc


class FakeFetcher:
    def __init__(self, behaviour):
        """``behaviour`` maps url -> 'ok' | 'fail' | 'raise'; default for unknown urls is 'ok'."""
        self.behaviour = behaviour
        self.client = FakeClient()

    async def fetch(self, url, clean_fn):
        mode = self.behaviour.get(url, self.behaviour.get("*", "ok"))
        if mode == "raise":
            raise RuntimeError(f"fetcher exploded on {url}")
        if mode == "fail":
            return FetchResult(url, "", "none", 503, "site unreachable", 0.1), ""
        # Per-URL content, so the cross-page boilerplate pass has something real
        # to distinguish. Serving byte-identical pages would be an unrealistic
        # input that the safety floor in the cleaner handles separately.
        html = GOOD_HTML.replace(
            "<h1>Acme Data</h1>",
            f"<h1>Acme Data</h1><p>This is unique content for the page at {url}, "
            f"describing that section of the business in enough detail to matter.</p>",
        )
        return FetchResult(url, html, "http", 200, None, 0.1), clean_fn(html)


class FakeExtractor:
    def __init__(self, result=VALID_EXTRACTION, errors=None, attempts=1, raises=False):
        self.result, self.errors, self.attempts, self.raises = result, errors or [], attempts, raises

    async def extract(self, domain, pages, harvested):
        if self.raises:
            raise RuntimeError("groq exploded")
        return self.result, UsageRecord(llm_calls=1, total_tokens=1200,
                                        estimated_cost_usd=0.0009, model="test"), self.errors, self.attempts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Per-domain isolation
# --------------------------------------------------------------------------- #

def test_happy_path_produces_a_scored_record():
    record = run(enrich_domain("acme.io", FakeFetcher({"*": "ok"}), FakeExtractor(), SETTINGS))
    assert record.status in ("success", "partial")
    assert record.company_overview
    assert record.data_confidence_score > 0
    assert record.usage.total_tokens == 1200


def test_dead_homepage_yields_failed_record_not_an_exception():
    record = run(enrich_domain("dead.test", FakeFetcher({"*": "fail"}), FakeExtractor(), SETTINGS))
    assert record.status == "failed"
    assert record.domain == "dead.test"
    assert any("homepage unreachable" in e for e in record.errors)
    assert record.model_dump_json()


def test_exploding_fetcher_is_contained():
    record = run(enrich_domain("boom.test", FakeFetcher({"*": "raise"}), FakeExtractor(), SETTINGS))
    assert record.status == "failed"
    assert any("unhandled" in e or "exploded" in e for e in record.errors)


def test_exploding_llm_is_contained():
    record = run(enrich_domain("acme.io", FakeFetcher({"*": "ok"}), FakeExtractor(raises=True), SETTINGS))
    assert record.status == "failed"
    assert record.errors


def test_llm_returning_nothing_still_writes_a_record():
    extractor = FakeExtractor(result=None, errors=["schema validation failed on attempt 3"], attempts=3)
    record = run(enrich_domain("acme.io", FakeFetcher({"*": "ok"}), extractor, SETTINGS))
    assert record.status == "failed"
    assert record.data_confidence_score == 0.0
    assert record.usage.total_tokens == 1200  # a failed extraction still cost money


def test_broken_subpages_downgrade_to_partial_not_failed():
    fetcher = FakeFetcher({
        "https://acme.io": "ok",
        "https://acme.io/about": "fail",
        "https://acme.io/team": "fail",
        "*": "ok",
    })
    record = run(enrich_domain("acme.io", fetcher, FakeExtractor(), SETTINGS))
    assert record.status in ("success", "partial")
    assert record.company_overview, "partial data must still be returned"


def test_invalid_domain_string_is_handled():
    for bad in ("", "   ", "!!!"):
        record = run(enrich_domain(bad, FakeFetcher({"*": "ok"}), FakeExtractor(), SETTINGS))
        assert record.status == "failed"


def test_discovery_failure_falls_back_to_homepage_only():
    """The fake client always raises, so discovery cannot contribute any URLs."""
    record = run(enrich_domain("acme.io", FakeFetcher({"*": "ok"}), FakeExtractor(), SETTINGS))
    assert record.company_overview
    assert len(record.pages_crawled) >= 1


# --------------------------------------------------------------------------- #
# Whole-run behaviour
# --------------------------------------------------------------------------- #

def test_one_dead_domain_cannot_stop_the_others(tmp_path, monkeypatch):
    class PatchedFetcher(FakeFetcher):
        def __init__(self, settings):
            super().__init__({"https://dead.test": "fail", "*": "ok"})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

    monkeypatch.setattr("src.pipeline.Fetcher", PatchedFetcher)
    monkeypatch.setattr("src.pipeline.GroqExtractor", lambda settings: FakeExtractor())

    out = tmp_path / "out.json"
    summary = run(run_pipeline(["acme.io", "dead.test", "other.io"], SETTINGS, str(out)))

    assert summary.domains_requested == 3
    assert summary.domains_failed == 1
    assert summary.domains_succeeded + summary.domains_partial == 2
    assert len(summary.results) == 3


def test_output_file_is_always_written_and_valid(tmp_path, monkeypatch):
    class AllDead(FakeFetcher):
        def __init__(self, settings):
            super().__init__({"*": "fail"})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

    monkeypatch.setattr("src.pipeline.Fetcher", AllDead)
    monkeypatch.setattr("src.pipeline.GroqExtractor", lambda settings: FakeExtractor())

    out = tmp_path / "out.json"
    run(run_pipeline(["a.test", "b.test"], SETTINGS, str(out)))

    assert out.exists(), "an output file must exist even when every domain fails"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["domains_failed"] == 2
    assert len(data["results"]) == 2
    assert all(r["status"] == "failed" for r in data["results"])


def test_input_order_is_preserved(tmp_path, monkeypatch):
    class OK(FakeFetcher):
        def __init__(self, settings):
            super().__init__({"*": "ok"})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

    monkeypatch.setattr("src.pipeline.Fetcher", OK)
    monkeypatch.setattr("src.pipeline.GroqExtractor", lambda settings: FakeExtractor())

    out = tmp_path / "out.json"
    summary = run(run_pipeline(["c.io", "a.io", "b.io"], SETTINGS, str(out)))
    assert [r.domain for r in summary.results] == ["c.io", "a.io", "b.io"]


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #

def test_load_domains_dedupes_across_url_forms():
    assert load_domains(["postman.com", "https://www.postman.com/", "POSTMAN.com"], None) == ["postman.com"]


def test_load_domains_reads_files_and_skips_comments(tmp_path):
    f = tmp_path / "domains.txt"
    f.write_text("# targets\npostman.com\nsupabase.com,extra-column\n\n", encoding="utf-8")
    assert load_domains(None, str(f)) == ["postman.com", "supabase.com"]


# --------------------------------------------------------------------------- #
# Escalation logic
# --------------------------------------------------------------------------- #

def test_escalates_on_thin_content():
    result = FetchResult("https://a.com", "<html></html>", "http", 200, None, 0.1)
    assert Fetcher.should_escalate(result, 20) is not None


def test_escalates_on_bot_block():
    result = FetchResult("https://a.com", "blocked", "http", 403, None, 0.1)
    assert Fetcher.should_escalate(result, 5000) is not None


def test_does_not_escalate_on_good_html_page():
    result = FetchResult("https://a.com", "x" * 20000, "http", 200, None, 0.1)
    assert Fetcher.should_escalate(result, 5000) is None
