"""The search fallback is the feature most able to put a wrong URL next to a real
person's name, so the tests below are mostly about what it *refuses* to do.

Every test runs offline against a fake provider: no test in this suite touches a
search engine, so the suite is deterministic and safe to run in CI.
"""

import asyncio
from pathlib import Path

from src.config import Settings
from src.linkedin import (
    BraveHtmlProvider,
    LinkedInResolver,
    Match,
    SearchProvider,
    SearchResult,
    build_provider,
    canonical_profile_url,
    company_name_from_html,
    score_candidate,
)
from src.models import TeamMember
from src.utils import company_tokens, name_tokens


def run(coro):
    return asyncio.run(coro)


class FakeProvider(SearchProvider):
    """Returns canned results and records every query it was asked."""

    name = "fake"

    def __init__(self, results=None, error=None):
        self._results = results if results is not None else []
        self._error = error
        self.queries = []

    async def search(self, query, limit):
        self.queries.append(query)
        if self._error:
            raise self._error
        if callable(self._results):
            return self._results(query)[:limit]
        return list(self._results)[:limit]


def settings(**overrides) -> Settings:
    base = dict(
        groq_api_key="test-key",
        linkedin_search_delay=0.0,
        linkedin_search_max_lookups=5,
        linkedin_min_confidence=0.75,
    )
    base.update(overrides)
    return Settings(**base)


def resolver(provider, **overrides) -> LinkedInResolver:
    return LinkedInResolver(settings(**overrides), provider=provider)


def hit(name="Jane Doe", slug="jane-doe", company="Acme Data", role="CEO"):
    return SearchResult(
        title=f"{name} - {role} - {company} | LinkedIn",
        url=f"https://www.linkedin.com/in/{slug}",
        snippet=f"{name} is the {role} at {company}. View {name}'s profile on LinkedIn.",
    )


# --------------------------------------------------------------------------- #
# URL validation
# --------------------------------------------------------------------------- #


def test_company_pages_are_never_a_person():
    assert canonical_profile_url("https://www.linkedin.com/company/acme-data") is None


def test_profile_urls_are_canonicalised():
    for raw in (
        "http://linkedin.com/in/jane-doe/",
        "https://uk.linkedin.com/in/jane-doe",
        "www.linkedin.com/in/jane-doe?trk=public",
        "https://www.linkedin.com/in/jane-doe",
    ):
        assert canonical_profile_url(raw) == "https://www.linkedin.com/in/jane-doe"


def test_non_linkedin_urls_are_rejected():
    for raw in ("https://linkedin.com.evil.com/in/jane", "https://twitter.com/jane", "", "not a url"):
        assert canonical_profile_url(raw) is None


def test_name_tokens_drop_initials_and_suffixes():
    assert name_tokens("Jane Q. Doe Jr.") == ["jane", "doe"]
    assert name_tokens("José Muñoz") == ["jose", "munoz"]


def test_company_name_is_read_from_the_page_title():
    html = "<html><head><title>Acme Data | Pipelines that never break</title></head></html>"
    assert company_name_from_html(html, "acmedata.io") == "Acme Data"
    # A tagline masquerading as a title falls back to the domain label.
    tagline = "<html><head><title>The fastest way to ship your data pipelines today</title></head></html>"
    assert company_name_from_html(tagline, "acmedata.io") == "acmedata"


# --------------------------------------------------------------------------- #
# Candidate scoring: what gets rejected
# --------------------------------------------------------------------------- #


def test_name_only_match_is_rejected():
    """A namesake with no company or role corroboration is not good enough."""
    member = TeamMember(name="John Smith", title="CEO")
    label, tokens = company_tokens("acmedata.io", "Acme Data")
    result = SearchResult(
        title="John Smith | LinkedIn",
        url="https://www.linkedin.com/in/john-smith-9182",
        snippet="John Smith. Retired. Bournemouth, England.",
    )
    assert score_candidate(member, result, label, tokens) is None


def test_a_different_person_is_rejected():
    member = TeamMember(name="Jane Doe", title="CEO")
    label, tokens = company_tokens("acmedata.io", "Acme Data")
    result = hit(name="Michael Brown", slug="michael-brown")
    assert score_candidate(member, result, label, tokens) is None


def test_company_page_result_is_rejected_even_with_a_perfect_name():
    member = TeamMember(name="Jane Doe", title="CEO")
    label, tokens = company_tokens("acmedata.io", "Acme Data")
    result = SearchResult(
        title="Jane Doe - CEO - Acme Data | LinkedIn",
        url="https://www.linkedin.com/company/acme-data",
    )
    assert score_candidate(member, result, label, tokens) is None


def test_single_word_names_are_never_matched():
    member = TeamMember(name="Prince", title="CEO")
    label, tokens = company_tokens("acmedata.io", "Acme Data")
    assert score_candidate(member, hit(name="Prince", slug="prince"), label, tokens) is None


def test_name_and_company_match_is_accepted_with_evidence():
    member = TeamMember(name="Jane Doe", title="CEO")
    label, tokens = company_tokens("acmedata.io", "Acme Data")
    match = score_candidate(member, hit(), label, tokens)
    assert match is not None
    assert match.url == "https://www.linkedin.com/in/jane-doe"
    assert match.confidence >= 0.75
    assert any("company" in e for e in match.evidence)


def test_accented_names_match_their_folded_slug():
    member = TeamMember(name="José Muñoz", title="CTO")
    label, tokens = company_tokens("acmedata.io", "Acme Data")
    match = score_candidate(member, hit(name="José Muñoz", slug="jose-munoz", role="CTO"), label, tokens)
    assert match is not None


def test_slug_match_scores_higher_than_title_only():
    member = TeamMember(name="Jane Doe", title="CEO")
    label, tokens = company_tokens("acmedata.io", "Acme Data")
    slug_match = score_candidate(member, hit(slug="jane-doe"), label, tokens)
    vanity = score_candidate(member, hit(slug="jd-builds-things-4821"), label, tokens)
    assert slug_match.confidence > vanity.confidence


# --------------------------------------------------------------------------- #
# Resolver: verification of what the model returned
# --------------------------------------------------------------------------- #


def test_model_invented_url_is_discarded():
    """The URL looks perfect and was never on the site. It must not survive."""
    member = TeamMember(name="Jane Doe", title="CEO", linkedin_url="https://www.linkedin.com/in/jane-doe")
    team = run(resolver(FakeProvider()).resolve_team([member], "acmedata.io", verified_profiles=[]))
    assert team[0].linkedin_url is None
    assert team[0].linkedin_source is None
    assert any("discarded" in e for e in team[0].linkedin_evidence)


def test_url_harvested_from_the_site_is_kept():
    member = TeamMember(name="Jane Doe", title="CEO", linkedin_url="https://linkedin.com/in/jane-doe/")
    team = run(
        resolver(FakeProvider()).resolve_team(
            [member], "acmedata.io", verified_profiles=["https://www.linkedin.com/in/jane-doe"]
        )
    )
    assert team[0].linkedin_url == "https://www.linkedin.com/in/jane-doe"
    assert team[0].linkedin_source == "website"
    assert team[0].linkedin_confidence >= 0.9


def test_site_url_that_does_not_match_the_name_is_kept_but_less_confident():
    member = TeamMember(name="Jane Doe", title="CEO", linkedin_url="https://www.linkedin.com/in/jd-4821")
    team = run(
        resolver(FakeProvider()).resolve_team(
            [member], "acmedata.io", verified_profiles=["https://www.linkedin.com/in/jd-4821"]
        )
    )
    assert team[0].linkedin_source == "website"
    assert team[0].linkedin_confidence < 0.9


# --------------------------------------------------------------------------- #
# Resolver: the search fallback itself
# --------------------------------------------------------------------------- #


def test_missing_profile_is_recovered_by_search():
    provider = FakeProvider([hit()])
    member = TeamMember(name="Jane Doe", title="CEO")
    team = run(provider_resolve(provider, [member]))
    assert team[0].linkedin_url == "https://www.linkedin.com/in/jane-doe"
    assert team[0].linkedin_source == "search"
    assert team[0].linkedin_evidence


def provider_resolve(provider, members, domain="acmedata.io", verified=None, **overrides):
    return resolver(provider, **overrides).resolve_team(
        members, domain, verified or [], "Acme Data"
    )


def test_query_contains_the_name_company_and_site_filter():
    provider = FakeProvider([])
    run(provider_resolve(provider, [TeamMember(name="Jane Doe", title="CEO")]))
    query = provider.queries[0]
    assert '"Jane Doe"' in query
    assert '"Acme Data"' in query
    assert "site:linkedin.com/in" in query


def test_weak_results_leave_the_url_empty():
    weak = SearchResult(
        title="Jane Doe | LinkedIn",
        url="https://www.linkedin.com/in/jane-doe-77",
        snippet="Jane Doe. Student at an unrelated university.",
    )
    team = run(provider_resolve(FakeProvider([weak]), [TeamMember(name="Jane Doe", title="CEO")]))
    assert team[0].linkedin_url is None
    assert any("rejected" in e for e in team[0].linkedin_evidence)


def test_evidence_distinguishes_an_unavailable_engine_from_a_real_miss():
    """A null URL because the engine was rate-limited is not the same finding as
    a null URL because nobody matched, and the output must not blur the two."""
    member = TeamMember(name="Jane Doe", title="CEO")

    blocked = run(provider_resolve(FakeProvider(error=RuntimeError("status 429")), [member]))
    assert any("search unavailable" in e and "429" in e for e in blocked[0].linkedin_evidence)

    empty = run(provider_resolve(FakeProvider([]), [member]))
    assert any("no results" in e for e in empty[0].linkedin_evidence)

    weak = SearchResult(title="Jane Doe | LinkedIn", url="https://www.linkedin.com/in/jane-doe", snippet="")
    rejected = run(provider_resolve(FakeProvider([weak]), [member]))
    assert any("rejected" in e for e in rejected[0].linkedin_evidence)

    below = run(provider_resolve(FakeProvider([hit()]), [member], linkedin_min_confidence=0.99))
    assert any("threshold" in e for e in below[0].linkedin_evidence)


def test_confidence_threshold_is_enforced():
    """Raising the bar past what the evidence supports must reject the match."""
    member = TeamMember(name="Jane Doe", title="CEO")
    accepted = run(provider_resolve(FakeProvider([hit()]), [member]))
    assert accepted[0].linkedin_url is not None
    rejected = run(provider_resolve(FakeProvider([hit()]), [member], linkedin_min_confidence=0.99))
    assert rejected[0].linkedin_url is None


def test_best_candidate_wins_over_a_weaker_one():
    weak = SearchResult(
        title="Jane Doe - Acme Data",
        url="https://www.linkedin.com/in/jd-2214",
        snippet="Acme Data",
    )
    team = run(provider_resolve(FakeProvider([weak, hit()]), [TeamMember(name="Jane Doe", title="CEO")]))
    assert team[0].linkedin_url == "https://www.linkedin.com/in/jane-doe"


def test_search_failure_never_breaks_the_record():
    provider = FakeProvider(error=RuntimeError("search engine unreachable"))
    team = run(provider_resolve(provider, [TeamMember(name="Jane Doe", title="CEO")]))
    assert len(team) == 1
    assert team[0].name == "Jane Doe"
    assert team[0].linkedin_url is None


def test_lookup_budget_caps_the_number_of_queries():
    provider = FakeProvider([])
    members = [TeamMember(name=f"Person{i} Example", title="Engineer") for i in range(8)]
    team = run(provider_resolve(provider, members, linkedin_search_max_lookups=3))
    assert len(provider.queries) == 3
    assert len(team) == 8  # everyone is still returned, searched or not


def test_repeated_names_are_only_searched_once():
    provider = FakeProvider([])
    members = [TeamMember(name="Jane Doe", title="CEO"), TeamMember(name="Jane Doe", title="Founder")]
    run(provider_resolve(provider, members))
    assert len(provider.queries) == 1


def test_team_order_and_membership_are_preserved():
    provider = FakeProvider(lambda q: [hit()] if "Jane Doe" in q else [])
    members = [
        TeamMember(name="Alan Turing", title="CTO"),
        TeamMember(name="Jane Doe", title="CEO"),
        TeamMember(name="Grace Hopper", title="VP Engineering"),
    ]
    team = run(provider_resolve(provider, members))
    assert [m.name for m in team] == ["Alan Turing", "Jane Doe", "Grace Hopper"]
    assert team[1].linkedin_url == "https://www.linkedin.com/in/jane-doe"
    assert team[0].linkedin_url is None


def test_search_is_skipped_for_people_the_site_already_linked():
    provider = FakeProvider([hit()])
    member = TeamMember(name="Jane Doe", title="CEO", linkedin_url="https://www.linkedin.com/in/jane-doe")
    run(provider_resolve(provider, [member], verified=["https://www.linkedin.com/in/jane-doe"]))
    assert provider.queries == []


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


def test_search_can_be_switched_off_entirely():
    assert build_provider(None, settings(linkedin_search_provider="none")) is None
    assert build_provider(None, settings(linkedin_search_enabled=False)) is None


def test_an_api_provider_is_preferred_over_the_keyless_scraper():
    assert build_provider(None, settings(serper_api_key="k")).name == "serper"
    assert build_provider(None, settings(brave_api_key="k")).name == "brave-api"
    assert build_provider(None, settings(serper_api_key="k", brave_api_key="k")).name == "serper"
    assert build_provider(None, settings()).name == "brave-html"


def test_an_explicitly_named_provider_without_its_key_is_not_silently_swapped():
    assert build_provider(None, settings(linkedin_search_provider="serper")) is None
    assert build_provider(None, settings(linkedin_search_provider="brave-api")) is None


def test_disabled_resolver_returns_the_team_untouched():
    team = run(
        LinkedInResolver(settings()).resolve_team(
            [TeamMember(name="Jane Doe", title="CEO")], "acmedata.io", []
        )
    )
    assert len(team) == 1 and team[0].linkedin_url is None


def test_redirect_wrapped_result_links_are_unwrapped():
    body = """
    <html><body><div>
      <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fjane-doe">
        Jane Doe - CEO - Acme Data
      </a>
      <p>Jane Doe is the CEO at Acme Data and leads the engineering team there.</p>
    </div></body></html>
    """
    results = BraveHtmlProvider(None, 10.0)._parse(body)
    assert results[0].url == "https://www.linkedin.com/in/jane-doe"


def test_non_linkedin_results_are_ignored_by_the_parser():
    body = '<html><body><a href="https://example.com/team">Acme Data team page</a></body></html>'
    assert BraveHtmlProvider(None, 10.0)._parse(body) == []


# --------------------------------------------------------------------------- #
# End to end against a real captured results page
# --------------------------------------------------------------------------- #

SERP = (Path(__file__).parent / "fixtures" / "brave_serp.html").read_text(encoding="utf-8")


def test_real_results_page_yields_the_right_profile():
    """Parse a genuine SERP and match it, with no network and no LLM involved."""
    parsed = BraveHtmlProvider(None, 10.0)._parse(SERP)
    label, tokens = company_tokens("postman.com", "Postman")
    member = TeamMember(name="Ankit Sobti", title="Co-Founder & CTO")

    matches = [m for m in (score_candidate(member, r, label, tokens) for r in parsed) if m]
    assert len(matches) == 1
    assert matches[0].url == "https://www.linkedin.com/in/ankit-sobti"
    assert matches[0].confidence >= 0.9


def test_linkedin_chrome_links_on_a_real_page_are_all_rejected():
    """Login, Games and 'Top Content' links live on every SERP. None is a person."""
    parsed = BraveHtmlProvider(None, 10.0)._parse(SERP)
    assert len(parsed) > 1  # the page really does contain the distractors
    label, tokens = company_tokens("postman.com", "Postman")
    member = TeamMember(name="Ankit Sobti", title="Co-Founder & CTO")
    for result in parsed:
        if "/in/ankit-sobti" in result.url:
            continue
        assert score_candidate(member, result, label, tokens) is None


def test_a_real_page_does_not_match_someone_who_is_not_on_it():
    parsed = BraveHtmlProvider(None, 10.0)._parse(SERP)
    label, tokens = company_tokens("postman.com", "Postman")
    stranger = TeamMember(name="Jane Doe", title="CEO")
    assert all(score_candidate(stranger, r, label, tokens) is None for r in parsed)


# --------------------------------------------------------------------------- #
# Through the pipeline
# --------------------------------------------------------------------------- #


def test_pipeline_writes_a_searched_profile_into_the_record():
    """The whole path, with the model and the network faked but the SERP real.

    A team member the site names but does not link is carried through extraction,
    matched against a genuine captured results page, and lands in the output
    record with its source and evidence attached.
    """
    from tests.test_resilience import SETTINGS, FakeExtractor, FakeFetcher
    from src.models import LLMExtraction
    from src.pipeline import enrich_domain

    extraction = LLMExtraction(
        company_overview="Postman is an API platform. Teams use it to design and test APIs.",
        target_audience="Developers building and operating APIs.",
        contact_emails=[],
        team_members=[TeamMember(name="Ankit Sobti", title="Co-Founder & CTO")],
        self_reported_confidence=0.8,
    )

    class SerpProvider(SearchProvider):
        name = "captured-serp"

        async def search(self, query, limit):
            return BraveHtmlProvider(None, 10.0)._parse(SERP)[:limit]

    record = run(
        enrich_domain(
            "postman.com",
            FakeFetcher({"*": "ok"}),
            FakeExtractor(extraction),
            SETTINGS,
            LinkedInResolver(settings(), provider=SerpProvider()),
        )
    )

    member = record.team_members[0]
    assert member.linkedin_url == "https://www.linkedin.com/in/ankit-sobti"
    assert member.linkedin_source == "search"
    assert member.linkedin_confidence >= 0.9
    assert member.linkedin_evidence
    # and the record still serialises cleanly for output.json
    assert "linkedin_source" in record.model_dump_json()


def test_pipeline_survives_a_resolver_that_explodes():
    from tests.test_resilience import SETTINGS, FakeExtractor, FakeFetcher
    from src.models import LLMExtraction
    from src.pipeline import enrich_domain

    extraction = LLMExtraction(
        company_overview="Postman is an API platform. Teams use it to design and test APIs.",
        target_audience="Developers building and operating APIs.",
        team_members=[TeamMember(name="Ankit Sobti", title="CTO")],
        self_reported_confidence=0.8,
    )

    class Exploding(LinkedInResolver):
        async def resolve_team(self, *args, **kwargs):
            raise RuntimeError("resolver exploded")

    record = run(
        enrich_domain(
            "postman.com", FakeFetcher({"*": "ok"}), FakeExtractor(extraction),
            SETTINGS, Exploding(settings()),
        )
    )
    # The extraction we already paid for is kept; the failure is reported, not fatal.
    assert record.status in ("success", "partial")
    assert record.team_members[0].name == "Ankit Sobti"
    assert any("linkedin resolution failed" in e for e in record.errors)


def test_people_beyond_the_budget_say_so_rather_than_looking_like_a_miss():
    provider = FakeProvider([])
    members = [TeamMember(name=f"Person{i} Example", title="Engineer") for i in range(5)]
    team = run(provider_resolve(provider, members, linkedin_search_max_lookups=2))
    assert all("budget" in " ".join(m.linkedin_evidence) for m in team[2:])
    assert all("budget" not in " ".join(m.linkedin_evidence) for m in team[:2])


def test_a_disabled_search_is_recorded_as_such():
    team = run(
        LinkedInResolver(settings()).resolve_team(
            [TeamMember(name="Jane Doe", title="CEO")], "acmedata.io", []
        )
    )
    assert any("disabled" in e for e in team[0].linkedin_evidence)


# --------------------------------------------------------------------------- #
# Reading a job title off the result that proved the profile
# --------------------------------------------------------------------------- #

from src.linkedin import role_from_result  # noqa: E402


def role_for(name, title, domain="supabase.com", company="Supabase"):
    label, tokens = company_tokens(domain, company)
    return role_from_result(name, SearchResult(title=title, url=""), label, tokens)


def test_a_role_is_read_from_a_real_result_title():
    assert role_for("Jon Meyers", "Jon Meyers - DevRel Engineer - LinkedIn") == "DevRel Engineer"
    assert role_for(
        "Tyler Hillery", "Tyler Hillery - Software Engineer (Storage) @ Supabase - LinkedIn"
    ) == "Software Engineer (Storage)"
    assert role_for(
        "Amy Quek", "Amy Quek - Marketing Operations at Supabase - LinkedIn Singapore"
    ) == "Marketing Operations"


def test_the_employer_is_stripped_only_when_it_is_the_employer():
    """'Co-Founder, Postman' loses the company; 'VP, Engineering' must not lose half its title."""
    assert role_for("Ankit Sobti", "Ankit Sobti - Co-Founder, Postman | LinkedIn",
                    "postman.com", "Postman") == "Co-Founder"
    assert role_for("Jane Doe", "Jane Doe - VP, Engineering - LinkedIn") == "VP, Engineering"


def test_a_title_with_no_role_in_it_yields_nothing():
    """'Beng Eu - Supabase - LinkedIn' states an employer, not a job."""
    assert role_for("Beng Eu", "Beng Eu - Supabase - LinkedIn Singapore") is None


def test_a_fuller_spelling_of_the_name_is_not_a_job_title():
    """LinkedIn lists 'Chris Martin' as 'Christopher (Chris) Martin'. That is a name."""
    assert role_for("Chris Martin", "Christopher (Chris) Martin - Supabase - LinkedIn") is None
    assert role_for("Chris Martin", "Chris Martin - Christopher (Chris) Martin - LinkedIn") is None


def test_page_furniture_is_never_a_role():
    for title in ("Jane Doe | LinkedIn", "Jane Doe - Sign in - LinkedIn", "Jane Doe - Profile"):
        assert role_for("Jane Doe", title) is None


def test_a_sentence_is_not_a_role():
    long = ("Jane Doe - Jane has spent the last fifteen years building teams across "
            "three continents and is now - LinkedIn")
    assert role_for("Jane Doe", long) is None


def test_a_missing_title_is_filled_from_the_matched_result_and_marked_as_such():
    provider = FakeProvider([SearchResult(
        title="Jane Doe - Head of Product at Acme Data - LinkedIn",
        url="https://www.linkedin.com/in/jane-doe",
        snippet="Head of Product at Acme Data.",
    )])
    team = run(provider_resolve(provider, [TeamMember(name="Jane Doe")]))
    assert team[0].title == "Head of Product"
    assert team[0].title_source == "search"
    assert any("title read from" in e for e in team[0].linkedin_evidence)


def test_a_title_the_site_stated_is_never_overwritten_by_search():
    provider = FakeProvider([SearchResult(
        title="Jane Doe - Senior Widget Polisher at Acme Data - LinkedIn",
        url="https://www.linkedin.com/in/jane-doe", snippet="Acme Data",
    )])
    team = run(provider_resolve(provider, [TeamMember(name="Jane Doe", title="CEO")]))
    assert team[0].title == "CEO"
    assert team[0].title_source == "website"
