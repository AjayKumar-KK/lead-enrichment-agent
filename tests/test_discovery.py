"""URL scoring decides what the LLM ever sees, so the ranking must be right."""

from src.config import Settings
from src.discovery import score_url, urls_from_html
from src.utils import normalise_domain, normalise_url, same_registrable_domain, looks_like_html

SETTINGS = Settings()


def test_homepage_scores_highest():
    assert score_url("https://acme.com/", SETTINGS) == 100.0
    assert score_url("https://acme.com", SETTINGS) == 100.0


def test_relevant_pages_score_positive():
    for path in ("/about", "/team", "/contact", "/company", "/leadership", "/our-story"):
        assert score_url(f"https://acme.com{path}", SETTINGS) > 0, path


def test_irrelevant_pages_score_negative():
    for path in ("/blog/why-we-switched-to-rust", "/docs/api/v2", "/privacy", "/login", "/terms"):
        assert score_url(f"https://acme.com{path}", SETTINGS) < 0, path


def test_about_outranks_pricing():
    about = score_url("https://acme.com/about", SETTINGS)
    pricing = score_url("https://acme.com/pricing", SETTINGS)
    assert about > pricing


def test_deep_paths_are_penalised():
    shallow = score_url("https://acme.com/team", SETTINGS)
    deep = score_url("https://acme.com/company/culture/people/team", SETTINGS)
    assert shallow > deep


def test_unconventional_naming_still_found():
    """The point of scoring over a hard-coded path list."""
    assert score_url("https://acme.com/who-we-are", SETTINGS) > 0
    assert score_url("https://acme.com/company/leadership", SETTINGS) > 0


def test_anchor_extraction_resolves_relative_urls():
    html = """
    <a href="/about">About</a>
    <a href="team">Team</a>
    <a href="https://acme.com/contact">Contact</a>
    <a href="#main">Skip</a>
    <a href="mailto:x@acme.com">Email</a>
    <a href="javascript:void(0)">JS</a>
    """
    urls = urls_from_html("https://acme.com/", html)
    assert "https://acme.com/about" in urls
    assert "https://acme.com/team" in urls
    assert not any(u.startswith(("mailto:", "javascript:", "#")) for u in urls)


def test_anchor_extraction_never_raises_on_garbage():
    assert urls_from_html("https://acme.com/", "<<<>>>") == []


def test_url_normalisation():
    assert normalise_url("https://Acme.com/About/?utm=x#top") == "https://acme.com/About"
    assert normalise_url("https://acme.com//a//b/") == "https://acme.com/a/b"


def test_domain_normalisation():
    assert normalise_domain("https://www.Postman.com/pricing/") == "postman.com"
    assert normalise_domain("vapi.ai") == "vapi.ai"
    assert normalise_domain("http://sub.acme.co.uk:8080/x") == "sub.acme.co.uk"


def test_subdomains_count_as_same_site_but_other_hosts_do_not():
    assert same_registrable_domain("https://blog.acme.com/x", "acme.com")
    assert same_registrable_domain("https://www.acme.com/x", "acme.com")
    assert not same_registrable_domain("https://evil.com/acme.com", "acme.com")


def test_asset_urls_are_rejected():
    assert not looks_like_html("https://acme.com/whitepaper.pdf")
    assert not looks_like_html("https://acme.com/app.js")
    assert looks_like_html("https://acme.com/about")


# --------------------------------------------------------------------------- #
# Subdomains
# --------------------------------------------------------------------------- #
# A root URL and a subdomain root look identical to a path-only scorer, and every
# subdomain scored 100 - the value reserved for the homepage. Real runs spent
# their whole page budget on status pages and Discord redirects and never opened
# an /about page.


def test_only_the_real_homepage_scores_as_the_homepage():
    assert score_url("https://supabase.com/", SETTINGS, "supabase.com") == 100.0
    assert score_url("https://www.supabase.com/", SETTINGS, "supabase.com") == 100.0
    for subdomain in ("https://status.supabase.com/", "https://discord.supabase.com/",
                      "https://blog.supabase.com/"):
        assert score_url(subdomain, SETTINGS, "supabase.com") < 100.0, subdomain


def test_noise_subdomains_are_dropped_entirely():
    """Scores of zero or below never make the crawl list."""
    for subdomain in ("https://status.acme.io/", "https://blog.acme.io/",
                      "https://docs.acme.io/", "https://discord.acme.io/"):
        assert score_url(subdomain, SETTINGS, "acme.io") <= 0, subdomain


def test_a_useful_subdomain_is_still_worth_reading():
    assert score_url("https://careers.acme.io/", SETTINGS, "acme.io") > 0
    assert score_url("https://about.acme.io/", SETTINGS, "acme.io") > 0


def test_team_pages_now_outrank_subdomain_roots():
    """The whole point: /about and /team must win the budget."""
    about = score_url("https://supabase.com/about", SETTINGS, "supabase.com")
    team = score_url("https://supabase.com/team", SETTINGS, "supabase.com")
    for noise in ("https://status.supabase.com/", "https://discord.supabase.com/"):
        assert about > score_url(noise, SETTINGS, "supabase.com"), noise
        assert team > score_url(noise, SETTINGS, "supabase.com"), noise


def test_a_subpage_on_a_subdomain_ranks_below_the_same_page_on_the_main_site():
    main = score_url("https://acme.io/about", SETTINGS, "acme.io")
    sub = score_url("https://eu.acme.io/about", SETTINGS, "acme.io")
    assert main > sub


def test_scoring_without_a_domain_still_works():
    """The argument is optional, so older callers keep their behaviour."""
    assert score_url("https://acme.io/team", SETTINGS) > 0


def test_www_and_bare_hosts_are_the_same_url():
    """A real run crawled postman.com twice - once with www, once without -
    spending a page of the budget and an LLM context on a byte-identical copy."""
    assert normalise_url("https://www.postman.com/") == normalise_url("https://postman.com")
    assert normalise_url("https://WWW.Acme.com/About/") == "https://acme.com/About"
