"""Cleaning must remove chrome and keep content, and never raise on bad input."""

from pathlib import Path

from src.cleaner import html_to_markdown, remove_cross_page_boilerplate

FIXTURES = Path(__file__).parent / "fixtures"
TEAM_HTML = (FIXTURES / "company_team.html").read_text(encoding="utf-8")
SPA_HTML = (FIXTURES / "spa_shell.html").read_text(encoding="utf-8")


def test_scripts_styles_and_svg_are_removed():
    md = html_to_markdown(TEAM_HTML)
    for noise in ("dataLayer", "buildId", "background: #fff", "<circle", "function track"):
        assert noise not in md, f"{noise!r} survived cleaning"


def test_navigation_and_cookie_banner_are_removed():
    md = html_to_markdown(TEAM_HTML)
    assert "We use cookies" not in md
    assert "All rights reserved" not in md


def test_real_content_survives():
    md = html_to_markdown(TEAM_HTML)
    assert "Acme Data Systems builds a managed streaming database" in md
    for name in ("Priya Raghavan", "Daniel Okoye", "Mei Lin Chen"):
        assert name in md
    assert "Chief Executive Officer" in md


def test_headings_become_markdown():
    md = html_to_markdown(TEAM_HTML)
    assert "# Leadership" in md
    assert "### Priya Raghavan" in md


def test_high_value_hrefs_are_preserved_inline():
    md = html_to_markdown(TEAM_HTML)
    assert "mailto:hello@acmedata.io" in md
    assert "linkedin.com/in/priya-raghavan-acme" in md


def test_cleaning_shrinks_the_document_substantially():
    md = html_to_markdown(TEAM_HTML)
    assert len(md) < len(TEAM_HTML) * 0.55, "cleaning should cut the document by ~half or more"


def test_spa_shell_yields_almost_nothing():
    """The signal that triggers browser escalation."""
    md = html_to_markdown(SPA_HTML)
    assert len(md) < 600


def test_malformed_input_never_raises():
    for bad in ("", "   ", "<<<>>>", "not html at all", "<html><body><p>unclosed"):
        assert isinstance(html_to_markdown(bad), str)


def test_cross_page_boilerplate_removed():
    shared = "Home Product Pricing About Contact"
    pages = {
        "https://x.com/": f"{shared}\nWelcome to our homepage content here.",
        "https://x.com/about": f"{shared}\nWe were founded in 2019 in Berlin.",
        "https://x.com/team": f"{shared}\nOur leadership team is listed below.",
        "https://x.com/contact": f"{shared}\nReach us at hello@x.com any time.",
    }
    cleaned = remove_cross_page_boilerplate(pages)
    for text in cleaned.values():
        assert shared not in text
    assert "founded in 2019 in Berlin" in cleaned["https://x.com/about"]
    assert "leadership team" in cleaned["https://x.com/team"]


def test_boilerplate_pass_skipped_for_small_sites():
    """With fewer than three pages the repetition signal is unreliable."""
    pages = {"a": "shared line\nunique a", "b": "shared line\nunique b"}
    assert remove_cross_page_boilerplate(pages) == pages
