"""The harvester is what stops the LLM inventing contact data - test it hard."""

from pathlib import Path

from src.cleaner import html_to_markdown
from src.harvester import harvest_all, harvest_emails, harvest_linkedin
from src.models import clean_email, clean_linkedin

FIXTURES = Path(__file__).parent / "fixtures"
TEAM_HTML = (FIXTURES / "company_team.html").read_text(encoding="utf-8")
TEAM_MD = html_to_markdown(TEAM_HTML)


def test_finds_real_emails():
    emails = harvest_emails(TEAM_HTML, TEAM_MD)
    for expected in ("hello@acmedata.io", "sales@acmedata.io", "support@acmedata.io"):
        assert expected in emails


def test_strips_mailto_query_strings():
    emails = harvest_emails(TEAM_HTML, TEAM_MD)
    assert "sales@acmedata.io" in emails
    assert not any("subject=" in e for e in emails)


def test_deobfuscates_at_and_dot():
    emails = harvest_emails(TEAM_HTML, TEAM_MD)
    assert "press@acmedata.io" in emails


def test_rejects_asset_filenames_and_placeholders():
    emails = harvest_emails(TEAM_HTML, TEAM_MD)
    for junk in ("logo@2x.png", "noreply@example.com", "someone@yourcompany.com"):
        assert junk not in emails
    assert not any(e.endswith((".png", ".svg", ".jpg")) for e in emails)


def test_generic_mailboxes_are_ranked_first():
    emails = harvest_emails(TEAM_HTML, TEAM_MD)
    assert emails[0].split("@")[0] in {"contact", "hello", "sales", "support", "info", "press"}


def test_linkedin_profiles_separated_from_company_pages():
    links = harvest_linkedin(TEAM_HTML, TEAM_MD)
    assert "https://www.linkedin.com/company/acme-data-systems" in links["companies"]
    assert all("/company/" not in p for p in links["profiles"])
    assert len(links["profiles"]) == 3


def test_linkedin_urls_are_normalised():
    links = harvest_linkedin(TEAM_HTML, TEAM_MD)
    assert "https://www.linkedin.com/in/priya-raghavan-acme" in links["profiles"]
    assert "https://linkedin.com/in/danielokoye" in links["profiles"]  # http -> https
    assert "https://linkedin.com/in/meilinchen" in links["profiles"]   # trailing slash gone
    assert not any("?" in p for p in links["profiles"])


def test_clean_email_edge_cases():
    assert clean_email("MAILTO:Hello@Acme.IO") == "hello@acme.io"
    assert clean_email("hello@acme.io.") == "hello@acme.io"
    assert clean_email("not-an-email") is None
    assert clean_email("") is None
    assert clean_email("a@b") is None


def test_clean_linkedin_rejects_non_profile_urls():
    assert clean_linkedin("https://linkedin.com/feed") is None
    assert clean_linkedin("https://twitter.com/in/someone") is None
    assert clean_linkedin("linkedin.com/in/abc") == "https://linkedin.com/in/abc"


def test_harvest_all_survives_a_broken_page():
    pages = {
        "https://a.com/team": (TEAM_HTML, TEAM_MD),
        "https://a.com/broken": ("<<<not parseable>>>", ""),
    }
    result = harvest_all(pages)
    assert "hello@acmedata.io" in result["emails"]
    assert len(result["linkedin_profiles"]) == 3
