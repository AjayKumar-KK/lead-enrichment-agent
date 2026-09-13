"""Deterministic extraction of facts that must never be invented.

Emails and LinkedIn URLs are *verifiable* data: they either appear in the page or
they do not. Asking a language model to produce them invites plausible-looking
fabrications - ``careers@postman.com`` is exactly the sort of address a model will
emit whether or not it exists.

So we pull them out of the DOM ourselves, with regex over both hrefs and visible
text, and hand the model a closed list it is instructed to choose from. The model
still does the work it is actually good at: writing the summary, inferring the
ICP, and matching a person's name to their title and profile link. This split -
regex for facts, LLM for judgement - is what keeps the output trustworthy.
"""

from __future__ import annotations

import html
import re
from urllib.parse import unquote

from lxml import html as lxml_html

from .models import clean_email, clean_linkedin
from .utils import dedupe, get_logger

logger = get_logger("harvester")

# Matches addresses in visible text. Deliberately permissive; `clean_email`
# applies the strict validation and blocklist afterwards.
EMAIL_TEXT_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+\s?(?:@|\[at\]|\(at\))\s?[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)

LINKEDIN_TEXT_RE = re.compile(
    r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub|company)/[A-Za-z0-9\-_%.]+",
    re.IGNORECASE,
)

# Generic mailboxes worth surfacing first when a site lists many addresses.
PREFERRED_PREFIXES = (
    "contact", "sales", "support", "hello", "info", "help", "press",
    "partnerships", "security", "privacy", "careers", "jobs", "team",
)


_AT_RE = re.compile(r"\s*(?:\[at\]|\(at\)|\{at\})\s*", re.IGNORECASE)
_DOT_RE = re.compile(r"\s*(?:\[dot\]|\(dot\)|\{dot\})\s*", re.IGNORECASE)

# \u003e and friends, as embedded JSON payloads spell them.
_JSON_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _decode_escapes(value: str) -> str:
    """Turn escaped punctuation back into punctuation before scanning for addresses.

    Modern sites ship their content twice: once as HTML and once as a JSON blob
    for the client-side framework, where ``>`` is written ``\u003e``. The
    backslash is not a local-part character but ``u003e`` is, so the address
    pattern matched *into* the escape and produced ``u003einfo@postman.com``
    beside the real ``info@postman.com`` - a duplicate wearing a malformed name.
    Decoding first means the pattern only ever sees real punctuation, which it
    already knows to exclude.
    """
    def _replace(match: re.Match) -> str:
        code = int(match.group(1), 16)
        # Lone surrogates are not characters; leave those escapes alone.
        return match.group(0) if 0xD800 <= code <= 0xDFFF else chr(code)

    try:
        return html.unescape(_JSON_ESCAPE_RE.sub(_replace, value or ""))
    except Exception as exc:  # a malformed blob must not stop the harvest
        logger.debug("could not decode escapes: %s", exc)
        return value or ""


def _normalise_obfuscated(value: str) -> str:
    """Undo the common 'name [at] domain [dot] com' anti-scrape trick.

    Only the obfuscation tokens and the whitespace immediately around them are
    collapsed - stripping every space in the document would fuse ordinary words
    into addresses that were never on the page.
    """
    return _DOT_RE.sub(".", _AT_RE.sub("@", value or ""))


def harvest_emails(raw_html: str, markdown: str = "") -> list[str]:
    """Collect valid public email addresses from hrefs and visible text."""
    found: list[str] = []

    # mailto: hrefs are the highest-signal source.
    try:
        tree = lxml_html.fromstring(raw_html)
        for anchor in tree.xpath("//a[@href]"):
            href = unquote(str(anchor.get("href") or ""))
            if href.lower().startswith("mailto:"):
                cleaned = clean_email(href)
                if cleaned:
                    found.append(cleaned)
    except Exception as exc:
        logger.debug("email href pass failed: %s", exc)

    # Then visible text, from both the raw document and the cleaned markdown.
    # De-obfuscation runs on the whole blob first so that "a [at] b [dot] com"
    # becomes a matchable address before the pattern is applied.
    for blob in (raw_html, markdown):
        if not blob:
            continue
        for match in EMAIL_TEXT_RE.findall(_normalise_obfuscated(_decode_escapes(blob))):
            cleaned = clean_email(match.replace(" ", ""))
            if cleaned:
                found.append(cleaned)

    unique = dedupe(found)
    # Generic company mailboxes first - they are what a prospecting workflow wants.
    unique.sort(key=lambda e: (0 if e.split("@")[0] in PREFERRED_PREFIXES else 1, e))
    return unique


def harvest_linkedin(raw_html: str, markdown: str = "") -> dict[str, list[str]]:
    """Collect LinkedIn URLs, split into personal profiles and company pages."""
    found: list[str] = []

    try:
        tree = lxml_html.fromstring(raw_html)
        for anchor in tree.xpath("//a[@href]"):
            href = str(anchor.get("href") or "")
            if "linkedin.com/" in href.lower():
                cleaned = clean_linkedin(href)
                if cleaned:
                    found.append(cleaned)
    except Exception as exc:
        logger.debug("linkedin href pass failed: %s", exc)

    for blob in (raw_html, markdown):
        if not blob:
            continue
        for match in LINKEDIN_TEXT_RE.findall(blob):
            cleaned = clean_linkedin(match)
            if cleaned:
                found.append(cleaned)

    profiles = dedupe([u for u in found if "/company/" not in u.lower()])
    companies = dedupe([u for u in found if "/company/" in u.lower()])
    return {"profiles": profiles, "companies": companies}


def harvest_all(pages: dict[str, tuple[str, str]]) -> dict[str, list[str]]:
    """Run both harvesters across every page of a domain.

    ``pages`` maps url -> (raw_html, markdown).
    """
    emails: list[str] = []
    profiles: list[str] = []
    companies: list[str] = []

    for url, (raw, md) in pages.items():
        try:
            emails.extend(harvest_emails(raw, md))
            links = harvest_linkedin(raw, md)
            profiles.extend(links["profiles"])
            companies.extend(links["companies"])
        except Exception as exc:
            logger.debug("harvest failed for %s: %s", url, exc)

    result = {
        "emails": dedupe(emails),
        "linkedin_profiles": dedupe(profiles),
        "linkedin_companies": dedupe(companies),
    }
    logger.debug(
        "harvested %d emails, %d profiles, %d company pages",
        len(result["emails"]),
        len(result["linkedin_profiles"]),
        len(result["linkedin_companies"]),
    )
    return result
