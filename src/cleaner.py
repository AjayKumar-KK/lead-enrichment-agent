"""HTML -> clean markdown, and cross-page boilerplate removal.

This module exists to satisfy the hard rule in the brief: *never feed raw HTML to
the LLM*. It does three things, in order:

1. Structurally strip the DOM - scripts, styles, SVGs, navs, headers, footers,
   cookie banners and similar chrome are removed as elements, not regexed out.
2. Render what remains as compact markdown, keeping the signals that matter for
   this task (headings, list items, link text) and dropping the rest.
3. Remove boilerplate that survives step 1 by comparing every page of the same
   domain against every other: any short line appearing on most pages is
   navigation or footer text repeated site-wide, and is dropped.

Step 3 is the one that actually moves the token numbers, because modern sites put
their nav inside generic ``<div>``s that no selector list can anticipate.
"""

from __future__ import annotations

import re
from collections import Counter

from lxml import html as lxml_html

from .utils import get_logger

logger = get_logger("cleaner")

# Elements removed wholesale: they carry no company intelligence.
DROP_TAGS = (
    "script", "style", "noscript", "svg", "canvas", "iframe", "form",
    "picture", "source", "video", "audio", "template", "select", "button",
)

# Structural chrome. Removed by tag and by common role/class/id naming.
CHROME_TAGS = ("nav", "header", "footer", "aside")

CHROME_PATTERN = re.compile(
    r"(nav|menu|header|footer|cookie|consent|banner|breadcrumb|sidebar|"
    r"social|newsletter|subscribe|announce|skip-link|modal|popup|toast|"
    r"carousel-control|back-to-top)",
    re.IGNORECASE,
)

# lxml moved its HTML sanitiser into the separate `lxml_html_clean` package at
# lxml 5.2. We use it when present, but the pipeline does not depend on it: the
# structural strip below removes the same elements. Optional-import rather than a
# hard requirement so the project installs cleanly on any lxml version.
try:  # pragma: no cover - depends on the installed lxml build
    from lxml.html.clean import Cleaner  # type: ignore

    _CLEANER: object | None = Cleaner(
        scripts=True,
        javascript=True,
        comments=True,
        style=True,
        inline_style=True,
        links=False,
        meta=False,
        page_structure=False,
        embedded=True,
        frames=True,
        forms=False,
        annoying_tags=True,
        remove_unknown_tags=False,
        safe_attrs_only=False,
    )
except Exception:  # pragma: no cover
    _CLEANER = None

BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "td", "th", "br", "tr"}
HEADING_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


def _is_chrome(element) -> bool:
    """True if an element looks like site chrome rather than content."""
    if element.tag in CHROME_TAGS:
        return True
    for attr in ("class", "id", "role", "data-testid"):
        value = element.get(attr)
        if value and CHROME_PATTERN.search(str(value)):
            return True
    return False


def _strip_dom(tree) -> None:
    """Remove non-content elements in place."""
    for tag in DROP_TAGS:
        for element in tree.iter(tag):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)

    # Chrome removal is done on a materialised list because we mutate the tree.
    for element in list(tree.iter()):
        if not isinstance(element.tag, str):
            continue
        parent = element.getparent()
        if parent is None:
            continue
        if _is_chrome(element):
            # Keep the element if it is unusually text-heavy: some sites wrap
            # their entire body in a div whose class happens to contain "banner".
            if len(element.text_content() or "") < 1200:
                parent.remove(element)


def _render_markdown(tree) -> str:
    """Walk the stripped tree and emit compact markdown."""
    parts: list[str] = []

    def walk(element) -> None:
        if not isinstance(element.tag, str):
            return
        tag = element.tag.lower()

        if tag in HEADING_TAGS:
            text = re.sub(r"\s+", " ", element.text_content() or "").strip()
            if text:
                parts.append(f"\n{HEADING_TAGS[tag]} {text}\n")
            return

        if tag == "a":
            text = re.sub(r"\s+", " ", element.text_content() or "").strip()
            href = (element.get("href") or "").strip()
            # Keep mailto and LinkedIn hrefs inline: they are high-value signals
            # that would otherwise be lost when only visible text survives.
            if href.startswith("mailto:") or "linkedin.com/" in href.lower():
                parts.append(f" [{text}]({href}) ")
            elif text:
                parts.append(" " + text + " ")
            return

        if element.text and element.text.strip():
            parts.append(element.text.strip() + " ")

        for child in element:
            walk(child)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip() + " ")

        if tag in BLOCK_TAGS:
            parts.append("\n")

    walk(tree)
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_markdown(raw_html: str) -> str:
    """Convert a raw HTML document into clean markdown text.

    Returns an empty string rather than raising on malformed input: a page we
    cannot parse is one page lost, never a failed run.
    """
    if not raw_html or not raw_html.strip():
        return ""
    try:
        tree = lxml_html.fromstring(raw_html)
    except Exception as exc:  # malformed markup, binary content, empty body
        logger.debug("lxml could not parse document: %s", exc)
        return ""

    if _CLEANER is not None:
        try:
            tree = _CLEANER.clean_html(tree)
        except Exception as exc:
            logger.debug("sanitiser pass failed, continuing with raw tree: %s", exc)

    _strip_dom(tree)
    return _render_markdown(tree)


def remove_cross_page_boilerplate(
    pages: dict[str, str],
    threshold: float = 0.6,
    max_line_len: int = 200,
) -> dict[str, str]:
    """Drop lines that repeat across most pages of the same site.

    A line present on 60%+ of a domain's pages and short enough to be a nav item
    or footer link is boilerplate by definition. With fewer than three pages the
    signal is too weak to act on, so the input is returned untouched.
    """
    if len(pages) < 3:
        return pages

    counts: Counter[str] = Counter()
    for text in pages.values():
        unique_lines = {line.strip() for line in text.splitlines() if line.strip()}
        counts.update(unique_lines)

    cutoff = max(2, int(len(pages) * threshold))
    boilerplate = {
        line
        for line, count in counts.items()
        if count >= cutoff and len(line) <= max_line_len
    }

    if not boilerplate:
        return pages

    cleaned: dict[str, str] = {}
    for url, text in pages.items():
        kept = [
            line
            for line in text.splitlines()
            if line.strip() not in boilerplate
        ]
        stripped = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()

        # Safety floor. When a site serves near-identical content on every route -
        # a soft-404 that renders the homepage, or an SPA shell behind every path -
        # every line looks like boilerplate and this pass would delete the page
        # entirely. Losing more than 85% of a page means the signal was wrong, so
        # the original is kept and the LLM decides what is useful.
        if text.strip() and len(stripped) < len(text.strip()) * 0.15:
            logger.debug("boilerplate pass reverted for %s (would have removed too much)", url)
            cleaned[url] = text
        else:
            cleaned[url] = stripped

    removed = sum(len(v) for v in pages.values()) - sum(len(v) for v in cleaned.values())
    logger.debug(
        "boilerplate pass removed %d chars across %d pages (%d repeated lines)",
        removed,
        len(pages),
        len(boilerplate),
    )
    return cleaned
