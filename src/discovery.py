"""Sub-page discovery: find the pages worth reading, without hard-coding paths.

The naive approach is a fixed list - try /about, /team, /contact, give up. It
breaks the moment a site uses /our-story or /company/leadership, which is most of
them.

Instead we gather candidates from two sources (the XML sitemap, and the anchors on
the homepage), score every candidate against weighted keywords, penalise depth and
known-irrelevant sections, and crawl the top N. A site that names its about page
anything sensible gets found; a site with 4,000 blog posts in its sitemap does not
drown the crawl.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from lxml import etree
from lxml import html as lxml_html

from .config import USER_AGENT, Settings
from .utils import (
    get_logger,
    looks_like_html,
    normalise_url,
    same_registrable_domain,
    url_depth,
)

logger = get_logger("discovery")

SITEMAP_CANDIDATES = ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml")
MAX_SITEMAP_URLS = 3_000
MAX_NESTED_SITEMAPS = 5


def _keyword_score(subject: str, settings: Settings) -> float:
    """Score a path or a subdomain label against the keyword weights."""
    segment_set = set(re.split(r"[/\-_.]", subject))

    matched: list[float] = []
    for keyword, weight in settings.keywords.items():
        parts = keyword.split("-")
        if keyword in subject or (len(parts) == 1 and keyword in segment_set):
            matched.append(weight)

    # The strongest single signal decides the score; additional matches add only
    # half their weight. Summing them outright would let a deep, keyword-stuffed
    # path like /company/culture/people/team outrank the canonical /team.
    score = 0.0
    if matched:
        matched.sort(reverse=True)
        score = matched[0] + 0.5 * sum(matched[1:])

    for keyword, penalty in settings.negative_keywords.items():
        if keyword in segment_set or f"/{keyword}/" in f"/{subject}/":
            score += penalty

    return score


def score_url(url: str, settings: Settings, domain: str | None = None) -> float:
    """Score a candidate URL by how likely it is to hold company intelligence.

    ``domain`` is the site being enriched. Pass it: without it a root URL cannot
    be told apart from *any* subdomain's root, and every one of them scores as a
    homepage. That was a real bug - ``status.supabase.com/`` and
    ``discord.supabase.com/`` scored 100 while ``/about`` and ``/team`` scored 7,
    so the crawl spent its budget on a status page and an empty Discord redirect
    and never read a page with a person on it.

    A subdomain root is scored on its label instead, against the same keywords:
    ``careers.`` is worth reading, ``status.``, ``blog.`` and ``docs.`` are
    explicitly not, and an unrecognised one scores zero and is dropped.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0].removeprefix("www.")
    path = parsed.path.lower().strip("/")

    if not path:
        if not domain or host == domain:
            return 100.0  # the homepage itself is always worth reading
        label = host[: -(len(domain) + 1)] if host.endswith("." + domain) else host
        return round(_keyword_score(label, settings), 2)

    score = _keyword_score(path, settings)

    # Shallow pages are more likely to be the canonical about/team page.
    score -= url_depth(url) * 3.0

    # Very long slugs are almost always articles.
    if len(path) > 60:
        score -= 4.0

    # A sub-page on a subdomain is a step further from the company's own story.
    if domain and host != domain:
        score -= 2.0

    return round(score, 2)


async def _fetch_text(client, url: str, timeout: float) -> str | None:
    try:
        response = await client.get(url, timeout=timeout, follow_redirects=True)
        if response.status_code == 200 and response.text:
            return response.text
    except Exception as exc:
        logger.debug("fetch failed for %s: %s", url, exc)
    return None


async def fetch_robots(
    client, base: str, settings: Settings
) -> tuple[RobotFileParser | None, list[str]]:
    """Read ``/robots.txt``.

    Returns a parser for checking crawl permission, and any ``Sitemap:`` URLs the
    site declares there - which is where large sites usually point, rather than at
    the conventional ``/sitemap.xml``.

    A missing or unreadable robots.txt is normal and simply means no restrictions
    and no declared sitemaps; it never fails the crawl.
    """
    body = await _fetch_text(client, urljoin(base, "/robots.txt"), settings.http_timeout)
    if not body:
        return None, []

    declared = [
        line.split(":", 1)[1].strip()
        for line in body.splitlines()
        if line.lower().startswith("sitemap:") and ":" in line
    ]

    parser: RobotFileParser | None = None
    try:
        parser = RobotFileParser()
        parser.parse(body.splitlines())
    except Exception as exc:
        logger.debug("could not parse robots.txt for %s: %s", base, exc)
        parser = None

    if declared:
        logger.debug("robots.txt declares %d sitemap(s)", len(declared))
    return parser, declared


def _allowed(parser: RobotFileParser | None, url: str, settings: Settings) -> bool:
    """True when robots.txt permits fetching ``url`` (or when we are not checking)."""
    if parser is None or not settings.respect_robots:
        return True
    try:
        return parser.can_fetch(USER_AGENT, url)
    except Exception:
        return True  # a malformed rule must not block the crawl


async def _urls_from_sitemap(
    client, base: str, settings: Settings, declared: list[str] | None = None
) -> list[str]:
    """Read the sitemap, following one level of sitemap-index nesting."""
    collected: list[str] = []
    queue = list(declared or []) + [urljoin(base, path) for path in SITEMAP_CANDIDATES]
    seen_sitemaps: set[str] = set()
    nested_followed = 0

    while queue and len(collected) < MAX_SITEMAP_URLS:
        sitemap_url = queue.pop(0)
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)

        body = await _fetch_text(client, sitemap_url, settings.http_timeout)
        if not body:
            continue

        try:
            root = etree.fromstring(body.encode("utf-8", errors="ignore"))
        except Exception as exc:
            logger.debug("sitemap %s is not valid XML: %s", sitemap_url, exc)
            continue

        tag = etree.QName(root).localname if root.tag else ""
        locs = [
            (el.text or "").strip()
            for el in root.iter()
            if isinstance(el.tag, str) and etree.QName(el).localname == "loc"
        ]

        if tag == "sitemapindex":
            for loc in locs:
                if nested_followed >= MAX_NESTED_SITEMAPS:
                    break
                # Prefer child sitemaps whose names suggest pages, not posts.
                if re.search(r"(page|main|root|sitemap)", loc, re.IGNORECASE):
                    queue.append(loc)
                    nested_followed += 1
            if not queue and locs:
                queue.extend(locs[:MAX_NESTED_SITEMAPS])
                nested_followed += MAX_NESTED_SITEMAPS
        else:
            collected.extend(locs)

        if collected:
            logger.debug("sitemap %s yielded %d urls", sitemap_url, len(collected))

    return collected[:MAX_SITEMAP_URLS]


def urls_from_html(base_url: str, raw_html: str) -> list[str]:
    """Pull every internal anchor href out of a rendered page."""
    urls: list[str] = []
    try:
        tree = lxml_html.fromstring(raw_html)
    except Exception as exc:
        logger.debug("could not parse homepage anchors: %s", exc)
        return urls

    for anchor in tree.xpath("//a[@href]"):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        urls.append(urljoin(base_url, href))
    return urls


async def discover_subpages(
    client,
    domain: str,
    base_url: str,
    homepage_html: str,
    settings: Settings,
) -> list[tuple[str, float]]:
    """Return ``[(url, score), ...]`` for the best sub-pages, highest score first.

    The homepage is not included - the caller already has it.
    """
    candidates: list[str] = []
    parser: RobotFileParser | None = None

    try:
        parser, declared = await fetch_robots(client, base_url, settings)
    except Exception as exc:
        declared = []
        logger.debug("robots.txt lookup failed for %s: %s", domain, exc)

    try:
        candidates.extend(await _urls_from_sitemap(client, base_url, settings, declared))
    except Exception as exc:
        logger.debug("sitemap discovery failed for %s: %s", domain, exc)

    candidates.extend(urls_from_html(base_url, homepage_html))

    home = normalise_url(base_url)
    scored: dict[str, float] = {}
    disallowed = 0

    for raw in candidates:
        try:
            url = normalise_url(raw)
        except Exception:
            continue
        if url == home or url in scored:
            continue
        if not same_registrable_domain(url, domain):
            continue
        if not looks_like_html(url):
            continue
        if url_depth(url) > 3:
            continue
        if not _allowed(parser, url, settings):
            disallowed += 1
            continue
        scored[url] = score_url(url, settings, domain)

    if disallowed:
        logger.debug("%s: skipped %d url(s) disallowed by robots.txt", domain, disallowed)

    # Only pages with a positive signal are worth an LLM's attention.
    ranked = sorted(
        ((url, score) for url, score in scored.items() if score > 0),
        key=lambda pair: pair[1],
        reverse=True,
    )

    limit = max(0, settings.max_pages_per_domain - 1)
    selected = ranked[:limit]
    logger.info(
        "%s: %d candidate urls -> %d selected (top: %s)",
        domain,
        len(scored),
        len(selected),
        ", ".join(urlparse(u).path or "/" for u, _ in selected[:4]) or "none",
    )
    return selected
