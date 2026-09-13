"""LinkedIn resolution: validate what the site gave us, search for what it did not.

A team page usually links its people's profiles, and when it does the harvester
has already pulled those URLs out of the DOM. Plenty of sites do not: they list
"Jane Doe, CEO" as plain text and nothing else. The obvious fix - asking the model
to fill the gap - is exactly the wrong one, because a language model will happily
emit ``linkedin.com/in/jane-doe`` whether or not that profile exists or belongs to
this Jane Doe. That is a fabrication with a plausible shape, which is the most
expensive kind of error in a prospecting dataset.

So this module closes the gap without ever letting the model near it:

1. **Validate.** A URL the model attributed to a person is kept only if that exact
   URL was harvested from the company's own HTML. Anything else the model produced
   was invented and is dropped, whatever it looks like.
2. **Search.** For people still missing a profile, a search engine is queried with
   the person's name, the company, and ``site:linkedin.com/in``. Which engine
   matters: LinkedIn's robots.txt lets Google index profile pages and blocks most
   others, so Bing and DuckDuckGo return nothing usable here however well you
   parse them. The providers below are Google (via serper.dev), Brave's own index
   (API or, keylessly, its results page), in that order.
3. **Corroborate.** A result is accepted only when the person's first *and* last
   name appear in the profile slug or the result title, *and* the company or their
   role is independently corroborated in the title or snippet. Name-only matches
   are rejected - "John Smith" is thousands of people.

Every accepted URL carries its source, a confidence number and the human-readable
evidence behind it, so a reviewer can audit any single value in the output.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from lxml import html as lxml_html

from .config import USER_AGENT, Settings
from .models import EnrichedTeamMember, TeamMember, clean_linkedin
from .utils import (
    company_tokens,
    compact_error,
    fold_text,
    get_logger,
    name_tokens,
    squash_text,
    text_tokens,
)

logger = get_logger("linkedin")

# Only personal profiles can belong to a person. /company/ pages are a different
# kind of object and are never a valid answer here.
PROFILE_PATH_RE = re.compile(r"^/(?:in|pub)/[^/]+", re.IGNORECASE)

# Words that appear in job titles everywhere and corroborate nothing.
ROLE_STOPWORDS = {
    "the", "and", "for", "our", "team", "with", "from", "all", "new", "global",
    "senior", "lead", "head", "staff", "member", "manager", "director",
}


def company_name_from_html(raw_html: str, domain: str) -> str:
    """Best-effort trading name from the homepage <title>.

    '"Jane Doe" "Acme Data" site:linkedin.com/in' is a far better query than one
    built from the bare domain label, so it is worth one cheap parse. Everything
    after the first separator is tagline, not name, and is dropped. Falls back to
    the domain label when the title is missing or useless.
    """
    fallback = domain.split(".")[0]
    try:
        tree = lxml_html.fromstring(raw_html)
        title = (tree.findtext(".//title") or "").strip()
    except Exception:
        return fallback
    if not title:
        return fallback
    head = re.split(r"\s*[|\u2013\u2014\u00b7:\u2022-]\s+", title)[0].strip()
    # A "title" that is really a sentence is a tagline, not a company name.
    if not head or len(head) > 40 or len(head.split()) > 5:
        return fallback
    return head


# --------------------------------------------------------------------------- #
# Search results and providers
# --------------------------------------------------------------------------- #


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""

    @property
    def text(self) -> str:
        return f"{self.title} {self.snippet}"


class SearchProvider:
    """Interface for anything that can answer a web query."""

    name = "none"

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        raise NotImplementedError


class SerperProvider(SearchProvider):
    """Google results via serper.dev. Used when SERPER_API_KEY is configured.

    Google is the only major index that carries LinkedIn profile pages in bulk
    (LinkedIn's robots.txt permits it and blocks most others), so a Google-backed
    API is the strongest provider available here.
    """

    name = "serper"

    def __init__(self, client: httpx.AsyncClient, api_key: str, timeout: float) -> None:
        self._client = client
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        response = await self._client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max(limit, 5)},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = [
            SearchResult(
                title=str(item.get("title") or ""),
                url=str(item.get("link") or ""),
                snippet=str(item.get("snippet") or ""),
            )
            for item in (payload.get("organic") or [])
        ]
        return results[:limit]


class BraveApiProvider(SearchProvider):
    """Brave's official Search API. Used when BRAVE_API_KEY is configured.

    Brave runs its own index rather than reselling Bing's, and that index does
    contain LinkedIn profiles. It is the documented, rate-limit-honest version of
    what :class:`BraveHtmlProvider` scrapes.
    """

    name = "brave-api"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, client: httpx.AsyncClient, api_key: str, timeout: float) -> None:
        self._client = client
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        response = await self._client.get(
            self.ENDPOINT,
            params={"q": query, "count": max(limit, 5)},
            headers={"X-Subscription-Token": self._api_key, "Accept": "application/json"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = [
            SearchResult(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                snippet=str(item.get("description") or ""),
            )
            for item in ((payload.get("web") or {}).get("results") or [])
        ]
        return results[:limit]


class BraveHtmlProvider(SearchProvider):
    """Keyless fallback: Brave's public results page, parsed as HTML.

    This exists so the feature does something useful for a reviewer with no search
    API key, and it is deliberately the *last* choice. Two measured facts shaped
    it:

    * DuckDuckGo and Bing were tried first and are not viable at all - neither
      index carries ``linkedin.com/in`` pages, so even a perfectly parsed response
      contains no profiles to match. Brave's own index does carry them.
    * Brave's HTML endpoint starts returning 429 after a handful of requests from
      one address, and occasionally serves a challenge page with no results.

    So this provider is best-effort by construction: a block, a challenge page or
    a parse failure all resolve to "no confident match", which leaves
    ``linkedin_url`` null exactly as if the person had no profile. Nothing
    downstream depends on it succeeding. For a real run, set SERPER_API_KEY or
    BRAVE_API_KEY; to make no outbound search calls at all, set
    ``LINKEDIN_SEARCH_PROVIDER=none``.
    """

    name = "brave-html"
    ENDPOINTS = ("https://search.brave.com/search?q={q}",)

    def __init__(self, client: httpx.AsyncClient, timeout: float) -> None:
        self._client = client
        self._timeout = timeout

    @staticmethod
    def _unwrap(href: str) -> str:
        """Follow the redirect wrappers some engines put around outbound links."""
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if parsed.path.startswith(("/l/", "/url")):
            query = parse_qs(parsed.query)
            for key in ("uddg", "url", "u", "q"):
                if query.get(key):
                    return query[key][0]
        return href

    def _parse(self, body: str) -> list[SearchResult]:
        """Pull every LinkedIn result out of a results page.

        Anchors are used rather than engine-specific result classes, and the
        snippet is taken from the nearest ancestor that holds more text than the
        link itself. That keeps the parser working when the markup is restyled -
        which it will be - and across engines.
        """
        results: list[SearchResult] = []
        seen: set[str] = set()
        try:
            tree = lxml_html.fromstring(body)
        except Exception as exc:
            logger.debug("could not parse search results: %s", exc)
            return results

        for anchor in tree.xpath("//a[@href]"):
            href = self._unwrap(str(anchor.get("href") or ""))
            if "linkedin.com/" not in href.lower() or href in seen:
                continue
            seen.add(href)
            title = re.sub(r"\s+", " ", " ".join(anchor.itertext())).strip()
            snippet = ""
            node = anchor.getparent()
            for _ in range(4):
                if node is None:
                    break
                text = re.sub(r"\s+", " ", " ".join(node.itertext())).strip()
                if len(text) > len(title) + 20:
                    snippet = text[:600]
                    break
                node = node.getparent()
            results.append(SearchResult(title=title, url=href, snippet=snippet))
        return results

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        last_error: Exception | None = None
        for template in self.ENDPOINTS:
            try:
                response = await self._client.get(
                    template.format(q=quote_plus(query)),
                    timeout=self._timeout,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                    follow_redirects=True,
                )
                if response.status_code != 200 or not response.text:
                    last_error = RuntimeError(f"status {response.status_code}")
                    continue
                parsed = self._parse(response.text)
                if parsed:
                    return parsed[:limit]
            except Exception as exc:
                last_error = exc
        if last_error:
            raise last_error
        return []


def build_provider(client: httpx.AsyncClient, settings: Settings) -> SearchProvider | None:
    """Pick a provider from settings. Returns None when search is switched off.

    ``auto`` prefers a configured API over the keyless scraper, because only the
    API is dependable enough to run on every domain of a real batch.
    """
    choice = (settings.linkedin_search_provider or "auto").lower()
    if not settings.linkedin_search_enabled or choice == "none":
        return None

    if choice in ("auto", "serper") and settings.serper_api_key:
        return SerperProvider(client, settings.serper_api_key, settings.linkedin_search_timeout)
    if choice in ("auto", "brave-api", "brave") and settings.brave_api_key:
        return BraveApiProvider(client, settings.brave_api_key, settings.linkedin_search_timeout)

    if choice == "serper":
        logger.warning("LINKEDIN_SEARCH_PROVIDER=serper but SERPER_API_KEY is not set")
        return None
    if choice == "brave-api":
        logger.warning("LINKEDIN_SEARCH_PROVIDER=brave-api but BRAVE_API_KEY is not set")
        return None

    return BraveHtmlProvider(client, settings.linkedin_search_timeout)


# --------------------------------------------------------------------------- #
# Candidate validation
# --------------------------------------------------------------------------- #


@dataclass
class Match:
    url: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    role: str | None = None


def canonical_profile_url(raw: str) -> str | None:
    """Normalise a personal-profile URL, or None if it is not one.

    ``clean_linkedin`` already rejects non-LinkedIn strings; this additionally
    rejects company pages and localised prefixes so every stored URL has the same
    shape.
    """
    cleaned = clean_linkedin(raw)
    if not cleaned:
        return None
    parsed = urlparse(cleaned)
    if not PROFILE_PATH_RE.match(parsed.path):
        return None
    slug = parsed.path.split("/")[2]
    if not slug or len(slug) < 3:
        return None
    return f"https://www.linkedin.com/in/{slug}"


def profile_slug(url: str) -> str:
    """The slug, percent-decoded so 'st%C3%A9phane-dubois' matches 'Stéphane Dubois'."""
    try:
        return unquote(urlparse(url).path.split("/")[2])
    except IndexError:
        return ""


# Segments of a search-result title that are never a job role.
NON_ROLE_SEGMENTS = frozenset({
    "linkedin", "profile", "profiles", "experience", "education", "home",
    "sign in", "log in", "view profile", "posts", "activity",
})

# "Role at Company" / "Role @ Company" / "Role, Company" - split, so the employer
# half can be dropped only when it really is the employer.
_EMPLOYER_SPLIT = re.compile(r"^(.*?)(?:\s+@\s+|\s+at\s+|\s*,\s+)(.+)$", re.IGNORECASE)


def _strip_employer(segment: str, label: str, tokens: set[str]) -> str:
    """Drop a trailing employer from 'Co-Founder, Postman' without touching 'VP, Engineering'.

    The tail is only removed when it names the company we are enriching. A blind
    split on the comma would turn "VP, Engineering" into "VP" and quietly lose
    half the title.
    """
    match = _EMPLOYER_SPLIT.match(segment)
    if not match:
        return segment
    head, tail = match.group(1).strip(), match.group(2).strip()
    tail_tokens = text_tokens(tail)
    if head and tail_tokens and (tail_tokens <= tokens or squash_text(tail) == label):
        return head
    return segment


def role_from_result(name: str, result: SearchResult, label: str, tokens: set[str]) -> str | None:
    """Read a job title off a search result, or return None.

    LinkedIn result titles are formatted "Person - Role - LinkedIn", sometimes
    "Person - Role at Company - LinkedIn Country", and sometimes with no role at
    all ("Beng Eu - Supabase - LinkedIn"). This returns the role only when the
    title genuinely contains one, so a missing title stays missing rather than
    becoming the company name or a fragment of page furniture.
    """
    text = re.sub(r"\s*[-|–]\s*LinkedIn.*$", "", result.title or "", flags=re.IGNORECASE).strip()
    if not text:
        return None

    person = set(name_tokens(name))
    for raw in re.split(r"\s+[-–|]\s+", text):
        segment = raw.strip().strip(",;")
        if not segment or segment.lower() in NON_ROLE_SEGMENTS:
            continue
        # The person's own name, in either direction: a segment can be a subset
        # of it, or a fuller variant of it - "Chris Martin" is listed on LinkedIn
        # as "Christopher (Chris) Martin", which is a name, not a job title.
        segment_tokens = text_tokens(segment)
        if segment_tokens and (segment_tokens <= person or person <= segment_tokens):
            continue

        candidate = _strip_employer(segment, label, tokens).strip()
        if not candidate or candidate.lower() in NON_ROLE_SEGMENTS:
            continue
        candidate_tokens = text_tokens(candidate)
        if not candidate_tokens or candidate_tokens <= person or person <= candidate_tokens:
            continue
        # "Beng Eu - Supabase": the only thing left is the employer, not a role
        if candidate_tokens <= tokens or squash_text(candidate) == label:
            continue
        if not re.search(r"[A-Za-z]", candidate):
            continue
        # A role is a phrase, not a sentence scraped out of a snippet.
        if len(candidate) > 80 or len(candidate.split()) > 10:
            continue
        return candidate
    return None


def score_candidate(
    member: TeamMember,
    result: SearchResult,
    label: str,
    tokens: set[str],
) -> Match | None:
    """Decide whether a search hit is *this* person, and how sure we are.

    Two independent things must both hold, and neither is negotiable:

    * the person's first and last name appear in the profile slug or the title;
    * the company or their role is corroborated in the slug, title or snippet.

    A name-only match is rejected on purpose. Common names return dozens of real
    profiles, and picking the first one is guessing with extra steps.
    """
    url = canonical_profile_url(result.url)
    if not url:
        return None

    parts = name_tokens(member.name)
    if len(parts) < 2:
        return None  # a single name cannot be matched safely
    first, last = parts[0], parts[-1]

    slug_tokens = text_tokens(profile_slug(url))
    slug_squashed = squash_text(profile_slug(url))
    title_tokens = text_tokens(result.title)
    body = result.text

    def _present(token: str, pool: set[str], squashed: str) -> bool:
        return token in pool or token in squashed

    in_slug = _present(first, slug_tokens, slug_squashed) and _present(last, slug_tokens, slug_squashed)
    in_title = first in title_tokens and last in title_tokens
    if not (in_slug or in_title):
        return None

    evidence: list[str] = []
    confidence = 0.5
    if in_slug and in_title:
        confidence = 0.8
        evidence.append("name matches profile slug and result title")
    elif in_slug:
        confidence = 0.7
        evidence.append("name matches profile slug")
    else:
        confidence = 0.55
        evidence.append("name matches result title")

    # Corroboration. The company is the strong signal; the role is a weaker one
    # that still rules out an unrelated namesake.
    body_squashed = squash_text(body)
    body_tokens = text_tokens(body)

    company_hit = bool(label and label in body_squashed) or bool(
        {t for t in tokens if len(t) >= 4} & (body_tokens | slug_tokens)
    )
    role_hit = False
    if member.title:
        role = {t for t in text_tokens(member.title, minimum=3) if t not in ROLE_STOPWORDS}
        role_hit = bool(role & body_tokens)

    if company_hit:
        confidence += 0.2
        evidence.append("company corroborated in search result")
    if role_hit:
        confidence += 0.1
        evidence.append("role corroborated in search result")

    if not (company_hit or role_hit):
        # Name alone. Reject rather than guess.
        return None

    return Match(
        url=url,
        confidence=round(min(confidence, 0.95), 2),
        evidence=evidence,
        role=role_from_result(member.name, result, label, tokens),
    )


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #


class LinkedInResolver:
    """Validates site-sourced profile URLs and searches for the missing ones."""

    def __init__(self, settings: Settings, provider: SearchProvider | None = None) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None
        self._provider = provider
        self._owns_client = provider is None
        self._cache: dict[str, tuple[Match | None, str]] = {}
        self._lock = asyncio.Lock()
        self._last_query_at = 0.0

    async def __aenter__(self) -> "LinkedInResolver":
        if self._owns_client:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=self.settings.linkedin_search_timeout,
                follow_redirects=True,
            )
            self._provider = build_provider(self._client, self.settings)
            if self._provider:
                logger.info("linkedin search fallback enabled via %s", self._provider.name)
            else:
                logger.info("linkedin search fallback disabled")
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._provider is not None

    def build_query(self, member: TeamMember, domain: str, company_name: str) -> str:
        """Name and company as exact phrases; the role only as a hint.

        Quoting the role too is tempting and counter-productive: a site that says
        "Co-Founder & CTO" against a profile headlined "Co-founder, Postman" turns
        an exact-phrase match into zero results. Unquoted it still ranks the right
        person first, and the corroboration check does the deciding either way.
        """
        company = company_name or domain.split(".")[0]
        parts = [f'"{member.name}"', f'"{company}"']
        if member.title:
            parts.append(member.title[:40])
        parts.append("site:linkedin.com/in")
        return " ".join(parts)

    async def _throttled_search(self, query: str) -> list[SearchResult]:
        """One query at a time, spaced out - we are a guest on someone's index."""
        async with self._lock:
            wait = self.settings.linkedin_search_delay - (time.monotonic() - self._last_query_at)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                return await self._provider.search(query, self.settings.linkedin_search_results)
            finally:
                self._last_query_at = time.monotonic()

    async def lookup(
        self, member: TeamMember, domain: str, company_name: str = ""
    ) -> tuple[Match | None, str]:
        """Find a profile for one person. Returns ``(match, note)`` and never raises.

        The note is what ends up in ``linkedin_evidence`` when nothing is found,
        and it distinguishes the three outcomes that all look like a null URL:
        the search engine was unreachable, it returned nothing, or it returned
        candidates that failed the corroboration rules. A reviewer reading the
        output should not have to guess which happened.
        """
        if not self.enabled:
            return None, "linkedin search is disabled"

        cache_key = f"{domain}|{fold_text(member.name)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        label, tokens = company_tokens(domain, company_name)
        query = self.build_query(member, domain, company_name)

        try:
            results = await self._throttled_search(query)
        except Exception as exc:
            reason = compact_error(exc, 80)
            logger.debug("search failed for %r: %s", member.name, reason)
            outcome = (None, f"search unavailable: {reason}")
            self._cache[cache_key] = outcome
            return outcome

        best: Match | None = None
        for result in results:
            match = score_candidate(member, result, label, tokens)
            if match and (best is None or match.confidence > best.confidence):
                best = match

        if best is None:
            note = (
                f"{len(results)} search result(s) rejected: no corroborated name match"
                if results
                else "search returned no results"
            )
        elif best.confidence < self.settings.linkedin_min_confidence:
            logger.debug(
                "rejecting %s for %s: confidence %.2f below threshold %.2f",
                best.url, member.name, best.confidence, self.settings.linkedin_min_confidence,
            )
            note = (
                f"best candidate scored {best.confidence:.2f}, below the "
                f"{self.settings.linkedin_min_confidence:.2f} threshold"
            )
            best = None
        else:
            note = ""

        outcome = (best, note)
        self._cache[cache_key] = outcome
        return outcome

    async def search_for_member(
        self, member: TeamMember, domain: str, company_name: str = ""
    ) -> Match | None:
        """Convenience wrapper around :meth:`lookup` for callers that only want the URL."""
        match, _ = await self.lookup(member, domain, company_name)
        return match

    async def resolve_team(
        self,
        members: list[TeamMember],
        domain: str,
        verified_profiles: list[str],
        company_name: str = "",
    ) -> list[EnrichedTeamMember]:
        """Return the team with every LinkedIn URL either verified or searched for.

        Members are returned in their original order whatever happens, and a failed
        lookup simply leaves ``linkedin_url`` as None.
        """
        allowed = {u for u in (canonical_profile_url(p) for p in verified_profiles) if u}
        enriched: list[EnrichedTeamMember] = []
        pending: list[int] = []

        for member in members:
            candidate = canonical_profile_url(member.linkedin_url or "")
            if candidate and candidate in allowed:
                # Came off the company's own pages: the strongest evidence there is.
                slug_tokens = text_tokens(profile_slug(candidate))
                parts = name_tokens(member.name)
                matches_name = bool(parts) and all(
                    p in slug_tokens or p in squash_text(profile_slug(candidate)) for p in (parts[0], parts[-1])
                )
                enriched.append(
                    EnrichedTeamMember(
                        **member.model_dump(exclude={"linkedin_url"}),
                        linkedin_url=candidate,
                        title_source="website" if member.title else None,
                        linkedin_source="website",
                        linkedin_confidence=0.95 if matches_name else 0.75,
                        linkedin_evidence=[
                            "url harvested from the company's own HTML"
                            + (" and matches the person's name" if matches_name else ""),
                        ],
                    )
                )
                continue

            note: list[str] = []
            if member.linkedin_url:
                # The model produced a URL that is not in the verified list. That is
                # an invention, and it is discarded regardless of how real it looks.
                note.append("model-supplied url discarded: not found in page HTML")
                logger.debug(
                    "%s: discarding unverified url %r for %s",
                    domain, member.linkedin_url, member.name,
                )
            enriched.append(
                EnrichedTeamMember(
                    **member.model_dump(exclude={"linkedin_url"}),
                    linkedin_url=None,
                    title_source="website" if member.title else None,
                    linkedin_evidence=note,
                )
            )
            pending.append(len(enriched) - 1)

        if not pending:
            return enriched

        if not self.enabled:
            for index in pending:
                enriched[index].linkedin_evidence.append("linkedin search is disabled")
            return enriched

        # Cap the outbound requests per domain. A 50-person team page is not worth
        # 50 search queries, and the people listed first are the leadership.
        budget = pending[: self.settings.linkedin_search_max_lookups]
        for index in pending[len(budget):]:
            enriched[index].linkedin_evidence.append(
                f"not searched: per-domain lookup budget of "
                f"{self.settings.linkedin_search_max_lookups} reached"
            )
        if len(pending) > len(budget):
            logger.debug(
                "%s: %d member(s) left unsearched (lookup budget %d)",
                domain, len(pending) - len(budget), self.settings.linkedin_search_max_lookups,
            )

        outcomes = await asyncio.gather(
            *(self.lookup(enriched[i], domain, company_name) for i in budget),
            return_exceptions=True,
        )

        found = 0
        for index, outcome in zip(budget, outcomes):
            if isinstance(outcome, BaseException):
                enriched[index].linkedin_evidence.append(
                    f"search failed: {compact_error(outcome, 80)}"
                )
                logger.debug("%s: search task failed: %s", domain, compact_error(outcome))
                continue
            match, note = outcome
            if match is None:
                enriched[index].linkedin_evidence.append(note)
                continue
            enriched[index].linkedin_url = match.url
            enriched[index].linkedin_source = "search"
            enriched[index].linkedin_confidence = match.confidence
            enriched[index].linkedin_evidence.extend(match.evidence)

            # The result that proved the profile usually states the role too, and
            # it is better evidence than a page that never mentioned one. Only
            # ever fills a gap: a title the site stated is left alone.
            if match.role and not enriched[index].title:
                enriched[index].title = match.role
                enriched[index].title_source = "search"
                enriched[index].linkedin_evidence.append(
                    f"title read from the matched search result: {match.role}"
                )
            found += 1

        if found:
            logger.info("%s: search fallback resolved %d/%d missing profile(s)", domain, found, len(budget))
        return enriched
