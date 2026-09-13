"""Central configuration.

Every tunable in the pipeline is defined here and sourced from environment
variables, so no module below this one reads ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------------------------------- #
# Sub-page relevance weights
# --------------------------------------------------------------------------- #
# Discovery does not hard-code a list of paths to try. It scores every candidate
# URL found in the sitemap and the homepage navigation against these keywords and
# crawls the highest scorers, so a site using /our-story instead of /about is
# still handled correctly.

PAGE_KEYWORDS: dict[str, float] = {
    "about": 10.0,
    "about-us": 10.0,
    "our-story": 9.0,
    "who-we-are": 9.0,
    "team": 10.0,
    "our-team": 10.0,
    "leadership": 9.5,
    "founders": 9.5,
    "management": 8.0,
    "people": 7.0,
    "company": 8.0,
    "contact": 9.0,
    "contact-us": 9.0,
    "support": 6.0,
    "pricing": 6.5,
    "plans": 5.5,
    "customers": 5.0,
    "solutions": 4.5,
    "product": 4.5,
    "platform": 4.5,
    "use-cases": 4.0,
    # Weighted like an about/contact page, not like a footer link: on sites with
    # no /about or /company section, /careers is often the only page that talks
    # about the company rather than the product, and it carries a real contact
    # address (vapi.ai publishes talent@vapi.ai there and nowhere else reachable).
    "careers": 8.0,
    "press": 3.0,
    "investors": 3.0,
}

# Paths that are almost never worth an LLM call for company intelligence.
NEGATIVE_KEYWORDS: dict[str, float] = {
    "blog": -8.0,
    "docs": -8.0,
    "documentation": -8.0,
    "changelog": -7.0,
    "release": -6.0,
    "terms": -9.0,
    "privacy": -9.0,
    "legal": -8.0,
    "cookie": -9.0,
    "login": -9.0,
    "signup": -9.0,
    "sign-up": -9.0,
    "register": -9.0,
    "download": -6.0,
    "status": -6.0,
    "api": -5.0,
    "reference": -6.0,
    "tutorial": -6.0,
    "guide": -5.0,
    "webinar": -5.0,
    "event": -5.0,
    "search": -8.0,
    "tag": -7.0,
    "category": -6.0,
    "author": -7.0,
}

# Per-million-token prices in USD. Groq publishes these per model; they are kept
# here (not hard-coded at the call site) so the cost report stays accurate when
# the model changes. Update if Groq revises pricing.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model name: (input $/1M tokens, output $/1M tokens)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "moonshotai/kimi-k2-instruct": (1.00, 3.00),
}

DEFAULT_PRICING: tuple[float, float] = (0.59, 0.79)

# Browser identity. A real UA string avoids trivially triggering bot filters;
# we still respect robots.txt and rate-limit ourselves.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings, built once at startup."""

    # --- LLM -------------------------------------------------------------- #
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    llm_timeout: float = 90.0
    llm_max_attempts: int = 3
    temperature: float = 0.1

    # --- Crawling --------------------------------------------------------- #
    max_pages_per_domain: int = 6
    http_timeout: float = 20.0
    browser_timeout: float = 30.0
    domain_concurrency: int = 3
    page_concurrency: int = 3
    politeness_delay: float = 0.4
    max_fetch_attempts: int = 3
    respect_robots: bool = True

    # --- Token budget ------------------------------------------------------ #
    # Hard ceilings enforced before the LLM is ever called. This is the
    # "do not feed raw HTML to the model" requirement, made measurable.
    max_chars_per_page: int = 6_000
    max_chars_per_domain: int = 24_000

    # --- LinkedIn search fallback ------------------------------------------ #
    # When a team member is found with no profile link on the site, a search
    # engine is queried for one. Capped and throttled: this is a small number of
    # extra requests per domain, not a second crawl.
    linkedin_search_enabled: bool = True
    # auto | serper | brave-api | brave-html | none
    linkedin_search_provider: str = "auto"
    serper_api_key: str = ""
    brave_api_key: str = ""
    linkedin_search_timeout: float = 15.0
    linkedin_search_results: int = 6
    linkedin_search_max_lookups: int = 5
    linkedin_search_delay: float = 1.2
    # Below this, a candidate is discarded rather than guessed at.
    linkedin_min_confidence: float = 0.75

    # --- Output ------------------------------------------------------------ #
    output_path: str = "output/output.json"

    keywords: dict[str, float] = field(default_factory=lambda: dict(PAGE_KEYWORDS))
    negative_keywords: dict[str, float] = field(
        default_factory=lambda: dict(NEGATIVE_KEYWORDS)
    )

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            llm_timeout=_env_float("LLM_TIMEOUT", 90.0),
            llm_max_attempts=_env_int("LLM_MAX_ATTEMPTS", 3),
            temperature=_env_float("LLM_TEMPERATURE", 0.1),
            max_pages_per_domain=_env_int("MAX_PAGES_PER_DOMAIN", 6),
            http_timeout=_env_float("HTTP_TIMEOUT", 20.0),
            browser_timeout=_env_float("BROWSER_TIMEOUT", 30.0),
            domain_concurrency=_env_int("DOMAIN_CONCURRENCY", 3),
            page_concurrency=_env_int("PAGE_CONCURRENCY", 3),
            politeness_delay=_env_float("POLITENESS_DELAY", 0.4),
            max_fetch_attempts=_env_int("MAX_FETCH_ATTEMPTS", 3),
            respect_robots=os.getenv("RESPECT_ROBOTS", "true").lower() != "false",
            max_chars_per_page=_env_int("MAX_CHARS_PER_PAGE", 6_000),
            max_chars_per_domain=_env_int("MAX_CHARS_PER_DOMAIN", 24_000),
            linkedin_search_enabled=os.getenv("LINKEDIN_SEARCH_ENABLED", "true").lower() != "false",
            linkedin_search_provider=os.getenv("LINKEDIN_SEARCH_PROVIDER", "auto").strip().lower(),
            serper_api_key=os.getenv("SERPER_API_KEY", "").strip(),
            brave_api_key=os.getenv("BRAVE_API_KEY", "").strip(),
            linkedin_search_timeout=_env_float("LINKEDIN_SEARCH_TIMEOUT", 15.0),
            linkedin_search_results=_env_int("LINKEDIN_SEARCH_RESULTS", 6),
            linkedin_search_max_lookups=_env_int("LINKEDIN_SEARCH_MAX_LOOKUPS", 5),
            linkedin_search_delay=_env_float("LINKEDIN_SEARCH_DELAY", 1.2),
            linkedin_min_confidence=_env_float("LINKEDIN_MIN_CONFIDENCE", 0.75),
            output_path=os.getenv("OUTPUT_PATH", "output/output.json"),
        )

    def price_for(self, model: str | None = None) -> tuple[float, float]:
        """Return (input, output) $/1M tokens for a model, with a safe default."""
        return MODEL_PRICING.get(model or self.model, DEFAULT_PRICING)
