"""Small shared helpers: logging, async retry, URL normalisation, token estimation."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import unicodedata
from functools import wraps
from typing import Awaitable, Callable, Iterable, TypeVar
from urllib.parse import urlparse, urlunparse

T = TypeVar("T")

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once. Third-party noise is turned down."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #


def async_retry(
    attempts: int = 3,
    base_delay: float = 0.8,
    max_delay: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Retry an async callable with exponential backoff and jitter.

    Written by hand rather than pulled from ``tenacity`` to keep the dependency
    list minimal and the behaviour explicit: the delay is
    ``base_delay * 2**n`` capped at ``max_delay``, plus up to 25% jitter so
    concurrent workers do not retry in lockstep against the same host.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203 - retry is the point
                    last = exc
                    if attempt == attempts:
                        break
                    retry_after = getattr(exc, "retry_after", None)
                    if retry_after is not None:
                        delay = max(float(retry_after), 0.0)
                    else:
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                        delay *= 1 + random.random() * 0.25
                    delay *= 1 + random.random() * 0.25
                    if logger:
                        logger.debug(
                            "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                            func.__name__,
                            attempt,
                            attempts,
                            exc,
                            delay,
                        )
                    await asyncio.sleep(delay)
            assert last is not None
            raise last

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# URLs
# --------------------------------------------------------------------------- #


HOSTNAME_RE = re.compile(
    r"^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)+$"
)


def normalise_domain(raw: str) -> str:
    """'https://www.Postman.com/pricing/' -> 'postman.com'.

    Raises ``ValueError`` on anything that is not a plausible hostname, so junk
    input is rejected at the edge rather than turning into a doomed HTTP request.
    """
    value = (raw or "").strip().lower()
    if not value:
        raise ValueError("empty domain")
    if "://" not in value:
        value = "https://" + value
    host = urlparse(value).netloc or urlparse(value).path
    host = host.split("@")[-1].split(":")[0]
    host = host.removeprefix("www.").strip("/")
    if not HOSTNAME_RE.match(host):
        raise ValueError(f"not a valid hostname: {raw!r}")
    return host


def root_url(domain: str) -> str:
    return f"https://{normalise_domain(domain)}"


def normalise_url(url: str) -> str:
    """Drop fragments, query strings, trailing slashes and ``www.`` so we never crawl twice.

    Stripping ``www.`` matters more than it looks: sitemaps and navigation often
    mix the two spellings, and without this ``https://www.postman.com/`` is a
    different URL from ``https://postman.com/``. A real run crawled the homepage
    twice for that reason, spending a page of its budget and a full LLM context
    on a byte-identical copy. Redirects handle whichever spelling the site
    actually prefers.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return url.strip()
    path = re.sub(r"/{2,}", "/", parsed.path or "/").rstrip("/") or "/"
    netloc = parsed.netloc.lower().removeprefix("www.")
    return urlunparse((parsed.scheme.lower(), netloc, path, "", "", ""))


def same_registrable_domain(url: str, domain: str) -> bool:
    """True when ``url`` belongs to ``domain`` (subdomains allowed, others not)."""
    try:
        host = urlparse(url).netloc.lower().split(":")[0].removeprefix("www.")
    except ValueError:
        return False
    return host == domain or host.endswith("." + domain)


def url_depth(url: str) -> int:
    path = urlparse(url).path.strip("/")
    return 0 if not path else len(path.split("/"))


NON_HTML_SUFFIXES = (
    ".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".css", ".js", ".json", ".xml", ".mp4", ".mp3", ".woff", ".woff2", ".ttf",
    ".dmg", ".exe", ".pkg", ".deb", ".rpm", ".tar", ".gz",
)


def looks_like_html(url: str) -> bool:
    return not urlparse(url).path.lower().endswith(NON_HTML_SUFFIXES)


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #


def estimate_tokens(text: str) -> int:
    """Rough pre-flight token estimate (~4 chars/token).

    Used only to enforce the budget *before* calling the API; the numbers written
    to the output file are the exact counts the API reports back, never this.
    """
    return max(1, len(text or "") // 4)


def truncate_to_chars(text: str, limit: int) -> str:
    """Cut text at a paragraph boundary near ``limit`` so we never split mid-word."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    for sep in ("\n\n", "\n", ". "):
        idx = window.rfind(sep)
        if idx > limit * 0.6:
            return window[:idx].rstrip()
    return window.rstrip()


def compact_error(error: BaseException | str, limit: int = 180) -> str:
    """Flatten an exception into one short line.

    Playwright in particular raises errors carrying a multi-line "Call log:"
    dump. Left alone it makes the terminal report unreadable and bloats every
    failed record in output.json, so errors are collapsed and truncated at the
    point they are captured.
    """
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error)
    text = re.sub(r"\s+", " ", text).strip()
    # Playwright appends the navigation log after the useful message.
    text = text.split("Call log:")[0].strip()
    return text[: limit - 1] + "…" if len(text) > limit else text


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Name and company text matching
# --------------------------------------------------------------------------- #
# Shared by LinkedIn resolution and team filtering: both have to decide whether
# two pieces of free text name the same person or the same company, across
# accents, punctuation, percent-encoding and run-together spellings.


def fold_text(text: str) -> str:
    """Lowercase and strip accents, so 'Jose Munoz' matches a 'jose-munoz' slug."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


def text_tokens(text: str, *, minimum: int = 2) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", fold_text(text)) if len(t) >= minimum}


def squash_text(text: str) -> str:
    """'Acme Data Systems' -> 'acmedatasystems', for matching run-together names."""
    return re.sub(r"[^a-z0-9]+", "", fold_text(text))


# Tokens that are part of a name but carry no identifying signal.
NAME_NOISE = {
    "jr", "sr", "ii", "iii", "iv", "phd", "mba", "md", "dr", "mr", "ms", "mrs",
    "prof", "the",
}

# Generic words in a company's name that would match half the internet.
COMPANY_STOPWORDS = {
    "the", "inc", "llc", "ltd", "labs", "lab", "group", "technologies", "tech",
    "software", "systems", "solutions", "company", "corp", "io", "ai", "app",
    "com", "co", "get", "try", "hq", "cloud", "digital", "studio", "agency",
}


def name_tokens(name: str) -> list[str]:
    """Identifying parts of a person's name, in order, initials and suffixes dropped."""
    parts = [p for p in re.split(r"[^a-z0-9]+", fold_text(name)) if p]
    return [p for p in parts if len(p) >= 2 and p not in NAME_NOISE]


def company_tokens(domain: str, company_name: str = "") -> tuple[str, set[str]]:
    """Return the company's squashed label plus its distinctive tokens.

    The label is the registrable name from the domain ('acmedata' from
    'acmedata.io'); it is matched against squashed text so 'Acme Data' still hits.
    """
    label = squash_text(domain.split(".")[0])
    tokens = {
        t for t in text_tokens(company_name) | text_tokens(domain)
        if t not in COMPANY_STOPWORDS
    }
    if label and label not in COMPANY_STOPWORDS:
        tokens.add(label)
    return label, tokens
