"""Two-tier page fetching: cheap HTTP first, headless browser only when needed.

Launching Chromium for every URL is correct but wasteful - most marketing pages
are server-rendered and a plain GET returns the full document in ~200ms. Some are
not: ``vapi.ai`` and similar React/Next apps return an empty ``<div id="root">``
shell that no amount of parsing will help with.

So tier 1 is httpx. If the response is blocked, errored, or suspiciously thin once
stripped, tier 2 escalates the *same* URL to Playwright with images, fonts and
media blocked at the network layer so the render is as fast as it can be. The tier
used is recorded per page in the output, which makes the trade-off visible instead
of assumed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from .config import USER_AGENT, Settings
from .utils import async_retry, compact_error, get_logger

logger = get_logger("fetcher")

# Status codes that mean "a browser might get through where this did not".
ESCALATE_STATUSES = {401, 403, 405, 406, 429, 500, 502, 503, 520, 521, 522, 530}

# Below this many characters of markdown, assume the page needs JavaScript.
THIN_CONTENT_CHARS = 600

BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet", "websocket", "manifest"}


@dataclass
class FetchResult:
    url: str
    html: str
    tier: str  # "http" | "browser" | "none"
    http_status: int | None
    error: str | None
    duration: float

    @property
    def ok(self) -> bool:
        return bool(self.html) and self.error is None


class Fetcher:
    """Owns the HTTP client and a single shared browser instance for the whole run.

    Used as an async context manager so the browser is always closed, including on
    exceptions and Ctrl-C.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None
        self._playwright = None
        self._browser = None
        self._browser_lock = asyncio.Lock()
        self._browser_failed = False

    async def __aenter__(self) -> "Fetcher":
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=self.settings.http_timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._client:
            await self._client.aclose()
        if self._browser:
            try:
                await self._browser.close()
            except Exception as exc:
                logger.debug("browser close failed: %s", exc)
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as exc:
                logger.debug("playwright stop failed: %s", exc)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Fetcher must be used as an async context manager")
        return self._client

    # ----------------------------------------------------------------- tier 1
    async def _get(self, url: str) -> httpx.Response:
        """GET with backoff on transient failures only.

        The retry policy is built per call rather than as a class-level decorator
        so it can read ``max_fetch_attempts`` from settings. Only timeouts and
        network errors are retried - a malformed URL or a refused connection will
        fail the same way every time, and retrying it just wastes the budget.
        """

        @async_retry(
            attempts=self.settings.max_fetch_attempts,
            base_delay=0.6,
            max_delay=6.0,
            exceptions=(httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError),
            logger=logger,
        )
        async def _attempt() -> httpx.Response:
            return await self.client.get(url)

        return await _attempt()

    async def fetch_http(self, url: str) -> FetchResult:
        started = time.perf_counter()
        try:
            response = await self._get(url)
            elapsed = time.perf_counter() - started
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "xml" not in content_type:
                return FetchResult(
                    url, "", "http", response.status_code,
                    f"non-html content-type: {content_type or 'unknown'}", elapsed,
                )
            return FetchResult(
                url, response.text, "http", response.status_code, None, elapsed
            )
        except httpx.TimeoutException:
            return FetchResult(url, "", "http", None, "http timeout", time.perf_counter() - started)
        except Exception as exc:
            return FetchResult(
                url, "", "http", None, compact_error(exc),
                time.perf_counter() - started,
            )

    # ----------------------------------------------------------------- tier 2
    async def _ensure_browser(self):
        """Lazily start one Chromium instance, shared by every page in the run."""
        if self._browser is not None:
            return self._browser
        if self._browser_failed:
            return None

        async with self._browser_lock:
            if self._browser is not None:
                return self._browser
            try:
                from playwright.async_api import async_playwright

                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                    ],
                )
                logger.info("headless chromium started")
            except Exception as exc:
                # A missing browser binary degrades the run to HTTP-only rather
                # than killing it. The reason is logged once and surfaced per page.
                self._browser_failed = True
                logger.warning(
                    "could not start Playwright (%s). Falling back to HTTP-only. "
                    "Run 'playwright install chromium' to enable JS rendering.",
                    exc,
                )
                return None
        return self._browser

    async def fetch_browser(self, url: str) -> FetchResult:
        started = time.perf_counter()
        browser = await self._ensure_browser()
        if browser is None:
            return FetchResult(
                url, "", "browser", None, "playwright unavailable",
                time.perf_counter() - started,
            )

        context = None
        try:
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1366, "height": 900},
                locale="en-US",
                java_script_enabled=True,
            )

            # Block heavy assets at the network layer: we only ever want the DOM.
            async def _route(route):
                try:
                    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
                        await route.abort()
                    else:
                        await route.continue_()
                except Exception:
                    pass

            await context.route("**/*", _route)

            page = await context.new_page()
            timeout_ms = int(self.settings.browser_timeout * 1000)
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            # Give client-side rendering a moment to settle, but never block on it:
            # networkidle legitimately never fires on sites with polling or chat
            # widgets, so a timeout here is expected and ignored.
            try:
                await page.wait_for_load_state("networkidle", timeout=4_000)
            except Exception:
                pass

            html = await page.content()
            status = response.status if response else None
            return FetchResult(url, html, "browser", status, None, time.perf_counter() - started)
        except Exception as exc:
            return FetchResult(
                url, "", "browser", None, compact_error(exc),
                time.perf_counter() - started,
            )
        finally:
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass

    # ------------------------------------------------------------- escalation
    @staticmethod
    def should_escalate(result: FetchResult, clean_length: int) -> str | None:
        """Return the reason to retry in a browser, or None if HTTP was enough."""
        if result.error:
            return f"http error ({result.error})"
        if result.http_status in ESCALATE_STATUSES:
            return f"status {result.http_status}"
        if clean_length < THIN_CONTENT_CHARS:
            return f"thin content ({clean_length} chars)"
        return None

    async def fetch(self, url: str, clean_fn) -> tuple[FetchResult, str]:
        """Fetch ``url`` and return the result plus its cleaned markdown.

        ``clean_fn`` is injected rather than imported so this module stays
        independent of the cleaning strategy and is trivial to test.
        """
        http_result = await self.fetch_http(url)
        markdown = clean_fn(http_result.html) if http_result.html else ""

        reason = self.should_escalate(http_result, len(markdown))
        if reason is None:
            return http_result, markdown

        logger.debug("escalating %s to browser: %s", url, reason)
        browser_result = await self.fetch_browser(url)
        if browser_result.ok:
            browser_markdown = clean_fn(browser_result.html)
            # Only keep the browser render if it actually produced more content.
            if len(browser_markdown) >= len(markdown):
                return browser_result, browser_markdown
            return http_result, markdown

        # Browser failed too. Return whatever HTTP managed, with both reasons.
        if http_result.html:
            return http_result, markdown
        combined = compact_error(f"{http_result.error or reason}; browser: {browser_result.error}", 220)
        return (
            FetchResult(url, "", "none", http_result.http_status, combined,
                        http_result.duration + browser_result.duration),
            "",
        )
