"""robots.txt parsing: declared sitemaps, Disallow rules, and safe failure."""

import asyncio

from src.config import Settings
from src.discovery import _allowed, fetch_robots

SETTINGS = Settings(groq_api_key="x")

ROBOTS = """
User-agent: *
Disallow: /admin/
Disallow: /internal
Allow: /

Sitemap: https://acme.io/sitemap_pages.xml
Sitemap: https://acme.io/sitemap_posts.xml
"""


class FakeClient:
    def __init__(self, body=None, exc=None):
        self.body, self.exc = body, exc

    async def get(self, url, **kwargs):
        if self.exc:
            raise self.exc

        class Response:
            status_code = 200
            text = self.body

        return Response()


def run(coro):
    return asyncio.run(coro)


def test_declared_sitemaps_are_returned():
    parser, sitemaps = run(fetch_robots(FakeClient(ROBOTS), "https://acme.io", SETTINGS))
    assert sitemaps == [
        "https://acme.io/sitemap_pages.xml",
        "https://acme.io/sitemap_posts.xml",
    ]
    assert parser is not None


def test_disallowed_paths_are_blocked():
    parser, _ = run(fetch_robots(FakeClient(ROBOTS), "https://acme.io", SETTINGS))
    assert not _allowed(parser, "https://acme.io/admin/users", SETTINGS)
    assert not _allowed(parser, "https://acme.io/internal", SETTINGS)
    assert _allowed(parser, "https://acme.io/about", SETTINGS)
    assert _allowed(parser, "https://acme.io/team", SETTINGS)


def test_missing_robots_blocks_nothing():
    parser, sitemaps = run(fetch_robots(FakeClient(exc=RuntimeError("404")), "https://acme.io", SETTINGS))
    assert parser is None
    assert sitemaps == []
    assert _allowed(None, "https://acme.io/anything", SETTINGS)


def test_malformed_robots_never_blocks_the_crawl():
    parser, sitemaps = run(fetch_robots(FakeClient("<<<garbage>>>"), "https://acme.io", SETTINGS))
    assert sitemaps == []
    assert _allowed(parser, "https://acme.io/about", SETTINGS)


def test_respect_robots_can_be_disabled():
    parser, _ = run(fetch_robots(FakeClient(ROBOTS), "https://acme.io", SETTINGS))
    relaxed = Settings(groq_api_key="x", respect_robots=False)
    assert _allowed(parser, "https://acme.io/admin/users", relaxed)
