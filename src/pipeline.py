"""Orchestration: run every domain, isolate every failure, write results as they land.

The controlling requirement from the brief is *"the script must never crash midway
when one site fails"*. That is enforced structurally here rather than by scattering
try/except through the codebase:

* each domain runs inside ``enrich_domain``, which catches everything and returns a
  ``DomainRecord`` with ``status="failed"`` instead of propagating;
* ``asyncio.gather(..., return_exceptions=True)`` means even an exception escaping
  that guard cannot take down its siblings;
* results are written to disk after each domain completes, so a Ctrl-C or a power
  cut leaves a usable partial output file rather than nothing.

A domain is ``success`` when the LLM returned a valid extraction and every page
fetched, ``partial`` when it returned data but some pages failed, and ``failed``
when there is no extraction at all.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .cleaner import html_to_markdown, remove_cross_page_boilerplate
from .config import Settings
from .contacts import split_contacts
from .discovery import discover_subpages
from .extractor import GroqExtractor
from .fetcher import Fetcher
from .harvester import harvest_all
from .linkedin import LinkedInResolver, company_name_from_html
from .models import DomainRecord, EnrichedTeamMember, PageRecord, RunSummary, UsageRecord
from .scoring import compute_confidence
from .team import drop_unevidenced, filter_own_team, rank_team
from .utils import get_logger, normalise_domain, root_url

logger = get_logger("pipeline")


async def _fetch_page(
    fetcher: Fetcher,
    url: str,
    score: float,
    semaphore: asyncio.Semaphore,
    delay: float,
) -> tuple[PageRecord, str, str]:
    """Fetch one page under the concurrency limit. Returns (record, raw_html, markdown)."""
    async with semaphore:
        if delay:
            await asyncio.sleep(delay)
        result, markdown = await fetcher.fetch(url, html_to_markdown)

    record = PageRecord(
        url=url,
        status="ok" if (result.ok and markdown) else "failed",
        tier=result.tier,
        http_status=result.http_status,
        raw_chars=len(result.html),
        clean_chars=len(markdown),
        relevance_score=score,
        error=result.error or (None if markdown else "no extractable text"),
        duration_seconds=round(result.duration, 2),
    )
    return record, result.html, markdown


async def enrich_domain(
    raw_domain: str,
    fetcher: Fetcher,
    extractor: GroqExtractor,
    settings: Settings,
    resolver: LinkedInResolver | None = None,
) -> DomainRecord:
    """Process a single domain end to end. Never raises."""
    started = time.perf_counter()
    try:
        domain = normalise_domain(raw_domain)
        base = root_url(domain)
    except Exception as exc:
        return DomainRecord.failed(str(raw_domain), "", f"invalid domain: {exc}")

    logger.info("=" * 58)
    logger.info("starting %s", domain)

    errors: list[str] = []
    page_records: list[PageRecord] = []
    raw_pages: dict[str, tuple[str, str]] = {}
    markdown_pages: dict[str, str] = {}

    try:
        # --- 1. homepage -------------------------------------------------- #
        home_record, home_html, home_md = await _fetch_page(
            fetcher, base, 100.0, asyncio.Semaphore(1), 0.0
        )
        page_records.append(home_record)

        if home_record.status != "ok":
            errors.append(f"homepage unreachable: {home_record.error}")
            logger.warning("%s: homepage failed (%s)", domain, home_record.error)
            return DomainRecord(
                domain=domain, url=base, status="failed",
                pages_crawled=page_records, errors=errors,
                duration_seconds=round(time.perf_counter() - started, 2),
            )

        raw_pages[base] = (home_html, home_md)
        markdown_pages[base] = home_md
        logger.info("%s: homepage ok via %s (%d chars clean)", domain, home_record.tier, len(home_md))

        # --- 2. discover and fetch sub-pages ------------------------------ #
        try:
            targets = await discover_subpages(fetcher.client, domain, base, home_html, settings)
        except Exception as exc:
            targets = []
            errors.append(f"subpage discovery failed: {exc}")
            logger.warning("%s: discovery failed (%s), continuing with homepage only", domain, exc)

        if targets:
            semaphore = asyncio.Semaphore(settings.page_concurrency)
            results = await asyncio.gather(
                *(
                    _fetch_page(fetcher, url, score, semaphore, settings.politeness_delay)
                    for url, score in targets
                ),
                return_exceptions=True,
            )
            for item in results:
                if isinstance(item, BaseException):
                    errors.append(f"page task crashed: {item}")
                    continue
                record, raw_html, markdown = item
                page_records.append(record)
                if record.status == "ok":
                    raw_pages[record.url] = (raw_html, markdown)
                    markdown_pages[record.url] = markdown

        ok_count = sum(1 for p in page_records if p.status == "ok")
        logger.info("%s: %d/%d pages usable", domain, ok_count, len(page_records))

        # --- 3. clean and harvest ----------------------------------------- #
        markdown_pages = remove_cross_page_boilerplate(markdown_pages)
        harvested = harvest_all(raw_pages)

        total_chars = sum(len(v) for v in markdown_pages.values())
        logger.info(
            "%s: %d chars of clean text, %d emails, %d linkedin profiles",
            domain, total_chars,
            len(harvested["emails"]), len(harvested["linkedin_profiles"]),
        )

        if total_chars < 200:
            errors.append("insufficient text extracted for a meaningful LLM call")
            return DomainRecord(
                domain=domain, url=base, status="failed",
                pages_crawled=page_records, errors=errors,
                duration_seconds=round(time.perf_counter() - started, 2),
            )

        # --- 4. extract ---------------------------------------------------- #
        extraction, usage, llm_errors, attempts = await extractor.extract(
            domain, markdown_pages, harvested
        )
        errors.extend(llm_errors)

        # --- 5. sort addresses into contact points and personal inboxes ---- #
        contact_emails, personal_emails = split_contacts(
            list(extraction.contact_emails) if extraction else []
        )

        # --- 6. drop other companies' people ------------------------------- #
        # Before any lookup, because searching for a customer's PM against this
        # company can only ever fail - and it would burn a lookup doing it.
        team_notes: list[str] = []
        company_name = company_name_from_html(home_html, domain)
        if extraction is not None:
            kept, team_notes = filter_own_team(extraction.team_members, domain, company_name)
            # Rank before resolving, so the finite lookup budget is spent on the
            # founder rather than on whoever the page happened to list first.
            extraction = extraction.model_copy(update={"team_members": rank_team(kept)})

        # --- 7. resolve LinkedIn profiles ---------------------------------- #
        # Runs after extraction and before scoring, so a profile recovered by
        # search counts towards the confidence score exactly like one the site
        # linked itself. The model is not involved: URLs are either verified
        # against the harvested HTML or corroborated from a search result.
        if extraction is not None:
            try:
                active = resolver or LinkedInResolver(settings, provider=None)
                team = await active.resolve_team(
                    extraction.team_members,
                    domain,
                    harvested.get("linkedin_profiles", []),
                    company_name,
                )
            except Exception as exc:
                # LinkedIn resolution is an enhancement; it must never cost us the
                # extraction we already paid the model for.
                errors.append(f"linkedin resolution failed: {exc}")
                logger.warning("%s: linkedin resolution failed (%s)", domain, exc)
                # Fall back to the model's own members with no provenance claimed,
                # rather than losing the team entirely.
                team = [
                    EnrichedTeamMember(
                        **m.model_dump(exclude={"linkedin_url"}),
                        linkedin_url=None,
                        title_source="website" if m.title else None,
                    )
                    for m in extraction.team_members
                ]
            else:
                searched = sum(1 for m in team if m.linkedin_source == "search")
                if searched:
                    logger.info("%s: %d profile(s) recovered by search fallback", domain, searched)
            # The evidence rule, applied last: by now a person has had every
            # chance to acquire a title or a verified profile. A name with
            # neither was never established as a person at this company.
            team, evidence_notes = drop_unevidenced(team)
            team_notes.extend(evidence_notes)
            extraction = extraction.model_copy(update={"team_members": rank_team(team)})

        if extraction is not None:
            # Scoring counts usable contact points; a colleague's inbox is not one.
            extraction = extraction.model_copy(
                update={"contact_emails": [c.email for c in contact_emails]}
            )

        confidence = compute_confidence(
            extraction=extraction,
            pages=page_records,
            target_pages=settings.max_pages_per_domain,
            attempts_used=attempts,
            max_attempts=settings.llm_max_attempts,
            errors=llm_errors,
            extra_notes=team_notes,
        )

        if extraction is None:
            return DomainRecord(
                domain=domain, url=base, status="failed",
                pages_crawled=page_records, usage=usage, errors=errors,
                confidence_breakdown=confidence,
                duration_seconds=round(time.perf_counter() - started, 2),
            )

        failed_pages = sum(1 for p in page_records if p.status != "ok")
        status = "partial" if (failed_pages or errors) else "success"

        return DomainRecord(
            domain=domain,
            url=base,
            status=status,
            company_overview=extraction.company_overview,
            target_audience=extraction.target_audience,
            contact_emails=contact_emails,
            personal_emails=personal_emails,
            team_members=extraction.team_members,
            data_confidence_score=confidence.final,
            confidence_breakdown=confidence,
            pages_crawled=page_records,
            usage=usage,
            errors=errors,
            duration_seconds=round(time.perf_counter() - started, 2),
        )

    except Exception as exc:  # the final guard: nothing escapes a domain
        logger.exception("%s: unhandled error", raw_domain)
        errors.append(f"unhandled: {type(exc).__name__}: {exc}")
        return DomainRecord(
            domain=str(raw_domain), url="", status="failed",
            pages_crawled=page_records, errors=errors,
            duration_seconds=round(time.perf_counter() - started, 2),
        )


def _write_output(path: Path, summary: RunSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Write to a temp file then replace, so an interrupted write can never leave a
    # truncated output.json behind.
    tmp.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def _build_summary(
    records: list[DomainRecord], started_at: datetime, model: str, requested: int
) -> RunSummary:
    finished_at = datetime.now(timezone.utc)
    return RunSummary(
        run_started_at=started_at.isoformat(timespec="seconds"),
        run_finished_at=finished_at.isoformat(timespec="seconds"),
        duration_seconds=round((finished_at - started_at).total_seconds(), 2),
        model=model,
        domains_requested=requested,
        domains_succeeded=sum(1 for r in records if r.status == "success"),
        domains_partial=sum(1 for r in records if r.status == "partial"),
        domains_failed=sum(1 for r in records if r.status == "failed"),
        total_tokens=sum(r.usage.total_tokens for r in records),
        total_estimated_cost_usd=round(sum(r.usage.estimated_cost_usd for r in records), 6),
        results=records,
    )


async def run_pipeline(
    domains: list[str], settings: Settings, output_path: str | None = None
) -> RunSummary:
    """Entry point: enrich every domain and write the output file."""
    started_at = datetime.now(timezone.utc)
    path = Path(output_path or settings.output_path)
    records: list[DomainRecord] = []
    lock = asyncio.Lock()

    async with (
        Fetcher(settings) as fetcher,
        GroqExtractor(settings) as extractor,
        LinkedInResolver(settings) as resolver,
    ):
        semaphore = asyncio.Semaphore(settings.domain_concurrency)

        async def worker(domain: str) -> None:
            async with semaphore:
                record = await enrich_domain(domain, fetcher, extractor, settings, resolver)
            async with lock:
                records.append(record)
                # Incremental write: the file on disk is always valid and current.
                _write_output(path, _build_summary(records, started_at, settings.model, len(domains)))

        results = await asyncio.gather(
            *(worker(d) for d in domains), return_exceptions=True
        )
        for domain, outcome in zip(domains, results):
            if isinstance(outcome, BaseException):
                logger.error("worker for %s crashed: %s", domain, outcome)
                records.append(DomainRecord.failed(str(domain), "", f"worker crashed: {outcome}"))

    # Preserve the caller's input order in the final file.
    order = {normalise_domain_safe(d): i for i, d in enumerate(domains)}
    records.sort(key=lambda r: order.get(r.domain, 999))

    summary = _build_summary(records, started_at, settings.model, len(domains))
    _write_output(path, summary)
    return summary


def normalise_domain_safe(value: str) -> str:
    try:
        return normalise_domain(value)
    except Exception:
        return str(value)


def load_domains(raw: list[str] | None, input_file: str | None) -> list[str]:
    """Read domains from CLI args or a text/CSV file, deduplicated, order preserved."""
    domains: list[str] = list(raw or [])

    if input_file:
        content = Path(input_file).read_text(encoding="utf-8")
        for line in content.splitlines():
            value = line.split(",")[0].strip()
            if value and not value.startswith("#"):
                domains.append(value)

    seen: set[str] = set()
    unique: list[str] = []
    for domain in domains:
        key = normalise_domain_safe(domain)
        if key not in seen:
            seen.add(key)
            unique.append(domain)
    return unique
