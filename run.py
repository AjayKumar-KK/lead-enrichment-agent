#!/usr/bin/env python3
"""Autonomous Lead Enrichment Agent - command line entry point.

Examples
--------
    python run.py                                   # the three assignment targets
    python run.py --domains stripe.com figma.com
    python run.py --input domains.txt --out output/run2.json
    python run.py --domains postman.com --max-pages 8 --verbose
    python run.py --list-models
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace

from src.config import Settings
from src.extractor import list_available_models
from src.models import RunSummary
from src.pipeline import load_domains, run_pipeline
from src.utils import get_logger, setup_logging

logger = get_logger("run")

DEFAULT_DOMAINS = ["postman.com", "supabase.com", "vapi.ai"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Crawl company websites and extract structured intelligence with an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domains", "-d", nargs="+", metavar="DOMAIN",
        help="One or more domains, e.g. --domains postman.com supabase.com",
    )
    parser.add_argument(
        "--input", "-i", metavar="FILE",
        help="Path to a .txt or .csv file with one domain per line.",
    )
    parser.add_argument(
        "--out", "-o", metavar="FILE", help="Output JSON path (default: output/output.json)",
    )
    parser.add_argument("--max-pages", type=int, help="Max pages to crawl per domain (default: 6)")
    parser.add_argument("--concurrency", type=int, help="Domains processed in parallel (default: 3)")
    parser.add_argument("--model", help="Override the Groq model id")
    parser.add_argument("--list-models", action="store_true", help="List models your API key can use, then exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def run_totals(summary: RunSummary) -> dict[str, int | float]:
    """Count what the run actually produced.

    Kept separate from the printing so the numbers on the dashboard are testable
    rather than assembled inline in an f-string.
    """
    records = summary.results
    pages = [page for r in records for page in r.pages_crawled]
    team = [m for r in records for m in r.team_members]
    scored = [r.data_confidence_score for r in records if r.status != "failed"]

    return {
        "domains": summary.domains_requested,
        "succeeded": summary.domains_succeeded,
        "partial": summary.domains_partial,
        "failed": summary.domains_failed,
        "pages_crawled": sum(1 for p in pages if p.status == "ok"),
        "pages_attempted": len(pages),
        "pages_by_browser": sum(1 for p in pages if p.status == "ok" and p.tier == "browser"),
        "team_members": len(team),
        "titles": sum(1 for m in team if m.title),
        "linkedin_urls": sum(1 for m in team if m.linkedin_url),
        "linkedin_by_search": sum(1 for m in team if m.linkedin_source == "search"),
        "contact_emails": sum(len(r.contact_emails) for r in records),
        "personal_emails": sum(len(r.personal_emails) for r in records),
        "mean_confidence": round(sum(scored) / len(scored), 2) if scored else 0.0,
        "tokens": summary.total_tokens,
        "cost": summary.total_estimated_cost_usd,
        "runtime": summary.duration_seconds,
    }


def print_dashboard(summary: RunSummary) -> None:
    """The headline: what the agent accomplished, in one glance.

    A reviewer should not have to read three screens of per-domain detail to
    learn whether the run worked. Every row is an outcome - things found, pages
    read, money spent - rather than a restatement of the configuration.
    """
    t = run_totals(summary)
    width = 46  # inner width, excluding the border characters

    def row(label: str, value: str) -> str:
        # 1 leading space + 22 label + value + 1 trailing space == width
        return f"│ {label:<22}{value:>{width - 24}} │"

    def rule(left: str, fill: str, right: str) -> str:
        return f"{left}{fill * width}{right}"

    title = "AI LEAD ENRICHMENT AGENT"
    lines = [
        "",
        rule("┌", "─", "┐"),
        f"│{title.center(width)}│",
        rule("├", "─", "┤"),
        row("Domains processed", str(t["domains"])),
        row("  successful", str(t["succeeded"])),
    ]
    if t["partial"]:
        lines.append(row("  partial", str(t["partial"])))
    lines += [
        row("  failed", str(t["failed"])),
        rule("├", "─", "┤"),
        row("Pages crawled", f"{t['pages_crawled']}/{t['pages_attempted']}"),
        row("Team members found", str(t["team_members"])),
        row("  with a job title", str(t["titles"])),
        row("LinkedIn URLs found", str(t["linkedin_urls"])),
        row("  recovered by search", str(t["linkedin_by_search"])),
        row("Public emails found", str(t["contact_emails"])),
    ]
    if t["personal_emails"]:
        lines.append(row("  personal (excluded)", str(t["personal_emails"])))
    lines += [
        row("Mean confidence", f"{t['mean_confidence']:.2f}"),
        rule("├", "─", "┤"),
        row("Total tokens", f"{t['tokens']:,}"),
        row("Estimated cost", f"${t['cost']:.5f}"),
        row("Runtime", f"{t['runtime']:.2f} sec"),
        rule("└", "─", "┘"),
    ]
    print("\n".join(lines))


def print_report(summary: RunSummary) -> None:
    """Human-readable terminal summary - this is what the Loom walkthrough shows."""
    print_dashboard(summary)

    line = "=" * 72
    print(f"\n{line}\n PER-DOMAIN DETAIL   (model: {summary.model})\n{line}")

    for record in summary.results:
        icon = {"success": "[ OK ]", "partial": "[WARN]", "failed": "[FAIL]"}[record.status]
        print(f"\n{icon} {record.domain}   confidence {record.data_confidence_score:.2f}")

        if record.company_overview:
            print(f"   overview : {record.company_overview[:150]}")
        if record.target_audience:
            print(f"   ICP      : {record.target_audience[:150]}")
        if record.contact_emails:
            shown = ", ".join(f"{c.email} [{c.type}]" for c in record.contact_emails[:4])
            print(f"   emails   : {shown}")
        if record.personal_emails:
            print(f"   personal : {', '.join(record.personal_emails[:3])} (not contact points)")
        if record.team_members:
            print(f"   team     : {len(record.team_members)} found")
            searched = sum(1 for m in record.team_members if m.linkedin_source == "search")
            if searched:
                print(f"              ({searched} profile(s) recovered by search fallback)")
            for member in record.team_members[:3]:
                if member.linkedin_url:
                    tail = (
                        f" | {member.linkedin_url}"
                        f" [{member.linkedin_source}, {member.linkedin_confidence:.2f}]"
                    )
                else:
                    tail = ""
                print(f"              - {member.name} ({member.title or 'title n/a'}){tail}")

        ok_pages = sum(1 for p in record.pages_crawled if p.status == "ok")
        tiers = {p.tier for p in record.pages_crawled if p.status == "ok"}
        print(
            f"   pages    : {ok_pages}/{len(record.pages_crawled)} ok"
            f" via {', '.join(sorted(tiers)) or 'n/a'}"
            f"  |  {record.usage.total_tokens:,} tokens"
            f"  |  ${record.usage.estimated_cost_usd:.5f}"
            f"  |  {record.duration_seconds}s"
        )
        if record.errors:
            print(f"   notes    : {record.errors[0][:160]}")
            for err in record.errors[1:3]:
                print(f"              {err[:160]}")

    print(f"\n{line}\n")


async def main_async() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    settings = Settings.from_env()
    if args.model:
        settings = replace(settings, model=args.model)
    if args.max_pages:
        settings = replace(settings, max_pages_per_domain=args.max_pages)
    if args.concurrency:
        settings = replace(settings, domain_concurrency=args.concurrency)

    if not settings.groq_api_key:
        print(
            "\nERROR: GROQ_API_KEY is not set.\n\n"
            "  1. Get a free key at https://console.groq.com/keys\n"
            "  2. Copy .env.example to .env\n"
            "  3. Put the key in it as GROQ_API_KEY=gsk_...\n",
            file=sys.stderr,
        )
        return 2

    if args.list_models:
        try:
            for model_id in await list_available_models(settings):
                print(model_id)
            return 0
        except Exception as exc:
            print(f"Could not list models: {exc}", file=sys.stderr)
            return 1

    domains = load_domains(args.domains, args.input) or DEFAULT_DOMAINS
    logger.info("enriching %d domain(s): %s", len(domains), ", ".join(domains))

    summary = await run_pipeline(domains, settings, args.out)
    print_report(summary)

    out_path = args.out or settings.output_path
    print(f"Results written to {out_path}")
    print("Open viewer.html in a browser and load that file to view it as a table.\n")

    # Non-zero exit only if nothing at all worked, so CI can distinguish a total
    # failure from a run where one site was simply unreachable.
    return 0 if summary.domains_succeeded + summary.domains_partial else 1


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results were saved to the output file.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
