"""The dashboard is the first thing a reviewer reads, so its numbers have to be
right and its box has to line up. Counting is tested separately from printing.
"""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from run import print_dashboard, run_totals
from src.models import (
    ConfidenceBreakdown,
    ContactEmail,
    DomainRecord,
    EnrichedTeamMember,
    PageRecord,
    RunSummary,
    UsageRecord,
)


def page(status="ok", tier="http"):
    return PageRecord(url="https://acme.io/x", status=status, tier=tier, clean_chars=900)


def record(domain="acme.io", status="success", **kw):
    defaults = dict(
        domain=domain, url=f"https://{domain}", status=status,
        pages_crawled=[page(), page(), page("failed")],
        usage=UsageRecord(total_tokens=1000, estimated_cost_usd=0.002),
        data_confidence_score=0.8,
        confidence_breakdown=ConfidenceBreakdown(final=0.8),
    )
    defaults.update(kw)
    return DomainRecord(**defaults)


def summary(records):
    return RunSummary(
        run_started_at="2026-09-13T00:00:00+00:00",
        run_finished_at="2026-09-13T00:01:00+00:00",
        duration_seconds=60.0,
        model="test-model",
        domains_requested=len(records),
        domains_succeeded=sum(1 for r in records if r.status == "success"),
        domains_partial=sum(1 for r in records if r.status == "partial"),
        domains_failed=sum(1 for r in records if r.status == "failed"),
        total_tokens=sum(r.usage.total_tokens for r in records),
        total_estimated_cost_usd=sum(r.usage.estimated_cost_usd for r in records),
        results=records,
    )


# --------------------------------------------------------------------------- #
# The numbers
# --------------------------------------------------------------------------- #


def test_totals_count_what_the_run_actually_produced():
    team = [
        EnrichedTeamMember(name="A One", title="CEO", title_source="website",
                           linkedin_url="https://www.linkedin.com/in/a-one",
                           linkedin_source="website", linkedin_confidence=0.95),
        EnrichedTeamMember(name="B Two", title="CTO", title_source="search",
                           linkedin_url="https://www.linkedin.com/in/b-two",
                           linkedin_source="search", linkedin_confidence=0.9),
        EnrichedTeamMember(name="C Three"),
    ]
    records = [
        record(team_members=team,
               contact_emails=[ContactEmail(email="sales@acme.io", type="sales")],
               personal_emails=["danny@acme.io"]),
        record("other.io", status="failed", pages_crawled=[page("failed")],
               data_confidence_score=0.0),
    ]
    t = run_totals(summary(records))

    assert t["domains"] == 2 and t["succeeded"] == 1 and t["failed"] == 1
    assert t["pages_crawled"] == 2        # ok pages only
    assert t["pages_attempted"] == 4
    assert t["team_members"] == 3
    assert t["titles"] == 2               # the third has none
    assert t["linkedin_urls"] == 2
    assert t["linkedin_by_search"] == 1
    assert t["contact_emails"] == 1
    assert t["personal_emails"] == 1


def test_failed_domains_are_left_out_of_mean_confidence():
    """A zero from a domain that never ran would drag the average into nonsense."""
    records = [record(data_confidence_score=0.9),
               record("dead.io", status="failed", data_confidence_score=0.0)]
    assert run_totals(summary(records))["mean_confidence"] == 0.9


def test_an_empty_run_does_not_divide_by_zero():
    t = run_totals(summary([]))
    assert t["mean_confidence"] == 0.0 and t["domains"] == 0


# --------------------------------------------------------------------------- #
# The box
# --------------------------------------------------------------------------- #


def render(records) -> list[str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_dashboard(summary(records))
    return [line for line in buffer.getvalue().splitlines() if line.strip()]


def test_every_border_lines_up():
    """One row a character out and the whole box looks broken."""
    for records in (
        [record()],
        [record(team_members=[EnrichedTeamMember(name="A One", title="Chief Executive Officer")],
                personal_emails=["a@b.io"])],
        [record(status="partial"), record("b.io", status="failed")],
        [],
    ):
        lines = render(records)
        widths = {len(line) for line in lines}
        assert len(widths) == 1, f"ragged box: {sorted(widths)}\n" + "\n".join(lines)
        for line in lines:
            assert line[0] in "┌├└│" and line[-1] in "┐┤┘│"


def test_large_numbers_are_thousands_separated_and_still_fit():
    records = [record(usage=UsageRecord(total_tokens=9_876_543, estimated_cost_usd=1234.5))]
    lines = render(records)
    assert any("9,876,543" in line for line in lines)
    assert len({len(line) for line in lines}) == 1


def test_optional_rows_appear_only_when_they_say_something():
    plain = "\n".join(render([record()]))
    assert "partial" not in plain
    assert "personal (excluded)" not in plain

    with_extras = "\n".join(render([record(status="partial", personal_emails=["a@b.io"])]))
    assert "partial" in with_extras
    assert "personal (excluded)" in with_extras


def test_it_reports_the_headline_outcomes():
    lines = "\n".join(render([record()]))
    for label in ("Domains processed", "Pages crawled", "Team members found",
                  "LinkedIn URLs found", "Public emails found", "Total tokens",
                  "Estimated cost", "Runtime"):
        assert label in lines, label


def test_it_renders_the_real_output_file_if_one_exists():
    path = Path(__file__).resolve().parent.parent / "output" / "output.json"
    if not path.exists():
        return
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_dashboard(RunSummary(**json.loads(path.read_text(encoding="utf-8"))))
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert len({len(line) for line in lines}) == 1
