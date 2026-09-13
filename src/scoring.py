"""Deterministic data-confidence scoring.

The brief asks for "an estimated score between 0.0 and 1.0 indicating the
quality/completeness of the extracted data". The obvious implementation is to ask
the model for a number - and that number is close to meaningless, because the
model has no way to know which pages failed to load, how many of its claims
survived validation, or how much of the site it actually saw.

So the score here is computed from evidence the pipeline holds and the model does
not:

* **completeness**   - which fields came back populated, weighted by how much each
                       one matters to a prospecting workflow.
* **source coverage** - how much of the intended crawl actually succeeded.
* **validation health** - whether the model produced valid output first time or
                       had to be corrected.
* **evidence strength** - how well the data that *is* present is backed by
                       something other than the model's assertion: a profile URL
                       corroborated by search or harvested from the page, a title
                       sourced rather than guessed.

The model's own ``self_reported_confidence`` is recorded in the output for
comparison but is **not** part of the score. It was worth 10% until it was
removed: a model cannot know which pages failed to load, how many of its claims
survived validation, or how much of the site it actually saw, so its number was
noise wearing the costume of a signal. Every input to the score is now measured by
the pipeline, which makes the same record always produce the same number and
makes every point of it explainable.

Components with nothing to measure are skipped and the remaining weights are
renormalised - a company with no named team is scored on the parts that exist,
not punished twice for the same gap.

Every component is written to the output alongside the final number, so a reviewer
can see exactly why a domain scored what it did.
"""

from __future__ import annotations

from .models import ConfidenceBreakdown, LLMExtraction, PageRecord

# Field weights sum to 1.0. Team data is weighted highest because names, titles
# and profile URLs are the hardest part of the task and the most valuable output.
FIELD_WEIGHTS = {
    "overview": 0.18,
    "audience": 0.18,
    "emails": 0.16,
    "team": 0.28,
    "team_linkedin": 0.12,
    "team_titles": 0.08,
}

# Component weights for the final blend. Deterministic inputs only; they are
# renormalised over whichever components actually apply to a record.
W_COMPLETENESS = 0.50
W_COVERAGE = 0.20
W_EVIDENCE = 0.15
W_VALIDATION = 0.15


def _completeness(extraction: LLMExtraction) -> tuple[float, list[str]]:
    score = 0.0
    notes: list[str] = []

    if extraction.company_overview:
        score += FIELD_WEIGHTS["overview"]
    else:
        notes.append("no company overview")

    if extraction.target_audience:
        score += FIELD_WEIGHTS["audience"]
    else:
        notes.append("no target audience")

    if extraction.contact_emails:
        # Two or more addresses is treated as full credit; one is partial.
        ratio = min(len(extraction.contact_emails) / 2.0, 1.0)
        score += FIELD_WEIGHTS["emails"] * ratio
        if len(extraction.contact_emails) == 1:
            notes.append("only one contact email found")
    else:
        notes.append("no contact emails found")

    team = extraction.team_members
    if team:
        # Three or more named people is full credit.
        score += FIELD_WEIGHTS["team"] * min(len(team) / 3.0, 1.0)

        # A LinkedIn URL is credited in proportion to the evidence behind it, so a
        # profile the site linked itself is worth slightly more than one recovered
        # by search, and neither is worth as much as the other if it is missing.
        with_linkedin = sum(1 for m in team if m.linkedin_url)
        credit = sum(
            min(float(getattr(m, "linkedin_confidence", 0.0) or 1.0), 1.0)
            for m in team
            if m.linkedin_url
        )
        score += FIELD_WEIGHTS["team_linkedin"] * (credit / len(team))
        if with_linkedin == 0:
            notes.append("no LinkedIn URLs resolved for team members")
        else:
            searched = sum(1 for m in team if getattr(m, "linkedin_source", None) == "search")
            if searched:
                notes.append(f"{searched} LinkedIn URL(s) recovered by search fallback")

        with_title = sum(1 for m in team if m.title)
        score += FIELD_WEIGHTS["team_titles"] * (with_title / len(team))
    else:
        notes.append("no team members identified")

    return min(score, 1.0), notes


def _coverage(pages: list[PageRecord], target_pages: int) -> tuple[float, list[str]]:
    notes: list[str] = []
    if not pages:
        return 0.0, ["no pages fetched"]

    ok_pages = [p for p in pages if p.status == "ok"]
    if not ok_pages:
        return 0.0, ["every page fetch failed"]

    failed = len(pages) - len(ok_pages)
    if failed:
        notes.append(f"{failed} of {len(pages)} page fetches failed")

    # Coverage is how much usable content we gathered against the crawl target,
    # with a floor so a successful single-page fetch is not scored as zero.
    ratio = len(ok_pages) / max(target_pages, 1)

    # Thin pages should not count as full coverage even when they returned 200.
    substantial = sum(1 for p in ok_pages if p.clean_chars >= 800)
    if substantial < len(ok_pages):
        notes.append(f"{len(ok_pages) - substantial} page(s) had little usable text")
        ratio *= 0.5 + 0.5 * (substantial / len(ok_pages))

    return min(ratio, 1.0), notes


def _evidence_strength(extraction: LLMExtraction) -> tuple[float | None, list[str]]:
    """How well the extracted people are backed by something verifiable.

    Returns ``None`` when there is nothing to measure - a company with no named
    team is not penalised here a second time for the gap ``completeness`` already
    counted. The weight is redistributed instead.

    Per person: half for a title whose origin is recorded (the site said it, or
    it was read off the search result that proved their profile), half for a
    LinkedIn URL scaled by the confidence that it is really them. A name with a
    guessed title and no profile contributes nothing, which is correct - nothing
    about it was established.
    """
    team = extraction.team_members
    if not team:
        return None, []

    total = 0.0
    for member in team:
        sourced_title = 0.5 if getattr(member, "title_source", None) else 0.0
        profile = 0.5 * min(float(getattr(member, "linkedin_confidence", 0.0) or 0.0), 1.0)
        total += sourced_title + profile

    strength = total / len(team)
    notes: list[str] = []
    if strength < 0.5:
        unverified = sum(1 for m in team if not getattr(m, "linkedin_url", None))
        if unverified:
            notes.append(f"{unverified} of {len(team)} team member(s) have no verified profile")
    return strength, notes


def _validation_health(attempts_used: int, max_attempts: int, errors: list[str]) -> tuple[float, list[str]]:
    notes: list[str] = []
    if attempts_used <= 1:
        health = 1.0
    else:
        # Each extra attempt costs a fixed fraction of the component.
        health = max(0.0, 1.0 - (attempts_used - 1) * (1.0 / max(max_attempts, 2)))
        notes.append(f"model needed {attempts_used} attempts to produce valid output")

    schema_errors = [e for e in errors if "validation" in e or "json" in e]
    if schema_errors:
        health *= 0.9
    return round(health, 3), notes


def compute_confidence(
    extraction: LLMExtraction | None,
    pages: list[PageRecord],
    target_pages: int,
    attempts_used: int,
    max_attempts: int,
    errors: list[str],
    extra_notes: list[str] | None = None,
) -> ConfidenceBreakdown:
    """Blend the four components into a final 0.0-1.0 score with its rationale."""
    if extraction is None:
        return ConfidenceBreakdown(
            completeness=0.0,
            source_coverage=round(_coverage(pages, target_pages)[0], 3),
            validation_health=0.0,
            llm_self_reported=0.0,
            final=0.0,
            notes=["extraction failed; no data to score"] + list(extra_notes or []),
        )

    completeness, completeness_notes = _completeness(extraction)
    coverage, coverage_notes = _coverage(pages, target_pages)
    evidence, evidence_notes = _evidence_strength(extraction)
    validation, validation_notes = _validation_health(attempts_used, max_attempts, errors)

    # Only components that apply to this record, renormalised so they still sum
    # to one. A domain with no team is scored on what it does have.
    applicable = [
        (W_COMPLETENESS, completeness),
        (W_COVERAGE, coverage),
        (W_VALIDATION, validation),
    ]
    if evidence is not None:
        applicable.append((W_EVIDENCE, evidence))
    else:
        evidence_notes = ["no team members to assess evidence for; weight redistributed"]

    total_weight = sum(weight for weight, _ in applicable)
    final = sum(weight * value for weight, value in applicable) / total_weight

    # Recorded for comparison, deliberately excluded from the score above.
    self_reported = float(extraction.self_reported_confidence or 0.0)

    return ConfidenceBreakdown(
        completeness=round(completeness, 3),
        source_coverage=round(coverage, 3),
        evidence_strength=round(evidence, 3) if evidence is not None else 0.0,
        validation_health=round(validation, 3),
        llm_self_reported=round(self_reported, 3),
        final=round(min(max(final, 0.0), 1.0), 3),
        notes=(
            completeness_notes + coverage_notes + evidence_notes
            + validation_notes + list(extra_notes or [])
        ),
    )
