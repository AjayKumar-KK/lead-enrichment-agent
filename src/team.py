"""Keeping other companies' people out of the team list.

A marketing site names far more people than it employs. Supabase's homepage
carries five customer testimonials in the form "Bryan Byrne, Product Manager,
Lovable"; a product screenshot on the same site shows a demo table of names. An
extractor reading that page sees "person + job title" and reasonably concludes
these are team members. They are not, and the consequence is worse than a wrong
row: a prospecting workflow ends up contacting a competitor's PM believing they
work here.

The page itself states the answer - the employer is written right next to the
name - so the model is asked to copy it into ``affiliation`` and this module
decides, deterministically, whether that employer is the company being enriched.
The split is the same one used everywhere else here: the model reads, the code
verifies.

Someone whose employer the page never states is kept. An absent affiliation is
not evidence of anything, and a team page listing "Jane Doe, CEO" with no company
attached is the normal case, not a suspicious one.
"""

from __future__ import annotations

from .models import TeamMember
from .utils import company_tokens, get_logger, squash_text, text_tokens

logger = get_logger("team")


def is_own_employee(
    affiliation: str | None, label: str, tokens: set[str]
) -> bool:
    """True unless the stated employer clearly names a *different* organisation."""
    if not affiliation:
        return True  # the page did not say; absence of evidence is not evidence

    stated_squashed = squash_text(affiliation)
    if not stated_squashed:
        return True

    # Match generously in both directions: "Supabase" vs "supabase.com" vs
    # "Supabase Inc" vs a domain label that runs the words together.
    if label and (label in stated_squashed or stated_squashed in label):
        return True
    if tokens & text_tokens(affiliation, minimum=3):
        return True
    return False


def filter_own_team(
    members: list[TeamMember], domain: str, company_name: str = ""
) -> tuple[list[TeamMember], list[str]]:
    """Split out people the page attributes to another organisation.

    Returns ``(kept, notes)``. The notes name who was removed and why, so the
    exclusion shows up in the output rather than silently shrinking the team.
    """
    label, tokens = company_tokens(domain, company_name)
    kept: list[TeamMember] = []
    removed: list[str] = []

    for member in members:
        if is_own_employee(member.affiliation, label, tokens):
            kept.append(member)
        else:
            removed.append(f"{member.name} ({member.affiliation})")

    notes: list[str] = []
    if removed:
        notes.append(
            f"{len(removed)} name(s) excluded as other companies' people: "
            + ", ".join(removed[:5])
            + ("…" if len(removed) > 5 else "")
        )
        logger.info("%s: excluded %d non-employee name(s): %s",
                    domain, len(removed), ", ".join(removed[:5]))
    return kept, notes


# --------------------------------------------------------------------------- #
# Seniority
# --------------------------------------------------------------------------- #
# A prospecting workflow cares about the founder far more than the twelfth
# engineer, and the per-domain lookup budget is finite. Ranking the team before
# anything else is spent on it means the budget goes to the people who matter,
# and the output reads leadership-first.

SENIORITY_TIERS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("founder", "co-founder", "cofounder", "co founder", "owner")),
    (1, ("ceo", "chief executive")),
    (2, ("cto", "cpo", "coo", "cfo", "cmo", "cro", "ciso", "cio", "chief ")),
    (3, ("president", "managing director", "general manager")),
    (4, ("svp", "evp", "vp", "vice president")),
    (5, ("head of", "director")),
    (6, ("principal", "staff ", "lead ", "manager", "general counsel")),
)

UNTITLED_RANK = 99


def seniority_rank(title: str | None) -> int:
    """Lower is more senior. Untitled people sort last."""
    if not title:
        return UNTITLED_RANK
    lowered = f" {title.lower()} "
    for rank, markers in SENIORITY_TIERS:
        if any(marker in lowered for marker in markers):
            return rank
    return len(SENIORITY_TIERS)  # titled, but not a role we rank


def rank_team(members: list[TeamMember]) -> list[TeamMember]:
    """Sort by seniority, preserving page order within a tier.

    ``sorted`` is stable, so two co-founders stay in the order the site listed
    them - which is usually deliberate - while a VP never outranks a CEO.
    """
    return sorted(members, key=lambda m: seniority_rank(m.title))


# --------------------------------------------------------------------------- #
# The evidence rule
# --------------------------------------------------------------------------- #


def drop_unevidenced(members: list) -> tuple[list, list[str]]:
    """Remove people the run could establish nothing about.

    A row of "Name — — —" is not a lead. It happens when a name is lifted from
    somewhere that was never a team listing: a demo table in a product
    screenshot, a changelog byline, a photo caption. Supabase's
    ``/solutions/innovation-teams`` page ships a sample ``NAME | PUBLICATION``
    table, and six of its rows arrived in an earlier run as team members.

    A person is kept when the run established *something* verifiable about them:
    a job title, or a corroborated LinkedIn profile. Failing both, there is no
    evidence they work there at all, and a name on its own is not worth the row.
    """
    kept, dropped = [], []
    for member in members:
        has_title = bool(member.title)
        has_profile = bool(getattr(member, "linkedin_url", None))
        if has_title or has_profile:
            kept.append(member)
        else:
            dropped.append(member.name)

    notes: list[str] = []
    if dropped:
        notes.append(
            f"{len(dropped)} name(s) dropped with no title and no verifiable profile: "
            + ", ".join(dropped[:5])
            + ("…" if len(dropped) > 5 else "")
        )
        logger.info("dropped %d unevidenced name(s): %s", len(dropped), ", ".join(dropped[:5]))
    return kept, notes
