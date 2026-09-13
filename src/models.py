"""Pydantic schemas.

Two families of models live here and the split is deliberate:

``LLMExtraction`` and its children describe *only* what the language model is
allowed to return. Its JSON Schema is generated from the class itself and sent to
the model, and every response is parsed back through it, so a malformed or
hallucinated shape is a caught validation error rather than a silent bad record.

``DomainRecord`` is the envelope we actually write to disk. It wraps the model's
answer with provenance the model never sees: which pages were fetched and how,
token usage, cost, timings, and a confidence score computed from the data itself.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Validation patterns
# --------------------------------------------------------------------------- #

EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)

LINKEDIN_RE = re.compile(
    r"^https?://(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub|company)/[^/?#\s]+",
    re.IGNORECASE,
)

# Strings that look like addresses but are noise: tracking pixels, asset
# filenames, placeholder docs, and bundler artefacts.
EMAIL_BLOCKLIST_SUBSTRINGS = (
    "example.com",
    "domain.com",
    "yourcompany",
    "email.com",
    "sentry.io",
    "wixpress.com",
    "@2x",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".js",
    ".mjs",
    ".cjs",
    ".css",
    ".woff",
    ".map",
    ".vue",
)


# A leftover unicode escape at the front of a local part: 'u003einfo@x.com'.
ESCAPE_REMNANT_RE = re.compile(r"^(?:u00[0-9a-f]{2})+")


def clean_email(raw: str) -> str | None:
    """Normalise an email string, or return None if it is not a usable address."""
    if not raw:
        return None
    # '&gt;info@x.com' and '\u003einfo@x.com' are the same artifact in different
    # encodings; decode both before deciding whether this is an address.
    value = html.unescape(raw.strip()).lower()
    value = value.removeprefix("mailto:").split("?")[0].strip().strip(".,;:<>()[]\"'")
    # Belt and braces: the harvester decodes escapes before matching, but an
    # address can also arrive from the model, which copies what it was shown.
    value = ESCAPE_REMNANT_RE.sub("", value)
    if not EMAIL_RE.match(value):
        return None
    if any(bad in value for bad in EMAIL_BLOCKLIST_SUBSTRINGS):
        return None
    if len(value) > 120:
        return None

    # Bundled asset names parse as addresses: 'vue-shadow-dom@4.2.0.c6ed52f5c4de.mjs'
    # has a local part, an @ and dotted labels. Real hostnames never contain a
    # purely numeric label, so a version string in the domain is the giveaway.
    domain_part = value.split("@", 1)[1]
    if any(label.isdigit() for label in domain_part.split(".")):
        return None
    return value


def clean_linkedin(raw: str) -> str | None:
    """Normalise a LinkedIn URL, or return None if it is not a profile/company URL."""
    if not raw:
        return None
    value = raw.strip().split("?")[0].rstrip("/")
    if value.startswith("//"):
        value = "https:" + value
    if value.startswith("www."):
        value = "https://" + value
    if value.startswith("linkedin.com"):
        value = "https://" + value
    if not LINKEDIN_RE.match(value):
        return None
    return value.replace("http://", "https://")


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# --------------------------------------------------------------------------- #
# What the LLM is allowed to return
# --------------------------------------------------------------------------- #


class TeamMember(BaseModel):
    """A named person found on the company's own pages."""

    name: str = Field(description="Full name of the person, as written on the page.")
    title: str | None = Field(
        default=None, description="Their role or job title, e.g. 'Co-Founder & CEO'."
    )
    linkedin_url: str | None = Field(
        default=None,
        description=(
            "Their LinkedIn profile URL. Use ONLY a URL present in the supplied "
            "verified LinkedIn list. Never guess or construct one."
        ),
    )
    affiliation: str | None = Field(
        default=None,
        description=(
            "The organisation this person is stated to work for, copied exactly as "
            "written on the page (e.g. 'Lovable' in 'Bryan Byrne, Product Manager, "
            "Lovable'). Null when the page does not say. Do not infer it."
        ),
    )

    @field_validator("affiliation", mode="before")
    @classmethod
    def _affiliation_sane(cls, v):
        if v is None:
            return None
        return collapse_ws(str(v))[:80] or None

    @field_validator("name")
    @classmethod
    def _name_sane(cls, v: str) -> str:
        v = collapse_ws(v)
        if len(v) < 2 or len(v) > 80:
            raise ValueError("name must be between 2 and 80 characters")
        return v

    @field_validator("title", mode="before")
    @classmethod
    def _title_sane(cls, v):
        if v is None:
            return None
        v = collapse_ws(str(v))
        return v[:120] or None

    @field_validator("linkedin_url", mode="before")
    @classmethod
    def _linkedin_sane(cls, v):
        # Invalid LinkedIn URLs are dropped rather than raised on: a bad URL for
        # one person should not invalidate an otherwise good extraction.
        if v is None:
            return None
        return clean_linkedin(str(v))


class LLMExtraction(BaseModel):
    """The exact structure the model must produce. Its JSON Schema is the contract."""

    company_overview: str = Field(
        description="Exactly two sentences describing what the company does."
    )
    target_audience: str = Field(
        description=(
            "One sentence naming the ideal customer profile, e.g. "
            "'Developers building backend applications'."
        )
    )
    contact_emails: list[str] = Field(
        default_factory=list,
        description=(
            "Public/generic email addresses. Use ONLY addresses from the supplied "
            "verified email list. Return an empty list if none apply."
        ),
    )
    team_members: list[TeamMember] = Field(
        default_factory=list,
        description="Named leadership or team members found in the page content.",
    )
    self_reported_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Your own 0.0-1.0 estimate of how complete and reliable the source "
            "content was for this extraction."
        ),
    )

    @field_validator("company_overview", "target_audience")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = collapse_ws(v)
        if len(v) < 15:
            raise ValueError("must be a meaningful sentence, not a placeholder")
        return v

    @field_validator("contact_emails", mode="before")
    @classmethod
    def _clean_emails(cls, v):
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for item in v:
            cleaned = clean_email(str(item))
            if cleaned and cleaned not in out:
                out.append(cleaned)
        return out

    @model_validator(mode="after")
    def _dedupe_team(self) -> "LLMExtraction":
        seen: set[str] = set()
        unique: list[TeamMember] = []
        for member in self.team_members:
            key = member.name.lower()
            if key not in seen:
                seen.add(key)
                unique.append(member)
        object.__setattr__(self, "team_members", unique)
        return self


# --------------------------------------------------------------------------- #
# Provenance and telemetry the model never sees
# --------------------------------------------------------------------------- #

EmailCategory = Literal[
    "sales", "support", "contact", "info", "press",
    "partnerships", "careers", "security", "privacy", "other",
]


class ContactEmail(BaseModel):
    """A public mailbox, labelled with what it is for.

    The category is derived in :mod:`src.contacts` from the address itself, never
    asked of the model. Personal inboxes do not appear here at all - they are kept
    in ``DomainRecord.personal_emails`` so they are not lost, but they are not
    company contact points.
    """

    email: str
    type: EmailCategory = "other"


LinkedInSource = Literal["website", "search"]


class EnrichedTeamMember(TeamMember):
    """A team member after LinkedIn resolution, carrying the evidence for its URL.

    This is deliberately *not* part of ``LLMExtraction``: the model is never shown
    these fields and never gets to assert them. They are filled in afterwards by
    :mod:`src.linkedin`, from the harvested HTML or from a corroborated search
    result, so ``linkedin_url`` in the output file always has an auditable origin.
    """

    title_source: LinkedInSource | None = Field(
        default=None,
        description=(
            "Where the title came from: the company's own pages, or the search "
            "result its profile was matched from. Null when no title is known."
        ),
    )
    linkedin_source: LinkedInSource | None = Field(
        default=None,
        description="Where the URL came from: the company's own HTML, or a search engine.",
    )
    linkedin_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How strongly the evidence ties this URL to this person.",
    )
    linkedin_evidence: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons the URL was accepted, or why none was.",
    )


FetchTier = Literal["http", "browser", "none"]
PageStatus = Literal["ok", "failed", "skipped"]
DomainStatus = Literal["success", "partial", "failed"]


class PageRecord(BaseModel):
    """One crawled URL: how we got it, and what it cost us in characters."""

    url: str
    status: PageStatus
    tier: FetchTier = "none"
    http_status: int | None = None
    raw_chars: int = 0
    clean_chars: int = 0
    relevance_score: float = 0.0
    error: str | None = None
    duration_seconds: float = 0.0


class UsageRecord(BaseModel):
    """Token and cost accounting for a single domain (bonus requirement)."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""


class ConfidenceBreakdown(BaseModel):
    """Why the confidence score is what it is. Auditable, not a black box."""

    completeness: float = 0.0
    source_coverage: float = 0.0
    evidence_strength: float = 0.0
    validation_health: float = 0.0
    llm_self_reported: float = Field(
        default=0.0,
        description="What the model claimed. Recorded for comparison; NOT part of `final`.",
    )
    final: float = 0.0
    notes: list[str] = Field(default_factory=list)


class DomainRecord(BaseModel):
    """The unit written to output.json: one per input domain, always present."""

    domain: str
    url: str
    status: DomainStatus
    company_overview: str | None = None
    target_audience: str | None = None
    contact_emails: list[ContactEmail] = Field(default_factory=list)
    personal_emails: list[str] = Field(
        default_factory=list,
        description="Individual employees' addresses: kept, but not contact points.",
    )
    team_members: list[EnrichedTeamMember] = Field(default_factory=list)
    data_confidence_score: float = 0.0
    confidence_breakdown: ConfidenceBreakdown = Field(
        default_factory=ConfidenceBreakdown
    )
    pages_crawled: list[PageRecord] = Field(default_factory=list)
    usage: UsageRecord = Field(default_factory=UsageRecord)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @staticmethod
    def failed(domain: str, url: str, error: str, duration: float = 0.0) -> "DomainRecord":
        """A domain that could not be processed still produces a record, never a crash."""
        return DomainRecord(
            domain=domain,
            url=url,
            status="failed",
            errors=[error],
            duration_seconds=round(duration, 2),
        )


class RunSummary(BaseModel):
    """Top-level wrapper for the output file."""

    run_started_at: str
    run_finished_at: str
    duration_seconds: float
    model: str
    domains_requested: int
    domains_succeeded: int
    domains_partial: int
    domains_failed: int
    total_tokens: int
    total_estimated_cost_usd: float
    results: list[DomainRecord]
