"""Schema validation and the deterministic confidence score."""

from contextlib import contextmanager

from pydantic import ValidationError

from src.models import EnrichedTeamMember, ConfidenceBreakdown, DomainRecord, LLMExtraction, PageRecord, TeamMember
from src.scoring import compute_confidence


@contextmanager
def raises(exc_type):
    """Local stand-in for pytest.raises, so this module needs no test framework."""
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def make_page(status="ok", chars=2000, tier="http"):
    return PageRecord(url="https://a.com/x", status=status, tier=tier, clean_chars=chars)


FULL = LLMExtraction(
    company_overview="Acme builds a managed streaming database. It is used by engineering teams worldwide.",
    target_audience="Backend engineers building real-time data applications.",
    contact_emails=["hello@acme.io", "sales@acme.io"],
    # Resolved members, as the pipeline hands them to the scorer: a title whose
    # origin is known and a profile URL with the confidence it was accepted at.
    team_members=[
        EnrichedTeamMember(name="Priya Raghavan", title="CEO", title_source="website",
                           linkedin_url="https://linkedin.com/in/priya",
                           linkedin_source="website", linkedin_confidence=0.95),
        EnrichedTeamMember(name="Daniel Okoye", title="CTO", title_source="website",
                           linkedin_url="https://linkedin.com/in/dan",
                           linkedin_source="website", linkedin_confidence=0.95),
        EnrichedTeamMember(name="Mei Lin Chen", title="VP Engineering", title_source="search",
                           linkedin_url="https://linkedin.com/in/mei",
                           linkedin_source="search", linkedin_confidence=0.9),
    ],
    self_reported_confidence=0.9,
)

SPARSE = LLMExtraction(
    company_overview="Acme builds developer tools. It was founded recently.",
    target_audience="Software developers who need better tooling.",
    contact_emails=[],
    team_members=[],
    self_reported_confidence=0.3,
)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def test_invalid_emails_are_dropped_not_fatal():
    model = LLMExtraction(
        company_overview="Acme builds a managed streaming database for teams.",
        target_audience="Backend engineers building real-time applications.",
        contact_emails=["good@acme.io", "not-an-email", "logo@2x.png", "x@example.com"],
    )
    assert model.contact_emails == ["good@acme.io"]


def test_fabricated_linkedin_url_is_nulled():
    member = TeamMember(name="Jane Doe", title="CEO", linkedin_url="https://acme.com/jane")
    assert member.linkedin_url is None


def test_placeholder_overview_is_rejected():
    with raises(ValidationError):
        LLMExtraction(company_overview="N/A", target_audience="Backend engineers everywhere.")


def test_confidence_outside_range_is_rejected():
    with raises(ValidationError):
        LLMExtraction(
            company_overview="Acme builds a managed streaming database for teams.",
            target_audience="Backend engineers building real-time applications.",
            self_reported_confidence=1.7,
        )


def test_duplicate_team_members_collapse():
    model = LLMExtraction(
        company_overview="Acme builds a managed streaming database for teams.",
        target_audience="Backend engineers building real-time applications.",
        team_members=[TeamMember(name="Jane Doe"), TeamMember(name="jane doe")],
    )
    assert len(model.team_members) == 1


def test_schema_is_generated_and_complete():
    schema = LLMExtraction.model_json_schema()
    for field in ("company_overview", "target_audience", "contact_emails",
                  "team_members", "self_reported_confidence"):
        assert field in schema["properties"]


def test_failed_record_factory_produces_valid_output():
    record = DomainRecord.failed("broken.test", "https://broken.test", "dns failure")
    assert record.status == "failed"
    assert record.data_confidence_score == 0.0
    assert record.errors == ["dns failure"]
    assert record.model_dump_json()  # serialises cleanly


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #

def test_rich_extraction_scores_high():
    result = compute_confidence(FULL, [make_page() for _ in range(6)], 6, 1, 3, [])
    assert result.final > 0.85
    assert result.completeness > 0.95


def test_sparse_extraction_scores_low():
    result = compute_confidence(SPARSE, [make_page()], 6, 1, 3, [])
    assert result.final < 0.5
    assert "no contact emails found" in result.notes
    assert "no team members identified" in result.notes


def test_rich_always_beats_sparse():
    rich = compute_confidence(FULL, [make_page() for _ in range(6)], 6, 1, 3, [])
    sparse = compute_confidence(SPARSE, [make_page()], 6, 1, 3, [])
    assert rich.final > sparse.final


def test_failed_pages_reduce_the_score():
    good = compute_confidence(FULL, [make_page() for _ in range(6)], 6, 1, 3, [])
    bad_pages = [make_page(), make_page(status="failed"), make_page(status="failed")]
    degraded = compute_confidence(FULL, bad_pages, 6, 1, 3, [])
    assert degraded.final < good.final
    assert any("page fetches failed" in n for n in degraded.notes)


def test_retries_reduce_validation_health():
    clean = compute_confidence(FULL, [make_page() for _ in range(6)], 6, 1, 3, [])
    retried = compute_confidence(
        FULL, [make_page() for _ in range(6)], 6, 3, 3, ["schema validation failed on attempt 1"]
    )
    assert retried.validation_health < clean.validation_health
    assert retried.final < clean.final


def test_thin_pages_do_not_earn_full_coverage():
    thin = compute_confidence(FULL, [make_page(chars=100) for _ in range(6)], 6, 1, 3, [])
    thick = compute_confidence(FULL, [make_page(chars=3000) for _ in range(6)], 6, 1, 3, [])
    assert thin.source_coverage < thick.source_coverage


def test_no_extraction_scores_zero_with_a_reason():
    result = compute_confidence(None, [make_page()], 6, 3, 3, ["llm failed"])
    assert result.final == 0.0
    assert result.notes == ["extraction failed; no data to score"]


def test_score_is_always_within_bounds():
    for extraction in (FULL, SPARSE):
        for pages in ([], [make_page()], [make_page() for _ in range(20)]):
            result = compute_confidence(extraction, pages, 6, 1, 3, [])
            assert 0.0 <= result.final <= 1.0


def test_breakdown_is_always_serialisable():
    result = compute_confidence(FULL, [make_page()], 6, 1, 3, [])
    assert isinstance(ConfidenceBreakdown.model_validate(result.model_dump()), ConfidenceBreakdown)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
# The score must be a function of what the pipeline measured, and nothing else.
# These tests are the guarantee behind that claim.


def test_the_models_self_report_cannot_move_the_score():
    """The model may claim anything; the number it produces is not an input."""
    pages = [make_page() for _ in range(6)]
    scores = set()
    for claimed in (0.0, 0.25, 0.5, 0.75, 1.0):
        extraction = FULL.model_copy(update={"self_reported_confidence": claimed})
        result = compute_confidence(extraction, pages, 6, 1, 3, [])
        scores.add(result.final)
        # still reported, so a reviewer can compare claim against measurement
        assert result.llm_self_reported == claimed
    assert len(scores) == 1, f"self-report leaked into the score: {scores}"


def test_the_same_record_always_scores_the_same():
    pages = [make_page() for _ in range(6)]
    runs = {compute_confidence(FULL, pages, 6, 1, 3, []).final for _ in range(25)}
    assert len(runs) == 1


def test_every_component_is_measured_by_the_pipeline_not_asserted():
    result = compute_confidence(FULL, [make_page() for _ in range(6)], 6, 1, 3, [])
    measured = (
        result.completeness,
        result.source_coverage,
        result.evidence_strength,
        result.validation_health,
    )
    assert all(0.0 <= v <= 1.0 for v in measured)
    # the blend is reproducible from the parts that are published
    from src.scoring import W_COMPLETENESS, W_COVERAGE, W_EVIDENCE, W_VALIDATION

    expected = (
        W_COMPLETENESS * result.completeness
        + W_COVERAGE * result.source_coverage
        + W_EVIDENCE * result.evidence_strength
        + W_VALIDATION * result.validation_health
    )
    assert abs(result.final - expected) < 0.01


# --------------------------------------------------------------------------- #
# Evidence strength
# --------------------------------------------------------------------------- #


def test_verified_people_score_above_unverified_ones():
    """Two identical teams; only the provenance differs."""
    pages = [make_page() for _ in range(6)]
    bare = FULL.model_copy(update={
        "team_members": [
            EnrichedTeamMember(name=m.name, title=m.title) for m in FULL.team_members
        ]
    })
    verified = compute_confidence(FULL, pages, 6, 1, 3, [])
    unverified = compute_confidence(bare, pages, 6, 1, 3, [])
    assert verified.evidence_strength > unverified.evidence_strength
    assert verified.final > unverified.final


def test_a_sourced_title_counts_even_without_a_profile():
    pages = [make_page() for _ in range(6)]
    titled = FULL.model_copy(update={
        "team_members": [EnrichedTeamMember(name="A Person", title="CEO", title_source="website")]
    })
    nothing = FULL.model_copy(update={"team_members": [EnrichedTeamMember(name="A Person")]})
    assert compute_confidence(titled, pages, 6, 1, 3, []).evidence_strength == 0.5
    assert compute_confidence(nothing, pages, 6, 1, 3, []).evidence_strength == 0.0


def test_a_company_with_no_team_is_not_punished_twice_for_it():
    """Completeness already counts the missing team; evidence must not re-charge it."""
    pages = [make_page() for _ in range(6)]
    teamless = FULL.model_copy(update={"team_members": []})
    result = compute_confidence(teamless, pages, 6, 1, 3, [])
    assert any("weight redistributed" in n for n in result.notes)
    # the remaining components still blend to a full-weight score
    from src.scoring import W_COMPLETENESS, W_COVERAGE, W_VALIDATION

    total = W_COMPLETENESS + W_COVERAGE + W_VALIDATION
    expected = (
        W_COMPLETENESS * result.completeness
        + W_COVERAGE * result.source_coverage
        + W_VALIDATION * result.validation_health
    ) / total
    assert abs(result.final - expected) < 0.01
