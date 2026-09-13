"""Marketing pages name other companies' people constantly. None of them belong
in `team_members`, and the page always says so - these tests pin that reading.

The real failure this prevents: a prospecting workflow contacting a competitor's
product manager in the belief that they work at the company being researched.
"""

from src.models import TeamMember
from src.team import filter_own_team, is_own_employee
from src.utils import company_tokens


def member(name, title=None, affiliation=None):
    return TeamMember(name=name, title=title, affiliation=affiliation)


def supabase():
    return company_tokens("supabase.com", "Supabase")


# --------------------------------------------------------------------------- #
# The rule itself
# --------------------------------------------------------------------------- #


def test_another_companys_employee_is_not_on_this_team():
    label, tokens = supabase()
    assert is_own_employee("Lovable", label, tokens) is False
    assert is_own_employee("Phoenix Energy", label, tokens) is False
    assert is_own_employee("eXp Realty", label, tokens) is False


def test_the_company_itself_is_recognised_however_it_is_written():
    label, tokens = supabase()
    for spelling in ("Supabase", "supabase", "Supabase Inc", "Supabase, Inc.", "supabase.com"):
        assert is_own_employee(spelling, label, tokens) is True, spelling


def test_a_run_together_domain_still_matches_the_spaced_out_name():
    label, tokens = company_tokens("acmedata.io", "Acme Data")
    assert is_own_employee("Acme Data Systems", label, tokens) is True
    assert is_own_employee("Acme Data", label, tokens) is True


def test_an_unstated_employer_is_never_treated_as_disqualifying():
    """'Jane Doe, CEO' on a team page is the normal case, not a suspicious one."""
    label, tokens = supabase()
    assert is_own_employee(None, label, tokens) is True
    assert is_own_employee("", label, tokens) is True
    assert is_own_employee("   ", label, tokens) is True


# --------------------------------------------------------------------------- #
# The filter
# --------------------------------------------------------------------------- #


def test_the_real_supabase_homepage_testimonials_are_removed():
    """These five are the actual names and employers from supabase.com."""
    team = [
        member("Jon Meyers", "Developer Advocate"),
        member("Bryan Byrne", "Product Manager", "Lovable"),
        member("Seth Siegler", "Chief Innovation Officer", "eXp Realty"),
        member("Kris Woods", "CTO", "Phoenix Energy"),
        member("Yasser Elsaid", "Founder and CEO", "Chatbase"),
        member("Thiago Peres", "Founder & CTO", "Rally"),
        member("Tyler Hillery", "Software Engineer", "Supabase"),
    ]
    kept, notes = filter_own_team(team, "supabase.com", "Supabase")
    assert [m.name for m in kept] == ["Jon Meyers", "Tyler Hillery"]
    assert len(notes) == 1
    assert "5 name(s) excluded" in notes[0]
    assert "Bryan Byrne (Lovable)" in notes[0]


def test_a_clean_team_is_left_completely_alone():
    team = [member("Jane Doe", "CEO"), member("Alan Turing", "CTO", "Acme Data")]
    kept, notes = filter_own_team(team, "acmedata.io", "Acme Data")
    assert len(kept) == 2
    assert notes == []


def test_order_is_preserved_and_nothing_is_invented():
    team = [member("A One"), member("B Two", affiliation="Elsewhere Corp"), member("C Three")]
    kept, _ = filter_own_team(team, "acmedata.io", "Acme Data")
    assert [m.name for m in kept] == ["A One", "C Three"]


def test_an_empty_team_is_handled():
    assert filter_own_team([], "acmedata.io", "Acme Data") == ([], [])


def test_the_exclusion_is_reported_rather_than_silent():
    """A team that quietly shrinks is worse than one that explains itself."""
    team = [member(f"Person{i} X", affiliation="Other Co") for i in range(8)]
    kept, notes = filter_own_team(team, "acmedata.io", "Acme Data")
    assert kept == []
    assert "8 name(s) excluded" in notes[0]
    assert notes[0].endswith("…")  # long lists are truncated, not dumped


# --------------------------------------------------------------------------- #
# Through the pipeline
# --------------------------------------------------------------------------- #


def test_excluded_people_never_reach_the_output_or_the_lookup_budget():
    import asyncio

    from src.config import Settings
    from src.linkedin import LinkedInResolver, SearchProvider
    from src.models import LLMExtraction
    from src.pipeline import enrich_domain
    from tests.test_resilience import SETTINGS, FakeExtractor, FakeFetcher

    extraction = LLMExtraction(
        company_overview="Acme Data builds a streaming database. Engineering teams use it.",
        target_audience="Backend engineers building real-time applications.",
        team_members=[
            TeamMember(name="Jane Doe", title="CEO"),
            TeamMember(name="Bryan Byrne", title="Product Manager", affiliation="Lovable"),
        ],
        self_reported_confidence=0.8,
    )

    searched: list[str] = []

    class RecordingProvider(SearchProvider):
        name = "recording"

        async def search(self, query, limit):
            searched.append(query)
            return []

    record = asyncio.run(
        enrich_domain(
            "acmedata.io", FakeFetcher({"*": "ok"}), FakeExtractor(extraction), SETTINGS,
            LinkedInResolver(Settings(groq_api_key="k", linkedin_search_delay=0.0),
                             provider=RecordingProvider()),
        )
    )

    assert [m.name for m in record.team_members] == ["Jane Doe"]
    # the customer was never looked up - that lookup could only ever have failed
    assert len(searched) == 1 and "Jane Doe" in searched[0]
    # and the exclusion is visible to whoever reads the output
    assert any("excluded" in n for n in record.confidence_breakdown.notes)


# --------------------------------------------------------------------------- #
# Seniority
# --------------------------------------------------------------------------- #

from src.team import drop_unevidenced, rank_team, seniority_rank  # noqa: E402


def test_the_priority_order_is_respected():
    """Founder outranks CEO outranks CTO outranks President outranks VP outranks Head of."""
    ladder = ["Co-Founder", "CEO", "CTO", "President", "VP of Sales", "Head of Engineering"]
    ranks = [seniority_rank(t) for t in ladder]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)  # each tier is genuinely distinct


def test_chief_officers_share_a_tier_however_they_are_spelled():
    for title in ("CTO", "CPO", "COO", "Chief Product Officer", "Chief Innovation Officer"):
        assert seniority_rank(title) == seniority_rank("CTO"), title


def test_untitled_people_sort_last_but_are_not_lost():
    team = [
        member("No Title"),
        member("An Engineer", "Software Engineer"),
        member("The Founder", "Co-Founder & CEO"),
    ]
    assert [m.name for m in rank_team(team)] == ["The Founder", "An Engineer", "No Title"]


def test_ranking_is_stable_within_a_tier():
    """Two co-founders keep the order the site listed them in - usually deliberate."""
    team = [member("First Listed", "Co-Founder"), member("Second Listed", "Co-Founder")]
    assert [m.name for m in rank_team(team)] == ["First Listed", "Second Listed"]


def test_seniority_never_raises_on_odd_titles():
    for title in (None, "", "   ", "🙂", "a" * 200):
        assert isinstance(seniority_rank(title), int)


# --------------------------------------------------------------------------- #
# The evidence rule
# --------------------------------------------------------------------------- #


def test_a_name_with_no_title_and_no_profile_is_dropped():
    """The 'Name — — —' row: nothing was established, so it is not a lead."""
    kept, notes = drop_unevidenced([member("Riccardo Bussetti")])
    assert kept == []
    assert "Riccardo Bussetti" in notes[0]


def test_a_title_alone_is_enough_evidence():
    kept, notes = drop_unevidenced([member("Tracy Lane", "General Counsel")])
    assert [m.name for m in kept] == ["Tracy Lane"]
    assert notes == []


def test_a_verified_profile_alone_is_enough_evidence():
    from src.models import EnrichedTeamMember

    person = EnrichedTeamMember(
        name="Beng Eu", linkedin_url="https://www.linkedin.com/in/thebengeu",
        linkedin_source="search", linkedin_confidence=0.95,
    )
    kept, notes = drop_unevidenced([person])
    assert [m.name for m in kept] == ["Beng Eu"]
    assert notes == []


def test_the_drop_is_reported_not_silent():
    kept, notes = drop_unevidenced([member(f"Person{i} X") for i in range(7)])
    assert kept == []
    assert "7 name(s) dropped" in notes[0] and notes[0].endswith("…")
