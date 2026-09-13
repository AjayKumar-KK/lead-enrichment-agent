"""Not every address on a page is a way to contact the company.

`sales@` and `support@` are; `danny@postman.com` is one employee's inbox that
happens to be published. Presenting the second as the first is how outreach lands
in the wrong person's mailbox, so the split is tested as carefully as the
extraction that produced the addresses.
"""

from src.contacts import categorise, split_contacts
from src.models import clean_email


# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #


def test_the_common_generic_mailboxes_land_in_their_own_category():
    expected = {
        "sales@acme.io": "sales",
        "support@acme.io": "support",
        "contact@acme.io": "contact",
        "hello@acme.io": "contact",
        "info@acme.io": "info",
        "press@acme.io": "press",
        "partnerships@acme.io": "partnerships",
        "careers@acme.io": "careers",
        "security@acme.io": "security",
        "privacy@acme.io": "privacy",
    }
    for email, category in expected.items():
        assert categorise(email) == category, email


def test_synonyms_reach_the_right_category():
    assert categorise("helpdesk@acme.io") == "support"
    assert categorise("abuse@acme.io") == "security"          # security reports
    assert categorise("dpo@acme.io") == "privacy"             # data protection officer
    assert categorise("media@acme.io") == "press"
    assert categorise("resellers@acme.io") == "partnerships"


def test_regional_and_plus_addressed_mailboxes_keep_their_category():
    assert categorise("sales-emea@acme.io") == "sales"
    assert categorise("sales+uk@acme.io") == "sales"
    assert categorise("support2@acme.io") == "support"


def test_a_word_that_merely_starts_with_a_role_is_not_that_role():
    """'sales-emea' is a sales mailbox; 'salesforce' is a word that begins with it."""
    assert categorise("salesforce@acme.io") != "sales"


def test_careers_is_not_swallowed_by_the_care_prefix():
    """Both are real categories, and 'care' must not capture 'careers'."""
    assert categorise("careers@acme.io") == "careers"
    assert categorise("careers-uk@acme.io") == "careers"
    assert categorise("care@acme.io") == "support"
    assert categorise("customercare@acme.io") == "support"


def test_recruiting_synonyms_reach_careers():
    for email in ("jobs@acme.io", "hiring@acme.io", "recruiting@acme.io",
                  "talent@acme.io", "hr@acme.io", "joinus@acme.io"):
        assert categorise(email) == "careers", email


def test_function_mailboxes_without_a_category_are_other_not_personal():
    for email in ("billing@acme.io", "legal@acme.io",
                  "noreply@acme.io", "academy@acme.io", "events@acme.io"):
        assert categorise(email) == "other", email


# --------------------------------------------------------------------------- #
# People
# --------------------------------------------------------------------------- #


def test_individual_inboxes_are_recognised_as_people():
    for email in ("danny@postman.com", "jane.doe@acme.io", "m.okafor@acme.io",
                  "priya_raghavan@acme.io", "daniel-okoye@acme.io"):
        assert categorise(email) == "personal", email


def test_personal_addresses_are_kept_but_never_contact_points():
    contacts, personal = split_contacts(
        ["sales@acme.io", "danny@postman.com", "jane.doe@acme.io"]
    )
    assert [c.email for c in contacts] == ["sales@acme.io"]
    assert personal == ["danny@postman.com", "jane.doe@acme.io"]


def test_the_real_postman_output_is_sorted_correctly():
    """The two addresses that prompted this change, plus the junk beside them."""
    contacts, personal = split_contacts(["academy@postman.com", "danny@postman.com"])
    assert [(c.email, c.type) for c in contacts] == [("academy@postman.com", "other")]
    assert personal == ["danny@postman.com"]


# --------------------------------------------------------------------------- #
# Ordering and shape
# --------------------------------------------------------------------------- #


def test_contact_points_are_ordered_by_usefulness_not_alphabetically():
    contacts, _ = split_contacts(
        ["privacy@acme.io", "sales@acme.io", "security@acme.io",
         "contact@acme.io", "careers@acme.io"]
    )
    assert [c.type for c in contacts] == [
        "contact", "sales", "careers", "security", "privacy",
    ]


def test_an_empty_list_is_handled():
    assert split_contacts([]) == ([], [])


def test_every_contact_serialises_with_its_type():
    contacts, _ = split_contacts(["sales@acme.io"])
    assert contacts[0].model_dump() == {"email": "sales@acme.io", "type": "sales"}


# --------------------------------------------------------------------------- #
# The junk that was reaching the output
# --------------------------------------------------------------------------- #


def test_bundled_asset_names_are_not_email_addresses():
    """'vue-shadow-dom@4.2.0.c6ed52f5c4de.mjs' appeared in a real postman.com run."""
    assert clean_email("vue-shadow-dom@4.2.0.c6ed52f5c4de.mjs") is None


def test_version_numbers_in_the_domain_are_rejected():
    for junk in ("a@bundle.4.2.0.js", "x@1.2.3.4", "chunk@2.0.1.esm.mjs"):
        assert clean_email(junk) is None, junk


def test_real_addresses_still_survive_the_new_checks():
    for good in ("sales@acme.io", "danny@postman.com", "jane.doe@sub.example.co.uk"):
        assert clean_email(good) == good, good


# --------------------------------------------------------------------------- #
# Through the pipeline
# --------------------------------------------------------------------------- #


def test_the_record_separates_the_two_kinds_of_address():
    import asyncio

    from src.models import LLMExtraction
    from src.pipeline import enrich_domain
    from tests.test_resilience import SETTINGS, FakeExtractor, FakeFetcher

    extraction = LLMExtraction(
        company_overview="Acme Data builds a streaming database. Engineering teams use it.",
        target_audience="Backend engineers building real-time applications.",
        contact_emails=["danny@acme.io", "sales@acme.io", "security@acme.io"],
        self_reported_confidence=0.8,
    )
    record = asyncio.run(
        enrich_domain("acmedata.io", FakeFetcher({"*": "ok"}), FakeExtractor(extraction), SETTINGS)
    )

    assert [(c.email, c.type) for c in record.contact_emails] == [
        ("sales@acme.io", "sales"),
        ("security@acme.io", "security"),
    ]
    assert record.personal_emails == ["danny@acme.io"]
    assert '"type":"sales"' in record.model_dump_json().replace(" ", "")


# --------------------------------------------------------------------------- #
# Duplicates and extraction artifacts
# --------------------------------------------------------------------------- #
# A real postman.com run produced six contact points where there were three:
# help@, info@ and info-jp@, each also appearing as 'u003ehelp@postman.com' and
# so on, filed under OTHER. Modern sites ship their content twice - as HTML and
# as a JSON payload where '>' is written '>' - and the address pattern
# matched into the escape, because 'u003e' is made of legal local-part
# characters even though the backslash before it is not.


def test_a_json_escape_stuck_to_an_address_is_removed():
    assert clean_email("u003einfo@postman.com") == "info@postman.com"
    assert clean_email("u003ehelp@postman.com") == "help@postman.com"
    assert clean_email("u003eu003einfo@postman.com") == "info@postman.com"


def test_a_leading_angle_bracket_is_removed():
    assert clean_email(">help@postman.com") == "help@postman.com"
    assert clean_email("<sales@acme.io>") == "sales@acme.io"


def test_html_entities_are_decoded_before_validation():
    assert clean_email("&gt;info@acme.io") == "info@acme.io"
    assert clean_email("&#62;help@acme.io") == "help@acme.io"


def test_addresses_that_merely_start_with_u_are_left_alone():
    """The remnant pattern is 'u00' plus two hex digits, not any leading 'u'."""
    for good in ("ulrich@acme.io", "u2@acme.io", "updates@acme.io", "u00@acme.io"):
        assert clean_email(good) == good, good


def test_the_same_address_twice_appears_once():
    contacts, _ = split_contacts(["sales@acme.io", "sales@acme.io"])
    assert [c.email for c in contacts] == ["sales@acme.io"]


def test_duplicates_are_collapsed_by_normalised_form_not_by_string():
    """'help@', 'HELP@' and 'u003ehelp@' are one mailbox, not three."""
    contacts, _ = split_contacts(
        ["help@postman.com", "HELP@postman.com", "u003ehelp@postman.com", ">help@postman.com"]
    )
    assert [c.email for c in contacts] == ["help@postman.com"]


def test_a_repaired_address_keeps_its_real_category():
    """The artifact filed help@ under 'other'; cleaned first, it is support."""
    contacts, _ = split_contacts(["u003ehelp@postman.com"])
    assert [(c.email, c.type) for c in contacts] == [("help@postman.com", "support")]


def test_a_repaired_personal_address_is_still_personal():
    contacts, personal = split_contacts(["u003edanny@postman.com"])
    assert contacts == []
    assert personal == ["danny@postman.com"]


def test_careers_survives_the_deduplication_path():
    contacts, _ = split_contacts([">careers@acme.io", "careers@acme.io", "jobs@acme.io"])
    assert [(c.email, c.type) for c in contacts] == [
        ("careers@acme.io", "careers"),
        ("jobs@acme.io", "careers"),
    ]


def test_the_exact_postman_regression_produces_three_contacts_not_six():
    """The precise list that came out of the run that prompted this fix."""
    contacts, personal = split_contacts([
        "help@postman.com", "info-jp@postman.com", "info@postman.com",
        "u003ehelp@postman.com", "u003einfo-jp@postman.com", "u003einfo@postman.com",
        "accommodations@postman.com",
    ])
    assert [(c.email, c.type) for c in contacts] == [
        ("help@postman.com", "support"),
        ("info-jp@postman.com", "info"),
        ("info@postman.com", "info"),
    ]
    assert personal == ["accommodations@postman.com"]
    assert not any(c.type == "other" for c in contacts)


def test_things_that_are_not_addresses_are_dropped_not_deduplicated():
    contacts, personal = split_contacts(["not-an-email", "", "sales@acme.io"])
    assert [c.email for c in contacts] == ["sales@acme.io"]
    assert personal == []


def test_the_harvester_never_produces_the_artifact_in_the_first_place():
    """Prevention, not just repair: decode escapes before the pattern runs."""
    from src.harvester import harvest_emails

    html_blob = (
        '<html><body><script>window.__DATA__ = '
        '{"copy":"Email us at \\u003einfo@postman.com\\u003c for help"};</script>'
        '<p>or &gt;help@postman.com</p></body></html>'
    )
    found = harvest_emails(html_blob, "")
    assert "info@postman.com" in found
    assert "help@postman.com" in found
    assert not any(e.startswith("u003e") for e in found), found
    assert len(found) == len(set(found))
