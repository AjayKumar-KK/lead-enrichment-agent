"""Sorting harvested addresses into usable contact points.

A flat list of every address on a site is not what a prospecting workflow wants.
``sales@`` and ``security@`` are both real, both public, and useful for entirely
different things; ``danny@postman.com`` is neither - it is one employee's inbox
that happens to appear in a page, and presenting it as the company's contact
point is how outreach ends up in the wrong person's mailbox.

So addresses are split two ways here, deterministically from the local part:

* **generic mailboxes** are categorised - sales, support, contact, info, press,
  partnerships, careers, security, privacy, or other - and kept as contact points;
* **personal mailboxes** are kept separately, so the information is not lost but
  is never presented as a way to contact the company.

No model is involved. A mailbox's purpose is spelled out in its own name, and a
lookup table cannot hallucinate a category the way a model can.
"""

from __future__ import annotations

import re

from .models import ContactEmail, clean_email
from .utils import get_logger

logger = get_logger("contacts")

# Local parts that name a function rather than a person. The first match wins, so
# more specific words must not be shadowed by broader ones.
ROLE_VOCABULARY: dict[str, tuple[str, ...]] = {
    "sales": ("sales", "sale", "buy", "purchase", "quote", "quotes", "deals", "upgrade"),
    "support": (
        "support", "help", "helpdesk", "service", "servicedesk", "care",
        "customercare", "customersuccess", "success", "assistance", "техподдержка",
    ),
    "contact": ("contact", "contactus", "hello", "hallo", "hi", "hey", "reachus", "enquire"),
    "info": (
        "info", "information", "general", "office", "admin", "mail", "email",
        "inquiry", "inquiries", "enquiry", "enquiries", "ask", "questions",
    ),
    "press": ("press", "media", "pr", "news", "newsroom", "communications", "comms", "journalists"),
    "partnerships": (
        "partner", "partners", "partnership", "partnerships", "alliances",
        "affiliate", "affiliates", "resellers", "bizdev", "businessdevelopment",
    ),
    # Checked before "support": the word-boundary rule already keeps 'careers'
    # clear of the 'care' prefix, but the intent is clearer stated in the order.
    "careers": (
        "careers", "career", "jobs", "job", "recruiting", "recruitment",
        "recruiter", "hiring", "talent", "apply", "applications", "hr",
        "peopleops", "workwithus", "joinus",
    ),
    "security": (
        "security", "infosec", "abuse", "vulnerability", "vulnerabilities",
        "vulns", "psirt", "soc", "cert", "bugbounty", "disclosure",
    ),
    "privacy": ("privacy", "dpo", "gdpr", "dataprotection", "dsar", "ccpa"),
}

# Function mailboxes that do not fit the categories above. They are still company
# contact points, not people, so they are kept and labelled "other" rather than
# being mistaken for someone's personal inbox.
DEPARTMENT_WORDS = frozenset({
    "people", "team", "teams", "billing", "invoices", "invoice",
    "accounts", "accounting", "finance", "payments", "ap", "ar",
    "legal", "compliance", "trust", "safety", "moderation", "copyright", "dmca",
    "marketing", "events", "event", "webinars", "community", "social",
    "feedback", "suggestions", "ideas", "research", "academy", "training",
    "education", "learn", "docs", "developers", "developer", "dev", "devs",
    "api", "engineering", "eng", "it", "ops", "devops", "sre", "status",
    "newsletter", "subscribe", "unsubscribe", "noreply", "no-reply", "donotreply",
    "postmaster", "webmaster", "hostmaster", "admin", "root", "abuse-desk",
    "orders", "shipping", "returns", "refunds", "bookings", "reservations",
})

# first.last / j.doe / first_last - the canonical shape of a work address
# belonging to one human being. A single-letter first part is allowed because
# initial-plus-surname is a very common corporate convention.
PERSON_PATTERN = re.compile(r"^[a-z]+[._-][a-z]{2,}(?:[._-][a-z]{2,})?$")


def _normalise_local(local: str) -> str:
    """Drop plus-addressing and punctuation: 'sales+eu' and 'sales-eu' -> 'saleseu'."""
    return re.sub(r"[^a-z0-9]+", "", local.split("+", 1)[0].lower())


def categorise(email: str) -> str:
    """Return the category for an address: one of the nine types, or 'personal'.

    Matching is on the whole normalised local part first, then on a prefix, so
    ``sales-emea@`` and ``sales+uk@`` land in "sales" while ``salesforce-admin@``
    does not accidentally become a sales mailbox by containing the word.
    """
    local = email.split("@", 1)[0].lower()
    squashed = _normalise_local(local)
    if not squashed:
        return "other"

    # Exact matches first, in both vocabularies. Prefix matching only afterwards,
    # or 'careers@' is swallowed by the 'care' prefix and becomes support.
    for category, words in ROLE_VOCABULARY.items():
        if squashed in words:
            return category
    if squashed in DEPARTMENT_WORDS or local in DEPARTMENT_WORDS:
        return "other"

    # A prefix only counts when the rest is a separate word - 'sales-emea' is a
    # sales mailbox, 'salesforce' is not.
    for category, words in ROLE_VOCABULARY.items():
        if any(re.match(rf"^{word}(?![a-z])", local) for word in words):
            return category

    if PERSON_PATTERN.match(local):
        return "personal"

    # A bare word we do not recognise as a function. Treated as a person's inbox:
    # wrongly labelling 'danny@' a contact point is the more expensive mistake,
    # and nothing is discarded either way.
    if local.isalpha():
        return "personal"

    return "other"


# The order contact points are presented in: what a prospecting workflow reaches
# for first, not alphabetical.
CATEGORY_ORDER = (
    "contact", "sales", "support", "info", "partnerships", "press", "careers",
    "security", "privacy", "other",
)


def split_contacts(emails: list[str]) -> tuple[list[ContactEmail], list[str]]:
    """Split addresses into categorised contact points and personal inboxes.

    Each address is normalised first and deduplicated on the normalised form, so
    a mailbox that was harvested twice - or once cleanly and once with an HTML or
    JSON escape stuck to it - appears exactly once, classified by what it really
    is. Anything that does not survive normalisation was not an address.

    Returns ``(contact_emails, personal_emails)``. Contact points are ordered by
    usefulness; personal addresses keep the order they were found in.
    """
    contacts: list[ContactEmail] = []
    personal: list[str] = []
    seen: set[str] = set()

    for email in emails:
        # Deduplicate on the normalised address. The same mailbox can reach here
        # twice - once cleanly and once via a malformed spelling that cleaning
        # has since repaired, or simply harvested from two pages - and a contact
        # list that repeats itself reads as a bug whatever caused it.
        normalised = clean_email(email or "")
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)

        # Classify the cleaned address, not the raw one: 'u003ehelp@postman.com'
        # is not an unrecognised mailbox, it is 'help@postman.com' with an escape
        # stuck to the front, and it belongs in support like any other help@.
        category = categorise(normalised)
        if category == "personal":
            personal.append(normalised)
        else:
            contacts.append(ContactEmail(email=normalised, type=category))

    contacts.sort(key=lambda c: (CATEGORY_ORDER.index(c.type), c.email))
    if personal:
        logger.debug("%d personal inbox(es) kept out of contact_emails", len(personal))
    return contacts, personal
