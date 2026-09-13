# Autonomous Lead Enrichment Agent

Takes a list of company domains, crawls their public web presence with a headless
browser, and returns structured company intelligence extracted by an LLM under a
strict schema.

```bash
python run.py --domains postman.com supabase.com vapi.ai
```

```
[ OK ] postman.com   confidence 0.87
   overview : Postman is an API platform for building and using APIs...
   ICP      : Developers and API teams building, testing and documenting APIs.
   emails   : help@postman.com, sales@postman.com
   team     : 4 found
              - Abhinav Asthana (Co-Founder & CEO) | https://linkedin.com/in/abhinavasthana
   pages    : 6/6 ok via browser, http  |  19,991 tokens  |  $0.00494  |  97.6s
```

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Setup](#setup)
- [Environment variables](#environment-variables)
- [Running it](#running-it)
- [Output schema](#output-schema)
- [LinkedIn resolution](#linkedin-resolution)
- [Who counts as a contact](#who-counts-as-a-contact)
- [Design decisions](#design-decisions)
- [Cost and token tracking](#cost-and-token-tracking)
- [Resilience](#resilience)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Limitations](#limitations)

---

## What it does

For each domain the agent:

1. **Fetches the homepage**, escalating from a plain HTTP request to headless
   Chromium when the page turns out to be JavaScript-rendered or bot-blocked.
2. **Discovers the sub-pages that matter** by reading the XML sitemap and the
   homepage navigation, then scoring every candidate URL for relevance — rather
   than guessing at a fixed list of paths.
3. **Strips each page to clean markdown**, removing scripts, styles, SVGs and
   navigation chrome, then removing the boilerplate that repeats across pages.
4. **Harvests emails and LinkedIn URLs deterministically** with regex over the
   DOM, so those fields cannot be hallucinated.
5. **Calls an LLM under a generated JSON Schema**, re-prompting with the specific
   validation errors when the response does not conform.
6. **Sorts contacts and team members** — categorising generic mailboxes, keeping
   individuals' inboxes out of the contact list, and dropping people the page
   attributes to another company.
7. **Resolves a LinkedIn profile for every named person** — verifying the URLs the
   site provided, and searching for the ones it did not, accepting a result only
   when the name *and* the company or role are corroborated in it.
8. **Scores its own output** from measured completeness and crawl coverage, not
   from a number the model invented.

Everything is written to `output/output.json`, including the pages crawled, which
tier fetched each one, token usage and estimated cost.

---

## Architecture

```
                    ┌──────────────┐
  domains  ────────▶│    run.py    │  CLI: args, env overrides, report
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  pipeline.py │  concurrency, per-domain isolation,
                    └──────┬───────┘  incremental writes
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
 │discovery.py │   │  fetcher.py  │   │  cleaner.py  │
 │             │   │              │   │              │
 │ sitemap.xml │   │ tier 1 httpx │   │ lxml strip   │
 │ nav anchors │──▶│      ↓       │──▶│      ↓       │
 │ relevance   │   │ tier 2 Play- │   │ markdown     │
 │ scoring     │   │ wright       │   │      ↓       │
 └─────────────┘   └──────────────┘   │ cross-page   │
                                      │ dedup        │
                                      └──────┬───────┘
                                             │
                        ┌────────────────────┴───────┐
                        ▼                            ▼
                 ┌──────────────┐            ┌──────────────┐
                 │ harvester.py │            │ extractor.py │
                 │              │            │              │
                 │ emails +     │───────────▶│ Groq + JSON  │
                 │ LinkedIn     │  verified  │ Schema +     │
                 │ via regex    │   lists    │ retry loop   │
                 └──────────────┘            └──────┬───────┘
                                                    ▼
                                             ┌──────────────┐
                                             │ contacts.py  │
                                             │   team.py    │
                                             │ categorise + │
                                             │ exclude      │
                                             └──────┬───────┘
                                                    ▼
                                             ┌──────────────┐
                                             │ linkedin.py  │
                                             │ verify site  │
                                             │ urls, search │
                                             │ for the rest │
                                             └──────┬───────┘
                                                    ▼
                                             ┌──────────────┐
                                             │  scoring.py  │
                                             │ deterministic│
                                             │ confidence   │
                                             └──────┬───────┘
                                                    ▼
                                            output/output.json
```

`models.py` defines the schemas every stage passes through; `config.py` holds
every tunable; `utils.py` holds retry, logging and URL helpers.

---

## Setup

**Requirements:** Python 3.10 or newer, and about 200 MB of disk for the headless
browser. No database, no Docker, no paid account — the whole thing runs from a
virtual environment on a laptop.

### 1. Get the code and create a virtual environment

```bash
git clone https://github.com/AjayKumar-KK/lead-enrichment-agent.git
cd lead-enrichment-agent
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activate script, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that window.

### 2. Install dependencies and the browser

```bash
pip install -r requirements.txt
playwright install chromium
```

`playwright install chromium` downloads the browser binary (~150 MB). It is what
lets the agent read JavaScript-rendered sites such as `vapi.ai`. Skip it and the
agent still runs: it logs one warning and degrades to HTTP-only fetching, which
costs you the sites that render client-side.

### 3. Create your `.env`

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then open `.env` and set **one** required value:

```ini
GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free key at **https://console.groq.com/keys** — Google sign-in, no credit
card. A full three-domain run costs roughly $0.004, so the free tier covers
hundreds of runs.

`.env` is listed in `.gitignore` and must never be committed. Everything else in
the file is optional and documented below.

### 4. Optionally, add a search key for LinkedIn lookups

The agent searches for LinkedIn profiles it cannot find on the site itself. Without
a search key it falls back to scraping Brave's public results page, which
rate-limits after a handful of queries — fine for a demo, not for a real batch.

```ini
SERPER_API_KEY=your_serper_key      # https://serper.dev — free tier, Google results
```

This is genuinely optional: with no key, missing profiles simply stay `null` and
the reason is recorded in the output. See
[Choosing a search provider](#choosing-a-search-provider).

### 5. Verify the install

```bash
python run.py --list-models     # confirms the API key works
python tests/run_tests.py       # 199 tests, no network or API key needed
```

If `--list-models` prints a list, you are ready. If the model named in `.env` is
not in that list, set `GROQ_MODEL` to one that is.

---

## Environment variables

Every tunable lives in `.env` and is read once at startup by `src/config.py`. No
module below it reads `os.environ` directly, so this table is the complete set.
All of them are optional except `GROQ_API_KEY`.

### Required

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq key. The run exits with instructions if it is missing. |

### Model

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model id. Run `python run.py --list-models` to see what your key can call. |
| `LLM_TEMPERATURE` | `0.1` | Low by design: this is extraction, not writing. |
| `LLM_TIMEOUT` | `90` | Seconds to wait for a completion. |
| `LLM_MAX_ATTEMPTS` | `3` | Schema-validation retries. Each failure is fed back to the model with the specific errors. |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint works. |

### LinkedIn search fallback

| Variable | Default | Purpose |
|---|---|---|
| `LINKEDIN_SEARCH_ENABLED` | `true` | Set `false` to make no outbound search requests at all. |
| `LINKEDIN_SEARCH_PROVIDER` | `auto` | `auto`, `serper`, `brave-api`, `brave-html`, or `none`. `auto` prefers whichever API key is set. |
| `SERPER_API_KEY` | — | Google results via [serper.dev](https://serper.dev). The best option; free tier. |
| `BRAVE_API_KEY` | — | [Brave Search API](https://api.search.brave.com), free tier. |
| `LINKEDIN_SEARCH_MAX_LOOKUPS` | `5` | People searched per domain. Raise it for sites with large team pages. |
| `LINKEDIN_MIN_CONFIDENCE` | `0.75` | Below this a candidate is discarded rather than guessed at. |
| `LINKEDIN_SEARCH_DELAY` | `1.2` | Seconds between queries. |
| `LINKEDIN_SEARCH_TIMEOUT` | `15` | Seconds per search request. |
| `LINKEDIN_SEARCH_RESULTS` | `6` | Results examined per query. |

### Crawling

| Variable | Default | Purpose |
|---|---|---|
| `MAX_PAGES_PER_DOMAIN` | `6` | Pages per domain, homepage included. |
| `HTTP_TIMEOUT` | `20` | Seconds for a tier-1 HTTP fetch. |
| `BROWSER_TIMEOUT` | `30` | Seconds for a tier-2 headless render. |
| `DOMAIN_CONCURRENCY` | `3` | Domains processed in parallel. |
| `PAGE_CONCURRENCY` | `3` | Pages fetched in parallel within one domain. |
| `POLITENESS_DELAY` | `0.4` | Seconds between requests to the same host. |
| `MAX_FETCH_ATTEMPTS` | `3` | Retries for timeouts and network errors only. |
| `RESPECT_ROBOTS` | `true` | Honour `robots.txt`. Leave it on. |

### Token budget

Hard character ceilings applied **before** any LLM call — this is the "do not
feed raw HTML to the model" requirement, made measurable.

| Variable | Default | Purpose |
|---|---|---|
| `MAX_CHARS_PER_PAGE` | `6000` | Per-page ceiling on cleaned text. |
| `MAX_CHARS_PER_DOMAIN` | `24000` | Total ceiling across all pages of one domain. |

### Output

| Variable | Default | Purpose |
|---|---|---|
| `OUTPUT_PATH` | `output/output.json` | Where results are written. `--out` overrides it. |

---

## Running it

```bash
# The three assignment targets (the default when no domains are given)
python run.py

# Explicit domains
python run.py --domains stripe.com figma.com linear.app

# From a file, one domain per line
python run.py --input domains.txt --out output/batch.json

# Crawl deeper, watch every decision
python run.py --domains postman.com --max-pages 10 --verbose
```

| Flag | Default | Purpose |
|---|---|---|
| `--domains`, `-d` | the 3 test targets | One or more domains |
| `--input`, `-i` | — | Read domains from a `.txt` / `.csv` file |
| `--out`, `-o` | `output/output.json` | Output path |
| `--max-pages` | `6` | Pages crawled per domain, homepage included |
| `--concurrency` | `3` | Domains processed in parallel |
| `--model` | `openai/gpt-oss-120b` | Override the Groq model |
| `--list-models` | — | List models your key can use, then exit |
| `--verbose`, `-v` | off | Debug logging |

Command-line flags win over `.env`, which wins over the defaults in `config.py`.

### What a run looks like

Three domains take about 90 seconds and cost around $0.004. Results are written
to `output/output.json` **incrementally**, after each domain completes, so a
Ctrl-C leaves a valid file rather than nothing.

Every run ends with a dashboard of what the agent accomplished, followed by
per-domain detail:

```
┌──────────────────────────────────────────────┐
│           AI LEAD ENRICHMENT AGENT           │
├──────────────────────────────────────────────┤
│ Domains processed                          3 │
│   successful                               3 │
│   failed                                   0 │
├──────────────────────────────────────────────┤
│ Pages crawled                          18/18 │
│ Team members found                         8 │
│   with a job title                         7 │
│ LinkedIn URLs found                        7 │
│   recovered by search                      7 │
│ Public emails found                        9 │
│   personal (excluded)                      1 │
│ Mean confidence                         0.89 │
├──────────────────────────────────────────────┤
│ Total tokens                          19,991 │
│ Estimated cost                      $0.00494 │
│ Runtime                            97.59 sec │
└──────────────────────────────────────────────┘
```

Every row is an outcome — things found, pages read, money spent — rather than a
restatement of the configuration. Rows that would say nothing are omitted: no
`partial` line when nothing was partial, no `personal (excluded)` line when no
individual inboxes were found. Failed domains are left out of the mean confidence,
since a zero from a domain that never ran would drag the average into nonsense.

To browse the results, open **`viewer.html`** in any browser and load the
JSON file — it renders each domain as a card with the confidence breakdown, the
category of every contact address, and where each LinkedIn URL came from.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ERROR: GROQ_API_KEY is not set` | No `.env`, or the key line is blank. Step 3 above. |
| `Groq rejected the API key (401)` | Wrong or revoked key. Check for a stray space, and that it starts with `gsk_`. |
| `Model '...' was not found on Groq` | Run `python run.py --list-models` and set `GROQ_MODEL` to one of them. |
| `429 Too Many Requests` | Groq's free-tier rate limit. The run retries with the server's own `retry-after`; wait a minute and re-run. |
| `could not start Playwright` | Run `playwright install chromium`. The agent continues HTTP-only until you do. |
| `search unavailable: status 429` in the output | The keyless Brave fallback is rate-limited. Add `SERPER_API_KEY`, or accept the missing profiles. |
| A domain reports `status: "failed"` | Read its `errors` array. Every domain always produces a record; one failure never stops the run. |
| No team members found for a site | Common and honest: some marketing sites name nobody. The `confidence_breakdown.notes` say what was and was not found. |

---

## Output schema

```jsonc
{
  "run_started_at": "2026-09-11T09:14:02+00:00",
  "duration_seconds": 34.7,
  "model": "openai/gpt-oss-120b",
  "domains_requested": 3,
  "domains_succeeded": 2,
  "domains_partial": 1,
  "domains_failed": 0,
  "total_tokens": 51204,
  "total_estimated_cost_usd": 0.03542,
  "results": [
    {
      "domain": "postman.com",
      "url": "https://postman.com",
      "status": "success",              // success | partial | failed

      "company_overview": "...",        // exactly two sentences
      "target_audience": "...",         // the ICP, one sentence
      "contact_emails": [                 // generic mailboxes, categorised
        { "email": "sales@postman.com", "type": "sales" },
        { "email": "security@postman.com", "type": "security" }
      ],
      "personal_emails": ["danny@postman.com"],   // kept, but not contact points
      "team_members": [
        {
          "name": "Ankit Sobti",
          "title": "Co-Founder & CTO",
          "affiliation": "Postman",      // as stated on the page; checked, not trusted
          "title_source": "search",      // website | search | null
          "linkedin_url": "https://www.linkedin.com/in/ankit-sobti",
          "linkedin_source": "search",   // website | search | null
          "linkedin_confidence": 0.95,   // strength of the evidence for THIS url
          "linkedin_evidence": [
            "name matches profile slug and result title",
            "company corroborated in search result"
          ]
        }
      ],

      "data_confidence_score": 0.87,
      "confidence_breakdown": {
        "completeness": 0.92,           // which fields came back populated
        "source_coverage": 0.83,        // how much of the crawl succeeded
        "evidence_strength": 0.85,      // how well the data present is verified
        "validation_health": 1.0,       // valid on the first attempt?
        "llm_self_reported": 0.85,      // recorded for comparison, NOT scored
        "final": 0.87,
        "notes": ["only one contact email found"]
      },

      "pages_crawled": [
        {
          "url": "https://postman.com/about",
          "status": "ok",
          "tier": "http",               // http | browser | none
          "http_status": 200,
          "raw_chars": 412088,          // before cleaning
          "clean_chars": 4102,          // after cleaning - a 99% reduction
          "relevance_score": 7.0,
          "duration_seconds": 0.94
        }
      ],

      "usage": {
        "llm_calls": 1,
        "prompt_tokens": 16204,
        "completion_tokens": 2000,
        "total_tokens": 18204,
        "estimated_cost_usd": 0.01249
      },

      "errors": [],
      "duration_seconds": 11.4
    }
  ]
}
```

Every input domain always produces a record. A domain that failed completely has
`status: "failed"`, a confidence of `0.0`, and the reason in `errors`.

---

## LinkedIn resolution

Team members are the most valuable field in the output and the easiest to get
wrong. A language model asked for someone's profile URL will produce
`linkedin.com/in/jane-doe` whether or not that profile exists or belongs to this
Jane Doe — a fabrication with a completely plausible shape. So the model is kept
out of this decision entirely and `src/linkedin.py` resolves it in three steps.

```
              LLM returns a named person
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
  linkedin_url supplied          no url supplied
          │                           │
          ▼                           │
  Was this exact url harvested        │
  from the company's own HTML?        │
     ├── no ──▶ discard it ───────────┤   (the model invented it)
     └── yes                          │
          ▼                           ▼
  source = "website"           search: "Name" "Company" site:linkedin.com/in
  confidence 0.75 – 0.95                     │
                                             ▼
                              For each result, both must hold:
                                first AND last name in the slug or title
                                company OR role corroborated in the text
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                        score ≥ threshold              anything less
                        source = "search"              linkedin_url stays null
                                                       reason recorded in evidence
```

**Verification comes before search.** A URL the model attributed to a person is
kept only when that exact URL appears in the list the harvester pulled out of the
page HTML. This is not theoretical: the check is what catches a model that
ignores its instructions and constructs a profile URL from a person's name.

**A name match alone is never enough.** "John Smith" returns dozens of real
profiles, and picking the first is guessing with extra steps. A result is accepted
only when the company or the person's role also appears in it. Single-word names
are never matched at all.

**Every accepted URL is auditable.** `linkedin_source`, `linkedin_confidence` and
`linkedin_evidence` travel with each person into `output.json`, and the confidence
score credits a URL in proportion to the evidence behind it. When nothing is
found, the evidence says which of the three things happened — the engine was
unreachable, it returned nothing, or its results failed corroboration.

### Choosing a search provider

LinkedIn's `robots.txt` permits Google to index profile pages and blocks most
other crawlers, which rules out more engines than you would expect: Bing and
DuckDuckGo return no `linkedin.com/in` results at all, however well you parse
them. Brave runs its own index and does carry them.

| `LINKEDIN_SEARCH_PROVIDER` | Needs a key | Notes |
|---|---|---|
| `serper` | `SERPER_API_KEY` | Google results via [serper.dev](https://serper.dev). Best quality; free tier. |
| `brave-api` | `BRAVE_API_KEY` | [Brave Search API](https://api.search.brave.com), free tier. |
| `brave-html` | no | Default. Brave's public results page. Rate-limits after a handful of queries. |
| `none` | — | Makes no outbound search requests at all. |

`auto` (the default) uses an API key if one is configured and falls back to
`brave-html` otherwise, so the feature does something useful out of the box. The
keyless path is best-effort by construction: a 429, a challenge page or a parse
failure all resolve to "no match", which leaves `linkedin_url` null exactly as if
the person had no profile. For a real batch, set a key.

Lookups are capped at `LINKEDIN_SEARCH_MAX_LOOKUPS` people per domain, spaced by
`LINKEDIN_SEARCH_DELAY` seconds, cached per name, and skipped entirely for anyone
whose profile the site already linked.

---

## Who counts as a contact

Two fields in the output are easy to fill with technically-true rubbish, and both
are fixed the same way: the model reports what the page says, and code decides
what that means.

### Emails are categorised, and people are not contact points

A flat list of every address on a site is not what a prospecting workflow wants.
`sales@` and `security@` are both public and useful for completely different
things, and `danny@postman.com` is one employee's inbox that happens to be
published — presenting it as the company's contact point is how outreach lands in
the wrong person's mailbox.

So `src/contacts.py` splits harvested addresses deterministically, from the local
part alone:

```jsonc
"contact_emails": [
  { "email": "sales@acme.io",    "type": "sales" },
  { "email": "security@acme.io", "type": "security" }
],
"personal_emails": ["danny@acme.io"]
```

Categories are `sales`, `support`, `contact`, `info`, `press`, `partnerships`,
`careers`, `security`, `privacy` and `other`. Synonyms map onto them (`helpdesk@`
→ support, `abuse@` → security, `dpo@` → privacy, `jobs@`/`hiring@`/`hr@` →
careers), as do regional and plus-addressed variants (`sales-emea@`,
`sales+uk@`). Function mailboxes with no category of their own — `billing@`,
`legal@`, `noreply@` — are `other` rather than being mistaken for a person.

Two rules stop the obvious misfires: a prefix only counts when the rest is a
separate word, so `sales-emea@` is sales and `salesforce@` is not; and exact
matches are checked before prefixes, so `careers@` stays in careers instead of
being swallowed by the `care` prefix into support.
Personal addresses are kept in their own field — nothing is discarded, it is just
never presented as a way to contact the company.

Every address is normalised before it is classified, and deduplicated on the
**normalised** form. That matters more than it sounds. Modern sites ship their
copy twice — once as HTML and once as a JSON payload for the client framework,
where `>` is written `\u003e`. The backslash is not a legal local-part character
but `u003e` is, so the address pattern matched *into* the escape and a real run
produced six contact points where there were three:

```
help@postman.com      SUPPORT        u003ehelp@postman.com      OTHER
info@postman.com      INFO           u003einfo@postman.com      OTHER
info-jp@postman.com   INFO           u003einfo-jp@postman.com   OTHER
```

Duplicates wearing a malformed name, and misfiled as `other` because
`u003ehelp@` is not a mailbox anyone recognises. The harvester now decodes JSON
escapes and HTML entities before the pattern runs, `clean_email` strips a leading
`>` or `u00XX` remnant from anything that still reaches it, and classification
happens on the cleaned address — so the repaired `help@` lands in `support` where
it belongs. The guard is deliberately narrow: `ulrich@`, `updates@` and `u2@` are
untouched.

No model is involved in any of this. A mailbox's purpose is spelled out in its
own name, and a lookup table cannot hallucinate a category.

### A name on its own is not a lead

A row reading `Jane Doe — — —` is not a contact, it is a string that survived
parsing. It happens when a name is lifted from somewhere that was never a team
listing. Supabase's `/solutions/innovation-teams` page ships a sample
`NAME | PUBLICATION` table inside a product screenshot, and six of its rows
arrived in an early run as team members.

Three rules, applied in order:

1. **The model never guesses a title.** If the page does not state a role,
   `title` is null. A null is correct and recoverable; an invented title is not.
2. **A missing title is filled from the search result that proved the profile.**
   When a LinkedIn match is accepted, the result title usually states the role —
   `"Tyler Hillery - Software Engineer (Storage) @ Supabase"` — and that is better
   evidence than a page which never mentioned one. `title_source` records whether
   a title came from the company's own pages or from search, and a title the site
   stated is never overwritten.
3. **What is left with no evidence at all is dropped.** After every chance to
   acquire a title or a corroborated profile, a person with neither was never
   established as working there. The drop is reported in
   `confidence_breakdown.notes`, never silent.

Parsing a role out of a result title is fussier than it looks, and two cases are
pinned by tests: the employer is stripped only when it *is* the employer, so
`"Co-Founder, Postman"` becomes `Co-Founder` while `"VP, Engineering"` keeps both
halves; and a fuller spelling of the person's own name is not a job title —
LinkedIn lists Chris Martin as "Christopher (Chris) Martin".

### Leadership first

Teams are ranked by seniority before anything is spent on them: Founder and
Co-Founder, then CEO, then CTO / CPO / COO and other chief officers, then
President, then VP, then Head of / Director, then everyone else, with untitled
people last. Sorting is stable, so two co-founders keep the order the site listed
them in.

This runs *before* the LinkedIn lookups, so a finite budget is spent on the
founder rather than on whoever the page happened to mention first — and the
output reads leadership-first.

On supabase.com these rules took the team from 13 names carrying a single job
title to a handful of people who nearly all have both a title and a verified
profile — and the confidence score from 0.91 to around 0.96.

The vagueness is deliberate. The exact count varies between runs, because the
model judges borderline cases differently each time: supabase.com renders a demo
`NAME | PUBLICATION` table inside a product screenshot whose rows happen to be
real staff names, and it is genuinely ambiguous whether those are team data. The
scoring is deterministic; the extraction feeding it is not, and a README that
quoted one run's numbers as if they were fixed would be misleading.

### Other companies' people are not team members

Marketing pages name far more people than the company employs. Supabase's
homepage carries five customer testimonials in the form "Bryan Byrne, Product
Manager, Lovable"; an extractor reading that sees a name and a job title and
reasonably concludes it has found a team member. The consequence is worse than a
wrong row — a prospecting workflow contacts a competitor's PM believing they work
here.

The page states the answer, so the model copies it into `affiliation` and
`src/team.py` checks it against the company being enriched. Matching is generous
in both directions (`Supabase` / `supabase.com` / `Supabase, Inc.`, and
`acmedata` ↔ `Acme Data`), and an **absent** affiliation is never disqualifying —
"Jane Doe, CEO" on a team page is the normal case, not a suspicious one.

This runs *before* the LinkedIn lookups, so an excluded customer never consumes a
search that could only ever have failed. Exclusions appear in
`confidence_breakdown.notes` rather than the team quietly shrinking.

On supabase.com this removes five names in one pass — Bryan Byrne (Lovable),
Seth Siegler (eXp Realty), Kris Woods (Phoenix Energy), Yasser Elsaid (Chatbase)
and Thiago Peres (Rally) — none of whom work there.

---

## Design decisions

### Two-tier fetching instead of always using a browser

Launching Chromium for every URL works but is slow and wasteful — most marketing
pages are server-rendered and a plain GET returns the full document in about
200 ms. So tier 1 is `httpx`. Tier 2 escalates the same URL to Playwright only
when the response was blocked, errored, or produced suspiciously little text after
cleaning.

The tier used is recorded per page in the output, which makes the trade-off
visible rather than assumed: on a typical run `postman.com` and `supabase.com`
are largely served by tier 1, while JavaScript-heavy pages escalate to tier 2.

### Relevance scoring instead of a hard-coded path list

Trying `/about`, `/team`, `/contact` and giving up fails on any site that names
its pages differently, which is most of them. Candidate URLs are gathered from the
sitemaps declared in `robots.txt` (falling back to `/sitemap.xml`, and following
one level of sitemap-index nesting) and from the homepage anchors, then scored
against weighted keywords with penalties for depth and for known-irrelevant
sections such as `/blog`, `/docs` and `/privacy`. URLs disallowed by `robots.txt`
are dropped before scoring.

The strongest single keyword decides the score and further matches add only half
their weight — otherwise a deep, keyword-stuffed path like
`/company/culture/people/team` would outrank the canonical `/team`.

Two bugs in that scoring were worth more than any prompt change, and both were
invisible until the crawled-page list was read carefully:

**Every subdomain root scored as a homepage.** The scorer returned 100 — the value
reserved for the site's own front page — for any URL with an empty path, without
checking the host. So `status.supabase.com/` and `discord.supabase.com/` scored
100 while `/about` and `/team` scored 7. Supabase spent two of five page slots on
a status page and an empty Discord redirect; postman.com spent *all* of them on
subdomain roots and found no team members at all. A subdomain root is now scored
on its label against the same keywords: `careers.` is worth reading, `status.`,
`blog.` and `docs.` are explicitly not, and an unrecognised one scores zero and is
dropped.

**`www.` made a second copy of the homepage.** URL normalisation lowercased the
host but kept the `www.` prefix, so `https://www.postman.com/` and
`https://postman.com/` were different URLs. A real run fetched the homepage twice
and sent both byte-identical copies to the model.

Fixing the pair took postman.com from 0 team members to 3, and changed what
supabase.com reads from a status page to `/company`.

### Regex for facts, LLM for judgement

Emails and LinkedIn URLs are *verifiable*: they either appear in the page or they
do not. Asking a model to produce them invites plausible fabrications —
`careers@postman.com` is exactly the kind of address a model will emit whether or
not it exists.

So `harvester.py` extracts them from the DOM with regex, and the model is given a
closed list it is explicitly instructed to choose from, with matching validators on
the schema to drop anything outside it. The model still does what it is genuinely
good at: writing the summary, inferring the ICP, and connecting a person's name to
their title and profile link.

### The confidence score is computed, not asked for

The brief asks for "an estimated score between 0.0 and 1.0 indicating the
quality/completeness of the extracted data". The obvious implementation is to ask
the model for a number, and that number is close to meaningless: the model has no
way to know which pages failed to load, how many of its claims survived
validation, or how much of the site it actually saw.

So every input to the score is measured by the pipeline. Four components, blended
by fixed weights:

| Component | Weight | Measures |
|---|---|---|
| `completeness` | 0.50 | which fields came back populated, weighted by how much each matters |
| `source_coverage` | 0.20 | how much of the intended crawl actually succeeded, discounted for thin pages |
| `evidence_strength` | 0.15 | how well the data present is backed by something other than the model's word |
| `validation_health` | 0.15 | whether valid output came first time or needed correcting |

`evidence_strength` is what the provenance fields buy: per person, half for a
title whose origin is recorded and half for a LinkedIn URL scaled by the
confidence it is really them. A name with a guessed title and no profile
contributes nothing to it, which is correct — nothing about it was established.

**The model's `self_reported_confidence` is recorded in the output but is not part
of the score.** It was worth 10% until it was removed. Keeping it meant the same
extraction could score differently run to run for no measurable reason, and "the
model felt good about it" is not something you can defend to whoever acts on the
data. It is still written to `llm_self_reported` so a reviewer can compare what
the model claimed against what was measured — on the runs in this repository it
claims 0.80 for a record that measures 0.67.

Components with nothing to measure are skipped and the remaining weights
renormalised: a company with no named team is scored on the parts that exist
rather than charged twice for the same gap, once by `completeness` and again by
`evidence_strength`.

Three tests pin the guarantee: the score is unchanged across every possible
self-report value, identical inputs produce an identical number across repeated
runs, and `final` is reproducible by hand from the published components and
weights.

### A hand-rolled schema-retry loop

The JSON Schema sent to the model is generated from the Pydantic class itself
(`LLMExtraction.model_json_schema()`), so the contract cannot drift: changing a
field changes the prompt, the validation and the output together.

When a response fails validation, the specific Pydantic errors are appended to the
conversation and the model is asked to fix them. A single failed call would
otherwise be a lost domain; in practice this converts most first-attempt failures
into valid output on the second.

### Few dependencies

The Groq client, the retry logic, the HTML-to-markdown cleaner and the test runner
are implemented here rather than installed. Groq's endpoint is OpenAI-compatible,
so it needs `httpx` and nothing else; hand-rolled backoff is about twenty lines.
The result is six runtime dependencies, less to break on another machine, and the
logic being graded visible in the source.

---

## Cost and token tracking

Token counts come from the API's own `usage` field, never from an estimate, and
are attributed per domain. Retries are counted too — a retry genuinely costs money
and hiding it would misreport the run.

Prices live in `config.py` as `MODEL_PRICING`, so the cost report stays correct
when the model changes.

A measured three-domain run on `openai/gpt-oss-120b`:

| Domain | Input | Output | Total | Cost |
|---|---|---|---|---|
| postman.com | 5,685 | 1,175 | 6,860 | $0.00173 |
| supabase.com | 6,548 | 1,063 | 7,611 | $0.00178 |
| vapi.ai | 4,513 | 1,007 | 5,520 | $0.00143 |
| **Run total** | **16,746** | **3,245** | **19,991** | **$0.00494** |

That is about **$0.0016 per enriched company**, in roughly 98 seconds.

Input dominates at 84%, and output barely moves because the schema bounds it. Of
the input, ~71% is page content and ~28% is the fixed per-call overhead of the
system prompt plus the JSON schema. The schema is serialised compactly rather
than pretty-printed, which costs 237 fewer tokens on every call — 3.7% of a run,
for whitespace no model reads.

The character budget (`MAX_CHARS_PER_PAGE`, `MAX_CHARS_PER_DOMAIN`) is what keeps
this low. Raw HTML for these sites runs to 300–500 KB per page; after cleaning,
each page contributes at most 6,000 characters and each domain at most 24,000.
Across this run that turned 4.6 million raw characters into 55,623 sent — a
**99.9% reduction** before a single token is billed. Feeding the raw HTML instead
would have cost well over a million tokens.

---

## Resilience

The brief's hard requirement is that one failing site must never kill the run.
That is enforced structurally, not by scattering `try/except` around:

- **Per-domain isolation** — each domain runs inside `enrich_domain`, which
  catches everything and returns a `failed` record instead of propagating.
- **Per-task isolation** — `asyncio.gather(..., return_exceptions=True)` at both
  the domain and page level, so even an exception escaping that guard cannot take
  down its siblings.
- **Incremental writes** — results are written after each domain completes, via a
  temp file and atomic replace, so a crash or Ctrl-C leaves a valid partial
  `output.json` rather than nothing or a truncated file.
- **Graceful degradation** — a missing Playwright browser downgrades the run to
  HTTP-only with a warning; a failed sitemap falls back to homepage anchors; a
  failed discovery falls back to the homepage alone.
- **Backoff with jitter** on transient network errors and HTTP 429, so concurrent
  workers do not retry in lockstep against the same host.
- **Bounded everywhere** — timeouts on navigation, selectors and the LLM call;
  caps on pages per domain, sitemap size and characters per prompt.

Handled explicitly: 404s and other error statuses, bot blocks (403/429), DNS
failures, timeouts, non-HTML content types, malformed HTML, malformed JSON from
the model, schema violations, an invalid API key, an unknown model name, missing
page elements, and sites that return near-identical content on every route.

---

## Tests

199 tests, no network and no API key required.

```bash
pytest tests/ -q          # if pytest is installed
python tests/run_tests.py # zero-dependency fallback runner
```

| Module | Covers |
|---|---|
| `test_cleaner.py` | Chrome removal, markdown rendering, cross-page dedup, malformed input |
| `test_harvester.py` | Email and LinkedIn extraction, obfuscation, asset-filename rejection |
| `test_discovery.py` | URL scoring and ranking, subdomain handling, normalisation, same-site checks |
| `test_robots.py` | Declared sitemaps, `Disallow` enforcement, malformed input |
| `test_scoring_and_models.py` | Schema validation, fabricated-URL rejection, confidence maths, determinism |
| `test_linkedin_search.py` | URL verification, match corroboration, provider selection, search failure |
| `test_contacts.py` | Email categories, personal-inbox detection, asset-name rejection |
| `test_dashboard.py` | Outcome counts, box alignment, rows that omit themselves |
| `test_team_filter.py` | Excluding other companies' people, seniority order, the evidence rule |
| `test_resilience.py` | Dead sites, exploding fetchers, failed extractions, output integrity |
| `test_extractor_integration.py` | The real Groq HTTP path against a local fake API |

`test_linkedin_search.py` matches against `fixtures/brave_serp.html`, a real
captured results page: the correct profile is found, and the login / games /
"top content" links that appear on every LinkedIn-heavy SERP are all rejected.
No test in the suite issues a search request.

`test_extractor_integration.py` stands up a throwaway HTTP server speaking the
OpenAI-compatible shape and drives the client through success, markdown fences,
invalid JSON, schema violations, exhausted retries, a bad key and an unknown
model — so the network code is verified without a key or an internet connection.

---

## Project layout

```
lead-enrichment-agent/
├── run.py                  CLI entry point and terminal report
├── viewer.html             static viewer for output.json
├── requirements.txt
├── .env.example
├── domains.txt             sample input file
│
├── src/
│   ├── config.py           settings, keyword weights, model pricing
│   ├── models.py           Pydantic schemas and field validators
│   ├── fetcher.py          two-tier HTTP → browser fetching
│   ├── discovery.py        sitemap + nav discovery, relevance scoring
│   ├── cleaner.py          DOM strip → markdown, boilerplate dedup
│   ├── harvester.py        deterministic email / LinkedIn extraction
│   ├── extractor.py        Groq client, JSON Schema, retry loop
│   ├── linkedin.py         profile verification + search fallback
│   ├── contacts.py         email categorisation, personal-inbox split
│   ├── team.py             team filtering, seniority, the evidence rule
│   ├── scoring.py          deterministic confidence scoring
│   ├── pipeline.py         orchestration, isolation, incremental writes
│   └── utils.py            retry, logging, URL and name-matching helpers
│
├── tests/
│   ├── fixtures/           saved HTML pages
│   ├── run_tests.py        zero-dependency runner
│   └── test_*.py
│
└── output/
    └── output.json         generated
```

---

## Limitations

Worth stating plainly rather than leaving to be discovered:

- **The keyless search provider is best-effort.** Brave's public results page
  rate-limits after a handful of queries from one address, and a block silently
  becomes "no match found" (the reason is recorded in `linkedin_evidence`). Set
  `SERPER_API_KEY` or `BRAVE_API_KEY` for a batch that matters.
- **Search only looks for people the extraction already named.** If a person is
  not on the site at all, nothing will find them here.
- **Extraction is not reproducible run to run.** The scoring is deterministic,
  but the model feeding it is not: borderline cases — a demo table of names, a
  legal notice naming an officer — are included on one run and skipped on the
  next, so team counts move by one or two between identical runs. Confidence
  moves with them. The per-record `notes` always say what was and was not found.
- **Some companies simply do not publish this.** vapi.ai has no about, team,
  company or contact page anywhere in a 763-URL sitemap; its only reachable
  public address lives on `/careers`. A low score there is an honest report of a
  thin website, not a failed crawl — which is why the score is built from
  measured components rather than a single opaque number.
- **No JavaScript interaction.** Pages are rendered, but the agent does not click
  "load more", expand accordions or fill forms. Team pages behind a click are
  missed.
- **English-centric.** The keyword weights in `config.py` assume English URL
  slugs, and page selection is keyword-based: a product page about "sales teams"
  scores on `team` exactly as a real team page would. On sites with a `/company`
  namespace this is harmless, because `company` stacks with `about` and
  `contact`; on sites without one it can cost a slot.
- **Prices drift.** `MODEL_PRICING` is a snapshot and should be checked against
  Groq's pricing page if the cost figures matter precisely.
- **Crawl politeness is basic.** `robots.txt` `Disallow` rules and declared
  sitemaps are honoured, and requests are rate-limited per domain, but
  `Crawl-delay` is not read and there is no persistent cache between runs.
