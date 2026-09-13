"""LLM extraction against a strict JSON Schema, with a validation-retry loop.

The schema sent to the model is generated from the Pydantic class itself
(``LLMExtraction.model_json_schema()``), so the contract cannot drift: changing a
field changes the prompt, the validation and the output together.

The retry loop is the important part. A single call that returns malformed JSON is
a failed extraction; here, the Pydantic validation error is fed back to the model
as a follow-up turn ("your last response failed validation with these errors, fix
them") and it tries again. In practice this converts most first-attempt failures
into valid output on attempt two.

The Groq API is called over plain ``httpx`` rather than the vendor SDK - the
endpoint is OpenAI-compatible and one dependency fewer is one fewer thing to break
on a reviewer's machine.
"""

from __future__ import annotations

import json
import time

import httpx
from pydantic import ValidationError

from .config import Settings
from .models import LLMExtraction, UsageRecord
from .utils import async_retry, estimate_tokens, get_logger, truncate_to_chars

logger = get_logger("extractor")


SYSTEM_PROMPT = """You are a precise B2B company-intelligence extractor.

You will be given cleaned text from several pages of ONE company's website, plus
two VERIFIED lists that were extracted from the raw HTML by a deterministic parser.

Absolute rules:
1. Return ONLY a single JSON object matching the schema. No prose, no markdown
   fences, no explanation.
2. For `contact_emails`, you may ONLY use addresses that appear in the VERIFIED
   EMAILS list. If that list is empty, return an empty array. Never invent or
   guess an address, and never construct one from a person's name. Include every
   address from that list that belongs to the company - do not try to pick out
   the useful ones or judge which are generic. They are categorised (sales,
   support, press, and so on) and personal inboxes are separated out after you
   answer, so leaving one out only loses information.
3. For `linkedin_url`, you may ONLY use URLs from the VERIFIED LINKEDIN PROFILES
   list, and only when the page content clearly connects that URL to that person.
   If you cannot make the connection confidently, use null. A null here costs
   nothing: URLs are verified against the page HTML after you answer, and missing
   ones are looked up by a separate search step. A URL you invented will be
   detected and discarded, so guessing only loses you the rest of the record.
4. Only include people in `team_members` whose names actually appear in the page
   content. Do not add well-known executives from your own knowledge.
4b. `team_members` means people who work for THIS company. Marketing pages are
   full of named people who do not: customers giving testimonials, quoted users,
   partners, investors, and names used as sample data in product screenshots.
   A line like "Bryan Byrne, Product Manager, Lovable" on acme.com is a CUSTOMER,
   not an Acme employee - exclude them. When the page states who someone works
   for, copy it verbatim into `affiliation`; that field is checked against the
   company afterwards, so an honest `affiliation` is always better than omitting
   it.
4c. A person needs EVIDENCE to be listed. Include someone only when the page
   presents them as a person at this company - a team or leadership section, an
   author byline with a role, an "our founders" block. Names that appear without
   any such context are usually not staff: sample data in a product screenshot,
   a table of demo rows, a photo caption, a list of conference speakers. Leave
   them out.
4d. NEVER guess a title. If the page does not state someone's role, set `title`
   to null - a null is correct and is filled in later from verified sources; an
   invented title is not recoverable. Do not infer a role from a person's name,
   their position in a list, or what a company of this kind usually has.
4e. When the page names more people than you can list confidently, prioritise by
   seniority: Founder and Co-Founder first, then CEO, then CTO / CPO / COO and
   other chief officers, then President, then VP, then Head of Engineering /
   Head of Product and other heads and directors.
5. `company_overview` must be exactly two sentences, factual, no marketing
   adjectives you cannot source from the text.
6. `target_audience` must be one sentence naming who the product is built for.
7. `self_reported_confidence` should reflect how complete the SOURCE TEXT was.
   Low (0.2-0.4) if you saw little more than a homepage; high (0.8-1.0) if you saw
   clear about/team/contact content.

If a field genuinely cannot be determined, use null or an empty array. An honest
gap is correct; a plausible invention is a failure."""


def build_user_prompt(
    domain: str,
    pages: dict[str, str],
    harvested: dict[str, list[str]],
    settings: Settings,
) -> str:
    """Assemble the prompt, enforcing the per-page and per-domain char budgets."""
    sections: list[str] = [f"COMPANY DOMAIN: {domain}\n"]

    emails = harvested.get("emails", [])
    profiles = harvested.get("linkedin_profiles", [])
    companies = harvested.get("linkedin_companies", [])

    sections.append(
        "VERIFIED EMAILS (parsed from the HTML - the only ones you may use):\n"
        + ("\n".join(f"- {e}" for e in emails[:25]) if emails else "- (none found)")
    )
    sections.append(
        "VERIFIED LINKEDIN PROFILES (the only ones you may use):\n"
        + ("\n".join(f"- {p}" for p in profiles[:25]) if profiles else "- (none found)")
    )
    if companies:
        sections.append(
            "COMPANY LINKEDIN PAGES (context only, not for team_members):\n"
            + "\n".join(f"- {c}" for c in companies[:5])
        )

    budget = settings.max_chars_per_domain
    spent = 0
    page_blocks: list[str] = []

    for url, text in pages.items():
        if spent >= budget:
            logger.debug("domain char budget exhausted, %d pages omitted", len(pages) - len(page_blocks))
            break
        allowance = min(settings.max_chars_per_page, budget - spent)
        clipped = truncate_to_chars(text, allowance)
        if not clipped.strip():
            continue
        spent += len(clipped)
        page_blocks.append(f"--- PAGE: {url} ---\n{clipped}")

    sections.append("WEBSITE CONTENT:\n\n" + "\n\n".join(page_blocks))
    prompt = "\n\n".join(sections)

    # Pre-flight estimate only. The figures written to output.json are always the
    # exact counts the API reports back, never this approximation.
    logger.debug(
        "%s: prompt built from %d/%d pages, %d chars (~%d tokens, budget %d chars)",
        domain, len(page_blocks), len(pages), len(prompt),
        estimate_tokens(prompt), budget,
    )
    return prompt


class GroqExtractor:
    """Thin async client for Groq's OpenAI-compatible chat completions endpoint."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://console.groq.com/keys"
            )
        self.settings = settings
        self.schema = LLMExtraction.model_json_schema()
        self._client = httpx.AsyncClient(
            base_url=settings.groq_base_url,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            timeout=settings.llm_timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GroqExtractor":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    @async_retry(
        attempts=3,
        base_delay=2.0,
        max_delay=20.0,
        exceptions=(httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError),
        logger=logger,
    )
    async def _call(self, messages: list[dict]) -> dict:
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self.settings.model,
                "messages": messages,
                "temperature": self.settings.temperature,
                "response_format": {"type": "json_object"},
                "max_tokens": 2048,
            },
        )
        if response.status_code == 429:

            # Rate limited. Preserve Groq's server-provided retry delay so the
            # async retry decorator can wait for the correct amount of time.

            retry_after = response.headers.get("retry-after")
            logger.warning(
                "rate limited by Groq (retry-after=%s)",
                retry_after or "n/a",
            )
            exc = httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=response.request,
                response=response,
            )
            try:
                exc.retry_after = float(retry_after) if retry_after else None
            except (TypeError, ValueError):
                exc.retry_after = None
            raise exc
        if response.status_code == 404:
            raise RuntimeError(
                f"Model '{self.settings.model}' was not found on Groq. "
                "Run 'python run.py --list-models' to see what your key can use, "
                "then set GROQ_MODEL in .env."
            )
        if response.status_code == 401:
            raise RuntimeError("Groq rejected the API key (401). Check GROQ_API_KEY in .env.")
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _strip_fences(content: str) -> str:
        """Models occasionally wrap JSON in markdown fences despite instructions."""
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        return text.strip()

    async def extract(
        self,
        domain: str,
        pages: dict[str, str],
        harvested: dict[str, list[str]],
    ) -> tuple[LLMExtraction | None, UsageRecord, list[str], int]:
        """Run extraction.

        Returns ``(result, usage, errors, attempts_used)``. ``result`` is None only
        when every attempt failed; the caller still writes a record.
        """
        user_prompt = build_user_prompt(domain, pages, harvested, self.settings)
        # Serialised compactly, not pretty-printed. The schema is the same object
        # either way, but indent=2 spends ~227 tokens per call on whitespace the
        # model does not read - 28% of this prompt is fixed overhead already.
        schema_text = json.dumps(self.schema, separators=(",", ":"))

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{user_prompt}\n\n"
                    f"Respond with a single JSON object conforming to this JSON Schema:\n"
                    f"{schema_text}"
                ),
            },
        ]

        usage = UsageRecord(model=self.settings.model)
        errors: list[str] = []
        in_price, out_price = self.settings.price_for()

        for attempt in range(1, self.settings.llm_max_attempts + 1):
            started = time.perf_counter()
            try:
                payload = await self._call(messages)
            except Exception as exc:
                errors.append(f"llm call attempt {attempt}: {exc}")
                logger.warning("%s: LLM call failed on attempt %d: %s", domain, attempt, exc)
                break  # transport-level retries already happened inside _call

            # Account for usage on every attempt, including ones that fail
            # validation - a retry genuinely costs money and must be reported.
            reported = payload.get("usage") or {}
            usage.llm_calls += 1
            usage.prompt_tokens += int(reported.get("prompt_tokens", 0))
            usage.completion_tokens += int(reported.get("completion_tokens", 0))
            usage.total_tokens += int(reported.get("total_tokens", 0))
            usage.estimated_cost_usd = round(
                usage.prompt_tokens / 1_000_000 * in_price
                + usage.completion_tokens / 1_000_000 * out_price,
                6,
            )

            try:
                content = payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                errors.append(f"malformed api response on attempt {attempt}: {exc}")
                continue

            raw = self._strip_fences(content)

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid json on attempt {attempt}: {exc}")
                messages.append({"role": "assistant", "content": raw[:2000]})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That response was not valid JSON ({exc}). "
                        "Return ONLY the JSON object, with no other text."
                    ),
                })
                continue

            try:
                result = LLMExtraction.model_validate(parsed)
            except ValidationError as exc:
                detail = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                    for e in exc.errors()[:6]
                )
                errors.append(f"schema validation failed on attempt {attempt}: {detail}")
                logger.debug("%s: validation failed, re-asking - %s", domain, detail)
                messages.append({"role": "assistant", "content": raw[:2000]})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response failed schema validation:\n{detail}\n\n"
                        "Fix these specific problems and return the corrected JSON "
                        "object only."
                    ),
                })
                continue

            logger.info(
                "%s: extracted in %d attempt(s), %d tokens, $%.5f (%.1fs)",
                domain, attempt, usage.total_tokens, usage.estimated_cost_usd,
                time.perf_counter() - started,
            )
            return result, usage, errors, attempt

        return None, usage, errors, self.settings.llm_max_attempts


async def list_available_models(settings: Settings) -> list[str]:
    """Helper behind ``--list-models``: show what the key can actually call."""
    async with httpx.AsyncClient(
        base_url=settings.groq_base_url,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        timeout=30.0,
    ) as client:
        response = await client.get("/models")
        response.raise_for_status()
        data = response.json().get("data", [])
        return sorted(m.get("id", "") for m in data if m.get("id"))
