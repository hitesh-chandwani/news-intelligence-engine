"""Async, config-driven LLM client wrapper (issue #23).

Gives every pipeline stage one resilient way to call the LLM and get back
a validated Pydantic object, without any stage having to hand-roll rate
limiting, JSON parsing, or retry logic itself. See `_docs/design.md` §6.

Prompt content and per-stage message construction are out of scope here --
each stage (see `src/nie/llm/prompts/`, owned by issues #24 onward) builds
its own `messages` list and `response_model` and calls `call_structured()`.
"""

from __future__ import annotations

import asyncio
import json
import time

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from nie.config import Settings

# `_docs/design.md` §6: "spaced at ~5 seconds between LLM calls" -- looser
# than the exact 4s = 15 RPM boundary of Google AI Studio's free tier so
# normal timing jitter doesn't trip a 429. A named constant (not a magic
# number inline) so it's easy to tune later; also overridable per-instance,
# see `LLMClient.__init__`.
DEFAULT_MIN_INTERVAL_SECONDS = 5.0

_RETRY_INSTRUCTION = (
    "Your previous response could not be parsed as valid JSON matching the "
    "required schema. Validation error:\n{error}\n\n"
    "Please correct your answer and reply again with ONLY valid JSON that "
    "satisfies the required schema."
)


class RateLimiter:
    """Serializes calls so consecutive calls are >= `min_interval_seconds` apart.

    `min_interval_seconds` is injectable (constructor arg) so tests can use
    a tiny value instead of sleeping for the real ~5s production interval.
    """

    def __init__(self, min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS) -> None:
        self._min_interval_seconds = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call_at: float | None = None

    async def wait(self) -> None:
        """Block until at least `min_interval_seconds` has passed since the last call."""
        async with self._lock:
            now = time.monotonic()
            if self._last_call_at is not None:
                elapsed = now - self._last_call_at
                remaining = self._min_interval_seconds - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self._last_call_at = time.monotonic()


class LLMClient:
    """Resilient, config-driven wrapper around an OpenAI-compatible chat API.

    Reads `Settings().llm_base_url`, `Settings().llm_api_key`, and
    `Settings().llm_model` -- no other module should read those three
    settings directly.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    ) -> None:
        settings = settings if settings is not None else Settings()
        if not settings.llm_api_key:
            raise ValueError(
                "Settings().llm_api_key is not set -- required to construct an LLMClient."
            )
        # AsyncOpenAI retries HTTP 429/5xx responses with backoff by default
        # (max_retries=2 out of the box), so we deliberately do not
        # implement our own HTTP-level retry loop here -- this satisfies
        # `_docs/design.md` §6's "backoff-and-retry on 429 regardless".
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self._model = settings.llm_model
        self._rate_limiter = RateLimiter(min_interval_seconds)

    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[BaseModel]
    ) -> BaseModel:
        """Call the LLM and return a validated `response_model` instance.

        Requests JSON output from the model and parses+validates the
        response body against `response_model`. If the first attempt's
        response fails to parse as JSON or fails Pydantic validation, this
        retries **exactly once**: it re-sends the conversation with one
        additional message appended containing the validation error text
        and asking the model to correct its answer. If the retry's
        response also fails validation, the underlying
        `json.JSONDecodeError` or `pydantic.ValidationError` propagates to
        the caller -- there is no third attempt and no partial/invalid
        object is returned.
        """
        try:
            return await self._request(messages, response_model)
        except (json.JSONDecodeError, ValidationError) as first_error:
            retry_messages = [
                *messages,
                {"role": "user", "content": _RETRY_INSTRUCTION.format(error=first_error)},
            ]
            return await self._request(retry_messages, response_model)

    async def _request(
        self, messages: list[dict[str, str]], response_model: type[BaseModel]
    ) -> BaseModel:
        await self._rate_limiter.wait()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content)
        return response_model.model_validate(data)
