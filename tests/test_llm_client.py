"""Tests for src/nie/llm/client.py (issue #23).

Fully offline per `_docs/testing-guidelines.md`: no real network call and
no real API key. The HTTP layer is stubbed by monkeypatching
`AsyncOpenAI.chat.completions.create` on the constructed client with an
`AsyncMock` -- the issue's Constraints section explicitly allows
"monkeypatching ... the client method that issues the request" as an
alternative to stubbing the SDK's transport, and doing it this way needs
no knowledge of `openai`'s internal transport/httpx wiring.

`LLMClient` is built with `Settings(_env_file=None, ...)` (same pattern
`tests/test_config.py` uses) so the real shell/`.env` never leaks in, and
with a tiny `min_interval_seconds` so no test in this file sleeps for
anything close to the real ~5s production interval.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ValidationError

from nie.config import Settings
from nie.llm.client import LLMClient, RateLimiter


class _Verdict(BaseModel):
    label: str
    confidence: float


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_base_url="https://example-llm.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
    )


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeChatCompletion:
    """Duck-types the small slice of `openai`'s `ChatCompletion` we read."""

    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


def _client_with_stubbed_create(
    *, min_interval_seconds: float = 0.0
) -> tuple[LLMClient, AsyncMock]:
    client = LLMClient(settings=_settings(), min_interval_seconds=min_interval_seconds)
    stub_create = AsyncMock()
    client._client.chat.completions.create = stub_create  # type: ignore[method-assign]
    return client, stub_create


async def test_call_structured_returns_parsed_model_on_first_try_valid() -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion('{"label": "up", "confidence": 0.9}')

    result = await client.call_structured([{"role": "user", "content": "hi"}], _Verdict)

    assert result == _Verdict(label="up", confidence=0.9)
    assert stub_create.call_count == 1


async def test_call_structured_retries_once_then_returns_parsed_model() -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.side_effect = [
        _FakeChatCompletion('{"label": "up"}'),  # missing "confidence" -> ValidationError
        _FakeChatCompletion('{"label": "up", "confidence": 0.9}'),
    ]
    messages = [{"role": "user", "content": "hi"}]

    result = await client.call_structured(messages, _Verdict)

    assert result == _Verdict(label="up", confidence=0.9)
    assert stub_create.call_count == 2
    second_call_messages = stub_create.call_args_list[1].kwargs["messages"]
    assert second_call_messages[:1] == messages  # original conversation preserved
    retry_content = second_call_messages[-1]["content"]
    assert "confidence" in retry_content
    assert "Field required" in retry_content


async def test_call_structured_raises_when_retry_also_fails_validation() -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.side_effect = [
        _FakeChatCompletion('{"label": "up"}'),
        _FakeChatCompletion('{"label": "up"}'),
    ]

    with pytest.raises(ValidationError):
        await client.call_structured([{"role": "user", "content": "hi"}], _Verdict)

    assert stub_create.call_count == 2


async def test_rate_limiter_enforces_minimum_interval_between_calls() -> None:
    limiter = RateLimiter(min_interval_seconds=0.05)

    start = time.monotonic()
    await limiter.wait()
    await limiter.wait()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.05
