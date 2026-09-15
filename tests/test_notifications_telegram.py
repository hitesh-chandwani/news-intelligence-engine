"""Tests for src/nie/notifications/telegram.py (issue #32).

`render_telegram_message` tests are plain synchronous `pytest` functions --
no config, network, or Jinja setup beyond the module's own -- against an
in-code fixture `NotificationPayload` built the same style as
`tests/test_notifications_email.py`'s `_payload`/`_related_event`/`_source`
helpers.

`send_telegram_message` tests never make a live network call, per
`_docs/testing-guidelines.md`'s no-live-network rule: the missing-config
paths return before any HTTP call is attempted (asserted by making a
stubbed call raise `AssertionError` if reached, the same as
`test_send_email_no_call_attempted_when_config_missing`), and the
success/failure paths monkeypatch `httpx.AsyncClient.post` directly --
the "monkeypatch the client method that issues the request" precedent
`tests/test_llm_client.py` set.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

import httpx
import pytest

from nie.config import Settings
from nie.notifications.telegram import (
    TelegramSendError,
    render_telegram_message,
    send_telegram_message,
)
from nie.pipeline.notify import NotificationPayload, PayloadRelatedEvent, PayloadSource


def _payload(
    *,
    related_events: list[PayloadRelatedEvent] | None = None,
    sources: list[PayloadSource] | None = None,
    **overrides: Any,
) -> NotificationPayload:
    fields: dict[str, Any] = {
        "event_id": uuid.uuid4(),
        "title": "Fed hikes rates by 25bps",
        "fact_summary": "The Federal Reserve raised its benchmark rate by 25bps.",
        "interpretation": "This tightens financial conditions going into Q4.",
        "importance_rationale": "Rate moves are a primary driver of near-term market pricing.",
        "categories": ["macro", "rates"],
        "importance": "high",
        "impact_direction": "bearish",
        "impact_confidence": "medium",
        "related_events": related_events if related_events is not None else [_related_event()],
        "sources": sources if sources is not None else [_source("a"), _source("b")],
    }
    fields.update(overrides)
    return NotificationPayload(**fields)


def _related_event(
    *, label: str = "prior", event_date: datetime | None = None
) -> PayloadRelatedEvent:
    return PayloadRelatedEvent(
        event_id=uuid.uuid4(),
        title=f"Related event {label}",
        event_date=event_date,
        relation="precedes",
        rationale="Same underlying policy cycle.",
    )


def _source(label: str, *, published_at: datetime | None = None) -> PayloadSource:
    return PayloadSource(
        title=f"Source {label} title",
        url=f"https://example.test/{label}",
        source_name=f"Wire {label}",
        published_at=published_at,
    )


# ---------------------------------------------------------------------------
# render_telegram_message
# ---------------------------------------------------------------------------


def test_render_telegram_message_contains_core_fields_and_source_urls() -> None:
    payload = _payload()

    message = render_telegram_message(payload)

    assert payload.title in message
    assert payload.fact_summary in message
    assert payload.interpretation in message
    assert payload.importance_rationale in message
    assert payload.importance in message
    assert payload.impact_direction in message
    assert payload.impact_confidence in message
    for category in payload.categories:
        assert category in message
    for related in payload.related_events:
        assert related.title in message
        assert related.relation in message
        assert related.rationale in message
    for source in payload.sources:
        assert source.url in message
        assert source.title in message
        assert source.source_name in message


def test_render_telegram_message_empty_related_events_and_sources_does_not_raise() -> None:
    payload = _payload(related_events=[], sources=[])

    message = render_telegram_message(payload)

    assert payload.title in message
    assert "No related events." in message
    assert "No sources." in message


def test_render_telegram_message_none_dates_do_not_raise() -> None:
    payload = _payload(
        related_events=[_related_event(event_date=None)],
        sources=[_source("a", published_at=None)],
    )

    message = render_telegram_message(payload)

    assert payload.title in message
    assert "date unknown" in message


def test_render_telegram_message_escapes_html_in_free_text_fields() -> None:
    """Autoescaping is on -- a free-text LLM field containing `&`/`<`/`>`
    must be escaped, not rendered as live markup."""
    payload = _payload(fact_summary="Rates <up> & yields > expectations, <script>alert(1)</script>")

    message = render_telegram_message(payload)

    assert "<script>alert(1)</script>" not in message
    assert "&lt;script&gt;" in message
    assert "&amp;" in message
    assert "&lt;up&gt;" in message
    assert "&gt; expectations" in message


def test_render_telegram_message_never_exceeds_limit_and_has_no_unclosed_tags() -> None:
    """A payload with many sources/related events and oversized free-text
    fields would exceed 4096 chars unmodified -- render_telegram_message
    must shrink it down without ever leaving a dangling `<b>`/`<a>` tag."""
    long_text = "x" * 2000
    payload = _payload(
        fact_summary=long_text,
        interpretation=long_text,
        importance_rationale=long_text,
        sources=[_source(f"s{i}") for i in range(200)],
        related_events=[_related_event(label=f"r{i}") for i in range(200)],
    )

    message = render_telegram_message(payload)

    assert len(message) <= 4096
    assert message.count("<b>") == message.count("</b>")
    assert message.count("<a href=") == message.count("</a>")
    # No stray, unescaped angle brackets outside the template's own
    # literal <b>/</b>/<a href=.../</a> markup.
    for opening in re.finditer(r"<(?!b>|/b>|a href=|/a>)", message):
        pytest.fail(f"unexpected unescaped '<' at position {opening.start()}")


# ---------------------------------------------------------------------------
# send_telegram_message
# ---------------------------------------------------------------------------


async def test_send_telegram_message_raises_when_bot_token_missing() -> None:
    settings = Settings(telegram_bot_token=None, telegram_chat_id="12345")

    with pytest.raises(TelegramSendError):
        await send_telegram_message(_payload(), settings)


async def test_send_telegram_message_raises_when_chat_id_missing() -> None:
    settings = Settings(telegram_bot_token="fake-token", telegram_chat_id=None)

    with pytest.raises(TelegramSendError):
        await send_telegram_message(_payload(), settings)


async def test_send_telegram_message_no_call_attempted_when_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserts the missing-config paths return before any HTTP call --
    the fixture failing loudly if reached is stronger than merely
    asserting an exception type."""

    async def unexpected_post(*args: Any, **kwargs: Any) -> httpx.Response:
        raise AssertionError("AsyncClient.post should not be called when config is missing")

    monkeypatch.setattr(httpx.AsyncClient, "post", unexpected_post)

    with pytest.raises(TelegramSendError):
        await send_telegram_message(
            _payload(), Settings(telegram_bot_token=None, telegram_chat_id="12345")
        )

    with pytest.raises(TelegramSendError):
        await send_telegram_message(
            _payload(), Settings(telegram_bot_token="fake-token", telegram_chat_id=None)
        )


async def test_send_telegram_message_succeeds_and_posts_expected_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return httpx.Response(
            200, json={"ok": True, "result": {}}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    settings = Settings(telegram_bot_token="fake-token", telegram_chat_id="12345")
    payload = _payload()

    result = await send_telegram_message(payload, settings)  # type: ignore[func-returns-value]

    assert result is None
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.telegram.org/botfake-token/sendMessage"
    body = call["json"]
    assert body["chat_id"] == "12345"
    assert body["text"] == render_telegram_message(payload)
    assert body["parse_mode"] == "HTML"
    assert body["disable_web_page_preview"] is True


async def test_send_telegram_message_raises_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("simulated connection failure")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    settings = Settings(telegram_bot_token="fake-token", telegram_chat_id="12345")

    with pytest.raises(TelegramSendError) as exc_info:
        await send_telegram_message(_payload(), settings)
    assert exc_info.value.__cause__ is not None


async def test_send_telegram_message_raises_on_non_2xx_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            401,
            json={"ok": False, "description": "Unauthorized"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    settings = Settings(telegram_bot_token="fake-token", telegram_chat_id="12345")

    with pytest.raises(TelegramSendError) as exc_info:
        await send_telegram_message(_payload(), settings)
    assert exc_info.value.__cause__ is not None


async def test_send_telegram_message_raises_when_ok_false_in_200_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"ok": False, "description": "Bad Request: chat not found"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    settings = Settings(telegram_bot_token="fake-token", telegram_chat_id="12345")

    with pytest.raises(TelegramSendError):
        await send_telegram_message(_payload(), settings)
