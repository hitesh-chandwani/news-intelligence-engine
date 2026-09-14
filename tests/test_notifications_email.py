"""Tests for src/nie/notifications/email.py (issue #31).

`render_email` tests are plain synchronous `pytest` functions -- no DB,
no network, no Resend/Jinja config -- against an in-code fixture
`NotificationPayload` built the same style as `_event`/`_preference` in
`tests/test_pipeline_notify.py`.

`send_email` tests never make a live network call, per
`_docs/testing-guidelines.md`'s no-live-network rule: the missing-config
paths return before any Resend call is attempted, and the
success/failure paths monkeypatch `resend.Emails.send_async` directly
rather than exercising real HTTP.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pytest
import resend

from nie.config import Settings
from nie.notifications.email import EmailSendError, render_email, send_email
from nie.pipeline.notify import NotificationPayload, PayloadRelatedEvent, PayloadSource


def _payload(
    *,
    related_events: list[PayloadRelatedEvent] | None = None,
    sources: list[PayloadSource] | None = None,
) -> NotificationPayload:
    return NotificationPayload(
        event_id=uuid.uuid4(),
        title="Fed hikes rates by 25bps",
        fact_summary="The Federal Reserve raised its benchmark rate by 25bps.",
        interpretation="This tightens financial conditions going into Q4.",
        importance_rationale="Rate moves are a primary driver of near-term market pricing.",
        categories=["macro", "rates"],
        importance="high",
        impact_direction="bearish",
        impact_confidence="medium",
        related_events=related_events if related_events is not None else [_related_event()],
        sources=sources if sources is not None else [_source("a"), _source("b")],
    )


def _related_event(*, event_date: datetime | None = None) -> PayloadRelatedEvent:
    return PayloadRelatedEvent(
        event_id=uuid.uuid4(),
        title="Prior hike",
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
# render_email
# ---------------------------------------------------------------------------


def test_render_email_contains_core_fields_and_source_urls() -> None:
    payload = _payload()

    html = render_email(payload)

    assert payload.title in html
    assert payload.fact_summary in html
    assert payload.interpretation in html
    for source in payload.sources:
        assert source.url in html


def test_render_email_empty_related_events_and_sources_does_not_raise() -> None:
    payload = _payload(related_events=[], sources=[])

    html = render_email(payload)

    assert payload.title in html


def test_render_email_none_dates_do_not_raise() -> None:
    payload = _payload(
        related_events=[_related_event(event_date=None)],
        sources=[_source("a", published_at=None)],
    )

    html = render_email(payload)

    assert payload.title in html


def test_render_email_escapes_html_in_free_text_fields() -> None:
    """Autoescaping is on -- a free-text LLM field containing HTML must
    not be rendered as live markup."""
    payload = _payload().model_copy(update={"fact_summary": "<script>alert(1)</script>"})

    html = render_email(payload)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------


async def test_send_email_raises_when_resend_api_key_missing() -> None:
    settings = Settings(resend_api_key=None, notify_email_to="watcher@example.test")

    with pytest.raises(EmailSendError):
        await send_email(_payload(), settings)


async def test_send_email_raises_when_notify_email_to_missing() -> None:
    settings = Settings(resend_api_key="fake-key", notify_email_to=None)

    with pytest.raises(EmailSendError):
        await send_email(_payload(), settings)


async def test_send_email_succeeds_and_calls_resend_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_send_async(params: dict[str, Any]) -> dict[str, str]:
        calls.append(params)
        return {"id": "fake-email-id"}

    monkeypatch.setattr(resend.Emails, "send_async", fake_send_async)

    settings = Settings(resend_api_key="fake-key", notify_email_to="watcher@example.test")

    result = await send_email(_payload(), settings)  # type: ignore[func-returns-value]

    assert result is None
    assert len(calls) == 1
    assert calls[0]["to"] == "watcher@example.test"


async def test_send_email_wraps_resend_failure_in_email_send_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_send_async(params: dict[str, Any]) -> dict[str, str]:
        raise RuntimeError("simulated Resend failure")

    monkeypatch.setattr(resend.Emails, "send_async", fake_send_async)

    settings = Settings(resend_api_key="fake-key", notify_email_to="watcher@example.test")

    with pytest.raises(EmailSendError):
        await send_email(_payload(), settings)


async def test_send_email_no_call_attempted_when_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserts the missing-config paths return before any Resend call --
    the fixture failing loudly if reached is stronger than merely
    asserting an exception type."""

    async def unexpected_send_async(params: dict[str, Any]) -> dict[str, str]:
        raise AssertionError("send_async should not be called when config is missing")

    monkeypatch.setattr(resend.Emails, "send_async", unexpected_send_async)

    with pytest.raises(EmailSendError):
        await send_email(
            _payload(), Settings(resend_api_key=None, notify_email_to="watcher@example.test")
        )

    with pytest.raises(EmailSendError):
        await send_email(_payload(), Settings(resend_api_key="fake-key", notify_email_to=None))
