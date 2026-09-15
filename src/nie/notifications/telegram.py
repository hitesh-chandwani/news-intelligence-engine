"""Telegram notification channel (#32, `design.md` §10).

`render_telegram_message` turns a `NotificationPayload` (#30,
`src/nie/pipeline/notify.py`) into a Telegram-safe message body. It is
pure and synchronous -- no config, network, or DB access -- mirroring
`render_email`'s signature shape (`src/nie/notifications/email.py`, #31).

`send_telegram_message` (added separately, below) is the thin async
wrapper that actually delivers the rendered message through the raw
Telegram Bot API. Neither function is wired into the notify pipeline
stage yet -- that is #48's job, which will catch `TelegramSendError` per
channel (the same pattern `extract_stage` uses for `ExtractionError`,
see `src/nie/pipeline/extract.py`) and add `"telegram"` to
`Notification.channels_sent` only when `send_telegram_message` did not
raise.

Format choice -- Telegram's restricted HTML subset (`parse_mode="HTML"`:
`<b>`, `<i>`, `<a href="...">`, plain `\\n` line breaks), not MarkdownV2:
MarkdownV2 requires escaping ~15 special characters
(``_*[]()~`>#+-=|{}.!``) in every free-text LLM field (`fact_summary`,
`interpretation`, `importance_rationale`, each related event's
`rationale`), and a single missed character causes Telegram to reject
the *whole* message. The HTML subset only requires escaping
`&`/`<`/`>`, which a Jinja2 `Environment(autoescape=True)` handles
automatically -- the same approach `render_email` uses. Telegram's HTML
subset does **not** support block tags like `<p>`/`<ul>`/`<li>`/`<h1>`
(unlike the full HTML `render_email` produces), so this template uses
`<b>` headers and `\\n` for structure instead -- Telegram's API rejects
messages containing unsupported tags.
"""

from __future__ import annotations

import httpx
from jinja2 import Environment

from nie.config import Settings
from nie.pipeline.notify import NotificationPayload

_TELEGRAM_API_BASE_URL = "https://api.telegram.org"

_TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# Free-text fields eligible for the last-resort shrink step in
# `render_telegram_message`, in no particular priority order -- the
# longest one is always shrunk first. Populated purely from `payload`
# fields, so shrinking them can never touch the template's literal `<b>`/
# `<a>` markup and can never leave a tag unclosed.
_FREE_TEXT_FIELDS = ("title", "fact_summary", "interpretation", "importance_rationale")

# `autoescape=True` (rather than `select_autoescape`, which keys off a
# template filename this inline template doesn't have) so free-text LLM
# fields (`fact_summary`, `interpretation`, `importance_rationale`, each
# related event's `rationale`) can't inject markup Telegram doesn't
# support -- same approach `render_email` uses. `trim_blocks`/
# `lstrip_blocks` keep the plain-text output free of the blank lines/
# indentation Jinja's default whitespace handling would otherwise leave
# behind from the `{% ... %}` control lines below -- there's no browser
# to absorb that whitespace the way `render_email`'s HTML does.
_ENV = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)

_TEMPLATE_SOURCE = """\
<b>{{ payload.title }}</b>

{{ payload.fact_summary }}

{{ payload.interpretation }}

{{ payload.importance_rationale }}

<b>Importance:</b> {{ payload.importance }} | <b>Impact:</b> {{
  payload.impact_direction }} (confidence: {{
  payload.impact_confidence }})

<b>Categories:</b> {% if payload.categories %}{% for category in payload.categories %}{{
  category }}{% if not loop.last %}, {% endif %}{% endfor %}{% else %}none{% endif %}

<b>Related events</b>
{% if payload.related_events %}
{% for related in payload.related_events %}
- <b>{{ related.title }}</b> ({{
  related.event_date if related.event_date is not none else "date unknown" }}) -- {{
  related.relation }}: {{ related.rationale }}
{% endfor %}
{% else %}
No related events.
{% endif %}

<b>Sources</b>
{% if payload.sources %}
{% for source in payload.sources %}
- <a href="{{ source.url }}">{{ source.title }}</a> -- {{
  source.source_name }}{% if source.published_at is not none %} ({{
  source.published_at }}){% endif %}
{% endfor %}
{% else %}
No sources.
{% endif %}
"""

_TEMPLATE = _ENV.from_string(_TEMPLATE_SOURCE)


def _render(payload: NotificationPayload) -> str:
    return _TEMPLATE.render(payload=payload).strip("\n")


def render_telegram_message(payload: NotificationPayload) -> str:
    """Render `payload` into a Telegram `parse_mode="HTML"` message body.

    Pure and synchronous: no config, network, or DB access. Renders
    every field `design.md` §10 names -- `title`, `fact_summary`,
    `interpretation`, `importance_rationale`, `categories` (each slug),
    `importance`, `impact_direction`, `impact_confidence`, every
    `related_events` entry (`title`, `event_date`, `relation`,
    `rationale`), and every `sources` entry (`title` as a link to
    `url`, `source_name`, `published_at`) -- and renders without error
    when `related_events`/`sources` is `[]` or when an entry's optional
    date field (`event_date`/`published_at`) is `None`.

    Guarantees the result never exceeds Telegram's 4096-character
    `sendMessage` text limit and never contains an unclosed HTML tag,
    even for a payload with many `sources`/`related_events`. Strategy,
    applied in order, re-rendering after each step: (1) drop trailing
    `sources` entries one at a time, since an unbounded source list is
    the likeliest cause of an oversized message; (2) once `sources` is
    empty, drop trailing `related_events` entries the same way; (3) as a
    last resort -- both lists already empty, so the fixed template
    scaffolding plus `title`/`fact_summary`/`interpretation`/
    `importance_rationale` alone still exceed the limit -- repeatedly
    shrink whichever of those four fields is currently longest. Every
    step only ever removes a whole list entry or trims a plain-text
    field value *before* it is escaped and interpolated into the
    template's literal `<b>`/`<a>` markup, so no step can ever cut
    through, or leave dangling, one of those tags.
    """
    message = _render(payload)
    if len(message) <= _TELEGRAM_MAX_MESSAGE_LENGTH:
        return message

    working = payload

    sources = list(working.sources)
    while sources:
        sources.pop()
        working = working.model_copy(update={"sources": sources})
        message = _render(working)
        if len(message) <= _TELEGRAM_MAX_MESSAGE_LENGTH:
            return message

    related_events = list(working.related_events)
    while related_events:
        related_events.pop()
        working = working.model_copy(update={"related_events": related_events})
        message = _render(working)
        if len(message) <= _TELEGRAM_MAX_MESSAGE_LENGTH:
            return message

    while len(message) > _TELEGRAM_MAX_MESSAGE_LENGTH:
        field_name = max(_FREE_TEXT_FIELDS, key=lambda name: len(getattr(working, name)))
        current: str = getattr(working, field_name)
        if not current:
            # The fixed template scaffolding alone is far under the
            # limit, so this should be unreachable in practice -- this
            # guards against an infinite loop if it ever were.
            break
        trimmed = current[: max(0, len(current) - max(1, len(current) // 10))]
        working = working.model_copy(update={field_name: trimmed})
        message = _render(working)

    return message


class TelegramSendError(Exception):
    """Raised by `send_telegram_message` on missing config or a failed send.

    Covers both "can't attempt the send" (`settings.telegram_bot_token` or
    `settings.telegram_chat_id` is `None`) and "attempted and failed" (a
    network/transport error, a non-2xx HTTP status, or a 200 response whose
    JSON body has `"ok": false`) -- callers (#48's notify stage) only need
    to catch this one exception class to decide `"telegram"` stays out of
    `Notification.channels_sent`, the same one-exception-type-to-catch
    pattern `EmailSendError` sets (`src/nie/notifications/email.py`).
    """


async def send_telegram_message(
    payload: NotificationPayload, settings: Settings | None = None
) -> None:
    """Render `payload` and send it through the raw Telegram Bot API.

    Mirrors `send_email`'s `settings: Settings | None = None` pattern
    (`src/nie/notifications/email.py`): `settings` defaults to
    `Settings()` when omitted.

    Raises `TelegramSendError` -- never a bare/unwrapped exception -- when
    `settings.telegram_bot_token` or `settings.telegram_chat_id` is `None`,
    before attempting any HTTP call, and when the HTTP call itself fails
    (network/transport error, non-2xx status, or a 200 response whose JSON
    body has `"ok": false`); the underlying exception is chained via
    `from` so it isn't lost. Returns `None` on a confirmed-accepted send
    (`"ok": true`) and raises nothing.
    """
    settings = settings if settings is not None else Settings()

    if not settings.telegram_bot_token:
        raise TelegramSendError(
            "Settings().telegram_bot_token is not set -- required to send Telegram messages."
        )
    if not settings.telegram_chat_id:
        raise TelegramSendError(
            "Settings().telegram_chat_id is not set -- required to send Telegram messages."
        )

    text = render_telegram_message(payload)
    url = f"{_TELEGRAM_API_BASE_URL}/bot{settings.telegram_bot_token}/sendMessage"
    body = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=body)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise TelegramSendError(f"Telegram sendMessage call failed: {exc}") from exc

    if not data.get("ok", False):
        raise TelegramSendError(f"Telegram sendMessage returned ok=false: {data}")
