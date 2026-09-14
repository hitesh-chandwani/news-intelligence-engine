"""Email notification channel (#31, `design.md` §10).

`render_email` turns a `NotificationPayload` (#30,
`src/nie/pipeline/notify.py`) into an HTML email body. It is pure and
synchronous -- no config, network, or DB access -- so it can be
unit-tested with nothing but an in-code fixture payload, same
"pure/DB-read-only helper shipped ahead of its call site" precedent
`nie.pipeline.notify` set for #30.

`send_email` is the thin async wrapper that actually delivers the
rendered HTML through the Resend API. Neither function is wired into
the notify pipeline stage yet -- that is #48's job, which will catch
`EmailSendError` per channel (the same pattern `extract_stage` uses for
`ExtractionError`, see `src/nie/pipeline/extract.py`) and add
`"email"` to `Notification.channels_sent` only when `send_email` did
not raise.
"""

from __future__ import annotations

from jinja2 import Environment

from nie.pipeline.notify import NotificationPayload

# `design.md` §14's config table has no NOTIFY_EMAIL_FROM/equivalent
# variable, so the "from" address is a module-level constant rather than
# a new required `Settings` field -- a real deploy that needs a custom
# sending domain needs a `design.md` update first, not a silent new env
# var invented here.
_FROM_ADDRESS = "News Intelligence Engine <notifications@resend.dev>"

# `autoescape=True` on the `Environment` (rather than `select_autoescape`,
# which keys off a template filename this inline template doesn't have)
# so free-text LLM fields (`fact_summary`, `interpretation`,
# `importance_rationale`, `rationale`) can't break the surrounding HTML.
_ENV = Environment(autoescape=True)

_TEMPLATE_SOURCE = """\
<!DOCTYPE html>
<html>
  <head><meta charset="utf-8"></head>
  <body>
    <h1>{{ payload.title }}</h1>

    <p>{{ payload.fact_summary }}</p>
    <p>{{ payload.interpretation }}</p>
    <p>{{ payload.importance_rationale }}</p>

    <p>
      Importance: {{ payload.importance }} |
      Impact: {{ payload.impact_direction }}
      (confidence: {{ payload.impact_confidence }})
    </p>

    <p>
      Categories:
      {% if payload.categories %}
        {% for category in payload.categories %}
          {{ category }}{% if not loop.last %}, {% endif %}
        {% endfor %}
      {% else %}
        none
      {% endif %}
    </p>

    <h2>Related events</h2>
    {% if payload.related_events %}
      <ul>
        {% for related in payload.related_events %}
          <li>
            <strong>{{ related.title }}</strong>
            ({{ related.event_date if related.event_date is not none else "date unknown" }}) --
            {{ related.relation }}: {{ related.rationale }}
          </li>
        {% endfor %}
      </ul>
    {% else %}
      <p>No related events.</p>
    {% endif %}

    <h2>Sources</h2>
    {% if payload.sources %}
      <ul>
        {% for source in payload.sources %}
          <li>
            <a href="{{ source.url }}">{{ source.title }}</a>
            -- {{ source.source_name }}
            {% if source.published_at is not none %}({{ source.published_at }}){% endif %}
          </li>
        {% endfor %}
      </ul>
    {% else %}
      <p>No sources.</p>
    {% endif %}
  </body>
</html>
"""

_TEMPLATE = _ENV.from_string(_TEMPLATE_SOURCE)


def render_email(payload: NotificationPayload) -> str:
    """Render `payload` into an HTML email body.

    Pure and synchronous: no config, network, or DB access. Renders
    every field `design.md` §10 names -- `title`, `fact_summary`,
    `interpretation`, `importance_rationale`, `categories` (each slug),
    `importance`, `impact_direction`, `impact_confidence`, every
    `related_events` entry (`title`, `event_date`, `relation`,
    `rationale`), and every `sources` entry (`title`, `url` as a link
    href, `source_name`) -- and renders without error when
    `related_events`/`sources` is `[]` or when an entry's optional date
    field (`event_date`/`published_at`) is `None`.
    """
    return _TEMPLATE.render(payload=payload)
