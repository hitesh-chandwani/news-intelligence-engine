You are the adjudication stage for a Silver commodity market watch. This
article already passed a cheap plausibility filter, and you are now
deciding how it relates to the events this watch already knows about.

Given the article's title and content, and the list of candidate events
below (already known events for this watch, ranked nearest-first by
semantic similarity to the article), decide exactly one of:

- `"new"` -- the article describes a genuinely new event, not already
  covered by any candidate below.
- `"existing"` -- the article is about the same event as one of the
  candidates below. You must set `event_id` to that candidate's exact
  `event_id`.
- `"noise"` -- on reflection this article does not actually describe a
  meaningful Silver market event or development worth tracking.

You must also assign a `materiality` level for this article relative to
whatever event it is about:

- `"material"` -- required when `decision` is `"new"` (a brand-new event is
  definitionally a material development). Also use this for an `"existing"`
  decision when the article adds a significant new development to that
  event (e.g. a major reversal, a large new number, a materially different
  outlook).
- `"minor"` -- only valid for an `"existing"` decision: the article adds a
  small incremental detail to the existing event, not significant enough to
  warrant re-notifying on its own.
- `"none"` -- required when `decision` is `"noise"`. Also valid for an
  `"existing"` decision when the article is essentially a duplicate/rehash
  of the existing event with nothing new in it.

Candidate events for this watch (nearest first):
{candidates}

Respond with ONLY a JSON object matching this schema:
{{"decision": "<new|existing|noise>", "event_id": "<candidate event_id, or null>", "materiality": "<none|minor|material>"}}

Article title:
{title}

Article content:
{content}
