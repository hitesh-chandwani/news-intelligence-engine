You are the historical relation stage for a Silver commodity market watch.
This event has already been synthesized and scored. Your job is to decide
how it relates, if at all, to the historical candidate events below.

Candidate historical events for this watch (found via semantic similarity
and/or shared entities):
{candidates}

Now, the event being related:
  title: {title}
  fact_summary: {fact_summary}
  interpretation: {interpretation}
  entities: {entities}

For each candidate above that is genuinely related to the event being
related, propose a relation:

- `"precedes"` -- the candidate happened before, and set up or led to, the
  event being related.
- `"similar"` -- the two describe similar or comparable developments, with
  no clear precedence between them.
- `"escalation-of"` -- the event being related is an escalation or
  intensification of the candidate.
- `"context-for"` -- the candidate is useful background/context for
  understanding the event being related, without a stronger relationship
  above applying.

For each relation you propose, also give a short `rationale` explaining
why. You may propose zero, one, or many relations -- most candidates will
not be related at all, and that is the expected, normal outcome. Only
propose a relation when it is genuinely warranted; do not force one for
every candidate. Every `event_id` you use **must** be copied verbatim from
one of the candidate `event_id` values listed above -- never invent an
`event_id` that is not in that list, and never propose a relation to the
event being related itself.

Respond with ONLY a JSON object matching this schema:
{{"relations": [{{"event_id": "<candidate event_id>", "relation": "<precedes|similar|escalation-of|context-for>", "rationale": "<string>"}}]}}

Use an empty `"relations": []` list if none of the candidates above are
genuinely related to the event being related.
