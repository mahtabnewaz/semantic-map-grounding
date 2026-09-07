"""Versioned prompt registry -- the single home for all prompts.

Prompts live here, not scattered as string literals (CLAUDE.md). Every
prompt carries an explicit version string, and that string is written into
every result row so a table can be traced to the exact prompt that produced
it. Prompts are immutable: never edit one in place -- add a new version
(``resolve_v2``, ...) so prior results stay reproducible.

One prompt to start: ``resolve_v1``, which asks for a single destination
node given a serialised subgraph. Its output schema is::

    {"target_phrase": str, "node_id": str, "node_label": str,
     "relation": str|null, "anchor_id": str|null}

``relation`` and ``anchor_id`` are null for non-relational queries.
``target_phrase`` is required: verify.grounded() needs it to score the
proposal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    """An immutable, versioned prompt: system text + user template."""

    version: str
    system: str
    user_template: str

    def render(self, **fields) -> str:
        """Fill the user template. The template contains only the named
        placeholders below -- no stray braces -- so substituted values
        (e.g. structured JSON, which contains ``{}``) are inserted safely."""
        return self.user_template.format(**fields)


_RESOLVE_V1_SYSTEM = """\
You resolve a natural-language destination to a single node in an indoor
map. You are given a serialised subgraph: nodes with ids, labels, types,
and (depending on the encoding) attributes and spatial relations. Choose
the one node that best answers the query.

Return ONLY a JSON object, with no prose and no code fences, using exactly
these keys:
  {"target_phrase": "<the destination the query asks for, in your words>",
   "node_id": "<id of the node you choose, exactly as in the subgraph>",
   "node_label": "<that node's label>",
   "relation": "<adjacent_to | connects_to | nearest, or null>",
   "anchor_id": "<reference node id, or null>"}

Rules:
- target_phrase is required.
- For a query phrased relative to another place ("the room next to the
  kitchen"), set relation and anchor_id to the reference node. Otherwise
  set both to null.
- Never invent a node id that is not in the subgraph. Choose exactly one.
- Never output coordinates, paths, or step-by-step directions. You return
  a node identifier and, if relevant, a symbolic relation -- nothing else.\
"""

_RESOLVE_V1_USER = """\
Current node (robot position): {anchor_id}

Subgraph ({encoding} encoding):
{subgraph}

Query: {query}

Feedback from previous attempts:
{feedback}

Respond with the JSON object only."""


PROMPTS: dict[str, PromptSpec] = {
    "resolve_v1": PromptSpec(
        version="resolve_v1",
        system=_RESOLVE_V1_SYSTEM,
        user_template=_RESOLVE_V1_USER,
    ),
}


def get_prompt(version: str) -> PromptSpec:
    """Look up a prompt by version, with a clear error if it is unknown."""
    try:
        return PROMPTS[version]
    except KeyError:
        raise KeyError(
            f"unknown prompt version {version!r}; "
            f"available: {sorted(PROMPTS)}"
        ) from None
