"""Graph -> text, in three encodings compared in the ablation.

Each encoder takes a :class:`~src.graph.SemanticMap`, a set of node ids to
include, and an optional ``anchor_id`` (the robot's current position). The
three encodings carry the *same information content* modulo representation
(CLAUDE.md): do not enrich one and starve the others.

- structured  -- compact JSON records with explicit [x, y] positions
- relational  -- qualitative spatial relations, no numeric coordinates
- natural     -- prose description

Also here: ``token_estimate`` (compute budget) and ``scoped_subgraph``
(k-hop + lexical scoping under a token budget, dropping the furthest nodes
first). Tokenisation and stemming are reused from :mod:`src.verify`, not
duplicated. No LLM is called in this module.

Cardinal directions are derived from positions with the convention +x east,
+y north; the numeric positions themselves are never emitted by the
relational or natural encoders.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import networkx as nx

from .graph import SemanticMap
from .verify import _content_stems, _node_stem_set  # reuse; do not duplicate

_CHARS_PER_TOKEN = 4
_COMPASS = ["east", "northeast", "north", "northwest",
            "west", "southwest", "south", "southeast"]


# --------------------------------------------------------------------------
# compute budget
# --------------------------------------------------------------------------
def token_estimate(text: str) -> int:
    """Rough token count: ~4 characters per token."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _ordered(node_ids: Iterable[str]) -> list[str]:
    """Deterministic node ordering for stable output."""
    return sorted(node_ids)


def _cardinal(from_pos, to_pos) -> Optional[str]:
    """Compass direction of ``to`` relative to ``from`` (+x east, +y north),
    or None when the two coincide. Derived from positions; the numbers are
    not exposed by callers that must stay coordinate-free."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 0 and dy == 0:
        return None
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return _COMPASS[int((angle + 22.5) // 45) % 8]


def _included_edges(m: SemanticMap, ids: set[str]) -> list[tuple[str, str]]:
    edges = [
        (u, v) for u, v in m.graph.edges()
        if u in ids and v in ids
    ]
    return sorted((min(u, v), max(u, v)) for u, v in edges)


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------
def structured(m: SemanticMap, node_ids: set[str],
               anchor_id: Optional[str] = None) -> str:
    """Compact JSON records with explicit numeric positions."""
    ids = _ordered(node_ids)
    nodes = []
    for nid in ids:
        n = m.get_node(nid)
        if n is None:
            continue
        nodes.append({
            "id": n.id,
            "label": n.label,
            "type": n.type.value,
            "position": [n.position[0], n.position[1]],
            "attributes": list(n.attributes),
        })
    payload = {
        "anchor": anchor_id,
        "nodes": nodes,
        "edges": [[u, v] for u, v in _included_edges(m, set(node_ids))],
    }
    return json.dumps(payload, separators=(",", ":"))


def relational(m: SemanticMap, node_ids: set[str],
               anchor_id: Optional[str] = None) -> str:
    """Qualitative relations only. No numeric coordinates are emitted."""
    ids = _ordered(node_ids)
    lines: list[str] = []

    if anchor_id is not None and anchor_id in node_ids:
        a = m.get_node(anchor_id)
        if a is not None:
            lines.append(f"The robot is at {a.id} ({a.label}).")

    for nid in ids:
        n = m.get_node(nid)
        if n is None:
            continue
        sent = f"{n.id} ({n.label}) is {_article(n.type.value)} {n.type.value}"
        if n.attributes:
            sent += " with " + ", ".join(n.attributes)
        lines.append(sent + ".")

    for u, v in _included_edges(m, set(node_ids)):
        nu, nv = m.get_node(u), m.get_node(v)
        if nu is None or nv is None:
            continue
        direction = _cardinal(nu.position, nv.position)
        if direction is not None:
            lines.append(
                f"{nv.id} ({nv.label}) is {direction} of "
                f"{nu.id} ({nu.label}) and connected to it."
            )
        else:
            lines.append(
                f"{nv.id} ({nv.label}) is connected to {nu.id} ({nu.label})."
            )
    return "\n".join(lines)


def natural(m: SemanticMap, node_ids: set[str],
            anchor_id: Optional[str] = None) -> str:
    """Prose description of the same layout. No numeric coordinates."""
    ids = _ordered(node_ids)
    sentences: list[str] = []

    if anchor_id is not None and anchor_id in node_ids:
        a = m.get_node(anchor_id)
        if a is not None:
            sentences.append(
                f"The robot is currently at the {a.label} ({a.id})."
            )

    described = []
    for nid in ids:
        n = m.get_node(nid)
        if n is None:
            continue
        described.append(f"the {n.label} ({n.id})")
    if described:
        sentences.append(
            "The surrounding area includes " + _english_list(described) + "."
        )

    for nid in ids:
        n = m.get_node(nid)
        if n is None or not n.attributes:
            continue
        sentences.append(
            f"The {n.label} contains {_english_list(list(n.attributes))}."
        )

    for u, v in _included_edges(m, set(node_ids)):
        nu, nv = m.get_node(u), m.get_node(v)
        if nu is None or nv is None:
            continue
        direction = _cardinal(nu.position, nv.position)
        if direction is not None:
            sentences.append(
                f"The {nv.label} lies to the {direction} of the "
                f"{nu.label} and connects to it."
            )
        else:
            sentences.append(
                f"The {nv.label} connects to the {nu.label}."
            )
    return " ".join(sentences)


def _english_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


ENCODERS = {
    "structured": structured,
    "relational": relational,
    "natural": natural,
}


# --------------------------------------------------------------------------
# scoping under a token budget
# --------------------------------------------------------------------------
@dataclass
class ScopedSubgraph:
    """Result of :func:`scoped_subgraph`.

    ``node_ids`` is the retained set (the subgraph to serialise); ``dropped``
    lists nodes removed to fit the budget, furthest-first; ``tokens`` is the
    estimate for the retained set under ``encoding``.
    """

    node_ids: set[str]
    dropped: list[str] = field(default_factory=list)
    tokens: int = 0
    encoding: str = "structured"

    def __iter__(self):
        return iter(self.node_ids)

    def __contains__(self, item):
        return item in self.node_ids

    def __len__(self):
        return len(self.node_ids)


def scoped_subgraph(m: SemanticMap, anchor_id: str,
                    query_terms: Iterable[str], k: int,
                    token_budget: int,
                    encoding: str = "structured") -> ScopedSubgraph:
    """Scope a subgraph around ``anchor_id`` under a token budget.

    Includes the k-hop neighbourhood of the anchor plus any node whose label
    or attributes lexically match a query term (stemmed match, reusing
    :mod:`src.verify`). If the serialisation would exceed ``token_budget``,
    the furthest nodes (by hop distance from the anchor, then Euclidean
    distance) are dropped first; the anchor is never dropped. Dropped nodes
    are recorded on the result.
    """
    if not m.has_node(anchor_id):
        raise KeyError(anchor_id)
    if encoding not in ENCODERS:
        raise ValueError(f"unknown encoding {encoding!r}")

    khop = {n.id for n in m.k_hop(anchor_id, k).nodes()}

    query_stems: set[str] = set()
    for term in query_terms:
        query_stems.update(_content_stems(term))
    lexical: set[str] = set()
    if query_stems:
        for node in m.nodes():
            if _node_stem_set(m, node.id) & query_stems:
                lexical.add(node.id)

    included = set(khop) | lexical | {anchor_id}

    # Distance-from-anchor ranking for truncation order.
    hops = nx.single_source_shortest_path_length(m.graph, anchor_id)
    apos = m.get_node(anchor_id).position

    def dist_key(nid: str):
        p = m.get_node(nid).position
        euclid = (p[0] - apos[0]) ** 2 + (p[1] - apos[1]) ** 2
        return (hops.get(nid, math.inf), euclid)

    encoder = ENCODERS[encoding]
    dropped: list[str] = []
    while (token_estimate(encoder(m, included, anchor_id)) > token_budget
           and len(included) > 1):
        victim = max((nid for nid in included if nid != anchor_id),
                     key=dist_key)
        included.remove(victim)
        dropped.append(victim)

    return ScopedSubgraph(
        node_ids=included,
        dropped=dropped,
        tokens=token_estimate(encoder(m, included, anchor_id)),
        encoding=encoding,
    )
