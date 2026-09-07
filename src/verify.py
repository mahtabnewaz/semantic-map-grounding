"""Structural grounding checks + diagnostics.

Pure Python (invariant 2, CLAUDE.md): set membership, graph reachability,
lexical grounding, and structural relation checks against a
:class:`~src.graph.SemanticMap`, and nothing else. No LLM is called
anywhere in this module. If a prompt appears here, that is a bug.

Every failure returns a diagnostic (invariant 6) naming the failed
predicate and the entity involved -- the debugging surface an operator
reads, not a log line.

Predicates are individually callable; :func:`verify` selects them by
query type. Proposals are plain dicts (the agent returns only symbolic
content -- node ids and relations -- never coordinates, invariant 1):

    {
        "node_id":      "n7",            # the proposed answer node
        "claimed_label": "kitchen",      # optional, for label_consistent
        "target_phrase": "the kitchen",  # direct / attribute grounding
        "anchor_id":     "n3",           # relational: the reference entity
        "relation":      "adjacent_to",  # relational
        "from_id":       "n0",           # optional start for reachability
    }
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import Point

from data.lexicon import is_valid_target

from .graph import NodeType, SemanticClass, SemanticMap

# Words carrying no grounding signal. "room", "dock", etc. are NOT here --
# they are content words that legitimately ground a phrase.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "is", "are", "and",
    "or", "with", "for", "near", "by", "that", "this", "please", "go",
    "me", "i", "my", "we", "us", "it",
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_MIN_STEM_LEN = 3  # "eating" -> "eat" (len 3) must survive; 4 would drop it.
_SUFFIXES = ("ing", "ers", "er", "ed", "es", "s")


@dataclass
class VerificationResult:
    """Outcome of a verification check."""

    ok: bool
    failed_predicate: Optional[str] = None
    diagnostic: Optional[str] = None
    details: dict = field(default_factory=dict)


def _ok(predicate: str, **details) -> VerificationResult:
    return VerificationResult(ok=True, failed_predicate=None, diagnostic=None,
                              details=dict(details, predicate=predicate))


def _fail(predicate: str, diagnostic: str, **details) -> VerificationResult:
    return VerificationResult(ok=False, failed_predicate=predicate,
                              diagnostic=diagnostic,
                              details=dict(details, predicate=predicate))


# --------------------------------------------------------------------------
# lexical helpers (grounding)
# --------------------------------------------------------------------------
def _stem(word: str) -> str:
    """Crude suffix stripper. Strips a suffix only if >= 3 chars remain,
    so "eat"/"eating" both reduce to "eat" (min prefix length 3, not 4)."""
    w = word.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= _MIN_STEM_LEN:
            return w[: -len(suf)]
    return w


def _content_stems(text: str) -> list[str]:
    stems = []
    for tok in _WORD_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        stems.append(_stem(tok))
    return stems


def _node_stem_set(m: SemanticMap, node_id: str) -> set[str]:
    node = m.get_node(node_id)
    if node is None:
        return set()
    bag = [node.label] + list(node.attributes)
    stems: set[str] = set()
    for piece in bag:
        stems.update(_content_stems(piece))
    return stems


def _grounding_score(m: SemanticMap, node_id: str, target_phrase: str) -> float:
    """Fraction of the phrase's content words that match the node's
    label/attribute vocabulary (stem equality)."""
    target = _content_stems(target_phrase)
    if not target:
        return 0.0
    node_stems = _node_stem_set(m, node_id)
    matched = sum(1 for t in target if t in node_stems)
    return matched / len(target)


def _best_grounding(m: SemanticMap, target_phrase: str):
    """(node, score) of the best-scoring node in the map, or (None, 0.0)."""
    best_node = None
    best_score = 0.0
    for node in m.nodes():
        s = _grounding_score(m, node.id, target_phrase)
        if s > best_score or best_node is None:
            best_node, best_score = node, s
    return best_node, best_score


# --------------------------------------------------------------------------
# discriminator detection (runs before the overlap score)
# --------------------------------------------------------------------------
# A discriminator is a token that singles out one of several same-category
# rooms: an ordinal ("second"), a count ("two"), or a letter designator
# ("bedroom C"). A phrase carrying one grounds only if the map holds enough
# same-category rooms to satisfy it -- overlap score is irrelevant otherwise.
_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4,
                  "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
                  "ninth": 9, "tenth": 10}
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _detect_discriminator(phrase: str):
    """Return (required_count, human_phrase, category_stems, category_display)
    if ``phrase`` carries a discriminator, else None."""
    tokens = re.findall(r"[A-Za-z0-9]+", phrase)
    required = None
    disc_display = None
    disc_idx = None
    is_letter = False

    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in _ORDINAL_WORDS:
            required, disc_display, disc_idx = _ORDINAL_WORDS[low], low, i
            break
        mo = re.fullmatch(r"(\d+)(st|nd|rd|th)", low)
        if mo:
            required, disc_display, disc_idx = int(mo.group(1)), low, i
            break
        if low in _NUMBER_WORDS:
            required, disc_display, disc_idx = _NUMBER_WORDS[low], low, i
            break
        if re.fullmatch(r"\d+", low):
            required, disc_display, disc_idx = int(low), low, i
            break
    if required is None:
        # letter designator: a lone uppercase A-Z following a category word
        for i, tok in enumerate(tokens):
            if i > 0 and re.fullmatch(r"[A-Z]", tok):
                required, disc_display, disc_idx = ord(tok) - ord("A") + 1, tok, i
                is_letter = True
                break
    if required is None:
        return None

    cat_words = [t.lower() for j, t in enumerate(tokens)
                 if j != disc_idx and t.lower() not in _STOPWORDS]
    cat_stems = {_stem(w) for w in cat_words}
    if not cat_stems:
        return None

    cat_display = " ".join(cat_words)
    cat_singular = (" ".join(cat_words[:-1] + [_singular(cat_words[-1])])
                    if cat_words else cat_display)
    human = (f"{cat_display} {disc_display}" if is_letter
             else f"{disc_display} {cat_display}")
    return required, human, cat_stems, cat_singular


def _category_room_count(m: SemanticMap, category_stems: set[str]) -> int:
    """Number of ROOM nodes whose label covers ``category_stems``."""
    count = 0
    for node in m.nodes():
        if node.type is NodeType.ROOM:
            if category_stems <= set(_content_stems(node.label)):
                count += 1
    return count


# --------------------------------------------------------------------------
# predicates
# --------------------------------------------------------------------------
def node_exists(m: SemanticMap, node_id: str) -> VerificationResult:
    if m.has_node(node_id):
        return _ok("node_exists", node_id=node_id)
    return _fail(
        "node_exists",
        f"node '{node_id}' does not exist in the map",
        node_id=node_id,
    )


def label_consistent(m: SemanticMap, node_id: str,
                     claimed_label: str) -> VerificationResult:
    node = m.get_node(node_id)
    if node is None:
        return _fail("label_consistent",
                     f"node '{node_id}' does not exist in the map",
                     node_id=node_id)
    if node.label.strip().lower() == claimed_label.strip().lower():
        return _ok("label_consistent", node_id=node_id, label=node.label)
    return _fail(
        "label_consistent",
        f"node '{node_id}' is labelled '{node.label}', not '{claimed_label}'",
        node_id=node_id, actual_label=node.label, claimed_label=claimed_label,
    )


def reachable(m: SemanticMap, from_id: str, to_id: str) -> VerificationResult:
    for endpoint in (from_id, to_id):
        if not m.has_node(endpoint):
            return _fail("reachable",
                         f"cannot check reachability: node '{endpoint}' "
                         f"does not exist",
                         from_id=from_id, to_id=to_id)
    path = m.shortest_path(from_id, to_id)
    if path is not None:
        return _ok("reachable", from_id=from_id, to_id=to_id, path=path)
    return _fail(
        "reachable",
        f"no traversable path from '{from_id}' to '{to_id}'",
        from_id=from_id, to_id=to_id,
    )


def route_clear_of_keepout(m: SemanticMap, path: list[str]) -> VerificationResult:
    for node_id in path:
        node = m.get_node(node_id)
        if node is None:
            return _fail("route_clear_of_keepout",
                         f"route references unknown node '{node_id}'",
                         node_id=node_id)
        x, y = node.position
        for ann in m.annotations:
            if ann.semantic_class is SemanticClass.KEEP_OUT and \
                    ann.polygon.covers(Point(x, y)):
                return _fail(
                    "route_clear_of_keepout",
                    f"route passes through keep-out zone at node "
                    f"'{node_id}' ({node.label})",
                    node_id=node_id,
                )
    return _ok("route_clear_of_keepout", path=path)


def grounded(m: SemanticMap, node_id: str, target_phrase: str,
             threshold: float = 0.60) -> VerificationResult:
    """Does ``node_id``'s vocabulary lexically ground ``target_phrase``?"""
    node = m.get_node(node_id)
    if node is None:
        return _fail("grounded",
                     f"node '{node_id}' does not exist in the map",
                     node_id=node_id, target_phrase=target_phrase)

    # Discriminator gate (before overlap): a phrase naming an ordinal, count,
    # or letter the map cannot satisfy never grounds, whatever the overlap.
    disc = _detect_discriminator(target_phrase)
    if disc is not None:
        required, human, cat_stems, cat_display = disc
        actual = _category_room_count(m, cat_stems)
        if actual < required:
            return _fail(
                "grounded",
                f"phrase specifies '{human}' but map contains "
                f"{actual} {cat_display}",
                node_id=node_id, target_phrase=target_phrase,
                discriminator=human, required=required, actual=actual,
            )

    score = _grounding_score(m, node_id, target_phrase)
    if score >= threshold:
        return _ok("grounded", node_id=node_id, target_phrase=target_phrase,
                   score=score, threshold=threshold)
    # Enrich the diagnostic with the map-wide best match (invariant 6).
    best_node, best_score = _best_grounding(m, target_phrase)
    best_label = best_node.label if best_node is not None else "<none>"
    return _fail(
        "grounded",
        f"no node grounds to '{target_phrase}'; nearest match "
        f"'{best_label}' scored {best_score:.2f} (threshold {threshold:.2f})",
        node_id=node_id, target_phrase=target_phrase, score=score,
        threshold=threshold, best_match=best_label, best_score=best_score,
    )


def relation_holds(m: SemanticMap, node_id: str, anchor_id: str,
                   relation: str) -> VerificationResult:
    """Graph-structural relation check between ``node_id`` and ``anchor_id``.

    Supported: adjacent_to, connects_to, across_from, nearest. These are
    room-scoped: adjacent_to/connects_to hold when the answer is a room
    reachable from the anchor through connective space only (doorways,
    corridors, waypoints); nearest ranks only over target-eligible rooms.
    A doorway/corridor/waypoint is never a valid room-scoped answer.
    Rejects ``node_id == anchor_id`` for every relation.
    """
    if node_id == anchor_id:
        return _fail(
            "relation_holds",
            f"relation '{relation}' cannot hold between a node and itself "
            f"('{node_id}')",
            node_id=node_id, anchor_id=anchor_id, relation=relation,
        )
    for endpoint in (node_id, anchor_id):
        if not m.has_node(endpoint):
            return _fail("relation_holds",
                         f"relation '{relation}' references unknown node "
                         f"'{endpoint}'",
                         node_id=node_id, anchor_id=anchor_id,
                         relation=relation)

    if relation == "adjacent_to":
        return _rel_adjacent(m, node_id, anchor_id, relation)
    if relation == "connects_to":
        return _rel_connects(m, node_id, anchor_id, relation)
    if relation == "across_from":
        return _rel_across(m, node_id, anchor_id, relation)
    if relation == "nearest":
        return _rel_nearest(m, node_id, anchor_id, relation)
    return _fail("relation_holds",
                 f"unsupported relation '{relation}'",
                 node_id=node_id, anchor_id=anchor_id, relation=relation)


def _room_connected(m, anchor_id, node_id) -> bool:
    """True if a path exists from ``anchor_id`` to ``node_id`` passing through
    only non-ROOM nodes (doorways, corridors, waypoints). A direct edge is the
    zero-intermediate special case. Rooms block: the search never expands
    through another room."""
    seen = {anchor_id}
    q = deque([anchor_id])
    while q:
        cur = q.popleft()
        for nb in m.neighbours(cur):
            if nb == node_id:
                return True
            if nb in seen:
                continue
            seen.add(nb)
            other = m.get_node(nb)
            if other is not None and other.type is not NodeType.ROOM:
                q.append(nb)          # connective node -> keep exploring
            # a ROOM (that is not the target) is a wall; do not expand it
    return False


def _rel_room_path(m, node_id, anchor_id, relation):
    """Shared body for adjacent_to / connects_to: the answer must be a room
    reachable from the anchor through connective space only."""
    node = m.get_node(node_id)
    if node.type is not NodeType.ROOM:
        return _fail("relation_holds",
                     f"'{node_id}' is a {node.type.value}, not a room; "
                     f"'{relation}' requires a room answer",
                     node_id=node_id, anchor_id=anchor_id, relation=relation)
    if _room_connected(m, anchor_id, node_id):
        return _ok("relation_holds", node_id=node_id, anchor_id=anchor_id,
                   relation=relation)
    return _fail("relation_holds",
                 f"'{node_id}' is not {relation.replace('_', ' ')} "
                 f"'{anchor_id}' (no path through connective space only)",
                 node_id=node_id, anchor_id=anchor_id, relation=relation)


def _rel_adjacent(m, node_id, anchor_id, relation):
    return _rel_room_path(m, node_id, anchor_id, relation)


def _rel_connects(m, node_id, anchor_id, relation):
    return _rel_room_path(m, node_id, anchor_id, relation)


def _rel_across(m, node_id, anchor_id, relation):
    # "across from": both share a common doorway/junction neighbour.
    portal_types = {NodeType.DOORWAY, NodeType.JUNCTION}
    shared = set(m.neighbours(node_id)) & set(m.neighbours(anchor_id))
    portals = [
        s for s in shared
        if (m.get_node(s) is not None and m.get_node(s).type in portal_types)
    ]
    if portals:
        return _ok("relation_holds", node_id=node_id, anchor_id=anchor_id,
                   relation=relation, via=portals)
    return _fail("relation_holds",
                 f"'{node_id}' is not across from '{anchor_id}' "
                 f"(no shared doorway or junction)",
                 node_id=node_id, anchor_id=anchor_id, relation=relation)


def _rel_nearest(m, node_id, anchor_id, relation):
    """Nearest ranks only over target-eligible ROOM nodes; a doorway,
    corridor, or waypoint is never a valid nearest answer."""
    anchor = m.get_node(anchor_id)
    candidates = [n for n in m.nodes()
                  if n.id != anchor_id and n.type is NodeType.ROOM
                  and is_valid_target(n.label)]
    if not candidates:
        return _fail("relation_holds",
                     f"no target-eligible room to rank as nearest to "
                     f"'{anchor_id}'",
                     node_id=node_id, anchor_id=anchor_id, relation=relation)
    node = m.get_node(node_id)
    if node.type is not NodeType.ROOM or not is_valid_target(node.label):
        return _fail("relation_holds",
                     f"'{node_id}' is a {node.type.value}; nearest must be a "
                     f"target-eligible room",
                     node_id=node_id, anchor_id=anchor_id, relation=relation)
    nearest = min(candidates, key=lambda n: _dist2(n.position, anchor.position))
    if nearest.id == node_id:
        return _ok("relation_holds", node_id=node_id, anchor_id=anchor_id,
                   relation=relation)
    return _fail("relation_holds",
                 f"'{node_id}' is not the nearest room to '{anchor_id}'; "
                 f"'{nearest.id}' ({nearest.label}) is nearer",
                 node_id=node_id, anchor_id=anchor_id, relation=relation,
                 nearest=nearest.id)


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def verify(m: SemanticMap, proposal: dict,
           query_type: str, threshold: float = 0.60) -> VerificationResult:
    """Verify a symbolic ``proposal`` against the map for ``query_type``.

    query_type dispatch:
      - direct, attribute -> node_exists, grounded, (reachable if from_id)
      - relational        -> node_exists (node + anchor), relation_holds,
                             (reachable if from_id)
      - functional        -> node_exists, (reachable if from_id) ONLY. No
                             semantic predicate (see below).

    Returns the first failing predicate's result, or an ``ok`` result.
    """
    node_id = proposal.get("node_id")
    if node_id is None:
        return _fail("node_exists", "proposal has no node_id",
                     proposal=proposal)

    exists = node_exists(m, node_id)
    if not exists.ok:
        return exists

    if query_type in ("direct", "attribute"):
        target_phrase = proposal.get("target_phrase", "")
        g = grounded(m, node_id, target_phrase, threshold=threshold)
        if not g.ok:
            return g

    elif query_type == "relational":
        anchor_id = proposal.get("anchor_id")
        relation = proposal.get("relation")
        if anchor_id is None or relation is None:
            return _fail("relation_holds",
                         "relational proposal needs 'anchor_id' and 'relation'",
                         proposal=proposal)
        anchor_exists = node_exists(m, anchor_id)
        if not anchor_exists.ok:
            return _fail("node_exists",
                         f"anchor node '{anchor_id}' does not exist in the map",
                         node_id=anchor_id)
        r = relation_holds(m, node_id, anchor_id, relation)
        if not r.ok:
            return r

    elif query_type == "functional":
        # Functional / world-knowledge queries name a room by its function
        # with NO lexical overlap with the node by construction, so grounded()
        # cannot verify them. Do NOT add a semantic predicate here: checking
        # the node's category against the query's ground-truth category would
        # be an ORACLE -- with a unique node per category, "right category"
        # equals "right node" equals the answer, so the predicate would be
        # reading the label. This stratum therefore gets only the
        # model-independent structural guards, node_exists (above) and
        # reachable (below). The reduced verification coverage is intentional
        # and is recorded per result row so the analysis reports it rather
        # than averaging it in with strata that have a semantic predicate.
        pass

    else:
        return _fail("verify", f"unknown query_type '{query_type}'",
                     query_type=query_type)

    # Reachability is checked for every query type when a start is given.
    from_id = proposal.get("from_id")
    if from_id is None and query_type == "relational":
        from_id = proposal.get("anchor_id")
    if from_id is not None:
        reach = reachable(m, from_id, node_id)
        if not reach.ok:
            return reach

    return _ok("verify", node_id=node_id, query_type=query_type)
