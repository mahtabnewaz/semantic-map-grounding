"""Query generation + deterministic grading.

Generates natural-language spatial queries over loaded SemanticMaps. Ground
truth is computed deterministically from the graph, never model-generated
(CLAUDE.md invariant 3). No LLM is called in this module.

Strata:
- direct           names a target room by its label
- attribute        specifies a target room by a lexicon attribute, not label
- relational       relative to an anchor room; relations adjacent_to,
                   connects_to, nearest (room-scoped, see below)
- ungroundable_far names a category absent from the whole dataset
- ungroundable_near a plausible variant of a present room ("the second
                   kitchen", "bedroom C"); correct behaviour is abstention

Targeting rules:
- Only NodeType.ROOM nodes that are valid targets (is_valid_target on the
  primary category, so NON_TARGET_CATEGORIES like "Room" are excluded) are
  ever an expected answer or a relational anchor. Corridors and waypoints
  stay in the graph but are never asked for.

Ambiguity is rejected at generation time and counted by cause:
- direct       two target rooms of the same category in one plan
- attribute    an attribute matching more than one target room
- relational   a relation satisfied by more than one target room
- elevator     any query grounding to "elevator"/"lift" in a plan that
               contains BOTH elevator types

Room-scoped relations (deterministic, doorway-aware). HouseExpo topology is
room -> doorway -> room, so rooms are never directly edge-adjacent and a
naive nearest-over-all-nodes would return a doorway. Ground truth here is
therefore computed room-to-room:
- adjacent_to  rooms sharing a single doorway with the anchor
- connects_to  rooms reachable from the anchor through connective space
               (doorways/corridors/waypoints) without passing another room
- nearest      the unique nearest target room by Euclidean centroid distance

NOTE / alignment gap: verify.relation_holds defines adjacent_to as a direct
edge and nearest over all node types. Under the room->doorway->room topology
those never match a room answer, so the agent's verifier must be made
room-aware before it can confirm these benchmark answers. Do not "fix" this
by editing verify.py without approval (CLAUDE.md) -- it is flagged here.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from data.functional_queries import FUNCTIONAL_QUERIES
from data.lexicon import (
    NON_TARGET_CATEGORIES,
    TRANSPORT_CATEGORIES,
    is_valid_target,
)

from .graph import NodeType, SemanticMap
from .verify import _content_stems, _node_stem_set, grounded

GROUND_THRESHOLD = 0.60
RELATIONS = ["adjacent_to", "connects_to", "nearest"]

# Categories absent from the entire HouseExpo dataset (all 25 present ones
# checked). Used for ungroundable_far. A grounding guard rejects any that
# would still ground, so this list only needs to be plausible + absent.
ABSENT_CATEGORIES = [
    "server room", "laboratory", "operating theatre", "cockpit",
    "greenhouse", "sauna", "wine cellar", "recording studio", "armoury",
    "planetarium", "darkroom", "brewery",
]

_ORDINALS = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
             7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}

_ELEVATOR_WORDS = {"elevator", "lift"}

# All surface phrasings live here so they are reviewable in one place.
TEMPLATES: dict = {
    "direct": [
        "go to the {label}",
        "navigate to the {label}",
        "take me to the {label}",
        "head over to the {label}",
        "I need to reach the {label}",
    ],
    "attribute": [
        "take me to the room with {attribute}",
        "which room is used for {attribute}?",
        "find the area for {attribute}",
        "I'm looking for somewhere for {attribute}",
        "go to the space meant for {attribute}",
    ],
    "relational": {
        "adjacent_to": [
            "which room is next to the {anchor}?",
            "take me to the room adjacent to the {anchor}",
            "go to the room beside the {anchor}",
        ],
        "connects_to": [
            "which room connects to the {anchor}?",
            "take me to the room leading off the {anchor}",
            "go to the room joined to the {anchor}",
        ],
        "nearest": [
            "which room is closest to the {anchor}?",
            "take me to the nearest room to the {anchor}",
            "go to the room nearest the {anchor}",
        ],
    },
    "ungroundable_far": [
        "go to the {label}",
        "navigate to the {label}",
        "take me to the {label}",
        "I need to reach the {label}",
    ],
    "ungroundable_near": {
        "ordinal": [
            "take me to the {ordinal} {label}",
            "go to the {ordinal} {label}",
            "navigate to the {ordinal} {label}",
        ],
        "letter": [
            "go to {label} {letter}",
            "take me to {label} {letter}",
            "head to {label} {letter}",
        ],
    },
}


@dataclass
class Query:
    """One benchmark query with deterministic ground truth."""

    text: str
    stratum: str
    expected: Optional[str]          # target node id, or None for ungroundable
    anchor: Optional[str]            # relational anchor node id
    relation: Optional[str]          # relational relation name
    query_terms: list[str]           # salient words for retrieval scoping
    plan_id: str
    template_id: str
    split: Optional[str] = None      # "tuning" or "test"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _pretty(label: str) -> str:
    return label.replace("_", " ").lower()


def index_by_plan(maps: list[SemanticMap]) -> dict[str, SemanticMap]:
    """Map plan_id -> SemanticMap, for grading and tests."""
    return {m.plan_id: m for m in maps}


def _target_rooms(m: SemanticMap) -> list:
    return [n for n in m.nodes()
            if n.type is NodeType.ROOM and is_valid_target(n.label)]


def _rooms_by_category(rooms: list) -> dict[str, list]:
    d: dict[str, list] = defaultdict(list)
    for n in rooms:
        d[n.label].append(n)
    return d


def _plan_has_both_elevators(m: SemanticMap) -> bool:
    cats: set[str] = set()
    for n in m.nodes():
        cats.update(n.metadata.get("categories", []))
    return TRANSPORT_CATEGORIES <= cats


def _elevator_ambiguous(m: SemanticMap, terms: list[str]) -> bool:
    if not _plan_has_both_elevators(m):
        return False
    for t in terms:
        for w in t.lower().split():
            if w in _ELEVATOR_WORDS:
                return True
    return False


def _room_neighbours(m: SemanticMap, room_id: str) -> set[str]:
    """Target rooms sharing a single doorway with ``room_id``."""
    out: set[str] = set()
    for d in m.neighbours(room_id):
        if m.get_node(d).type is not NodeType.DOORWAY:
            continue
        for r in m.neighbours(d):
            if r == room_id:
                continue
            node = m.get_node(r)
            if node.type is NodeType.ROOM and is_valid_target(node.label):
                out.add(r)
    return out


def _connective_rooms(m: SemanticMap, anchor_id: str) -> set[str]:
    """Target rooms reachable from ``anchor_id`` through connective space
    only (never expanding through another room)."""
    seen = {anchor_id}
    out: set[str] = set()
    q = deque([anchor_id])
    while q:
        cur = q.popleft()
        for nb in m.neighbours(cur):
            if nb in seen:
                continue
            seen.add(nb)
            node = m.get_node(nb)
            if node.type is NodeType.ROOM:
                if is_valid_target(node.label):
                    out.add(nb)          # reached a room; do not expand it
            else:
                q.append(nb)             # connective node; keep exploring
    return out


def _nearest_room(m: SemanticMap, anchor) -> Optional[tuple]:
    """(node, is_unique) nearest target room to ``anchor`` by centroid, or
    None if there is no other target room."""
    ax, ay = anchor.position
    best = None
    best_d = None
    tie = False
    for n in _target_rooms(m):
        if n.id == anchor.id:
            continue
        d = (n.position[0] - ax) ** 2 + (n.position[1] - ay) ** 2
        if best_d is None or d < best_d:
            best, best_d, tie = n, d, False
        elif d == best_d:
            tie = True
    if best is None:
        return None
    return (best, not tie)


# --------------------------------------------------------------------------
# opportunity collection (deterministic; ambiguity tallied into rej)
# --------------------------------------------------------------------------
def _direct_ops(m, rej) -> list:
    ops = []
    for label, rooms in _rooms_by_category(_target_rooms(m)).items():
        if len(rooms) == 1:
            ops.append((m, rooms[0]))
        else:
            rej["direct_duplicate_category"] += 1
    return ops


def _attribute_ops(m, rej) -> list:
    ops = []
    targets = _target_rooms(m)
    stem_sets = {r.id: _node_stem_set(m, r.id) for r in targets}
    for room in targets:
        for attr in room.attributes:
            astems = set(_content_stems(attr))
            if not astems:
                continue
            matches = [r for r in targets if astems <= stem_sets[r.id]]
            if len(matches) == 1 and matches[0].id == room.id:
                ops.append((m, room, attr))
            elif len(matches) > 1:
                rej["attribute_multi_match"] += 1
    return ops


def _relational_ops(m, rej) -> dict[str, list]:
    ops = {r: [] for r in RELATIONS}
    targets = _target_rooms(m)
    for anchor in targets:
        # adjacent_to
        adj = _room_neighbours(m, anchor.id)
        if len(adj) == 1:
            ops["adjacent_to"].append((m, anchor, m.get_node(next(iter(adj)))))
        elif len(adj) > 1:
            rej["relation_adjacent_to_ambiguous"] += 1
        # connects_to
        con = _connective_rooms(m, anchor.id)
        if len(con) == 1:
            ops["connects_to"].append((m, anchor, m.get_node(next(iter(con)))))
        elif len(con) > 1:
            rej["relation_connects_to_ambiguous"] += 1
        # nearest
        near = _nearest_room(m, anchor)
        if near is not None:
            node, unique = near
            if unique:
                ops["nearest"].append((m, anchor, node))
            else:
                rej["relation_nearest_ambiguous"] += 1
    return ops


def _far_ops(m, rej) -> list:
    ops = []
    targets = _target_rooms(m)
    for fake in ABSENT_CATEGORIES:
        if any(grounded(m, r.id, fake, GROUND_THRESHOLD).ok for r in targets):
            rej["ungroundable_far_grounds"] += 1
            continue
        ops.append((m, fake))
    return ops


def _near_ops(m, rej) -> list:
    ops = []
    targets = _target_rooms(m)
    for label, rooms in _rooms_by_category(targets).items():
        k = len(rooms)
        pretty = _pretty(label)
        # ordinal overshoot: "the (k+1)th <label>"
        ordw = _ORDINALS.get(k + 1, f"{k + 1}th")
        phrase = f"{ordw} {pretty}"
        if any(grounded(m, r.id, phrase, GROUND_THRESHOLD).ok for r in targets):
            rej["ungroundable_near_grounds"] += 1
        else:
            ops.append((m, "ordinal", pretty, ordw, phrase))
        # letter overshoot: "<label> <letter beyond the count>"
        letter = chr(ord("A") + k)
        phrase = f"{pretty} {letter}"
        if any(grounded(m, r.id, phrase, GROUND_THRESHOLD).ok for r in targets):
            rej["ungroundable_near_grounds"] += 1
        else:
            ops.append((m, "letter", pretty, letter, phrase))
    return ops


def _functional_ops(m, rej) -> list:
    """Hand-authored functional phrasings (data/functional_queries.py).

    This applies AMBIGUITY rejection only -- it removes queries with no unique
    ground truth, never queries merely because the lexical baseline can answer
    them. A phrasing that grounds lexically to its OWN target category is kept
    (that is a measured property, reported, not a filter). Two rejection rules:

    1. Room-level (as `direct`): the target category must be uniquely present
       in the plan, else the expected node is ambiguous.
    2. Cross-category: reject if any OTHER category also present in the plan
       grounds the phrasing at/above threshold -- then a solver could
       legitimately name that competitor, so the ground truth is not unique.
       Rejections are logged by (target, competitor) pair (the intrinsically
       confusable clusters: sleep, meal, sanitation).
    """
    ops = []
    by_cat = _rooms_by_category(_target_rooms(m))
    for category, phrasings in FUNCTIONAL_QUERIES.items():
        if not phrasings:
            continue
        rooms = by_cat.get(category, [])
        if len(rooms) != 1:
            if len(rooms) > 1:
                rej["functional_duplicate_category"] += 1
            continue
        node = rooms[0]
        for i, phrasing in enumerate(phrasings):
            competitor = _functional_competitor(m, by_cat, category, phrasing)
            if competitor is not None:
                rej["functional_crosscat"] += 1
                rej[("functional_crosscat_pair", category, competitor)] += 1
                continue
            ops.append((m, node, category, i, phrasing))
    return ops


def _functional_competitor(m, by_cat, target, phrasing):
    """First OTHER category present in the plan that grounds ``phrasing`` at or
    above threshold, or None. Its presence means the ground truth is not
    unique -- the query is ambiguous and must be rejected."""
    for cat in sorted(by_cat):
        if cat == target:
            continue
        for room in by_cat[cat]:
            if grounded(m, room.id, phrasing, GROUND_THRESHOLD).ok:
                return cat
    return None


# --------------------------------------------------------------------------
# rendering (rng chooses the surface template)
# --------------------------------------------------------------------------
def _tpl(rng, templates: list, prefix: str):
    i = rng.randrange(len(templates))
    return templates[i], f"{prefix}#{i}"


def _render_direct(rng, op) -> Query:
    m, room = op
    tmpl, tid = _tpl(rng, TEMPLATES["direct"], "direct")
    label = _pretty(room.label)
    return Query(tmpl.format(label=label), "direct", room.id, None, None,
                 [label], m.plan_id, tid)


def _render_attribute(rng, op) -> Query:
    m, room, attr = op
    tmpl, tid = _tpl(rng, TEMPLATES["attribute"], "attribute")
    return Query(tmpl.format(attribute=attr), "attribute", room.id, None, None,
                 [attr], m.plan_id, tid)


def _render_relational(rng, op, relation: str) -> Query:
    m, anchor, other = op
    tmpl, tid = _tpl(rng, TEMPLATES["relational"][relation],
                     f"relational.{relation}")
    label = _pretty(anchor.label)
    return Query(tmpl.format(anchor=label), "relational", other.id, anchor.id,
                 relation, [label], m.plan_id, tid)


def _render_far(rng, op) -> Query:
    m, fake = op
    tmpl, tid = _tpl(rng, TEMPLATES["ungroundable_far"], "ungroundable_far")
    return Query(tmpl.format(label=fake), "ungroundable_far", None, None, None,
                 [fake], m.plan_id, tid)


def _render_near(rng, op) -> Query:
    m, style, pretty, qual, phrase = op
    tmpl, tid = _tpl(rng, TEMPLATES["ungroundable_near"][style],
                     f"ungroundable_near.{style}")
    if style == "ordinal":
        text = tmpl.format(ordinal=qual, label=pretty)
    else:
        text = tmpl.format(label=pretty, letter=qual)
    return Query(text, "ungroundable_near", None, None, None, [phrase],
                 m.plan_id, tid)


def _render_functional(rng, op) -> Query:
    # The authored phrasing IS the surface form; there is no template. The
    # phrasing is the query_terms too, so retrieval and the lexical baseline
    # see exactly what a user would say.
    m, node, category, i, phrasing = op
    return Query(phrasing, "functional", node.id, None, None, [phrasing],
                 m.plan_id, f"functional.{category}#{i}")


def _emit(ops, quota, rng, render_fn, rej, out):
    if not ops or quota <= 0:
        return
    for op in rng.sample(ops, min(quota, len(ops))):
        q = render_fn(rng, op)
        if _elevator_ambiguous(op[0], q.query_terms):
            rej["elevator_both"] += 1
            continue
        out.append(q)


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------
def _assign_splits(queries: list[Query], rng, tuning_frac: float = 0.20):
    plans = sorted({q.plan_id for q in queries})
    rng.shuffle(plans)
    if len(plans) <= 1:
        n_tune = 0
    else:
        n_tune = max(1, min(len(plans) - 1, round(len(plans) * tuning_frac)))
    tuning = set(plans[:n_tune])
    for q in queries:
        q.split = "tuning" if q.plan_id in tuning else "test"


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def generate(maps: list[SemanticMap], n_per_stratum: int,
             seed: int, *, report: bool = True) -> list[Query]:
    """Generate a reproducible, stratified benchmark over ``maps``.

    Aims for ``n_per_stratum`` queries per stratum (relational is split
    roughly evenly across its relations; relations that cannot fill their
    share simply contribute fewer). The result is split by plan id into a
    20% tuning / 80% test partition so no plan appears in both.
    """
    rng = random.Random(seed)
    rej: Counter = Counter()

    direct_ops, attribute_ops, far_ops, near_ops = [], [], [], []
    functional_ops = []
    relational_ops = {r: [] for r in RELATIONS}
    for m in sorted(maps, key=lambda x: x.plan_id):
        direct_ops += _direct_ops(m, rej)
        attribute_ops += _attribute_ops(m, rej)
        for r, lst in _relational_ops(m, rej).items():
            relational_ops[r] += lst
        far_ops += _far_ops(m, rej)
        near_ops += _near_ops(m, rej)
        functional_ops += _functional_ops(m, rej)

    out: list[Query] = []
    _emit(direct_ops, n_per_stratum, rng, _render_direct, rej, out)
    _emit(attribute_ops, n_per_stratum, rng, _render_attribute, rej, out)
    per_rel = max(1, n_per_stratum // len(RELATIONS))
    for r in RELATIONS:
        _emit(relational_ops[r], per_rel, rng,
              lambda rng, op, _r=r: _render_relational(rng, op, _r), rej, out)
    _emit(far_ops, n_per_stratum, rng, _render_far, rej, out)
    _emit(near_ops, n_per_stratum, rng, _render_near, rej, out)
    # Functional stratum is empty until the phrasings are authored+frozen.
    _emit(functional_ops, n_per_stratum, rng, _render_functional, rej, out)

    _assign_splits(queries=out, rng=rng)

    if report:
        _print_report(out, rej, seed)
    return out


def _print_report(queries: list[Query], rej: Counter, seed: int):
    by_stratum = Counter(q.stratum for q in queries)
    by_relation = Counter(q.relation for q in queries if q.relation)
    lengths = [len(q.text.split()) for q in queries]
    tune_plans = {q.plan_id for q in queries if q.split == "tuning"}
    test_plans = {q.plan_id for q in queries if q.split == "test"}
    tune_q = sum(1 for q in queries if q.split == "tuning")

    print(f"\n=== benchmark generation (seed={seed}) ===")
    print(f"total queries: {len(queries)}")
    print("queries per stratum:")
    for s in ["direct", "attribute", "relational",
              "ungroundable_far", "ungroundable_near", "functional"]:
        line = f"  {s:18s}: {by_stratum.get(s, 0)}"
        if s == "relational":
            rels = ", ".join(f"{r}={by_relation.get(r, 0)}" for r in RELATIONS)
            line += f"   ({rels})"
        print(line)
    print("ambiguity rejections by cause:")
    for cause in ["direct_duplicate_category", "attribute_multi_match",
                  "relation_adjacent_to_ambiguous",
                  "relation_connects_to_ambiguous",
                  "relation_nearest_ambiguous", "elevator_both"]:
        print(f"  {cause:34s}: {rej.get(cause, 0)}")
    print("near-miss rejections (would ground) : "
          f"{rej.get('ungroundable_near_grounds', 0)}")
    print("far rejections (would ground)       : "
          f"{rej.get('ungroundable_far_grounds', 0)}")
    if lengths:
        print(f"mean query length (words): {sum(lengths) / len(lengths):.2f}")
    print(f"split: tuning {len(tune_plans)} plans / {tune_q} queries ; "
          f"test {len(test_plans)} plans / {len(queries) - tune_q} queries")
