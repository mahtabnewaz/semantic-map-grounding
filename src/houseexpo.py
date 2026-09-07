"""Load HouseExpo floor plans into :class:`~src.graph.SemanticMap`.

Data (inspected, not assumed -- see the schema notes below):
  Each plan JSON (packed in ``HouseExpo/HouseExpo/json.tar.gz``, one file
  per house, id == PNG filename) has:
    - ``verts``: single outer wall outline of the whole house, metres,
      positive-quadrant frame. Not per-room; unused for the topology here.
    - ``room_category``: SUNCG category -> list of axis-aligned bboxes
      ``[xmin, ymin, xmax, ymax]`` in metres. Boxes may be identical across
      aliased categories (Living_Room/Dining_Room, Toilet/Bathroom) and may
      overlap or contain one another (~35% of plans contain a nested box).
    - ``room_num``, ``bbox``: counts / overall extent.

Because doorways and connectivity are ABSENT from the data, they are
synthesised geometrically. Design decisions (confirmed with the author):

  * Room nodes: boxes that are (near-)identical are MERGED into one room
    node (multi-label kept: primary label by :data:`PRIORITY`, the rest in
    ``metadata['alt_labels']``); attributes are the union of the lexicon
    entries of all merged categories.
  * Doorways: a doorway node is placed where two room boxes share a border
    (touch within ``ADJ_EPS`` along one axis and overlap >= ``MIN_OVERLAP``
    along the other). Edges are always room -> doorway -> room, never
    room -> room directly.
  * Frame: JSON metres taken as-is, +x east / +y north.

Attributes come from :data:`data.lexicon.LEXICON`, hand-authored, never
generated. With an empty lexicon, room attribute lists are empty.

Caveat on invariant 4 (CLAUDE.md): edges here are synthesised from bbox
geometry, not verified against an occupancy grid under a robot footprint.
Grid-level reachability gating belongs to the real-map pipeline
(``extract.py``); HouseExpo adjacency is a geometric proxy.
"""

from __future__ import annotations

import json
import math
import random
import tarfile
from collections import Counter
from pathlib import Path
from typing import Optional

import networkx as nx

from data.lexicon import (
    CIRCULATION_CATEGORIES,
    TRANSPORT_CATEGORIES,
    attributes_for,
)

from .graph import Node, NodeType, SemanticMap

# Location of the packed plan JSONs, relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_TAR = _REPO_ROOT / "HouseExpo" / "HouseExpo" / "json.tar.gz"

# Label priority when several categories share one merged box (lower index
# wins). Ordered by dataset frequency; used only to pick a primary label.
PRIORITY = [
    "Kitchen", "Bathroom", "Bedroom", "Toilet", "Living_Room", "Dining_Room",
    "Room", "Office", "Garage", "Hallway", "Hall", "Gym", "Child_Room",
    "Storage", "Wardrobe", "Guest_Room", "Balcony", "Lobby", "Entryway",
    "Terrace", "Loggia", "Boiler_room", "Aeration", "Passenger_elevator",
    "Freight_elevator",
]

# Geometry tolerances (metres). ADJ_EPS/MIN_OVERLAP were selected by a sweep
# over 300 plans against fragmentation rate (see experiments notes): loosening
# the gap tolerance from 0.30 to 0.75 cut the share of fragmented plans from
# ~64% to ~43% while keeping doorways/plan (~7) well under the rooms x 1.5
# sanity ceiling. Residual fragmentation is handled by restricting each plan
# to its largest connected component at load time.
MERGE_EPS = 0.15    # boxes closer than this on every coord are "the same room"
ADJ_EPS = 0.75      # max gap/overlap along the shared-wall axis to count as touching
MIN_OVERLAP = 0.25  # min overlap along the other axis (a passable opening)

_member_cache: Optional[list[str]] = None


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _box_close(a, b, eps: float = MERGE_EPS) -> bool:
    return all(abs(x - y) <= eps for x, y in zip(a, b))


def _centroid(box) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _interval_overlap(a0, a1, b0, b1) -> float:
    """>0 overlap length, <0 gap, ~0 touching."""
    return min(a1, b1) - max(a0, b0)


def _doorway_between(A, B, adj_eps: float = ADJ_EPS,
                     min_overlap: float = MIN_OVERLAP):
    """Position of a doorway on the shared wall of boxes A and B, or None."""
    ax0, ay0, ax1, ay1 = A
    bx0, by0, bx1, by1 = B
    ox = _interval_overlap(ax0, ax1, bx0, bx1)
    oy = _interval_overlap(ay0, ay1, by0, by1)
    # Vertical wall: boxes side by side (touch in x, overlap in y).
    if abs(ox) <= adj_eps and oy >= min_overlap:
        wall_x = (min(ax1, bx1) + max(ax0, bx0)) / 2.0
        mid_y = (max(ay0, by0) + min(ay1, by1)) / 2.0
        return (wall_x, mid_y)
    # Horizontal wall: boxes stacked (touch in y, overlap in x).
    if abs(oy) <= adj_eps and ox >= min_overlap:
        mid_x = (max(ax0, bx0) + min(ax1, bx1)) / 2.0
        wall_y = (min(ay1, by1) + max(ay0, by0)) / 2.0
        return (mid_x, wall_y)
    return None


def _priority_key(cat: str) -> int:
    try:
        return PRIORITY.index(cat)
    except ValueError:
        return len(PRIORITY)  # unknown categories sort last


def _node_type(primary_category: str) -> NodeType:
    """Topological type from the node's primary (label) category.

    Circulation categories become corridors and vertical transport becomes
    waypoints; everything else is a room. Typing follows the primary label
    (chosen by :data:`PRIORITY`) so the type is consistent with the label
    even for merged multi-label nodes.
    """
    if primary_category in CIRCULATION_CATEGORIES:
        return NodeType.CORRIDOR
    if primary_category in TRANSPORT_CATEGORIES:
        return NodeType.WAYPOINT
    return NodeType.ROOM


# --------------------------------------------------------------------------
# building a map from a parsed plan dict
# --------------------------------------------------------------------------
def _build_map(plan: dict, largest_component_only: bool = True) -> SemanticMap:
    """Build a SemanticMap from a parsed HouseExpo plan dict.

    With ``largest_component_only`` (default), the map is restricted to its
    largest connected component. Real buildings are connected; the residual
    fragmentation after the doorway-threshold sweep is an artefact of
    geometric inference, so a single navigable region is used. Every target,
    anchor, and robot start then lives in one reachable component.
    """
    m = SemanticMap()
    m.plan_id = plan.get("id")

    # 1. Merge (near-)identical boxes across categories into room clusters.
    clusters: list[dict] = []
    for cat, boxes in plan.get("room_category", {}).items():
        for box in boxes:
            box = [float(v) for v in box]
            for cl in clusters:
                if _box_close(cl["box"], box):
                    cl["cats"].add(cat)
                    break
            else:
                clusters.append({"box": box, "cats": {cat}})

    # Deterministic ordering by centroid, so ids are stable.
    clusters.sort(key=lambda cl: (round(cl["box"][0], 3), round(cl["box"][1], 3)))

    # 2. One room node per cluster.
    room_ids: list[str] = []
    for i, cl in enumerate(clusters):
        cats = cl["cats"]
        primary = min(cats, key=_priority_key)
        alt = sorted(cats - {primary}, key=_priority_key)
        # Union of lexicon attributes across all merged categories.
        attrs: list[str] = []
        for cat in [primary] + alt:
            for a in attributes_for(cat):
                if a not in attrs:
                    attrs.append(a)
        rid = f"r{i}"
        m.add_node(Node(
            id=rid,
            label=primary,
            type=_node_type(primary),
            position=_centroid(cl["box"]),
            attributes=attrs,
            metadata={"categories": sorted(cats), "alt_labels": alt,
                      "bbox": cl["box"]},
        ))
        room_ids.append(rid)

    # 3. Synthesise doorways on shared borders; edges room -> door -> room.
    door_count = 0
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            pos = _doorway_between(clusters[a]["box"], clusters[b]["box"])
            if pos is None:
                continue
            ra, rb = room_ids[a], room_ids[b]
            did = f"d{door_count}"
            door_count += 1
            m.add_node(Node(id=did, label="doorway", type=NodeType.DOORWAY,
                            position=pos, attributes=[],
                            metadata={"between": [ra, rb]}))
            m.add_edge(ra, did, cost=_dist(m.get_node(ra).position, pos))
            m.add_edge(did, rb, cost=_dist(pos, m.get_node(rb).position))

    if largest_component_only:
        _restrict_to_largest_component(m)
    return m


def _restrict_to_largest_component(m: SemanticMap) -> list[str]:
    """Drop all nodes outside the largest connected component. Returns the
    ids of the dropped nodes (for exclusion reporting)."""
    if m.graph.number_of_nodes() == 0:
        return []
    keep = max(nx.connected_components(m.graph), key=len)
    dropped = [n for n in list(m.graph.nodes()) if n not in keep]
    for n in dropped:
        m.graph.remove_node(n)
    return dropped


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# --------------------------------------------------------------------------
# public loaders
# --------------------------------------------------------------------------
def load_plan(path, largest_component_only: bool = True) -> SemanticMap:
    """Load a single HouseExpo plan JSON file into a SemanticMap."""
    with open(path) as fh:
        return _build_map(json.load(fh),
                          largest_component_only=largest_component_only)


def _plan_names() -> list[str]:
    global _member_cache
    if _member_cache is None:
        with tarfile.open(JSON_TAR, "r:gz") as tf:
            _member_cache = sorted(
                m.name for m in tf.getmembers() if m.name.endswith(".json")
            )
    return _member_cache


def scan_categories() -> "Counter[str]":
    """Distinct SUNCG categories in the full dataset, with plan counts.

    Streams every plan JSON once. ``counts[cat]`` is the number of plans
    containing at least one room of ``cat``. Feed the keys to
    :func:`data.lexicon.check_keys` to reconcile the lexicon against the
    live data: ``check_keys(list(scan_categories()))``.
    """
    counts: "Counter[str]" = Counter()
    with tarfile.open(JSON_TAR, "r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            plan = json.load(tf.extractfile(member))
            for cat in plan.get("room_category", {}):
                counts[cat] += 1
    return counts


def load_dataset(n: int, seed: int, *, min_rooms: int = 2,
                 largest_component_only: bool = True) -> list[SemanticMap]:
    """Sample ``n`` plans reproducibly and load them.

    A plan is skipped (and counted) if it fails to parse or yields fewer
    than ``min_rooms`` room nodes (counted after the largest-component
    restriction). The number skipped is printed. Returns the successfully
    loaded maps (<= n).
    """
    names = _plan_names()
    n = min(n, len(names))
    chosen = random.Random(seed).sample(names, n)

    maps: list[SemanticMap] = []
    skipped = 0
    with tarfile.open(JSON_TAR, "r:gz") as tf:
        for name in chosen:
            try:
                plan = json.load(tf.extractfile(tf.getmember(name)))
                m = _build_map(plan,
                               largest_component_only=largest_component_only)
                rooms = sum(1 for nd in m.nodes() if nd.type is NodeType.ROOM)
                if rooms < min_rooms:
                    skipped += 1
                    continue
                maps.append(m)
            except Exception:
                skipped += 1
    print(f"load_dataset(n={n}, seed={seed}): loaded {len(maps)}, "
          f"skipped {skipped}")
    return maps


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _report(n: int = 300, seed: int = 0) -> None:
    from statistics import mean

    from . import serialize as S

    maps = load_dataset(n, seed)
    room_counts, door_counts, tok_counts = [], [], []
    for m in maps:
        ids = {nd.id for nd in m.nodes()}
        room_counts.append(sum(1 for nd in m.nodes()
                               if nd.type is NodeType.ROOM))
        door_counts.append(sum(1 for nd in m.nodes()
                               if nd.type is NodeType.DOORWAY))
        tok_counts.append(S.token_estimate(S.structured(m, ids)))
    print(f"\nsample of {len(maps)} valid plans (seed={seed})")
    print(f"  mean rooms per plan     : {mean(room_counts):.2f}")
    print(f"  mean doorways per plan  : {mean(door_counts):.2f}")
    print(f"  mean structured tokens  : {mean(tok_counts):.1f}")


if __name__ == "__main__":
    _report()
