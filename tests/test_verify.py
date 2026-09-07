"""Tests for src.verify. Pure Python; no LLM is called (invariant 2)."""

from src import verify
from src.graph import Node, SemanticMap
from src.verify import (
    grounded,
    node_exists,
    relation_holds,
    verify as verify_entry,
)


def _living_room_map(n: int) -> SemanticMap:
    """A map with ``n`` living rooms and nothing else."""
    m = SemanticMap()
    for i in range(n):
        m.add_node(Node(f"lr{i}", "Living_Room", "room", (float(i), 0.0),
                        attributes=["lounge", "sofa"]))
    return m


# --- the four required scenarios ------------------------------------------
def test_valid_direct_answer(small_map):
    """A correct direct answer verifies: the node exists, grounds to the
    query phrase, and is reachable from the start."""
    proposal = {
        "node_id": "n_kitchen",
        "target_phrase": "the kitchen",
        "from_id": "n_office",
    }
    result = verify_entry(small_map, proposal, "direct")
    assert result.ok
    assert result.failed_predicate is None


def test_semantic_snap_exists_but_fails_grounding(small_map):
    """A node that exists but does not ground to the target phrase must
    fail on the grounding predicate, not on existence -- the 'semantic
    snap' failure mode. The diagnostic names the phrase and the nearest
    match (invariant 6)."""
    proposal = {"node_id": "n_office", "target_phrase": "server room"}

    assert node_exists(small_map, "n_office").ok  # existence passes

    result = verify_entry(small_map, proposal, "direct")
    assert not result.ok
    assert result.failed_predicate == "grounded"
    assert "server room" in result.diagnostic
    assert "threshold" in result.diagnostic
    assert result.details["score"] < result.details["threshold"]


def test_relational_answer_returns_anchor_itself(small_map):
    """A relational proposal where node_id == anchor_id must be rejected:
    a relation cannot hold between a node and itself."""
    proposal = {
        "node_id": "n_kitchen",
        "anchor_id": "n_kitchen",
        "relation": "adjacent_to",
    }
    result = verify_entry(small_map, proposal, "relational")
    assert not result.ok
    assert result.failed_predicate == "relation_holds"
    assert "itself" in result.diagnostic

    # And the predicate rejects it directly, too.
    direct = relation_holds(small_map, "n_kitchen", "n_kitchen", "adjacent_to")
    assert not direct.ok


def test_invented_node_id(small_map):
    """A proposal naming a node absent from the map fails on node_exists."""
    proposal = {"node_id": "n_ghost", "target_phrase": "the kitchen"}
    result = verify_entry(small_map, proposal, "direct")
    assert not result.ok
    assert result.failed_predicate == "node_exists"
    assert "n_ghost" in result.diagnostic


# --- supporting coverage ---------------------------------------------------
def test_grounding_stems_eat_eating(small_map):
    """The kitchen carries the attribute 'eating'; the phrase 'eat' must
    ground to it via stemming with min prefix length 3."""
    result = grounded(small_map, "n_kitchen", "eat", threshold=0.6)
    assert result.ok
    assert result.details["score"] == 1.0


def test_room_scoped_adjacency(small_map):
    # office and kitchen are joined through a doorway only -> related for
    # both adjacent_to and connects_to under the room-scoped definition.
    assert relation_holds(small_map, "n_kitchen", "n_office", "adjacent_to").ok
    assert relation_holds(small_map, "n_kitchen", "n_office", "connects_to").ok


def test_doorway_never_valid_room_relation_answer(small_map):
    # A doorway is never a valid answer for any room-scoped relation.
    for rel in ("adjacent_to", "connects_to", "nearest"):
        r = relation_holds(small_map, "n_door", "n_office", rel)
        assert not r.ok, rel


def test_relation_nearest_ranks_rooms_only(small_map):
    # Nearest room to the office is the kitchen; the doorway (nearer in
    # metric distance) and the dock (not a room) are both excluded.
    assert relation_holds(small_map, "n_kitchen", "n_office", "nearest").ok
    assert not relation_holds(small_map, "n_dock", "n_office", "nearest").ok


# --- discriminator gate in grounded() -------------------------------------
def test_discriminator_rejects_when_insufficient():
    m = _living_room_map(1)
    r = grounded(m, "lr0", "the second living room", 0.6)
    assert not r.ok
    assert r.failed_predicate == "grounded"
    assert "second living room" in r.diagnostic
    assert r.details["actual"] == 1 and r.details["required"] == 2


def test_discriminator_accepts_when_sufficient():
    m = _living_room_map(2)
    assert grounded(m, "lr0", "the second living room", 0.6).ok


def test_letter_discriminator():
    m = _living_room_map(2)
    # "living room C" needs 3 living rooms; only 2 exist -> rejected.
    assert not grounded(m, "lr0", "living room C", 0.6).ok
    # "living room B" needs 2; satisfied, and overlap still clears 0.6.
    assert grounded(m, "lr0", "living room B", 0.6).ok


def test_plain_label_unaffected_by_discriminator():
    m = _living_room_map(1)
    assert grounded(m, "lr0", "the living room", 0.6).ok


def test_ordinary_phrases_unchanged(small_map):
    # No discriminator -> identical to prior behaviour.
    assert grounded(small_map, "n_kitchen", "eat", 0.6).ok
    assert not grounded(small_map, "n_office", "server room", 0.6).ok


def test_functional_verify_is_structural_only(small_map):
    # Functional queries get node_exists + reachable ONLY -- no grounding.
    # A node that exists and is reachable verifies even though the functional
    # phrasing would never ground it lexically.
    ok = verify_entry(small_map,
                      {"node_id": "n_kitchen",
                       "target_phrase": "somewhere to cook a big feast",
                       "from_id": "n_office"},
                      "functional")
    assert ok.ok
    # existence still applies
    ghost = verify_entry(small_map,
                         {"node_id": "n_ghost", "from_id": "n_office"},
                         "functional")
    assert not ghost.ok and ghost.failed_predicate == "node_exists"


def test_relational_unreachable_anchor_missing(small_map):
    proposal = {
        "node_id": "n_kitchen",
        "anchor_id": "n_ghost",
        "relation": "adjacent_to",
    }
    result = verify_entry(small_map, proposal, "relational")
    assert not result.ok
    assert result.failed_predicate == "node_exists"
