"""Tests for src.serialize. No LLM is called (CLAUDE.md invariant 2)."""

from src.serialize import (
    ENCODERS,
    natural,
    relational,
    scoped_subgraph,
    structured,
    token_estimate,
)


def test_all_encodings_nonempty_same_subgraph(small_map):
    ids = {"n_office", "n_door", "n_kitchen"}
    s = structured(small_map, ids, anchor_id="n_office")
    r = relational(small_map, ids, anchor_id="n_office")
    n = natural(small_map, ids, anchor_id="n_office")
    assert s.strip() and r.strip() and n.strip()
    # Same entities named in every encoding (same information content).
    for nid in ids:
        assert nid in s and nid in r and nid in n


def test_relational_has_no_position_digits(small_map):
    """The fixture's ids/labels/types/attributes contain no digits, so any
    digit in the relational encoding would be a leaked coordinate. There
    must be none. The structured encoding, by contrast, carries them."""
    ids = {"n_office", "n_door", "n_kitchen", "n_hall", "n_dock"}
    r = relational(small_map, ids, anchor_id="n_office")
    assert not any(c.isdigit() for c in r), r
    # Sanity: the numeric information lives in `structured`.
    assert any(c.isdigit() for c in structured(small_map, ids, "n_office"))


def test_natural_has_no_position_digits(small_map):
    ids = {"n_office", "n_door", "n_kitchen"}
    n = natural(small_map, ids, anchor_id="n_office")
    assert not any(c.isdigit() for c in n), n


def test_token_estimate_scales_with_length():
    assert token_estimate("") == 0
    assert token_estimate("abcd") == 1
    assert token_estimate("a" * 40) == 10


def test_scoped_subgraph_respects_k(small_map):
    # Line graph: office - door - kitchen - hall - dock.
    res = scoped_subgraph(small_map, "n_office", query_terms=[], k=1,
                          token_budget=100_000)
    assert res.node_ids == {"n_office", "n_door"}
    assert "n_kitchen" not in res.node_ids
    assert "n_dock" not in res.node_ids
    assert res.dropped == []


def test_scoped_subgraph_lexical_pulls_distant_match(small_map):
    # "charging" grounds the far dock (hop 4) even outside the k=1 hull.
    res = scoped_subgraph(small_map, "n_office", query_terms=["charging"],
                          k=1, token_budget=100_000)
    assert "n_dock" in res.node_ids


def test_token_budget_truncation_drops_distant_first(small_map):
    res = scoped_subgraph(small_map, "n_office", query_terms=[], k=4,
                          token_budget=15)
    # Something was dropped and it was reported.
    assert res.dropped
    # The furthest node is always dropped first.
    assert "n_dock" in res.dropped
    # The anchor is never dropped.
    assert "n_office" in res.node_ids
    # Every retained node is at least as near as every dropped node
    # (hop distance == x-position in this line layout).
    retained_x = [small_map.get_node(i).position[0] for i in res.node_ids]
    dropped_x = [small_map.get_node(i).position[0] for i in res.dropped]
    assert max(retained_x) <= min(dropped_x)
    # Drops are ordered furthest-first.
    assert dropped_x == sorted(dropped_x, reverse=True)


def test_encoders_registry_matches_functions(small_map):
    ids = {"n_office", "n_door"}
    assert ENCODERS["structured"](small_map, ids, "n_office") == \
        structured(small_map, ids, "n_office")
    assert ENCODERS["relational"](small_map, ids, "n_office") == \
        relational(small_map, ids, "n_office")
    assert ENCODERS["natural"](small_map, ids, "n_office") == \
        natural(small_map, ids, "n_office")
