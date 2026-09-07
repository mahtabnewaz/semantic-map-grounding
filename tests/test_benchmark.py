"""Tests for src.benchmark. No LLM is called (CLAUDE.md invariant 2).

Ground truth is deterministic; these re-derive uniqueness independently to
prove no ambiguous query survived.
"""

import pytest

from data.lexicon import is_valid_target
from src.benchmark import (
    _connective_rooms,
    _nearest_room,
    _room_neighbours,
    _target_rooms,
    generate,
    index_by_plan,
)
from src.graph import NodeType
from src.houseexpo import load_dataset
from src.verify import grounded


@pytest.fixture(scope="module")
def maps():
    return load_dataset(40, seed=0)


@pytest.fixture(scope="module")
def bench(maps):
    queries = generate(maps, n_per_stratum=15, seed=0, report=False)
    return queries, index_by_plan(maps)


def test_expected_ids_exist_and_are_target_rooms(bench):
    queries, idx = bench
    for q in queries:
        if q.expected is None:
            continue
        m = idx[q.plan_id]
        node = m.get_node(q.expected)
        assert node is not None
        assert node.type is NodeType.ROOM
        assert is_valid_target(node.label)


def test_relational_anchors_are_target_rooms(bench):
    queries, idx = bench
    for q in queries:
        if q.stratum != "relational":
            continue
        m = idx[q.plan_id]
        anchor = m.get_node(q.anchor)
        assert anchor is not None
        assert anchor.type is NodeType.ROOM
        assert is_valid_target(anchor.label)
        assert q.expected != q.anchor


def test_ungroundable_targets_genuinely_absent(bench):
    queries, idx = bench
    for q in queries:
        if not q.stratum.startswith("ungroundable"):
            continue
        assert q.expected is None
        m = idx[q.plan_id]
        # Nothing in the plan grounds to the query's referring phrase.
        phrase = q.query_terms[0]
        assert not any(
            grounded(m, n.id, phrase, 0.6).ok for n in _target_rooms(m)
        ), f"{q.stratum} query grounded: {q.text!r} / {phrase!r}"


def test_no_ambiguous_query_survives(bench):
    queries, idx = bench
    for q in queries:
        m = idx[q.plan_id]
        targets = _target_rooms(m)
        if q.stratum == "direct":
            same = [r for r in targets if r.label == m.get_node(q.expected).label]
            assert len(same) == 1
        elif q.stratum == "attribute":
            attr = q.query_terms[0]
            matches = [r for r in targets
                       if grounded(m, r.id, attr, 0.6).ok]
            assert len(matches) == 1
            assert matches[0].id == q.expected
        elif q.stratum == "relational":
            if q.relation == "adjacent_to":
                sat = _room_neighbours(m, q.anchor)
            elif q.relation == "connects_to":
                sat = _connective_rooms(m, q.anchor)
            else:  # nearest
                node, unique = _nearest_room(m, m.get_node(q.anchor))
                assert unique and node.id == q.expected
                continue
            assert sat == {q.expected}


def test_same_seed_same_output(maps):
    a = generate(maps, n_per_stratum=15, seed=3, report=False)
    b = generate(maps, n_per_stratum=15, seed=3, report=False)
    key = lambda qs: [(q.text, q.stratum, q.expected, q.anchor, q.relation,
                       q.plan_id, q.template_id, q.split) for q in qs]
    assert key(a) == key(b)


def test_splits_share_no_plan(bench):
    queries, _ = bench
    tune = {q.plan_id for q in queries if q.split == "tuning"}
    test = {q.plan_id for q in queries if q.split == "test"}
    assert tune and test
    assert tune.isdisjoint(test)


def test_functional_stratum_generation(monkeypatch, maps):
    from src import benchmark as B
    monkeypatch.setattr(B, "FUNCTIONAL_QUERIES",
                        {"Kitchen": ["where I warm the soup at midnight",
                                     "the spot with the oven"]})
    qs = B.generate(maps, n_per_stratum=15, seed=0, report=False)
    fq = [q for q in qs if q.stratum == "functional"]
    assert fq, "expected functional queries once phrasings exist"
    idx = B.index_by_plan(maps)
    for q in fq:
        node = idx[q.plan_id].get_node(q.expected)
        assert node.type is NodeType.ROOM
        assert node.label == "Kitchen"          # the category IS the answer
        assert q.query_terms == [q.text]        # phrasing is the surface form
        assert q.relation is None and q.anchor is None


def test_every_query_has_required_fields(bench):
    queries, _ = bench
    for q in queries:
        assert q.text and q.stratum and q.plan_id and q.template_id
        assert q.query_terms
        assert q.split in ("tuning", "test")
