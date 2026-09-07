"""Tests for src.houseexpo. No LLM is called (CLAUDE.md invariant 2).

These read real HouseExpo plans from the packed JSON in the repo.
"""

import json
import math
import tarfile

import pytest

from src.graph import NodeType
from src.houseexpo import JSON_TAR, load_dataset, load_plan


@pytest.fixture(scope="module")
def sample():
    # A small reproducible sample; several plans so at least one has edges.
    return load_dataset(6, seed=0)


def test_loaded_plan_has_at_least_two_rooms(sample):
    assert sample, "sample should not be empty"
    for m in sample:
        rooms = [n for n in m.nodes() if n.type is NodeType.ROOM]
        assert len(rooms) >= 2


def test_every_edge_touches_a_doorway(sample):
    saw_edge = False
    for m in sample:
        for u, v in m.graph.edges():
            saw_edge = True
            types = {m.get_node(u).type, m.get_node(v).type}
            # room -> doorway -> room: exactly one endpoint is a doorway.
            assert NodeType.DOORWAY in types
            assert not (m.get_node(u).type is NodeType.DOORWAY
                        and m.get_node(v).type is NodeType.DOORWAY)
    assert saw_edge, "expected at least one edge across the sample"


def test_all_positions_finite(sample):
    for m in sample:
        for n in m.nodes():
            assert math.isfinite(n.position[0])
            assert math.isfinite(n.position[1])


def test_same_seed_same_sample():
    a = load_dataset(5, seed=7)
    b = load_dataset(5, seed=7)
    assert [m.plan_id for m in a] == [m.plan_id for m in b]


def test_different_seed_differs():
    a = load_dataset(8, seed=1)
    b = load_dataset(8, seed=2)
    assert [m.plan_id for m in a] != [m.plan_id for m in b]


def test_load_plan_from_path(tmp_path):
    # Extract one plan to disk and load it through the path-based API.
    with tarfile.open(JSON_TAR, "r:gz") as tf:
        name = next(m.name for m in tf.getmembers() if m.name.endswith(".json"))
        plan = json.load(tf.extractfile(tf.getmember(name)))
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    m = load_plan(p)
    assert m.plan_id == plan["id"]
    assert any(n.type is NodeType.ROOM for n in m.nodes())
