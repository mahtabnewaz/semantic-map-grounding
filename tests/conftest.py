"""Shared fixtures. No LLM is used anywhere in the test suite (invariant 2)."""

import pytest

from src.graph import Node, SemanticMap


@pytest.fixture
def small_map() -> SemanticMap:
    """A tiny hand-built map:

        office_a --door_1-- kitchen --hall-- charging_dock

    Positions are laid out on a line so 'nearest' is unambiguous.
    """
    m = SemanticMap()
    m.add_node(Node("n_office", "office A", "room", (0.0, 0.0),
                    attributes=["desk", "printer"]))
    m.add_node(Node("n_door", "doorway", "doorway", (1.0, 0.0)))
    m.add_node(Node("n_kitchen", "kitchen", "room", (2.0, 0.0),
                    attributes=["sink", "fridge", "eating"]))
    m.add_node(Node("n_hall", "hallway", "corridor", (3.0, 0.0)))
    m.add_node(Node("n_dock", "charging dock", "dock", (4.0, 0.0),
                    attributes=["charger"]))

    m.add_edge("n_office", "n_door")
    m.add_edge("n_door", "n_kitchen")
    m.add_edge("n_kitchen", "n_hall")
    m.add_edge("n_hall", "n_dock")
    return m
