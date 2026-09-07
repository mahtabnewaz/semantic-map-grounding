"""Node, edge, and annotation datastructures for the semantic map.

The semantic map pairs a metric occupancy grid with an annotated
topological graph: rooms, corridors, doorways, junctions, docks,
waypoints, plus semantic-zone polygons (keep-out, restricted, operating
area). This module defines those datastructures over a ``networkx.Graph``
and their JSON round-trip.

Invariant 4 (CLAUDE.md): an edge is admitted only if grid-level
reachability holds between its endpoints under the robot footprint. That
check is NOT implemented here -- ``SemanticMap.add_edge`` accepts an
injected ``reachable`` predicate and refuses edges that fail it, so this
module stays testable without an occupancy grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

import networkx as nx
from shapely.geometry import Point, Polygon, mapping, shape


class NodeType(str, Enum):
    """Closed set of topological node kinds."""

    ROOM = "room"
    CORRIDOR = "corridor"
    DOORWAY = "doorway"
    JUNCTION = "junction"
    DOCK = "dock"
    WAYPOINT = "waypoint"


class SemanticClass(str, Enum):
    """Closed set of annotation zone kinds."""

    KEEP_OUT = "keep_out"
    RESTRICTED = "restricted"
    OPERATING_AREA = "operating_area"


class EdgeRejectedError(ValueError):
    """Raised when ``add_edge`` refuses an edge (invariant 4)."""


def _coerce_enum(value, enum_cls):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError as exc:
        allowed = ", ".join(e.value for e in enum_cls)
        raise ValueError(
            f"{value!r} is not a valid {enum_cls.__name__}; allowed: {allowed}"
        ) from exc


@dataclass
class Node:
    """A topological node."""

    id: str
    label: str
    type: NodeType
    position: tuple[float, float]
    attributes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.type = _coerce_enum(self.type, NodeType)
        self.position = (float(self.position[0]), float(self.position[1]))
        self.attributes = list(self.attributes)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type.value,
            "position": list(self.position),
            "attributes": list(self.attributes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            id=d["id"],
            label=d["label"],
            type=d["type"],
            position=tuple(d["position"]),
            attributes=list(d.get("attributes", [])),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class Edge:
    """A topological edge between two node ids."""

    u: str
    v: str
    traversable: bool = True
    cost: float = 1.0
    constraints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "u": self.u,
            "v": self.v,
            "traversable": bool(self.traversable),
            "cost": float(self.cost),
            "constraints": dict(self.constraints),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(
            u=d["u"],
            v=d["v"],
            traversable=bool(d.get("traversable", True)),
            cost=float(d.get("cost", 1.0)),
            constraints=dict(d.get("constraints", {})),
        )


@dataclass
class Annotation:
    """A semantic-zone polygon."""

    polygon: Polygon
    semantic_class: SemanticClass

    def __post_init__(self):
        self.semantic_class = _coerce_enum(self.semantic_class, SemanticClass)

    def to_dict(self) -> dict:
        return {
            "semantic_class": self.semantic_class.value,
            "polygon": mapping(self.polygon),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(polygon=shape(d["polygon"]), semantic_class=d["semantic_class"])


# Predicate injected into add_edge: (u_id, v_id) -> bool.
ReachablePredicate = Callable[[str, str], bool]


class SemanticMap:
    """Topological graph + semantic annotations + a grid reference.

    The graph is a ``networkx.Graph``. Each node carries its :class:`Node`
    under the ``data`` attribute; each edge carries its :class:`Edge`
    under ``data`` and mirrors ``cost``/``traversable`` for algorithms.
    """

    def __init__(self, occupancy_grid=None):
        self.graph = nx.Graph()
        self.annotations: list[Annotation] = []
        self.occupancy_grid = occupancy_grid  # opaque reference (PGM+YAML, array, ...)

    # ---- nodes -----------------------------------------------------------
    def add_node(self, node: Node) -> None:
        self.graph.add_node(node.id, data=node)

    def remove_node(self, node_id: str) -> None:
        self.graph.remove_node(node_id)

    def has_node(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)

    def get_node(self, node_id: str) -> Optional[Node]:
        if not self.graph.has_node(node_id):
            return None
        return self.graph.nodes[node_id].get("data")

    def nodes(self) -> Iterable[Node]:
        for _, d in self.graph.nodes(data=True):
            if d.get("data") is not None:
                yield d["data"]

    # ---- edges -----------------------------------------------------------
    def add_edge(
        self,
        u: str,
        v: str,
        *,
        traversable: bool = True,
        cost: float = 1.0,
        constraints: Optional[dict] = None,
        reachable: Optional[ReachablePredicate] = None,
    ) -> Edge:
        """Add an edge, refusing it if the reachability predicate fails.

        ``reachable`` is an injected callable ``(u, v) -> bool`` (invariant
        4). When supplied and it returns False, no edge is added and
        :class:`EdgeRejectedError` is raised -- the graph must never assert
        connectivity the robot cannot realise.
        """
        if u == v:
            raise EdgeRejectedError(f"self-loop rejected on node {u!r}")
        for endpoint in (u, v):
            if not self.graph.has_node(endpoint):
                raise EdgeRejectedError(f"unknown endpoint {endpoint!r}")
        if reachable is not None and not reachable(u, v):
            raise EdgeRejectedError(
                f"edge {u!r}->{v!r} refused: fails grid reachability under footprint"
            )
        edge = Edge(u=u, v=v, traversable=traversable, cost=cost,
                    constraints=constraints or {})
        self.graph.add_edge(
            u, v, data=edge, cost=edge.cost, traversable=edge.traversable
        )
        return edge

    def remove_edge(self, u: str, v: str) -> None:
        self.graph.remove_edge(u, v)

    def get_edge(self, u: str, v: str) -> Optional[Edge]:
        if not self.graph.has_edge(u, v):
            return None
        return self.graph.edges[u, v].get("data")

    # ---- retrieval -------------------------------------------------------
    def k_hop(self, node_id: str, k: int) -> "SemanticMap":
        """Return the induced subgraph within ``k`` hops of ``node_id``.

        Used to scope a subgraph before serialisation. Annotations are
        carried over unfiltered (semantic zones are global).
        """
        if not self.graph.has_node(node_id):
            raise KeyError(node_id)
        ego = nx.ego_graph(self.graph, node_id, radius=k)
        sub = SemanticMap(occupancy_grid=self.occupancy_grid)
        sub.graph = self.graph.subgraph(ego.nodes).copy()
        sub.annotations = list(self.annotations)
        return sub

    def neighbours(self, node_id: str) -> list[str]:
        return list(self.graph.neighbors(node_id))

    def shortest_path(self, from_id: str, to_id: str) -> Optional[list[str]]:
        """Shortest traversable path by edge cost, or None if unreachable.

        Only edges marked ``traversable`` are considered.
        """
        if not (self.graph.has_node(from_id) and self.graph.has_node(to_id)):
            return None
        view = nx.subgraph_view(
            self.graph,
            filter_edge=lambda a, b: self.graph.edges[a, b].get("traversable", True),
        )
        try:
            return nx.shortest_path(view, from_id, to_id, weight="cost")
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def is_reachable(self, from_id: str, to_id: str) -> bool:
        return self.shortest_path(from_id, to_id) is not None

    # ---- lookup ----------------------------------------------------------
    def nodes_by_label(self, label: str) -> list[Node]:
        target = label.strip().lower()
        return [n for n in self.nodes() if n.label.strip().lower() == target]

    def nodes_with_attribute(self, attribute: str) -> list[Node]:
        target = attribute.strip().lower()
        return [
            n
            for n in self.nodes()
            if any(a.strip().lower() == target for a in n.attributes)
        ]

    # ---- annotations -----------------------------------------------------
    def add_annotation(self, annotation: Annotation) -> None:
        self.annotations.append(annotation)

    def zones_containing(self, x: float, y: float) -> list[Annotation]:
        p = Point(x, y)
        return [a for a in self.annotations if a.polygon.covers(p)]

    # ---- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes()],
            "edges": [
                self.graph.edges[u, v]["data"].to_dict()
                for u, v in self.graph.edges()
            ],
            "annotations": [a.to_dict() for a in self.annotations],
        }

    @classmethod
    def from_dict(cls, d: dict, occupancy_grid=None) -> "SemanticMap":
        m = cls(occupancy_grid=occupancy_grid)
        for nd in d.get("nodes", []):
            m.add_node(Node.from_dict(nd))
        for ed in d.get("edges", []):
            edge = Edge.from_dict(ed)
            # Bypass the reachability predicate on load: the stored graph
            # already satisfied invariant 4 when it was authored.
            m.graph.add_edge(
                edge.u, edge.v, data=edge, cost=edge.cost,
                traversable=edge.traversable,
            )
        for ad in d.get("annotations", []):
            m.add_annotation(Annotation.from_dict(ad))
        return m

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str, occupancy_grid=None) -> "SemanticMap":
        return cls.from_dict(json.loads(text), occupancy_grid=occupancy_grid)
