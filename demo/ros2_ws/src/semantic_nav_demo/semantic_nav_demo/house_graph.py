"""Hand-authored semantic topological map for the TurtleBot3 house world.

This stands in for the operator-authoring step of the research: a human labels
rooms, marks doorways, and gives each a metric centroid in the map frame. The
resulting graph is the SAME ``SemanticMap`` type the paper uses, so the live
demo runs the identical ``resolve()`` + ``verify()`` code path.

CALIBRATION (do this once, on your machine)
-------------------------------------------
The (x, y) coordinates below are approximate for ``turtlebot3_house`` in the
default map frame. To make navigation land in the right rooms:
  1. Launch the demo and open RViz.
  2. Use the "Publish Point" tool and click the centre of each room, or run
       ros2 topic echo /clicked_point
     and read off map-frame coordinates.
  3. Replace the (x, y) values in ROOMS below. Orientation does not matter for
     the goal; Nav2 plans the path.
Attributes are drawn from the paper's lexicon so functional queries have a
target; do not put the room's own name in its attribute list.
"""

# room_id -> (label, x, y, [attributes])
ROOMS = {
    "living":  ("Living_Room", 0.0,  0.0, ["lounge", "sitting", "sofa", "television"]),
    "kitchen": ("Kitchen",     3.2,  1.0, ["cooking", "food", "meals", "fridge"]),
    "dining":  ("Dining_Room", 3.0, -1.6, ["dining", "eating", "dinner"]),
    "bed1":    ("Bedroom",    -3.0,  2.2, ["sleeping", "bed", "night"]),
    "bath":    ("Bathroom",   -3.4, -1.0, ["washing", "shower", "sink", "hygiene"]),
    "study":   ("Office",      6.2,  0.4, ["work", "desk", "study", "computer"]),
    "hall":    ("Hallway",     1.4,  0.2, []),   # corridor; connective space
}

# undirected room-room adjacencies (a doorway is synthesised on each)
DOORWAYS = [
    ("living", "hall"),
    ("hall", "kitchen"),
    ("kitchen", "dining"),
    ("hall", "bed1"),
    ("living", "bath"),
    ("hall", "study"),
]

# Curated demo queries: (text, stratum, query_terms). Free-typed queries in the
# UI default to the 'direct' path, which resolves named rooms and safely
# abstains on the rest; these buttons showcase each behaviour explicitly.
EXAMPLE_QUERIES = [
    ("take me to the kitchen",              "direct",       ["kitchen"]),
    ("go to the bedroom",                   "direct",       ["bedroom"]),
    ("somewhere I can cook dinner",         "functional",   ["cook", "dinner"]),
    ("where would I wash my hands",         "functional",   ["wash", "hands"]),
    ("i need to get some work done",        "functional",   ["work", "get", "done"]),
    ("go to the operating theatre",         "direct",       ["operating", "theatre"]),
    ("take me to the second kitchen",       "direct",       ["second", "kitchen"]),
]


def build_house_map():
    """Return a src.graph.SemanticMap for the house (metres, +x east/+y north).

    CORRIDOR/WAYPOINT typing follows the lexicon (Hallway -> corridor). Every
    room-room adjacency becomes a room->doorway->room pair of edges, exactly as
    the HouseExpo loader does, so relational/room-scoped checks behave the same.
    """
    from data.lexicon import CIRCULATION_CATEGORIES, TRANSPORT_CATEGORIES
    from src.graph import Node, NodeType, SemanticMap

    def _type(label):
        if label in CIRCULATION_CATEGORIES:
            return NodeType.CORRIDOR
        if label in TRANSPORT_CATEGORIES:
            return NodeType.WAYPOINT
        return NodeType.ROOM

    m = SemanticMap()
    m.plan_id = "turtlebot3_house"
    for rid, (label, x, y, attrs) in ROOMS.items():
        m.add_node(Node(id=rid, label=label, type=_type(label),
                        position=(float(x), float(y)), attributes=list(attrs),
                        metadata={"world_xy": (float(x), float(y))}))
    for i, (a, b) in enumerate(DOORWAYS):
        pa, pb = ROOMS[a][1:3], ROOMS[b][1:3]
        mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        did = f"door{i}"
        m.add_node(Node(id=did, label="doorway", type=NodeType.DOORWAY,
                        position=mid, attributes=[],
                        metadata={"between": [a, b]}))
        m.add_edge(a, did)
        m.add_edge(did, b)
    return m


def world_xy(m, node_id):
    """Map-frame (x, y) for a resolved node, for the Nav2 goal."""
    n = m.get_node(node_id)
    return None if n is None else (n.position[0], n.position[1])
