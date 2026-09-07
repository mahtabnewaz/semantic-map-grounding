"""Occupancy grid -> topological graph.

Skeletonise the free space (scikit-image), detect rooms/junctions/
doorways, and build the topological graph. Enforces the grid-level
reachability invariant when admitting edges (CLAUDE.md #4): a path must
exist between endpoints in the occupancy grid under the robot footprint.

Input:  PGM+YAML occupancy grid (data/real_maps) or HouseExpo floor plan.
Output: a SemanticGraph (see graph.py).

TODO: skeletonisation, region segmentation, edge admission with footprint.
"""
