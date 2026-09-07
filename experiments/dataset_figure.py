#!/usr/bin/env python
"""Dataset/method figure: a HouseExpo floor plan and the semantic topological
map we derive from it. Left: merged room bounding boxes (the raw annotation).
Right: the topological graph -- rooms/corridors/waypoints as nodes at their
centroids, doorways synthesised on shared borders, edges room->doorway->room.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.graph import NodeType  # noqa: E402
from src.houseexpo import load_dataset  # noqa: E402

plt.rcParams.update({"font.size": 7, "font.family": "serif",
                     "savefig.bbox": "tight", "figure.dpi": 150})

TYPE_COLOR = {NodeType.ROOM: "#88CCEE", NodeType.CORRIDOR: "#DDCC77",
              NodeType.WAYPOINT: "#CC6677", NodeType.DOORWAY: "#333333"}


def _pretty(label):
    return label.replace("_", " ")


def pick_plan(maps, min_rooms=6, max_rooms=10):
    """A legible plan: connected, a handful of rooms, some doorways."""
    best = None
    for m in maps:
        rooms = [n for n in m.nodes() if n.type is NodeType.ROOM]
        doors = [n for n in m.nodes() if n.type is NodeType.DOORWAY]
        if min_rooms <= len(rooms) <= max_rooms and doors:
            if best is None or len(doors) > best[1]:
                best = (m, len(doors))
    return best[0] if best else maps[0]


def main():
    maps = load_dataset(120, seed=3)
    m = pick_plan(maps)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ---- left: room bounding boxes ----
    for n in m.nodes():
        if n.type is NodeType.DOORWAY:
            continue
        x0, y0, x1, y1 = n.metadata["bbox"]
        axL.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, facecolor=TYPE_COLOR.get(n.type, "#CCC"),
            edgecolor="black", linewidth=0.6, alpha=0.75))
        axL.text((x0 + x1) / 2, (y0 + y1) / 2, _pretty(n.label), ha="center",
                 va="center", fontsize=6)
    axL.set_title("(a) HouseExpo room annotation", fontsize=8)
    axL.set_aspect("equal"); axL.autoscale_view()
    axL.set_xlabel("metres"); axL.set_ylabel("metres")

    # ---- right: topological graph ----
    for u, v in m.graph.edges():
        pu, pv = m.get_node(u).position, m.get_node(v).position
        axR.plot([pu[0], pv[0]], [pu[1], pv[1]], "-", color="#666666",
                 lw=0.8, zorder=1)
    for n in m.nodes():
        p = n.position
        if n.type is NodeType.DOORWAY:
            axR.plot(p[0], p[1], "s", color=TYPE_COLOR[n.type], ms=3, zorder=2)
        else:
            axR.plot(p[0], p[1], "o", color=TYPE_COLOR.get(n.type, "#CCC"),
                     ms=9, markeredgecolor="black", markeredgewidth=0.5, zorder=3)
            axR.text(p[0], p[1], _pretty(n.label), ha="center", va="center",
                     fontsize=5, zorder=4)
    axR.set_title("(b) Derived semantic topological map", fontsize=8)
    axR.set_aspect("equal"); axR.autoscale_view()
    axR.set_xlabel("metres")

    handles = [mpatches.Patch(color=TYPE_COLOR[t], label=t.value)
               for t in (NodeType.ROOM, NodeType.CORRIDOR, NodeType.WAYPOINT)]
    handles.append(plt.Line2D([0], [0], marker="s", color="w",
                              markerfacecolor="#333333", markersize=5,
                              label="doorway"))
    axR.legend(handles=handles, frameon=False, fontsize=5.5, loc="best")

    out = ROOT / "paper2" / "figures" / "fig_dataset.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}  (plan {m.plan_id[:10]}, "
          f"{sum(1 for n in m.nodes() if n.type is NodeType.ROOM)} rooms)")


if __name__ == "__main__":
    main()
