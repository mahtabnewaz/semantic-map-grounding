"""Semantic navigation bridge: research agent <-> ROS 2 / Nav2 / web UI.

One process that:
  * loads the hand-authored house SemanticMap (house_graph.py);
  * runs the paper's resolve()+verify() over it, using a Groq or Ollama LLM;
  * serves a small web UI and JSON API (FastAPI) for typing queries;
  * publishes the semantic graph as RViz markers; and
  * on a verified answer, sends the goal to Nav2 so the robot drives there.
    An abstention drives nowhere and returns the verifier's diagnostic.

The LLM never produces a pose or a path: it returns a node id, the verifier
checks it against the map, and Nav2 does all metric planning. This is the
paper's separation, made live.
"""

import os
import sys
import threading
import types

# The research code lives in the mounted repo; make it importable.
sys.path.insert(0, os.environ.get("AMR_REPO", "/workspace/amr"))

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.agent import AgentConfig, resolve            # the paper's agent
from src.graph import NodeType
from src.llm import LLMClient
from semantic_nav_demo.house_graph import (
    EXAMPLE_QUERIES, build_house_map, world_xy)

_STOP = {"go", "to", "the", "a", "an", "take", "me", "please", "i", "need",
         "want", "get", "can", "you", "would", "where", "is", "my", "at", "in",
         "of", "for", "some", "somewhere", "navigate", "head", "drive"}


def _terms(text):
    """Content words from a free-typed query (drop command/stopwords)."""
    out = [w for w in "".join(c.lower() if c.isalnum() else " "
                              for c in text).split() if w not in _STOP]
    return out or [text]


class SemanticAgentBridge(Node):
    def __init__(self):
        super().__init__("semantic_agent_bridge")
        self.map = build_house_map()
        self.client = self._make_client()
        self.n_max = int(os.environ.get("AGENT_NMAX", "2"))
        self.threshold = float(os.environ.get("AGENT_THRESHOLD", "0.60"))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(MarkerArray, "semantic_graph", 1)
        self.create_timer(1.0, self._publish_markers)

        # Nav2 goal sender: an action client on THIS node. Sending goals async
        # via the node's own executor is thread-safe; BasicNavigator's internal
        # spin would collide with the main rclpy.spin ("generator already executing").
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._lock = threading.Lock()
        self.state = {"navigating": False, "goal": None, "last": None}

        self.get_logger().info(
            f"bridge up: {len(list(self.map.nodes()))} nodes, "
            f"provider={self.client.provider}, model={self.client.model}")

    # ---- LLM ----------------------------------------------------------
    def _make_client(self):
        provider = os.environ.get("LLM_PROVIDER", "").lower()
        model = os.environ.get("LLM_MODEL", "")
        if provider == "groq" or (not provider and os.environ.get("GROQ_API_KEY")):
            return LLMClient("groq", model or "openai/gpt-oss-20b",
                             max_tokens=int(os.environ.get("LLM_MAXTOK", "512")),
                             reasoning_effort=os.environ.get("LLM_EFFORT") or None)
        return LLMClient("ollama", model or "llama3.2:3b")

    # ---- pose / nearest node -----------------------------------------
    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
            return (t.transform.translation.x, t.transform.translation.y)
        except Exception:
            return None

    def _nearest_room(self):
        xy = self.robot_xy()
        rooms = [n for n in self.map.nodes() if n.type is NodeType.ROOM]
        if not rooms:
            return None
        if xy is None:
            return rooms[0].id
        return min(rooms, key=lambda n: (n.position[0] - xy[0]) ** 2
                   + (n.position[1] - xy[1]) ** 2).id

    # ---- resolve + drive ---------------------------------------------
    def handle_query(self, text, stratum=None, terms=None):
        q = types.SimpleNamespace(
            text=text, stratum=stratum or "direct",
            query_terms=terms or _terms(text), anchor=None, relation=None)
        cfg = AgentConfig(n_max=self.n_max, verify_enabled=True,
                          current_node=self._nearest_room(),
                          threshold=self.threshold)
        res = resolve(q, self.map, self.client, cfg)
        rec = res.attempt_records[-1] if res.attempt_records else None
        out = {
            "query": text, "stratum": q.stratum,
            "node_id": res.node_id, "abstained": res.abstained,
            "attempts": res.attempts, "status": res.status,
            "diagnostic": (rec.diagnostic if rec else None),
            "latency_s": round(res.latency_s, 2),
        }
        if res.node_id and not res.abstained:
            n = self.map.get_node(res.node_id)
            out["label"] = n.label
            xy = world_xy(self.map, res.node_id)
            out["goal"] = {"x": xy[0], "y": xy[1]}
            self._send_goal(*xy)
            out["action"] = "navigating"
        else:
            out["label"] = None
            out["action"] = "abstained"
        with self._lock:
            self.state["last"] = out
        return out

    def _send_goal(self, x, y):
        """Send a NavigateToPose goal without blocking or crashing.

        wait_for_server only polls a flag the main executor updates (no spin
        here), and send_goal_async merely enqueues the request, so this is safe
        to call from the web thread. If Nav2 is not up yet, we log and return
        instead of raising, so the query still gets a response.
        """
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.w = 1.0
        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Nav2 action server unavailable; goal not sent")
            with self._lock:
                self.state.update(navigating=False, goal={"x": x, "y": y})
            return
        self._nav_client.send_goal_async(goal)
        with self._lock:
            self.state.update(navigating=True, goal={"x": x, "y": y})

    def graph_json(self):
        nodes = [{"id": n.id, "label": n.label, "type": n.type.value,
                  "x": n.position[0], "y": n.position[1],
                  "attributes": n.attributes} for n in self.map.nodes()]
        edges = [{"u": u, "v": v} for u, v in self.map.graph.edges()]
        return {"nodes": nodes, "edges": edges, "robot": self.robot_xy(),
                "state": self.state}

    # ---- RViz markers -------------------------------------------------
    def _publish_markers(self):
        arr = MarkerArray()
        mid = 0
        col = {"room": (0.2, 0.6, 1.0), "corridor": (0.85, 0.8, 0.3),
               "waypoint": (0.8, 0.3, 0.4), "doorway": (0.3, 0.3, 0.3)}
        for n in self.map.nodes():
            c = col.get(n.type.value, (0.5, 0.5, 0.5))
            sph = self._mk(mid, n.position, "sphere",
                           0.35 if n.type is not NodeType.DOORWAY else 0.15, c)
            arr.markers.append(sph); mid += 1
            if n.type is not NodeType.DOORWAY:
                arr.markers.append(self._text(mid, n.position, n.label)); mid += 1
        line = Marker()
        line.header.frame_id = "map"; line.header.stamp = self.get_clock().now().to_msg()
        line.ns = "edges"; line.id = mid; mid += 1
        line.type = Marker.LINE_LIST; line.action = Marker.ADD
        line.scale.x = 0.03
        line.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.8)
        from geometry_msgs.msg import Point
        for u, v in self.map.graph.edges():
            pu, pv = self.map.get_node(u).position, self.map.get_node(v).position
            line.points.append(Point(x=pu[0], y=pu[1], z=0.05))
            line.points.append(Point(x=pv[0], y=pv[1], z=0.05))
        arr.markers.append(line)
        self.marker_pub.publish(arr)

    def _mk(self, mid, pos, _shape, size, rgb):
        m = Marker()
        m.header.frame_id = "map"; m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "nodes"; m.id = mid; m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = pos[0]; m.pose.position.y = pos[1]; m.pose.position.z = 0.2
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = size
        m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.9)
        return m

    def _text(self, mid, pos, label):
        m = Marker()
        m.header.frame_id = "map"; m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "labels"; m.id = mid; m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD; m.text = label.replace("_", " ")
        m.pose.position.x = pos[0]; m.pose.position.y = pos[1]; m.pose.position.z = 0.6
        m.pose.orientation.w = 1.0; m.scale.z = 0.35
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        return m


# ------------------------------------------------------------------ web
def build_app(bridge: SemanticAgentBridge) -> FastAPI:
    app = FastAPI(title="Semantic Navigation Demo")
    here = os.path.dirname(__file__)
    index = os.path.join(os.environ.get("WEB_DIR", "/workspace/web"), "index.html")

    class Q(BaseModel):
        query: str
        stratum: str | None = None
        terms: list[str] | None = None

    @app.get("/", response_class=HTMLResponse)
    def root():
        try:
            with open(index) as f:
                return f.read()
        except OSError:
            return "<h3>web/index.html not found; set WEB_DIR</h3>"

    @app.get("/graph")
    def graph():
        return JSONResponse(bridge.graph_json())

    @app.get("/examples")
    def examples():
        return [{"text": t, "stratum": s, "terms": k}
                for t, s, k in EXAMPLE_QUERIES]

    @app.post("/resolve")
    def do_resolve(q: Q):
        return JSONResponse(bridge.handle_query(q.query, q.stratum, q.terms))

    return app


def main():
    rclpy.init()
    bridge = SemanticAgentBridge()
    app = build_app(bridge)
    port = int(os.environ.get("WEB_PORT", "8088"))
    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port,
                                   log_level="warning"),
        daemon=True).start()
    bridge.get_logger().info(f"web UI on http://localhost:{port}")
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
