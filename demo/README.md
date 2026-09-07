# Language-Grounded Navigation — Live Demo

A Docker-packaged simulation that shows the paper's result end to end: a
TurtleBot3 explores a house in Gazebo, `slam_toolbox` builds the map live, and a
web console lets you type destinations in plain language. The research agent
(`src/agent.py` `resolve()` + `src/verify.py`) turns each request into a **map
node** or an **abstention with a reason**, and Nav2 drives the robot to verified
goals. The language model never emits a pose or a path — it returns a node id,
the verifier checks it, Nav2 plans the motion.

```
  Web query  ─▶  agent (LLM propose ─▶ Python verify)  ─┬─ verified ─▶ Nav2 goal ─▶ robot drives
  "go to the                                            └─ abstain  ─▶ robot stays, shows diagnostic
   kitchen"
```

> **Honest status.** This is a standard-pattern integration built to run, but it
> was *not* executed in the environment where it was written (no display/GPU/ROS
> there). Expect a short first-run shakedown on your machine: X11 permissions,
> GPU vs. software rendering, and a one-time **coordinate calibration** so goals
> land in the right rooms. Steps below.

---

## 1. Prerequisites
- **Native** Docker + Docker Compose on Linux with an X server (not the Docker
  Desktop VM — check `docker context show` prints `default`).
- ~6 GB disk for the image. Integrated Intel/AMD graphics are fine: the compose
  file passes `/dev/dri` for hardware OpenGL, so **no dedicated GPU is needed**.
- An LLM: put a **Groq API key** in `../.env` as `GROQ_API_KEY=gsk_...` (fast,
  recommended for a live demo), or use local **Ollama** with `llama3.2:3b`.

## 2. Run it (two commands)
```bash
cd ~/amr_research/demo
xhost +local:docker                         # once per login: let the container draw
docker compose up -d --build                # container comes up idle, GUI-ready
docker exec -it semantic_nav_demo bash /workspace/amr/demo/run_demo.sh
```
`run_demo.sh` starts Gazebo, SLAM, Nav2, and the agent one stage at a time (logs
in `/tmp/demo_logs/` inside the container) and then runs the agent + web UI in the
foreground. Web console: **http://localhost:8088**. `Ctrl-C` stops the stack; the
container stays up so you can re-run the script.

**Performance knobs** (prefix the `run_demo.sh` command). On a laptop with
integrated graphics the smoothest setup is Gazebo headless + RViz:
```bash
docker exec -it semantic_nav_demo bash -lc \
  'HEADLESS=1 RVIZ=1 bash /workspace/amr/demo/run_demo.sh'
```
- `HEADLESS=1` — run Gazebo physics with no 3D window (much lighter)
- `RVIZ=1` — open RViz (map, semantic-graph markers, planned path)
- `NAV=0` — skip Nav2; resolve/verify/abstain still work, the robot just won't drive
- If 3D windows are black/garbled, GL is the problem: re-run with
  `LIBGL_ALWAYS_SOFTWARE=1 docker compose up -d` then re-run the script.

To use local Ollama instead of Groq, bring the container up with
`LLM_PROVIDER=ollama LLM_MODEL=llama3.2:3b docker compose up -d`.

## 3. If a component misbehaves, run pieces by hand
The container stays alive, so open a shell and source the env first:
```bash
docker exec -it semantic_nav_demo bash
source /workspace/amr/demo/setup_env.sh     # ROS + workspace + repo on the path
```
Then run any single piece and read its output live:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py            # simulator
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true  # SLAM
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true \
    params_file:=/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml
rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz
ros2 run semantic_nav_demo agent_bridge                             # agent + web UI
```

## 4. Calibrate room coordinates (do this once)
The graph in
`ros2_ws/src/semantic_nav_demo/semantic_nav_demo/house_graph.py` has approximate
`(x, y)` per room. To make Nav2 land in the right places:
1. With the stack running, in RViz use **Publish Point** and click the centre of
   a room, or run `ros2 topic echo /clicked_point`.
2. Read the map-frame `x, y` and edit `ROOMS` in `house_graph.py`.
3. `symlink-install` means you only need to restart the `agent_bridge` node.

## 5. Demo script (what to show / record)
Drive the robot around for ~30 s first so SLAM fills in the map (teleop:
`ros2 run turtlebot3_teleop teleop_keyboard`), then use the web console. Suggested
sequence, which is also the paper's story in miniature:

| Type this | What the audience sees |
|---|---|
| `take me to the kitchen` | resolves the Kitchen node, robot drives there |
| `go to the bedroom` | resolves and drives |
| `somewhere I can cook dinner` | **functional** query — the LLM maps it to the kitchen with no keyword match |
| `where would I wash my hands` | functional — maps to the bathroom |
| `go to the operating theatre` | **abstains** — no such room; diagnostic shown, robot does not move |
| `take me to the second kitchen` | **abstains** — only one kitchen exists (discriminator gate) |

The left panel shows the semantic graph and the live robot dot; the right panel
shows each verdict (NAVIGATING / ABSTAINED), the target, and the verifier's
diagnostic. RViz shows the map, the graph markers, and the planned path.

## 6. How it maps to the paper
- **Semantic map** = `house_graph.py`, built with the same `src.graph.SemanticMap`
  used in the paper (rooms/corridors/doorways, room→doorway→room edges).
- **Agent** = `src.agent.resolve()` with `verify_enabled=True`; the LLM proposes
  a node, `src.verify` checks it (grounding, discriminator gate, reachability),
  and abstains after retries.
- **Separation of concerns** = the model returns symbols only; Nav2 owns all
  metric planning. Abstention is a first-class outcome, surfaced with a reason.

## 7. Troubleshooting
- **Blank Gazebo / segfault:** GPU issue — set `LIBGL_ALWAYS_SOFTWARE=1`, or enable
  the NVIDIA block in `docker-compose.yml` with `nvidia-container-toolkit`.
- **RViz can't open / "cannot connect to display":** rerun `xhost +local:docker`
  and confirm `echo $DISPLAY` is set on the host.
- **Goals go to the wrong spot:** calibrate coordinates (Section 4).
- **Every query abstains:** the LLM isn't reachable — check `GROQ_API_KEY`, or for
  Ollama that `llama3.2:3b` is pulled and the base URL is right.
- **Nav2 won't move:** it needs a little map first; teleop the robot briefly so
  SLAM has something to plan on, and set an initial pose if using localization.
```
