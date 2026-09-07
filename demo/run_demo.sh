#!/usr/bin/env bash
# Start / save / stop the demo stack from INSIDE the container.
#
#   MAP once, reuse forever:
#     1) build a map:   bash run_demo.sh            (SLAM; teleop the house)
#     2) save it:       bash run_demo.sh save       (-> demo/maps/house.{pgm,yaml})
#     3) reuse it:      MODE=localize bash run_demo.sh   (no teleop; just query)
#     stop:             bash run_demo.sh stop
#
# Components run in the BACKGROUND and survive this shell closing (one Ctrl-C
# won't tear the stack down). Logs in /tmp/demo_logs/. Web UI: http://localhost:8088
#
# Toggles (prefix the command):
#   HEADLESS=1   no Gazebo 3D window (lighter; watch RViz instead)
#   RVIZ=0       don't open RViz
#   NAV=0        SLAM mode only: skip Nav2 (resolve/verify still work; no driving)
#   MAP=<name>   map basename to save/load (default: house)
#   MODE=localize  load a saved map + AMCL instead of running SLAM
# NB: no `set -u` -- ROS setup.bash references unset vars.
source /workspace/amr/demo/setup_env.sh

LOG=/tmp/demo_logs; mkdir -p "$LOG"
MAPS_DIR=/workspace/amr/demo/maps
MAP_NAME=${MAP:-house}
# gazebo spawn of the TurtleBot3 in turtlebot3_house (= pose in the map frame)
SPAWN_X=-2.0; SPAWN_Y=-0.5

stop_all() {
  echo "[demo] stopping stack ..."
  pkill -9 -f gzserver 2>/dev/null; pkill -9 -f gzclient 2>/dev/null
  pkill -9 -f slam_toolbox 2>/dev/null; pkill -9 -f 'nav2' 2>/dev/null
  pkill -9 -f map_server 2>/dev/null; pkill -9 -f amcl 2>/dev/null
  pkill -9 -f rviz2 2>/dev/null; pkill -9 -f agent_bridge 2>/dev/null
  pkill -9 -f robot_state_publisher 2>/dev/null; pkill -9 -f spawn_entity 2>/dev/null
  sleep 2; echo "[demo] stopped."
}

# ---- subcommands ---------------------------------------------------------
case "${1:-}" in
  stop) stop_all; exit 0 ;;
  save)
    name=${2:-$MAP_NAME}
    mkdir -p "$MAPS_DIR"
    echo "[demo] saving current map -> $MAPS_DIR/$name.{pgm,yaml}"
    echo "       (SLAM must be running with a map you have driven)"
    ros2 run nav2_map_server map_saver_cli -f "$MAPS_DIR/$name" \
      --ros-args -p save_map_timeout:=10.0 && \
      { echo "[demo] saved:"; ls -la "$MAPS_DIR/$name."*; \
        echo "[demo] reuse with:  MODE=localize bash run_demo.sh"; } || \
      echo "[demo] SAVE FAILED -- is SLAM running and /map published?"
    exit 0 ;;
esac

HEADLESS=${HEADLESS:-0}
RVIZ=${RVIZ:-1}
NAV=${NAV:-1}
MODE=${MODE:-slam}

stop_all

bg() {  # bg "<name>" "<logfile>" "<command...>"
  echo "[demo] starting $1 ...  log: $LOG/$2.log"
  setsid bash -c "source /workspace/amr/demo/setup_env.sh; exec $3" \
    >"$LOG/$2.log" 2>&1 < /dev/null &
}

wait_topic() {  # wait_topic /topic <timeout_s>
  local t=0
  until ros2 topic list 2>/dev/null | grep -q "^$1$"; do
    sleep 1; t=$((t+1)); [ "$t" -ge "${2:-60}" ] && return 1
  done
}

echo "=================================================================="
echo " (1/4) Gazebo house world + TurtleBot3   [mode: $MODE]"
bg "Gazebo" gazebo "ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py"
echo "       waiting for the robot laser (/scan) ..."
if wait_topic /scan 90; then echo "       robot is up."; else
  echo "       !! /scan never came up -- see $LOG/gazebo.log"; fi
if [ "$HEADLESS" = "1" ]; then
  echo "       HEADLESS=1: closing the Gazebo 3D window (physics keeps running)"
  pkill -f gzclient 2>/dev/null
fi

if [ "$MODE" = "localize" ]; then
  # ---- reuse a saved map: map_server + AMCL + Nav2 (no SLAM, no teleop) ----
  MAPYAML="$MAPS_DIR/$MAP_NAME.yaml"
  if [ ! -f "$MAPYAML" ]; then
    echo " !! no saved map at $MAPYAML"
    echo "    build one first:  bash run_demo.sh   (teleop)  then  bash run_demo.sh save"
    exit 1
  fi
  echo " (2/4) Localization on saved map '$MAP_NAME' + Nav2"
  bg "Nav2+AMCL" nav2 "ros2 launch nav2_bringup bringup_launch.py use_sim_time:=true autostart:=true map:=$MAPYAML"
  echo "       waiting for AMCL / map server ..."
  wait_topic /amcl_pose 30 >/dev/null 2>&1; sleep 6
  echo "       setting initial pose to the robot's spawn ($SPAWN_X, $SPAWN_Y) ..."
  # rclpy publisher with a real sim-time stamp (CLI pub sends stamp=0 -> rejected)
  bg "initpose" initpose "python3 /workspace/amr/demo/set_initial_pose.py $SPAWN_X $SPAWN_Y"
  sleep 5
else
  # ---- SLAM: build the map live as you teleop --------------------------
  echo " (2/4) SLAM (slam_toolbox) -- builds the map as you drive"
  bg "SLAM" slam "ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true"
  wait_topic /map 40 || echo "       (/map appears once the robot starts moving)"
  if [ "$NAV" = "1" ]; then
    echo " (3/4) Nav2 -- drives the robot to verified goals"
    bg "Nav2" nav2 "ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true params_file:=/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml"
    sleep 8
  else
    echo " (3/4) Nav2 skipped (NAV=0)"
  fi
fi

if [ "$RVIZ" = "1" ]; then
  echo " RViz -- map, robot, semantic-graph markers, planned path"
  bg "RViz" rviz "rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz --ros-args -p use_sim_time:=true"
  sleep 3
fi

echo " (4/4) semantic agent + web UI"
bg "agent" agent "ros2 run semantic_nav_demo agent_bridge"
sleep 4

echo "=================================================================="
echo " Everything is running in the BACKGROUND. This shell is now free."
echo
if [ "$MODE" = "localize" ]; then
  echo "  Map loaded from $MAPS_DIR/$MAP_NAME.yaml -- NO teleop needed."
  echo "  * ASK QUESTIONS: open  http://localhost:8088"
  echo "  * If goals go to the wrong spot, set the pose in RViz (2D Pose Estimate)."
else
  echo "  * DRIVE to build the map (open a teleop terminal):"
  echo "      docker exec -it semantic_nav_demo bash -lc \\"
  echo "        'source /workspace/amr/demo/setup_env.sh; ros2 run turtlebot3_teleop teleop_keyboard'"
  echo "    Drive through every room, then SAVE the map so you never remap:"
  echo "      docker exec -it semantic_nav_demo bash /workspace/amr/demo/run_demo.sh save"
  echo "  * ASK QUESTIONS: open  http://localhost:8088"
fi
echo "  * STOP: docker exec -it semantic_nav_demo bash /workspace/amr/demo/run_demo.sh stop"
echo "=================================================================="
