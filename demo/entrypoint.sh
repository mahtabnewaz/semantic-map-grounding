#!/usr/bin/env bash
# Source ROS, build the demo workspace against the mounted repo, then run.
set -e
source /opt/ros/humble/setup.bash

WS="${AMR_REPO}/demo/ros2_ws"
if [ -d "${WS}/src" ]; then
  cd "${WS}"
  # symlink-install so edits to house_graph.py etc. take effect without rebuild
  colcon build --symlink-install --packages-select semantic_nav_demo \
    2>/dev/null || colcon build --symlink-install --packages-select semantic_nav_demo
  source "${WS}/install/setup.bash"
fi

# TurtleBot3 Gazebo models
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH}:/opt/ros/humble/share/turtlebot3_gazebo/models"
# so `import src.*` and `import data.*` resolve from the mounted repo
export PYTHONPATH="${AMR_REPO}:${PYTHONPATH}"

exec "$@"
