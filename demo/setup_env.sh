# Source this in any `docker exec` shell so ROS, the demo workspace, and the
# research repo are all on the path:  source /workspace/amr/demo/setup_env.sh
set +u                     # ROS setup scripts reference unset vars; tolerate them
source /opt/ros/humble/setup.bash
source /workspace/amr/demo/ros2_ws/install/setup.bash 2>/dev/null
# Gazebo's own setup adds the SYSTEM model path (ground_plane, sun) and resource
# paths. Without this, and with the online DB disabled, the world has no floor and
# the robot's wheels spin with no traction.
source /usr/share/gazebo/setup.sh 2>/dev/null
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH}:/opt/ros/humble/share/turtlebot3_gazebo/models"
# Never contact the defunct online model DB (it hangs and kills gzserver). Set
# AFTER sourcing gazebo/setup.sh, which would otherwise restore the dead default.
export GAZEBO_MODEL_DATABASE_URI=""
export PYTHONPATH="/workspace/amr:${PYTHONPATH}"
