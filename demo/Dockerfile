# Language-grounded navigation demo: ROS 2 Humble + Gazebo + TurtleBot3 + Nav2
# + slam_toolbox, plus the Python deps the research agent needs.
FROM osrf/ros:humble-desktop-full

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-turtlebot3 \
      ros-humble-turtlebot3-gazebo \
      ros-humble-turtlebot3-msgs \
      ros-humble-turtlebot3-teleop \
      ros-humble-nav2-bringup \
      ros-humble-nav2-simple-commander \
      ros-humble-slam-toolbox \
      mesa-utils libgl1-mesa-dri libgl1-mesa-glx \
      python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Deps for the research code (src/) and the web bridge. openai/httpx/httpcore are
# PINNED to real, known-good versions: an unpinned `openai` here resolved to a
# broken build pulling counterfeit "httpx2"/"httpcore2" that fail every request.
RUN pip3 install --no-cache-dir \
      networkx shapely numpy scipy \
      fastapi "uvicorn[standard]" pydantic python-dotenv \
      "openai==1.54.0" "httpx==0.27.2" "httpcore==1.0.9"

# This host's container network has no working IPv6 route, but DNS returns AAAA
# records first, so the HTTP client tries a dead IPv6 address and times out.
# Prefer IPv4 so calls to the Groq API connect.
RUN echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf

# GAZEBO_MODEL_DATABASE_URI="" => never contact the defunct models.gazebosim.org
# (an unreachable fetch that otherwise hangs and kills gzserver on world load).
ENV TURTLEBOT3_MODEL=waffle \
    AMR_REPO=/workspace/amr \
    WEB_DIR=/workspace/amr/demo/web \
    WEB_PORT=8088 \
    GAZEBO_MODEL_DATABASE_URI=""

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
