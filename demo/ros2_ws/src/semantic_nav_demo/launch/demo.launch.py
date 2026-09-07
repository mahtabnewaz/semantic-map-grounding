"""One-command demo: Gazebo house + SLAM + Nav2 + RViz + the agent bridge.

If the combined launch is flaky on your machine (timing, GPU, or a distro
mismatch), fall back to the reliable multi-terminal flow in the README; it runs
these same pieces one at a time.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim = LaunchConfiguration("use_sim_time", default="true")

    tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    nav2 = get_package_share_directory("nav2_bringup")
    slam = get_package_share_directory("slam_toolbox")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo, "launch", "turtlebot3_house.launch.py")))

    slam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam, "launch", "online_async_launch.py")),
        launch_arguments={"use_sim_time": use_sim}.items())

    # navigation_launch.py rewrites a params file; pass it explicitly so the
    # RewrittenYaml source never resolves to an empty path (the cause of the
    # "[Errno 2] No such file or directory: ''" abort in the combined launch).
    nav2_params = os.path.join(nav2, "params", "nav2_params.yaml")
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2, "launch", "navigation_launch.py")),
        launch_arguments={"use_sim_time": use_sim,
                          "params_file": nav2_params}.items())

    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", os.path.join(nav2, "rviz", "nav2_default_view.rviz")],
        parameters=[{"use_sim_time": True}], output="screen")

    bridge = Node(
        package="semantic_nav_demo", executable="agent_bridge",
        name="semantic_agent_bridge", output="screen",
        parameters=[{"use_sim_time": True}])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        gazebo,
        # give Gazebo a head start, then bring up SLAM, Nav2, RViz, bridge
        TimerAction(period=6.0, actions=[slam_node]),
        TimerAction(period=9.0, actions=[nav, rviz]),
        TimerAction(period=12.0, actions=[bridge]),
    ])
