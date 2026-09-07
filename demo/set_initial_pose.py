#!/usr/bin/env python3
"""Publish an AMCL /initialpose with a valid sim-time stamp.

`ros2 topic pub /initialpose ...` sends header.stamp = 0, which AMCL rejects
under use_sim_time (it can't look up a transform "at time 0"), so the robot
never localizes. This publishes with the current sim clock instead.

Usage:  python3 set_initial_pose.py [x] [y]   (default -2.0 -0.5 = house spawn)
"""
import sys
import time

import rclpy
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseWithCovarianceStamped


def main():
    x = float(sys.argv[1]) if len(sys.argv) > 1 else -2.0
    y = float(sys.argv[2]) if len(sys.argv) > 2 else -0.5
    rclpy.init()
    node = rclpy.create_node(
        "set_initial_pose",
        parameter_overrides=[Parameter("use_sim_time", value=True)])
    pub = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)

    # wait until the sim clock is available, else the stamp is 0 again
    t0 = time.time()
    while node.get_clock().now().nanoseconds == 0 and time.time() - t0 < 15:
        rclpy.spin_once(node, timeout_sec=0.2)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.w = 1.0
    msg.pose.covariance[0] = 0.25    # x
    msg.pose.covariance[7] = 0.25    # y
    msg.pose.covariance[35] = 0.068  # yaw
    # publish a few times so AMCL (which may still be discovering) gets it
    for _ in range(10):
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.2)
    print("initial pose set to (%.2f, %.2f)" % (x, y))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
