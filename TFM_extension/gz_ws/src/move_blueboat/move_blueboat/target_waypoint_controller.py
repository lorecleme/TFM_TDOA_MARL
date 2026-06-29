#!/usr/bin/env python3
"""
Target Vessel Waypoint Controller — moves the target vessel through a pre-defined
waypoint circuit using direct differential thrust commands.

Design choices:
- Bypasses ArduPilot entirely. The target vessel is a "dumb" boat — no guidance,
  no navigation stack, just open-loop thrust following a circuit.
- Uses a simple state machine: ROTATE_TO_WAYPOINT → MOVE_TO_WAYPOINT → NEXT_WAYPOINT.
- PID heading control with proportional gain for turning towards waypoints.
- Publishes directly to Gazebo thruster topics at 10 Hz. No ROS bridge needed
  for the target's thrust (the bridge is used for odometry only).
- Waypoints are hardcoded for the initial evaluation scenario. The circuit keeps
  the target moving within the ~300-800m range of the world.

Why not ArduPilot? Each SITL instance consumes ~1 GB RAM and requires its own
sim_vehicle.py process. For 3 hunters + 1 target, that's 4 instances — a heavy
load for development. Direct thrust control is lightweight and sufficient for
generating target motion patterns.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class TargetWaypointController(Node):
    """Open-loop waypoint-following controller for the target vessel."""

    def __init__(self):
        super().__init__('target_waypoint_controller')

        # Parameters
        self.declare_parameter('target_vessel', 'target_vessel')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('max_thrust', 8.0)
        self.declare_parameter('waypoint_tolerance', 10.0)  # metres
        self.declare_parameter('heading_kp', 3.0)

        self.vessel = self.get_parameter('target_vessel').value
        self.rate_hz = self.get_parameter('rate_hz').value
        self.max_thrust = self.get_parameter('max_thrust').value
        self.wp_tol = self.get_parameter('waypoint_tolerance').value
        self.heading_kp = self.get_parameter('heading_kp').value

        # Waypoints: [x, y] in Gazebo ENU frame (match world ~0-1000m)
        self.waypoints = np.array([
            [200.0, 0.0],
            [400.0, 100.0],
            [600.0, -50.0],
            [400.0, -150.0],
            [200.0, 50.0],
            [100.0, 0.0],
        ])
        self.current_wp_idx = 0

        # State
        self.position = None
        self.yaw = 0.0

        # Subscriber
        self.create_subscription(
            Odometry, f'/model/{self.vessel}/odometry',
            self._odom_callback, 10
        )

        # Publishers (direct Gazebo topics via bridge)
        self.port_pub = self.create_publisher(
            Float64, f'/model/{self.vessel}/joint/motor_port_joint/cmd_thrust', 10
        )
        self.stbd_pub = self.create_publisher(
            Float64, f'/model/{self.vessel}/joint/motor_stbd_joint/cmd_thrust', 10
        )

        # Fixed-rate control loop
        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f'Target controller ready: {len(self.waypoints)} waypoints, '
            f'{self.rate_hz}Hz'
        )

    def _odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.position = np.array([pos.x, pos.y])
        # Extract yaw from quaternion
        siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        self.yaw = math.atan2(siny, cosy)

    def _control_loop(self):
        if self.position is None:
            return

        wp = self.waypoints[self.current_wp_idx]
        delta = wp - self.position
        distance = np.linalg.norm(delta)

        if distance < self.wp_tol:
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
            self.get_logger().info(
                f'Waypoint {self.current_wp_idx - 1} reached. '
                f'Next: {self.waypoints[self.current_wp_idx]}'
            )
            return

        # Heading to waypoint
        target_heading = math.atan2(delta[1], delta[0])
        heading_error = target_heading - self.yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # Differential thrust: forward + turn correction
        forward = self.max_thrust
        turn = self.heading_kp * heading_error
        port_cmd = np.clip(forward - turn, -self.max_thrust, self.max_thrust)
        stbd_cmd = np.clip(forward + turn, -self.max_thrust, self.max_thrust)

        self.port_pub.publish(Float64(data=float(port_cmd)))
        self.stbd_pub.publish(Float64(data=float(stbd_cmd)))


def main(args=None):
    rclpy.init(args=args)
    node = TargetWaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
