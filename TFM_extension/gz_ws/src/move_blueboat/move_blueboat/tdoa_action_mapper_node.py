#!/usr/bin/env python3
"""TDOA Action Mapper — discrete TransfQMix actions -> differential thrust.

The MPE policy outputs a discrete action index 0-4. Training decodes it via a
ONE-HOT vector (environment.py:104-109 converts index->one-hot, then _set_action
with discrete_action_input=False at environment.py:218-219):

    u[0] += action[1] - action[2]   # x-axis
    u[1] += action[3] - action[4]   # y-axis

So the CORRECT world-frame mapping (verified against the training env) is:
    0 = NOOP, 1 = +x (RIGHT), 2 = -x (LEFT), 3 = +y (UP), 4 = -y (DOWN)

NOTE: this is the OPPOSITE of the discrete_action_input=True path
(environment.py:208-211) that BRIDGE_ACTION_MAP.md / CLAUDE.md originally
documented. The training env uses discrete_action_input=False, so the one-hot
path above is authoritative. A closed-loop MPE sim with this mapping reproduces
the trained surround (agents close from 300m to ~15m, surround quality ~0.9,
target inside the triangle).

A BlueBoat is body-frame (forward + turn), so each action is mapped to a desired
world-frame heading and a proportional heading controller produces differential
thrust, reusing the proven pattern from target_waypoint_controller.py:

    desired_heading = atan2(dy, dx)
    err             = wrap(desired_heading - yaw)
    port            = forward - kp * err
    stbd            = forward + kp * err

Publishes std_msgs/Float64 to:
    /model/{hunter}/joint/motor_port_joint/cmd_thrust
    /model/{hunter}/joint/motor_stbd_joint/cmd_thrust

No ArduPilot, no MAVLink, no SITL — the Gazebo Thruster plugin is the sole
actuator (valid because no SITL connects to the hunters' ArduPilotPlugin, so
that plugin stays idle; cf. target_vessel which is already driven this way).
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Int32MultiArray

# MPE action -> world-frame direction (dx, dy). Gazebo ENU: +x East, +y North.
# CORRECT mapping per training env one-hot decode (environment.py:218-219).
ACTION_DIR = {
    0: (0.0, 0.0),    # NOOP
    1: (1.0, 0.0),    # RIGHT -> face +x (East)
    2: (-1.0, 0.0),   # LEFT  -> face -x (West)
    3: (0.0, 1.0),    # UP    -> face +y (North)
    4: (0.0, -1.0),   # DOWN  -> face -y (South)
}


def wrap(angle):
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(qx, qy, qz, qw):
    """Yaw (rotation about Z) from a quaternion, radians."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class TDOAActionMapperNode(Node):
    def __init__(self):
        super().__init__('tdoa_action_mapper_node')

        self.declare_parameter('hunter_vessels', ['blueboat_1', 'blueboat_2', 'blueboat_3'])
        self.declare_parameter('max_thrust', 40.0)
        self.declare_parameter('heading_kp', 3.0)
        self.declare_parameter('rate_hz', 10.0)

        self.hunters = list(self.get_parameter('hunter_vessels').value)
        self.max_thrust = float(self.get_parameter('max_thrust').value)
        self.heading_kp = float(self.get_parameter('heading_kp').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.n = len(self.hunters)

        # Latest per-hunter state
        self._actions = [0] * self.n
        self._yaws = [0.0] * self.n
        self._has_action = [False] * self.n

        # Actions subscriber
        self.create_subscription(Int32MultiArray, '/tdoa/actions', self._action_cb, 10)

        # Per-hunter: odometry subscription + thrust publishers
        self._port_pubs = []
        self._stbd_pubs = []
        for i, v in enumerate(self.hunters):
            self.create_subscription(
                Odometry, f'/model/{v}/odometry',
                lambda msg, i=i: self._odom_cb(msg, i), 10)
            self._port_pubs.append(self.create_publisher(
                Float64, f'/model/{v}/joint/motor_port_joint/cmd_thrust', 10))
            self._stbd_pubs.append(self.create_publisher(
                Float64, f'/model/{v}/joint/motor_stbd_joint/cmd_thrust', 10))

        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)

        self.get_logger().info(
            f'TDOA Action Mapper: {self.n} hunters, direct thrust (no SITL), '
            f'max_thrust={self.max_thrust}, heading_kp={self.heading_kp}, '
            f'{self.rate_hz} Hz')

    def _action_cb(self, msg: Int32MultiArray):
        if len(msg.data) != self.n:
            self.get_logger().warn(
                f'Expected {self.n} actions, got {len(msg.data)} - ignoring')
            return
        for i, a in enumerate(msg.data):
            self._actions[i] = int(a)
            self._has_action[i] = True

    def _odom_cb(self, msg: Odometry, i: int):
        o = msg.pose.pose.orientation
        self._yaws[i] = yaw_from_quaternion(o.x, o.y, o.z, o.w)

    def _control_loop(self):
        for i in range(self.n):
            action = self._actions[i] if self._has_action[i] else 0
            dx, dy = ACTION_DIR.get(action, (0.0, 0.0))

            if dx == 0.0 and dy == 0.0:
                # NOOP or unknown action -> zero thrust (drift)
                port = stbd = 0.0
            else:
                desired = math.atan2(dy, dx)
                err = wrap(desired - self._yaws[i])
                forward = self.max_thrust * (0.3 + 0.7 * max(0.0, math.cos(err)))
                turn = self.heading_kp * err
                # Tank steer: preserve forward component by limiting turn
                # so both motors stay in [0, max_thrust]. This guarantees
                # forward motion at any heading error, matching MPE dynamics
                # where force is applied regardless of agent orientation.
                turn = float(np.clip(turn, -forward, forward))
                port = float(np.clip(forward - turn, 0.0, self.max_thrust))
                stbd = float(np.clip(forward + turn, 0.0, self.max_thrust))

            self._port_pubs[i].publish(Float64(data=port))
            self._stbd_pubs[i].publish(Float64(data=stbd))


def main(args=None):
    rclpy.init(args=args)
    node = TDOAActionMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
