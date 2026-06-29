#!/usr/bin/env python3
"""Randomize hunter spawns to match MPE training distribution (300m+ from target)."""

import math, subprocess, json, time, numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class RandomizeSpawn(Node):
    def __init__(self):
        super().__init__('randomize_spawn')
        self.declare_parameter('hunter_vessels', ['blueboat_1', 'blueboat_2', 'blueboat_3'])
        self.declare_parameter('target_vessel', 'target_vessel')
        self.declare_parameter('min_sep', 60.0)
        self.declare_parameter('world_span', 256.0)
        self.declare_parameter('seed', 0)

        self.hunters = list(self.get_parameter('hunter_vessels').value)
        self.target_name = self.get_parameter('target_vessel').value
        self.min_sep = float(self.get_parameter('min_sep').value)
        self.world_span = float(self.get_parameter('world_span').value)
        seed = int(self.get_parameter('seed').value)
        if seed == 0:
            seed = int(time.time() * 1000) % (2**31)
        self.rng = np.random.RandomState(seed)

        self.target_pos = None
        # Wait for target odometry
        self.create_subscription(
            Odometry, f'/model/{self.target_name}/odometry',
            self._target_odom_cb, 10)

        self.pending = list(self.hunters)
        self.timer = self.create_timer(1.0, self._teleport_tick)
        self.get_logger().info(f'RandomizeSpawn ready, waiting for target odom...')

    def _target_odom_cb(self, msg):
        if self.target_pos is not None:
            return
        p = msg.pose.pose.position
        self.target_pos = np.array([p.x, p.y])
        self.get_logger().info(f'Got target position: ({p.x:.1f}, {p.y:.1f})')

        # Generate random agent positions (matching MPE reset_world: 300m+ from target)
        margin = 50
        agent_positions = []
        for _ in range(len(self.hunters)):
            for _ in range(200):
                angle = self.rng.uniform(0, 2 * math.pi)
                dist = self.rng.uniform(self.min_sep, self.world_span * 0.45)
                pos = self.target_pos + dist * np.array([math.cos(angle), math.sin(angle)])
                pos = np.clip(pos, -self.world_span/2 + margin, self.world_span/2 - margin)
                if np.linalg.norm(pos - self.target_pos) >= self.min_sep:
                    agent_positions.append(pos)
                    break
            else:
                angle = self.rng.uniform(0, 2 * math.pi)
                pos = self.target_pos + (self.world_span * 0.4) * np.array([math.cos(angle), math.sin(angle)])
                agent_positions.append(pos)

        self.agent_poses = agent_positions
        self.get_logger().info(f'Generated {len(agent_positions)} random positions, min_sep={self.min_sep}m')

    def _teleport_tick(self):
        if self.target_pos is None or not self.pending:
            return

        hunter = self.pending.pop(0)
        idx = self.hunters.index(hunter)
        pos = self.agent_poses[idx]

        cmd = [
            'gz', 'service', '-s', '/world/waves/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', f'name: "{hunter}", position: {{x: {pos[0]:.1f}, y: {pos[1]:.1f}, z: 0}}'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if 'data: true' in result.stdout:
                self.get_logger().info(f'Teleported {hunter} to ({pos[0]:.1f}, {pos[1]:.1f})')
            else:
                self.get_logger().warn(f'Failed to teleport {hunter}: {result.stdout.strip()}')
                self.pending.insert(0, hunter)
        except Exception as e:
            self.get_logger().error(f'Error teleporting {hunter}: {e}')
            self.pending.insert(0, hunter)

        if not self.pending:
            self.get_logger().info('All agents teleported. Shutting down.')
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RandomizeSpawn()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
