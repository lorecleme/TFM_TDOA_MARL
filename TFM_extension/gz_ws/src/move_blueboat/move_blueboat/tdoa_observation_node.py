#!/usr/bin/env python3
"""
TDOA Observation Node — reconstructs MPE entity-observation format from Gazebo odometry.

Subscribes to /model/<vessel>/odometry for all hunter vessels and the target vessel,
computes noisy inter-vessel ranges (simulating TDOA hydrophone measurements), and
assembles the 4x6 entity observation matrix per agent that exactly matches the
entity_observation() function from the MPE training environment.

Design choices:
- Uses nav_msgs/Odometry pose.position (x,y) in ENU frame as ground truth position,
  then adds configurable Gaussian noise to ranges (not positions) — matching the
  MPE philosophy of "noisy ranges, clean relative positions".
- Coordinate scaling: Gazebo world is ~1000m (waves.sdf domain). The MPE world is
  1000x1000 abstract units. 1:1 mapping — no scaling needed.
- Publishes entity observations at a fixed 10Hz rate (matching MPE dt=0.1).
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from scipy.optimize import least_squares


class TDOAObservationNode(Node):
    """Reconstruct MPE entity_observation from Gazebo odometry."""

    def __init__(self):
        super().__init__('tdoa_observation_node')

        self.declare_parameter('hunter_vessels', ['blueboat_1', 'blueboat_2', 'blueboat_3'])
        self.declare_parameter('target_vessel', 'target_vessel')
        self.declare_parameter('noise_std', 0.05)
        self.declare_parameter('world_span', 1000.0)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('landmark_source', 'ground_truth')
        self.declare_parameter('baseline_scale', 1.0)

        self.hunter_vessels = list(self.get_parameter('hunter_vessels').value)
        self.target_vessel = self.get_parameter('target_vessel').value
        self.noise_std = float(self.get_parameter('noise_std').value)
        self.world_span = float(self.get_parameter('world_span').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.landmark_source = self.get_parameter('landmark_source').value
        self.baseline_scale = float(self.get_parameter('baseline_scale').value)

        self.world_diag = math.sqrt(2.0) * self.world_span
        self.n_agents = len(self.hunter_vessels)
        self.n_landmarks = 1
        self.n_entities = self.n_agents + self.n_landmarks
        self.feat_dim = 5 + self.n_landmarks

        self.tdoa_pairs = [(i, j) for i in range(self.n_agents)
                           for j in range(self.n_agents) if i < j]

        self.positions = {}
        self.velocities = {}

        all_vessels = list(self.hunter_vessels) + [self.target_vessel]
        for vessel in all_vessels:
            topic = f'/model/{vessel}/odometry'
            self.create_subscription(
                Odometry, topic,
                lambda msg, v=vessel: self._odom_callback(v, msg), 10)

        self.obs_pub = self.create_publisher(Float64MultiArray, '/tdoa/entity_obs', 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self._publish_observations)

        self._first_pub = True

        self.get_logger().info(
            f'TDOA Observation Node ready: {self.n_agents} hunters, 1 target, '
            f'noise_std={self.noise_std}, rate={self.rate_hz}Hz, '
            f'landmark_source={self.landmark_source}, '
            f'baseline_scale={self.baseline_scale}')

    def _odom_callback(self, vessel: str, msg: Odometry):
        """Store latest position and velocity for a vessel."""
        pos = msg.pose.pose.position
        twist = msg.twist.twist.linear
        if vessel not in self.positions:
            self.get_logger().info(
                f'[DBG] got odom for {vessel} ({len(self.positions) + 1}/{self.n_entities})')
        self.positions[vessel] = np.array([pos.x, pos.y])
        self.velocities[vessel] = np.array([twist.x, twist.y])

    def _noisy_ranges(self, agent_positions, target_pos):
        """Compute noisy range from each agent to target.

        Matches _noisy_ranges() + _noisy_ranges_per_target() in
        tdoa_tracking.py lines 414-432.
        """
        true_ranges = np.linalg.norm(agent_positions - target_pos, axis=1)
        noise = self._rng.normal(
            0.0, self.noise_std * self.world_span, size=true_ranges.shape)
        return np.maximum(true_ranges + noise, 0.0)

    def _compute_tdoa_estimate(self, noisy_ranges, agent_positions):
        """Estimate target position from noisy TDOA measurements.

        Uses the same scipy least_squares solver as the training scenario
        (tdoa_tracking.py::_tdoa_estimate_position).
        """
        if self.n_agents < 3:
            return agent_positions.mean(axis=0)

        tdoa = np.array([noisy_ranges[j] - noisy_ranges[i]
                         for i, j in self.tdoa_pairs])
        initial_guess = agent_positions.mean(axis=0)

        def _residuals(pos, receivers, tdoa_meas, pairs):
            dists = np.linalg.norm(receivers - pos, axis=1)
            return np.array([dists[j] - dists[i] - tdoa_meas[k]
                             for k, (i, j) in enumerate(pairs)])

        try:
            result = least_squares(
                _residuals, initial_guess,
                args=(agent_positions, tdoa, self.tdoa_pairs),
                method='trf', ftol=1e-6)
            est = result.x
            # Sanity clamp: if estimate is wildly far from agent centroid
            # (ill-conditioned multilateration from clustered boats),
            # fall back to centroid. Training agents are 300m+ apart so
            # this never triggers there; Gazebo boats start ~25m apart.
            if np.linalg.norm(est - initial_guess) > 3.0 * self.world_span:
                return initial_guess
            return est
        except Exception:
            return initial_guess

    def _build_entity_observation(self, agent_idx, agent_positions,
                                  target_pos, noisy_ranges, landmark_noisy_range):
        """Build entity observation for a single agent.

        Matches entity_observation() in src_june20 tdoa_tracking.py lines 469-533.
        Each agent gets independent noisy_ranges (drawn fresh per agent in training).
        The landmark-row range uses a SEPARATE noise draw (training line 509).
        Returns array of shape (n_entities, feat_dim) = (4, 6).
        """
        obs = np.zeros((self.n_entities, self.feat_dim))
        pos_a = agent_positions[agent_idx]

        for other_idx in range(self.n_agents):
            entity_pos = agent_positions[other_idx]
            scale = self.baseline_scale if other_idx != agent_idx else 1.0
            obs[other_idx, 0] = scale * (entity_pos[0] - pos_a[0]) / self.world_span
            obs[other_idx, 1] = scale * (entity_pos[1] - pos_a[1]) / self.world_span
            obs[other_idx, 2] = 1.0
            obs[other_idx, 3] = 1.0 if other_idx == agent_idx else 0.0
            obs[other_idx, 4] = noisy_ranges[other_idx] / self.world_diag

        obs[self.n_agents, 0] = (target_pos[0] - pos_a[0]) / self.world_span
        obs[self.n_agents, 1] = (target_pos[1] - pos_a[1]) / self.world_span
        obs[self.n_agents, 2] = 0.0
        obs[self.n_agents, 3] = 0.0
        obs[self.n_agents, 4] = landmark_noisy_range / self.world_diag

        return obs

    def _publish_observations(self):
        """Timer callback: assemble and publish entity observations.

        Noise structure matches src_june20 tdoa_tracking.py exactly:
        - TDOA estimate: separate noise draw (training _tdoa_estimate_position line 220)
        - Per-agent observation: independent noise draw for agent-row ranges
          (training _noisy_ranges_per_target called per entity_observation, line 474)
        - Landmark-row range: independent scalar draw per agent (training line 509)
        """
        required = list(self.hunter_vessels) + [self.target_vessel]
        for v in required:
            if v not in self.positions:
                return

        agent_positions = np.array([self.positions[h] for h in self.hunter_vessels])
        target_pos = np.array(self.positions[self.target_vessel])
        sigma = self.noise_std * self.world_span

        # TDOA estimate: separate noise draw (cached per step in training)
        tdoa_noisy_ranges = self._noisy_ranges(agent_positions, target_pos)
        est_pos = self._compute_tdoa_estimate(tdoa_noisy_ranges, agent_positions)

        # Mode A (ground_truth): use TRUE target position for landmark row
        #   → compatible with old checkpoints (turbo_normal_moving, etc.)
        # Mode B (tdoa_estimate): use TDOA-estimated position for landmark row
        #   → compatible with B1 checkpoints (target_pos_mode=tdoa_estimate)
        # Mode C (hidden): zero landmark rel_x/rel_y (agent infers from ranges)
        #   → compatible with zero-noise hidden checkpoints (target_pos_mode=hidden)
        if self.landmark_source == 'ground_truth':
            landmark_pos = target_pos
        elif self.landmark_source == 'hidden':
            landmark_pos = None  # handled per-agent: rel = 0
        else:
            landmark_pos = est_pos

        all_obs = []
        for agent_idx in range(self.n_agents):
            # Independent noise for this agent's agent-row ranges
            agent_noisy_ranges = self._noisy_ranges(agent_positions, target_pos)

            # Independent noise for landmark-row range (training line 509)
            land_range = float(np.linalg.norm(target_pos - agent_positions[agent_idx]))
            land_noise = self._rng.normal(0.0, sigma)
            landmark_noisy_range = max(land_range + land_noise, 0.0)

            obs = self._build_entity_observation(
                agent_idx, agent_positions,
                agent_positions[agent_idx] if landmark_pos is None else landmark_pos,
                agent_noisy_ranges, landmark_noisy_range)
            all_obs.append(obs.flatten())

        msg = Float64MultiArray()
        msg.data = np.concatenate(all_obs).tolist()
        self.obs_pub.publish(msg)

        if self._first_pub:
            self._first_pub = False
            err = float(np.linalg.norm(est_pos - target_pos))
            self.get_logger().info(
                f'[DBG] first obs published, {len(msg.data)} floats, '
                f'agent0=({agent_positions[0, 0]:.1f},{agent_positions[0, 1]:.1f}) '
                f'target=({target_pos[0]:.1f},{target_pos[1]:.1f}) '
                f'est=({est_pos[0]:.1f},{est_pos[1]:.1f}) err={err:.1f}m')

    @property
    def _rng(self):
        if not hasattr(self, '_rng_state'):
            self._rng_state = np.random.RandomState()
        return self._rng_state


def main(args=None):
    rclpy.init(args=args)
    node = TDOAObservationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
