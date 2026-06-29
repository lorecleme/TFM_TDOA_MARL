#!/usr/bin/env python3
"""
ROS 2 node: computes surround-quality metrics from vessel odometry and prints
them to stdout.  Also writes a final summary to /tmp/eval_result.json on exit.

Subscribes to the bridged odometry topics of the 3 hunters + 1 target vessel.

Metrics (computed every print_interval seconds):
  surround_quality  – angular gap metric [0, 1]  (1 = perfect 120° triangle)
  mean_dist         – mean agent-to-target distance  (m)
  min_dist          – closest agent-to-target distance  (m)
  max_gap_deg       – largest angular gap between adjacent hunters  (deg)
  in_triangle       – 1 if target is inside the hunters' triangle, else 0

If episode_length > 0 the node auto-shuts down after that many seconds.
"""

import math
import json
import sys
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry


class SurroundMetricsNode(Node):
    def __init__(self):
        super().__init__('surround_metrics_node')

        self.declare_parameter('hunter_vessels', ['blueboat_1', 'blueboat_2', 'blueboat_3'])
        self.declare_parameter('target_vessel', 'target_vessel')
        self.declare_parameter('print_interval', 5.0)
        self.declare_parameter('episode_length', 0)

        self._hunter_vessels = self.get_parameter('hunter_vessels').value
        self._target_vessel = self.get_parameter('target_vessel').value
        self._print_interval = self.get_parameter('print_interval').value
        self._episode_length = self.get_parameter('episode_length').value

        self._positions = {}
        self._metrics_history = []

        all_vessels = self._hunter_vessels + [self._target_vessel]
        for vessel in all_vessels:
            self.create_subscription(
                Odometry,
                f'/model/{vessel}/odometry',
                lambda msg, v=vessel: self._odom_callback(v, msg),
                10,
            )

        self._print_timer = self.create_timer(self._print_interval, self._print_metrics)
        self._start_time = self.get_clock().now().nanoseconds / 1e9

        if self._episode_length > 0:
            self._episode_timer = self.create_timer(self._episode_length, self._end_episode)

        self.get_logger().info(
            f'Surround Metrics: {len(self._hunter_vessels)} hunters, '
            f'target={self._target_vessel}, '
            f'print_interval={self._print_interval}s, '
            f'episode_length={self._episode_length}s (0=never auto-kill)'
        )

    # ------------------------------------------------------------------
    def _odom_callback(self, vessel, msg):
        self._positions[vessel] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

    # ------------------------------------------------------------------
    def _compute_metrics(self):
        if self._target_vessel not in self._positions:
            return None
        if not all(v in self._positions for v in self._hunter_vessels):
            return None

        tx, ty = self._positions[self._target_vessel]
        pts = [self._positions[v] for v in self._hunter_vessels]

        dists = [math.hypot(x - tx, y - ty) for x, y in pts]

        angles_rad = [math.atan2(y - ty, x - tx) for x, y in pts]
        angles_deg = sorted(math.degrees(a) for a in angles_rad)

        gaps = [(angles_deg[(i + 1) % 3] - angles_deg[i] + 360) % 360 for i in range(3)]
        max_gap = max(gaps)
        surround_quality = max(0.0, 1.0 - (max_gap - 120.0) / 240.0)

        # Barycentric point-in-triangle test
        h0, h1, h2 = pts
        area_tri = abs(h0[0] * (h1[1] - h2[1]) +
                       h1[0] * (h2[1] - h0[1]) +
                       h2[0] * (h0[1] - h1[1])) / 2.0

        a1 = abs(tx * (h1[1] - h2[1]) + h1[0] * (h2[1] - ty) + h2[0] * (ty - h1[1])) / 2.0
        a2 = abs(h0[0] * (ty - h2[1]) + tx * (h2[1] - h0[1]) + h2[0] * (h0[1] - ty)) / 2.0
        a3 = abs(h0[0] * (h1[1] - ty) + h1[0] * (ty - h0[1]) + tx * (h0[1] - h1[1])) / 2.0

        if area_tri < 1e-6:
            in_triangle = 0
        else:
            in_triangle = 1 if abs(area_tri - (a1 + a2 + a3)) < 1e-3 * area_tri else 0

        return {
            'surround_quality': surround_quality,
            'mean_dist': float(np.mean(dists)),
            'min_dist': float(np.min(dists)),
            'max_dist': float(np.max(dists)),
            'max_gap_deg': max_gap,
            'in_triangle': in_triangle,
            'angles_deg': angles_deg,
            'distances': dists,
        }

    # ------------------------------------------------------------------
    def _print_metrics(self):
        m = self._compute_metrics()
        if m is None:
            return

        t = self.get_clock().now().nanoseconds / 1e9 - self._start_time

        print(
            f'[t={t:5.0f}s]  '
            f'quality={m["surround_quality"]:.2f}  '
            f'mean_dist={m["mean_dist"]:5.0f}m  '
            f'min_dist={m["min_dist"]:5.0f}m  '
            f'gap={m["max_gap_deg"]:5.0f}°  '
            f'in_triangle={m["in_triangle"]}'
        )
        sys.stdout.flush()

        self._metrics_history.append({
            't': t,
            'quality': m['surround_quality'],
            'mean_dist': m['mean_dist'],
            'min_dist': m['min_dist'],
            'max_dist': m['max_dist'],
            'max_gap_deg': m['max_gap_deg'],
            'in_triangle': m['in_triangle'],
        })

    # ------------------------------------------------------------------
    def _end_episode(self):
        self.get_logger().info('Episode timer fired – shutting down.')
        self._save_summary()
        self.destroy_node()
        raise SystemExit(0)

    # ------------------------------------------------------------------
    def _save_summary(self):
        if not self._metrics_history:
            return

        qualities = [m['quality'] for m in self._metrics_history]
        mean_dists = [m['mean_dist'] for m in self._metrics_history]
        in_tri = [m['in_triangle'] for m in self._metrics_history]

        summary = {
            'episode_length_s': round(self._metrics_history[-1]['t'], 1),
            'n_samples': len(self._metrics_history),
            'mean_quality': round(float(np.mean(qualities)), 3),
            'max_quality': round(float(np.max(qualities)), 3),
            'final_quality': round(float(qualities[-1]), 3),
            'mean_dist_m': round(float(np.mean(mean_dists)), 1),
            'min_dist_m': round(float(np.min([m['min_dist'] for m in self._metrics_history])), 1),
            'target_in_triangle_pct': round(float(np.mean(in_tri)) * 100, 1),
        }

        print()
        print('=' * 62)
        print('  EVALUATION SUMMARY')
        print('=' * 62)
        print(f'  Episode length    {summary["episode_length_s"]:>8.1f} s')
        print(f'  Samples            {summary["n_samples"]:>8d}')
        print(f'  Mean quality       {summary["mean_quality"]:>8.3f}')
        print(f'  Max quality        {summary["max_quality"]:>8.3f}')
        print(f'  Final quality      {summary["final_quality"]:>8.3f}')
        print(f'  Mean distance      {summary["mean_dist_m"]:>8.1f} m')
        print(f'  Min distance       {summary["min_dist_m"]:>8.1f} m')
        print(f'  Target in triangle {summary["target_in_triangle_pct"]:>7.1f} %')
        print('=' * 62)

        with open('/tmp/eval_result.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'  Results written to /tmp/eval_result.json')
        print('=' * 62)
        print()


def main():
    rclpy.init()
    node = SurroundMetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._save_summary()
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
