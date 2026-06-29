import numpy as np
from itertools import combinations
from scipy.optimize import least_squares

from ..core import World, Agent, Landmark
from ..scenario import BaseScenario

# Domain: 1000 x 1000 metres
WORLD_MIN, WORLD_MAX = 0.0, 1000.0
WORLD_SPAN = WORLD_MAX - WORLD_MIN          # 1000
WORLD_DIAG = np.sqrt(2) * WORLD_SPAN       # ~1414 — max possible range


class TDOAWorld(World):
    """World that supports a wandering (or static) target landmark."""

    def __init__(self):
        super().__init__()
        self.cam_range = 500
        self._wander_dir = {}
        self._wander_timer = {}

    def step(self):
        for i, landmark in enumerate(self.landmarks):
            if landmark.movable and getattr(landmark, 'is_target', False):
                self._update_target(landmark, i)
        super().step()

    def _update_target(self, target, idx):
        timer = self._wander_timer.get(idx, 0)
        if timer <= 0:
            self._wander_timer[idx] = self.np_random.randint(20, 50)
            angle = self.np_random.uniform(0, 2 * np.pi)
            speed = 5 + self.np_random.uniform(0, 10)
            self._wander_dir[idx] = np.array([np.cos(angle), np.sin(angle)]) * speed
        else:
            self._wander_timer[idx] = timer - 1

        direction = self._wander_dir.get(idx, np.zeros(2))
        new_pos = target.state.p_pos + direction * getattr(target, 'speed_mult', 1.0)

        margin = 50
        for d in range(self.dim_p):
            if new_pos[d] > WORLD_MAX - margin:
                new_pos[d] = WORLD_MAX - margin
                if direction[d] > 0:
                    self._wander_dir[idx][d] *= -1
            elif new_pos[d] < WORLD_MIN + margin:
                new_pos[d] = WORLD_MIN + margin
                if direction[d] < 0:
                    self._wander_dir[idx][d] *= -1

        target.state.p_vel = (new_pos - target.state.p_pos) / self.dt
        target.state.p_pos = new_pos


class Scenario(BaseScenario):

    # ------------------------------------------------------------------
    #  Setup
    # ------------------------------------------------------------------

    def make_world(self, num_agents=3, num_landmarks=1,
                   target_movable=True, target_speed_mult=1.0, noise_std=0.05,
                   **kwargs):
        world = TDOAWorld()
        world.dim_c = 2
        world.collaborative = True

        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_entities = num_agents + num_landmarks
        self.num_entity_types = 3

        world.agents = [Agent() for _ in range(num_agents)]
        agent_max_speed = float(kwargs.get('agent_max_speed', 500.0))
        agent_accel = float(kwargs.get('agent_accel', 1500.0))
        for i, agent in enumerate(world.agents):
            agent.name = f'tracker_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 40
            agent.accel = agent_accel
            agent.max_speed = agent_max_speed

        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'target_{i}'
            landmark.collide = False
            landmark.movable = target_movable
            landmark.size = 25
            landmark.is_target = True
            landmark.speed_mult = target_speed_mult

        self.target_movable = target_movable
        self.target_speed_mult = target_speed_mult
        self.noise_std = noise_std * WORLD_SPAN       # fraction → metres

        # Reward coefficients — all configurable via env_args; defaults match current values
        self.reward_dist_coeff = float(kwargs.get('reward_dist_coeff', 0.01))
        self.reward_surround_coeff = float(kwargs.get('reward_surround_coeff', 2.0))
        self.reward_triangle_coeff = float(kwargs.get('reward_triangle_coeff', 1.0))
        self.reward_triangle_continuous = bool(kwargs.get('reward_triangle_continuous', False))
        self.reward_triangle_temperature = float(kwargs.get('reward_triangle_temperature', 100.0))
        self.reward_rmse_coeff = float(kwargs.get('reward_rmse_coeff', 0.01))
        self.reward_coverage_coeff = float(kwargs.get('reward_coverage_coeff', 0.5))
        self.reward_collision_coeff = float(kwargs.get('reward_collision_coeff', 0.5))
        self.surround_gate_distance = float(kwargs.get('surround_gate_distance', 200.0))
        self.surround_gate_softness = float(kwargs.get('surround_gate_softness', 0.0))
        self.reward_simple_dist_coeff = float(kwargs.get('reward_simple_dist_coeff', 0.0))
        self.reward_simple_dist_metric = str(kwargs.get('reward_simple_dist_metric', 'min'))

        self.soft_assign_temperature = float(kwargs.get('soft_assign_temperature', 50.0))  # metres
        self.reward_stillness_coeff = float(kwargs.get('reward_stillness_coeff', 0.0))
        self.reward_balance_coeff = float(kwargs.get('reward_balance_coeff', 0.0))  # penalize >3 agents per target
        # target_pos_mode: 'ground_truth' (Oracle) | 'hidden' (Direct TDOA) | 'tdoa_estimate' (Solver Pipeline)
        if 'target_pos_mode' in kwargs:
            self.target_pos_mode = kwargs['target_pos_mode']
        elif kwargs.get('hide_target_positions', False):
            self.target_pos_mode = 'hidden'  # backward compat
        else:
            self.target_pos_mode = 'ground_truth'
        self._prev_agent_positions = None  # for displacement penalty

        self.reset_world(world)

        self.entity_obs_feats = 5 + num_landmarks
        self.entity_state_feats = 5 + num_landmarks  # 5 base + per-target range features
        self.benchmark_circles = [50, 100, 150, 200, 250]

        # pre-compute TDOA pairs once (same for all steps)
        self._tdoa_pairs = list(combinations(range(num_agents), 2))

        return world

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    def reset_world(self, world):
        world._wander_dir.clear()
        world._wander_timer.clear()

        min_agent_target_sep = 300.0  # agents start at least this far from any target
        max_retries = 200

        # --- place targets ---
        n_targets = len(world.landmarks)
        if n_targets > 1:
            # multi-target: opposite corners (guaranteed >565m separation)
            corner_zones = [
                (100, 300, 100, 300),     # top-left corner
                (700, 900, 100, 300),     # top-right corner
                (100, 300, 700, 900),     # bottom-left corner
                (700, 900, 700, 900),     # bottom-right corner
            ]
            chosen = world.np_random.choice(len(corner_zones), size=min(n_targets, len(corner_zones)), replace=False)
            for idx, landmark in enumerate(world.landmarks):
                q = corner_zones[chosen[idx % len(chosen)]]
                landmark.state.p_pos = world.np_random.uniform([q[0], q[2]], [q[1], q[3]])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.85, 0.15, 0.15])
        else:
            # single target: original behaviour
            for landmark in world.landmarks:
                landmark.state.p_pos = world.np_random.uniform(200, 800, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.85, 0.15, 0.15])

        target_positions = np.array([t.state.p_pos for t in world.landmarks])

        # --- place agents far from all targets ---
        for agent in world.agents:
            for _ in range(max_retries):
                pos = world.np_random.uniform(100, 900, world.dim_p)
                if np.all(np.linalg.norm(target_positions - pos, axis=1) >= min_agent_target_sep):
                    agent.state.p_pos = pos
                    break
            else:
                agent.state.p_pos = world.np_random.uniform(100, 900, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.color = np.array([0.35, 0.35, 0.85])

    # ------------------------------------------------------------------
    #  TDOA  — scipy least-squares localisation
    # ------------------------------------------------------------------

    def _compute_tdoa_rmse(self, noisy_ranges, receiver_positions, target_true):
        """Returns (rmse_m, estimated_pos, converged).
        Works with any number of receivers (pairs computed from passed positions)."""
        n_rec = len(receiver_positions)
        if n_rec < 2:
            return float(WORLD_DIAG), receiver_positions.mean(axis=0), False
        pairs = list(combinations(range(n_rec), 2))
        tdoa = np.array([noisy_ranges[j] - noisy_ranges[i] for i, j in pairs])
        initial_guess = receiver_positions.mean(axis=0)

        def _residuals(pos, receivers, tdoa_meas, p):
            dists = np.linalg.norm(receivers - pos, axis=1)
            return np.array([dists[j] - dists[i] - tdoa_meas[k]
                             for k, (i, j) in enumerate(p)])

        try:
            result = least_squares(
                _residuals, initial_guess,
                args=(receiver_positions, tdoa, pairs),
                method='trf', ftol=1e-6
            )
            estimated = result.x
            rmse = np.linalg.norm(estimated - target_true)
            return rmse, estimated, result.success
        except Exception:
            return float(WORLD_DIAG), initial_guess, False

    def _tdoa_estimate_position(self, world, target_idx):
        """Estimate target position from noisy TDOA ranges — no ground truth needed.
        Returns (estimated_pos_xy, converged_bool)."""
        agent_positions = np.array([a.state.p_pos for a in world.agents])
        noisy_ranges = self._noisy_ranges(world, target_idx)
        n_rec = len(agent_positions)
        if n_rec < 3:
            return agent_positions.mean(axis=0), False
        pairs = list(combinations(range(n_rec), 2))
        tdoa = np.array([noisy_ranges[j] - noisy_ranges[i] for i, j in pairs])
        initial_guess = agent_positions.mean(axis=0)

        def _residuals(pos, receivers, tdoa_meas, p):
            dists = np.linalg.norm(receivers - pos, axis=1)
            return np.array([dists[j] - dists[i] - tdoa_meas[k]
                             for k, (i, j) in enumerate(p)])

        try:
            result = least_squares(
                _residuals, initial_guess,
                args=(agent_positions, tdoa, pairs),
                method='trf', ftol=1e-6
            )
            return result.x, result.success
        except Exception:
            return initial_guess, False

    # ------------------------------------------------------------------
    #  Point-in-triangle  (barycentric)
    # ------------------------------------------------------------------

    @staticmethod
    def _point_in_triangle(pt, a, b, c):
        v0 = c - a
        v1 = b - a
        v2 = pt - a
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-12:
            return False
        inv = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        return u >= 0 and v >= 0 and (u + v) <= 1

    # ------------------------------------------------------------------
    #  Continuous triangle helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point_segment_distance(pt, a, b):
        """Perpendicular distance from pt to line segment AB, clamped to segment endpoints."""
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq < 1e-12:
            return np.linalg.norm(pt - a)
        t = max(0.0, min(1.0, np.dot(pt - a, ab) / ab_sq))
        proj = a + t * ab
        return np.linalg.norm(pt - proj)

    @staticmethod
    def _triangle_edge_distance(pt, a, b, c):
        """Minimum perpendicular distance from pt to any edge of triangle ABC.
        Returns 0.0 when pt is inside the triangle."""
        return min(
            Scenario._point_segment_distance(pt, a, b),
            Scenario._point_segment_distance(pt, b, c),
            Scenario._point_segment_distance(pt, c, a),
        )

    # ------------------------------------------------------------------
    #  Reward
    # ------------------------------------------------------------------

    def reward(self, agent, world):
        """Global reward — soft-assignment-based per-target TDOA with coverage bonus."""
        if agent.name != 'tracker_0':
            return 0.0

        agents = world.agents
        agent_positions = np.array([a.state.p_pos for a in agents])
        n = len(agents)
        n_targets = len(world.landmarks)

        # --- collision (agent-wide, computed once) ---
        collision_penalty = 0.0
        for i in range(n):
            if not agents[i].collide:
                continue
            for j in range(i + 1, n):
                if agents[j].collide and self._is_collision(agents[i], agents[j]):
                    collision_penalty += self.reward_collision_coeff

        # --- soft assignment: each agent contributes to all targets weighted by proximity ---
        # temperature controls specialization: lower = sharper assignment
        soft_assign_temperature = self.soft_assign_temperature  # metres; distance at which assignment becomes ambiguous
        distances_to_targets = np.array([
            [np.linalg.norm(agent_positions[a_idx] - world.landmarks[t_idx].state.p_pos)
             for t_idx in range(n_targets)]
            for a_idx in range(n)
        ])  # (n_agents, n_targets)

        # softmax over negative distances for each agent
        exp_weights = np.exp(-distances_to_targets / soft_assign_temperature)  # (n_agents, n_targets)
        soft_weights = exp_weights / exp_weights.sum(axis=1, keepdims=True)  # normalize per-agent
        # effective number of agents assigned to each target
        effective_n = soft_weights.sum(axis=0)  # (n_targets,)

        total_reward = 0.0

        for t_idx, target in enumerate(world.landmarks):
            target_pos = target.state.p_pos

            # --- weighted mean distance (soft-assignment weighted) ---
            dists_to_t = distances_to_targets[:, t_idx]
            weights_for_t = soft_weights[:, t_idx]
            weighted_mean_dist = np.average(dists_to_t, weights=weights_for_t)

            # --- surround quality (per-target assigned agents, distance-gated) ---
            # Path B fix: use only agents primarily assigned to THIS target,
            # not all agents. Prevents cross-target angle pollution in multi-target.
            assigned_mask = weights_for_t > 0.3  # primary assignment
            assigned_positions = agent_positions[assigned_mask]
            n_assigned = len(assigned_positions)
            if n_assigned >= 2:
                angles = sorted(np.arctan2(assigned_positions[:, 1] - target_pos[1],
                                           assigned_positions[:, 0] - target_pos[0]))
                max_gap = max(
                    (angles[(i + 1) % n_assigned] - angles[i])
                    + (2 * np.pi if i == n_assigned - 1 else 0)
                    for i in range(n_assigned)
                )
                optimal_gap = 2 * np.pi / n_assigned
                surround_quality = max(0.0, 1.0 - (max_gap - optimal_gap)
                                       / (2 * np.pi - optimal_gap))
            elif n_assigned == 1:
                surround_quality = 0.0
            else:
                # fallback: all agents (should only happen very early in training)
                surround_quality = 0.0
            # distance gate: configurable hard clamp or sigmoid
            max_dist_all = max(float(np.max(dists_to_t)), 0.01)
            if self.surround_gate_softness <= 0:
                gate = min(1.0, self.surround_gate_distance / max_dist_all)
            else:
                gate = 1.0 / (1.0 + np.exp(
                    (max_dist_all - self.surround_gate_distance) / self.surround_gate_softness
                ))
            surround_quality *= gate

            # --- TDOA RMSE (all agents) ---
            if self.reward_rmse_coeff > 0 and n >= 3:
                noisy_ranges_all = self._noisy_ranges(world, t_idx)
                tdoa_rmse, _, converged = self._compute_tdoa_rmse(
                    noisy_ranges_all, agent_positions, target_pos
                )
            elif n >= 3:
                tdoa_rmse = 0.0  # won't be used, skip expensive least_squares
            else:
                tdoa_rmse = WORLD_DIAG
            rmse_clipped = min(tdoa_rmse, WORLD_DIAG) if self.reward_rmse_coeff > 0 else 0.0

            # --- triangle bonus (3 closest agents) with optional continuous version ---
            if n >= 3:
                closest_3_indices = np.argsort(dists_to_t)[:3]
                closest_3_positions = agent_positions[closest_3_indices]
                if self.reward_triangle_continuous:
                    if self._point_in_triangle(
                        target_pos, closest_3_positions[0], closest_3_positions[1], closest_3_positions[2]
                    ):
                        triangle_bonus = 1.0
                    else:
                        edge_dist = self._triangle_edge_distance(
                            target_pos, closest_3_positions[0], closest_3_positions[1], closest_3_positions[2]
                        )
                        triangle_bonus = float(np.exp(-edge_dist / self.reward_triangle_temperature))
                else:
                    triangle_bonus = 1.0 if self._point_in_triangle(
                        target_pos, closest_3_positions[0], closest_3_positions[1], closest_3_positions[2]
                    ) else 0.0
            else:
                triangle_bonus = 0.0

            total_reward += (
                -weighted_mean_dist * self.reward_dist_coeff
                + self.reward_surround_coeff * surround_quality
                + self.reward_triangle_coeff * triangle_bonus
                - self.reward_rmse_coeff * rmse_clipped
            )

            # --- simple distance reward (optional, for clean pursuit gradient) ---
            if self.reward_simple_dist_coeff > 0:
                if self.reward_simple_dist_metric == "min":
                    dist_val = float(np.min(dists_to_t))
                elif self.reward_simple_dist_metric == "mean":
                    dist_val = float(np.mean(dists_to_t))
                else:  # "weighted_mean"
                    dist_val = weighted_mean_dist
                total_reward += -self.reward_simple_dist_coeff * dist_val / WORLD_SPAN

        # --- coverage bonus: encourage at least 3 effective agents per target for triangulation ---
        coverage_bonus = sum(min(effective_n[t], 3.0) * self.reward_coverage_coeff
                             for t in range(n_targets))
        total_reward += coverage_bonus

        # --- balance penalty: penalize concentrating >3 agents on any one target ---
        #    3+3 split = 0 penalty; 4+2 = -5.0; 5+1 = -10.0; 6+0 = -15.0 (at coeff=5.0)
        if self.reward_balance_coeff > 0:
            excess = sum(max(0.0, effective_n[t] - 3.0) for t in range(n_targets))
            total_reward -= excess * self.reward_balance_coeff

        stillness_penalty = 0.0
        if self.reward_stillness_coeff > 0:
            curr_pos = np.array([a.state.p_pos for a in agents])
            if self._prev_agent_positions is not None:
                displacements = np.linalg.norm(curr_pos - self._prev_agent_positions, axis=1)
                stillness_penalty = self.reward_stillness_coeff * np.mean(displacements / WORLD_SPAN)
            self._prev_agent_positions = curr_pos.copy()

        return total_reward - collision_penalty - stillness_penalty

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _is_collision(self, a1, a2):
        delta = a1.state.p_pos - a2.state.p_pos
        return np.sqrt(np.sum(np.square(delta))) < (a1.size + a2.size)

    def _noisy_ranges(self, world, target_idx=0):
        target_pos = world.landmarks[target_idx].state.p_pos
        true_ranges = np.array([np.linalg.norm(a.state.p_pos - target_pos)
                                for a in world.agents])
        noise = world.np_random.normal(0, self.noise_std, size=len(true_ranges))
        return np.maximum(true_ranges + noise, 0.0)

    def _noisy_ranges_per_target(self, world):
        return np.column_stack([self._noisy_ranges(world, t)
                                for t in range(len(world.landmarks))])

    def _true_ranges(self, world, target_idx=0):
        target_pos = world.landmarks[target_idx].state.p_pos
        return np.array([np.linalg.norm(a.state.p_pos - target_pos)
                         for a in world.agents])

    # ------------------------------------------------------------------
    #  Entity observations
    # ------------------------------------------------------------------

    def entity_observation(self, agent, world):
        num_landmarks = len(world.landmarks)
        feats = np.zeros(self.entity_obs_feats * self.num_entities, dtype=np.float32)
        pos_a = agent.state.p_pos
        agent_idx = world.agents.index(agent)
        noisy_ranges_per_target = self._noisy_ranges_per_target(world)  # (n_agents, n_targets)
        i = 0

        # --- TDOA estimate cache (computed once per world step, reused across agents) ---
        world_id = id(world)
        if self.target_pos_mode == 'tdoa_estimate':
            cached = getattr(self, '_tdoa_cache', None)
            if cached is None or cached[0] != world_id:
                estimates = []
                for t_idx in range(num_landmarks):
                    est_pos, conv = self._tdoa_estimate_position(world, t_idx)
                    estimates.append(est_pos)
                self._tdoa_cache = (world_id, estimates)
            tdoa_estimates = self._tdoa_cache[1]

        for a_idx, a in enumerate(world.agents):
            if a is agent:
                feats[i] = 0.   # rel_x
                feats[i+1] = 0. # rel_y
                feats[i+2] = 1. # is_agent
                feats[i+3] = 1. # is_self
                # per-target ranges from THIS agent
                feats[i+4:i+4+num_landmarks] = noisy_ranges_per_target[agent_idx]
            else:
                delta = a.state.p_pos - pos_a
                feats[i] = delta[0]    # rel_x
                feats[i+1] = delta[1]  # rel_y
                feats[i+2] = 1.        # is_agent
                feats[i+3] = 0.        # is_self
                # per-target ranges from THAT agent
                feats[i+4:i+4+num_landmarks] = noisy_ranges_per_target[a_idx]
            i += self.entity_obs_feats

        for l_idx, landmark in enumerate(world.landmarks):
            land_range = np.linalg.norm(landmark.state.p_pos - pos_a)
            noise = world.np_random.normal(0, self.noise_std)
            noisy_range = max(land_range + noise, 0.0)
            if self.target_pos_mode == 'tdoa_estimate':
                delta = tdoa_estimates[l_idx] - pos_a
                feats[i] = delta[0]
                feats[i+1] = delta[1]
            elif self.target_pos_mode == 'hidden':
                feats[i] = 0.
                feats[i+1] = 0.
            else:  # 'ground_truth'
                delta = landmark.state.p_pos - pos_a
                feats[i] = delta[0]
                feats[i+1] = delta[1]
            feats[i+2] = 0.           # is_agent
            feats[i+3] = 0.           # is_self
            feats[i+4] = noisy_range  # range to this landmark
            # remaining range slots already 0 (np.zeros initialization)
            i += self.entity_obs_feats

        feats[0::self.entity_obs_feats] /= WORLD_SPAN      # rel_x  → [-1, 1]
        feats[1::self.entity_obs_feats] /= WORLD_SPAN      # rel_y  → [-1, 1]
        for t in range(num_landmarks):
            feats[4+t::self.entity_obs_feats] /= WORLD_DIAG

        return feats

    # ------------------------------------------------------------------
    #  Entity state  (mixer)
    # ------------------------------------------------------------------

    def entity_state(self, world):
        feats = np.zeros(self.entity_state_feats * self.num_entities, dtype=np.float32)
        i = 0

        for a in world.agents:
            pos, vel = a.state.p_pos, a.state.p_vel
            row = [pos[0], pos[1], vel[0], vel[1], 1.]
            # per-target ranges from this agent
            for target in world.landmarks:
                dist = np.linalg.norm(pos - target.state.p_pos)
                row.append(dist)
            feats[i:i + self.entity_state_feats] = row
            i += self.entity_state_feats

        for landmark in world.landmarks:
            pos, vel = landmark.state.p_pos, landmark.state.p_vel
            row = [pos[0], pos[1], vel[0], vel[1], 0.]
            # pad zeros for per-target range slots (not meaningful for landmarks)
            row.extend([0.0] * len(world.landmarks))
            feats[i:i + self.entity_state_feats] = row
            i += self.entity_state_feats

        feats[0::self.entity_state_feats] /= WORLD_SPAN
        feats[1::self.entity_state_feats] /= WORLD_SPAN
        feats[2::self.entity_state_feats] /= WORLD_SPAN
        feats[3::self.entity_state_feats] /= WORLD_SPAN
        # Normalize per-target range features (slots 5..5+num_landmarks)
        for t in range(len(world.landmarks)):
            feats[5 + t::self.entity_state_feats] /= WORLD_SPAN

        return feats

    # ------------------------------------------------------------------
    #  Flat observation  (RNN fallback)
    # ------------------------------------------------------------------

    def observation(self, agent, world):
        agent_idx = world.agents.index(agent)
        noisy_ranges = self._noisy_ranges_per_target(world)[agent_idx]

        if self.target_pos_mode == 'tdoa_estimate':
            entity_pos = []
            for t_idx in range(len(world.landmarks)):
                est_pos, _ = self._tdoa_estimate_position(world, t_idx)
                entity_pos.append((est_pos - agent.state.p_pos) / WORLD_SPAN)
        elif self.target_pos_mode == 'hidden':
            entity_pos = [np.zeros(world.dim_p) for _ in world.landmarks]
        else:
            entity_pos = [(ent.state.p_pos - agent.state.p_pos) / WORLD_SPAN
                          for ent in world.landmarks]
        other_pos = [(other.state.p_pos - agent.state.p_pos) / WORLD_SPAN
                     for other in world.agents if other is not agent]

        parts = [agent.state.p_vel / WORLD_SPAN,
                 agent.state.p_pos / WORLD_SPAN] + entity_pos + other_pos + [noisy_ranges / WORLD_DIAG]
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------
    #  Benchmark  — now includes actual TDOA RMSE
    # ------------------------------------------------------------------

    def world_benchmark_data(self, world, final=False):
        agent_positions = np.array([a.state.p_pos for a in world.agents])
        n = len(agent_positions)
        n_targets = len(world.landmarks)

        # per-target metrics, averaged
        all_surround = []
        all_in_triangle = []
        all_min_dists = []
        all_mean_dists = []

        # soft assignment for per-target surround
        distances_to_targets = np.array([
            [np.linalg.norm(agent_positions[a] - world.landmarks[t].state.p_pos)
             for t in range(n_targets)]
            for a in range(n)
        ])
        exp_weights = np.exp(-distances_to_targets / self.soft_assign_temperature)
        sw = exp_weights / exp_weights.sum(axis=1, keepdims=True)

        for t_idx, target in enumerate(world.landmarks):
            target_pos = target.state.p_pos
            dists = distances_to_targets[:, t_idx]
            all_min_dists.append(float(np.min(dists)))
            all_mean_dists.append(float(np.mean(dists)))

            # Per-target surround: use only agents primarily assigned to this target
            assigned_mask = sw[:, t_idx] > 0.3
            assigned_positions = agent_positions[assigned_mask]
            n_assigned = len(assigned_positions)
            if n_assigned >= 2:
                angles = sorted(np.arctan2(assigned_positions[:, 1] - target_pos[1],
                                           assigned_positions[:, 0] - target_pos[0]))
                max_gap = max(
                    (angles[(i + 1) % n_assigned] - angles[i])
                    + (2 * np.pi if i == n_assigned - 1 else 0)
                    for i in range(n_assigned)
                )
                surround = max(0.0, 1.0 - (max_gap - 2 * np.pi / n_assigned)
                              / (2 * np.pi - 2 * np.pi / n_assigned))
            else:
                surround = 0.0
            all_surround.append(surround)

            # Per-target triangle: convex hull of the 3 closest agents to THIS
            # target. Matches the reward's triangle-bonus definition and the
            # thesis metric. (Previously hardcoded to agents [0,1,2], which was
            # only correct for 3v1 and meaningless for 6v2/9v3.)
            if n >= 3:
                closest3 = agent_positions[np.argsort(dists)[:3]]
                in_tri = self._point_in_triangle(
                    target_pos, closest3[0], closest3[1], closest3[2]
                )
            else:
                in_tri = False
            all_in_triangle.append(float(in_tri))

        info = {
            'min_dist_to_target': float(np.mean(all_min_dists)),
            'mean_dist_to_target': float(np.mean(all_mean_dists)),
            'surround_quality': float(np.mean(all_surround)),
            'target_in_triangle': float(np.mean(all_in_triangle)),
        }

        # Always log per-target agent counts (not just final) for split diagnostics
        if n_targets > 1:
            nearest_counts = np.zeros(n_targets)
            for a_pos in agent_positions:
                nearest = np.argmin([np.linalg.norm(a_pos - t.state.p_pos) for t in world.landmarks])
                nearest_counts[nearest] += 1
            info['min_agents_per_target'] = float(np.min(nearest_counts))
            info['max_agents_per_target'] = float(np.max(nearest_counts))
            info['agent_distribution_entropy'] = float(
                -sum((nearest_counts[i] / n) * np.log(max(nearest_counts[i] / n, 1e-9))
                     for i in range(n_targets)) / np.log(n_targets)
            )
            for t_idx in range(n_targets):
                dists_to_t = np.linalg.norm(agent_positions - world.landmarks[t_idx].state.p_pos, axis=1)
                for c in [50, 100, 200]:
                    info[f'target{t_idx}_agents_within_{c}m'] = float(np.sum(dists_to_t <= c))

        if final:
            # TDOA RMSE per target, averaged
            all_rmse = []
            all_conv = []
            for t_idx in range(n_targets):
                noisy_ranges = self._noisy_ranges(world, t_idx)
                tdoa_rmse, est_pos, converged = self._compute_tdoa_rmse(
                    noisy_ranges, agent_positions, world.landmarks[t_idx].state.p_pos
                )
                all_rmse.append(tdoa_rmse)
                all_conv.append(float(converged))
            info['tdoa_rmse_m'] = float(np.mean(all_rmse))
            info['tdoa_converged'] = float(np.mean(all_conv))
            for c in self.benchmark_circles:
                info[f'occupied_{c}m'] = np.mean([1.0 if d <= c else 0.0 for d in all_min_dists])

            # per-target coverage: how many agents within benchmark circles
            for t_idx in range(n_targets):
                dists_to_t = np.linalg.norm(agent_positions - world.landmarks[t_idx].state.p_pos, axis=1)
                for c in [50, 100, 150, 200, 250]:
                    n_within = int(np.sum(dists_to_t <= c))
                    info[f'target{t_idx}_agents_within_{c}m'] = float(n_within)

            # agent distribution across targets (how many agents nearest to each target)
            nearest_counts = np.zeros(n_targets)
            for a_pos in agent_positions:
                nearest = np.argmin([np.linalg.norm(a_pos - t.state.p_pos) for t in world.landmarks])
                nearest_counts[nearest] += 1
            if n_targets > 1:
                info['agent_distribution_entropy'] = float(
                    -sum((nearest_counts[i] / n) * np.log(max(nearest_counts[i] / n, 1e-9))
                         for i in range(n_targets)) / np.log(n_targets)
                )
            else:
                info['agent_distribution_entropy'] = 0.0
            info['min_agents_per_target'] = float(np.min(nearest_counts))
            info['max_agents_per_target'] = float(np.max(nearest_counts))

        return info


def expand_agent_embedding(agent, old_feat_dim, new_feat_dim):
    """Expand Linear embedding layer from old_feat_dim to new_feat_dim.
    Copies existing weights, initializes new columns to zero.
    Call this when loading a pretrained checkpoint with a different entity_obs_feats."""
    import torch as th
    old_weight = agent.feat_embedding.weight.data  # (emb, old_feat_dim)
    old_bias = agent.feat_embedding.bias.data if agent.feat_embedding.bias is not None else None

    new_linear = th.nn.Linear(new_feat_dim, old_weight.shape[0])
    new_linear.weight.data[:, :old_feat_dim] = old_weight
    new_linear.weight.data[:, old_feat_dim:] = 0.0  # zero-init for new targets
    if old_bias is not None:
        new_linear.bias.data = old_bias

    agent.feat_embedding = new_linear
