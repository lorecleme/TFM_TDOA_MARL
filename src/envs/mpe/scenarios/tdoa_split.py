"""TDOA tracking with split targets — targets move in opposite directions."""
import numpy as np
from itertools import combinations
from .tdoa_tracking import (
    Scenario as TDOAScenario, WORLD_MIN, WORLD_MAX, WORLD_SPAN, WORLD_DIAG,
)
from ..core import World


class SplitWorld(World):
    """Two targets move toward opposite corners."""

    def __init__(self):
        super().__init__()
        self.cam_range = 500
        self._directions = {}

    def step(self):
        for i, lm in enumerate(self.landmarks):
            if not (lm.movable and getattr(lm, 'is_target', False)):
                continue
            direction = self._directions[i]
            new_pos = lm.state.p_pos + direction * getattr(lm, 'speed_mult', 1.0)

            margin = 50
            stopped = False
            for d in range(self.dim_p):
                if new_pos[d] > WORLD_MAX - margin:
                    new_pos[d] = WORLD_MAX - margin
                    stopped = True
                elif new_pos[d] < WORLD_MIN + margin:
                    new_pos[d] = WORLD_MIN + margin
                    stopped = True
            if stopped:
                direction = np.zeros(2)  # stop at edge
            self._directions[i] = direction

            lm.state.p_vel = (new_pos - lm.state.p_pos) / self.dt
            lm.state.p_pos = new_pos
        super().step()


class Scenario(TDOAScenario):
    """Same as TDOA tracking, but targets split apart."""

    def make_world(self, *args, **kwargs):
        kwargs['target_movable'] = True
        kwargs['target_speed_mult'] = 1.0

        # Create a SplitWorld first so TDOAScenario can use it
        sw = SplitWorld()
        sw.dim_c = 2
        sw.collaborative = True
        sw.dim_p = 2

        # Monkey-patch: make the parent scenario use our world
        # Temporarily replace what the parent sees, call parent, restore
        from ..core import Agent, Landmark
        num_agents = kwargs.get('num_agents', 6)
        num_landmarks = kwargs.get('num_landmarks', 2)
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_entities = num_agents + num_landmarks
        self.num_entity_types = 3

        sw.agents = [Agent() for _ in range(num_agents)]
        agent_max_speed = float(kwargs.get('agent_max_speed', 500.0))
        agent_accel = float(kwargs.get('agent_accel', 1500.0))
        for i, agent in enumerate(sw.agents):
            agent.name = f'tracker_{i}'
            agent.collide = True; agent.silent = True; agent.size = 40
            agent.accel = agent_accel; agent.max_speed = agent_max_speed

        sw.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, lm in enumerate(sw.landmarks):
            lm.name = f'target_{i}'
            lm.collide = False; lm.movable = True; lm.size = 25
            lm.is_target = True; lm.speed_mult = 1.0

        # Replicate the parent's make_world logic (reward, obs setup)
        self.target_movable = True
        self.target_speed_mult = 1.0
        self.noise_std = float(kwargs.get('noise_std', 0.0)) * WORLD_SPAN
        self.reward_dist_coeff = float(kwargs.get('reward_dist_coeff', 0.01))
        self.reward_surround_coeff = float(kwargs.get('reward_surround_coeff', 2.0))
        self.reward_triangle_coeff = float(kwargs.get('reward_triangle_coeff', 1.0))
        self.reward_triangle_continuous = bool(kwargs.get('reward_triangle_continuous', False))
        self.reward_triangle_temperature = float(kwargs.get('reward_triangle_temperature', 100.0))
        self.reward_rmse_coeff = float(kwargs.get('reward_rmse_coeff', 0.0))
        self.reward_coverage_coeff = float(kwargs.get('reward_coverage_coeff', 0.5))
        self.reward_collision_coeff = float(kwargs.get('reward_collision_coeff', 0.5))
        self.surround_gate_distance = float(kwargs.get('surround_gate_distance', 200.0))
        self.surround_gate_softness = float(kwargs.get('surround_gate_softness', 0.0))
        self.reward_simple_dist_coeff = float(kwargs.get('reward_simple_dist_coeff', 0.0))
        self.reward_simple_dist_metric = str(kwargs.get('reward_simple_dist_metric', 'min'))
        self.soft_assign_temperature = float(kwargs.get('soft_assign_temperature', 50.0))
        self.reward_stillness_coeff = float(kwargs.get('reward_stillness_coeff', 0.0))
        self.reward_balance_coeff = float(kwargs.get('reward_balance_coeff', 0.0))
        if 'target_pos_mode' in kwargs:
            self.target_pos_mode = kwargs['target_pos_mode']
        elif kwargs.get('hide_target_positions', False):
            self.target_pos_mode = 'hidden'
        else:
            self.target_pos_mode = 'ground_truth'
        self._prev_agent_positions = None
        self.entity_obs_feats = 5 + num_landmarks
        self.entity_state_feats = 5 + num_landmarks
        self.benchmark_circles = [50, 100, 150, 200, 250]
        self._tdoa_pairs = list(combinations(range(num_agents), 2))

        self.reset_world(sw)
        return sw

    def reset_world(self, world):
        world._directions.clear()
        # Both targets start together at center, move to opposite edges
        world.landmarks[0].state.p_pos = np.array([500.0, 500.0])
        world.landmarks[0].state.p_vel = np.zeros(2)
        world.landmarks[1].state.p_pos = np.array([500.0, 500.0])
        world.landmarks[1].state.p_vel = np.zeros(2)
        world._directions[0] = np.array([-1.0, -0.3]) / np.sqrt(1.09) * 10.0  # down-left
        world._directions[1] = np.array([1.0, 0.3]) / np.sqrt(1.09) * 10.0    # up-right
        for agent in world.agents:
            agent.state.p_pos = np.array([500.0, 400.0]) + world.np_random.uniform(-80, 80, 2)
            agent.state.p_vel = np.zeros(2)
            agent.state.c = np.zeros(world.dim_c)
