"""Four vessel residual-action coordination environment and geometric teacher."""
from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from bcod_sim.benchmark_v2.scenarios import Scenario, VesselPose, validate_scenario
from ..config import TaskConfig
from ..parallel_env import NavigationParallelEnv
from ..control.high_level import HeadingController, wrap_angle

OBS_VERSION = "fleet-residual-observation-v1"
ACT_VERSION = "speed-multiplier-heading-residual-v1"
ACTION_SPACE = spaces.Box(np.array([0., -1.], np.float32), np.array([1., 1.], np.float32))
OBS_FIELDS = ("goal_x_body", "goal_y_body", "surge", "sway", "yaw_rate",
              "previous_speed", "previous_heading_residual", "remaining_time", "agent_id",
              *(f"other_{i}_{f}" for i in range(3) for f in
                ("x", "y", "vx", "vy", "goal_x", "goal_y", "dcpa", "tcpa", "inactive")))
OBS_SPACE = spaces.Box(-1., 1., shape=(len(OBS_FIELDS),), dtype=np.float32)


def _rotate(x, y, heading):
    return x * math.cos(heading) + y * math.sin(heading), -x * math.sin(heading) + y * math.cos(heading)


def dcpa_tcpa(rx, ry, rvx, rvy):
    speed2 = rvx * rvx + rvy * rvy
    tcpa = 0. if speed2 < 1e-10 else float(np.clip(-(rx * rvx + ry * rvy) / speed2, 0., 60.))
    return math.hypot(rx + rvx * tcpa, ry + rvy * tcpa), tcpa


def make_four_vessel_scenario(rng: np.random.Generator, family: str = "mixture", split="train"):
    """Generate nonconflict, pair, corner, or four-way crossing cases."""
    if family == "mixture":
        family = rng.choice(["easy", "pair", "corner", "fourway", "random"],
                            p=[.2, .3, .15, .2, .15])
    seed = int(rng.integers(0, 2**31 - 1))
    d = float(rng.uniform(20., 25.))
    lane = float(rng.uniform(5., 8.))
    if family == "easy":
        starts = [(-d, -12.), (-d, -4.), (-d, 4.), (-d, 12.)]
        goals = [(d, -12.), (d, -4.), (d, 4.), (d, 12.)]
    elif family == "pair":
        starts = [(-d, 0.), (0., -d), (d, lane), (0., d)]
        goals = [(d, 0.), (0., d), (-d, lane), (0., -d)]
    elif family == "corner":
        starts = [(-d, -d), (-d, d), (d, d), (d, -d)]
        goals = [(d, d), (d, -d), (-d, -d), (-d, d)]
    else:
        starts = [(-d, 0.), (d, 0.), (0., -d), (0., d)]
        goals = [(d, 0.), (-d, 0.), (0., d), (0., -d)]
    angle = float(rng.uniform(-math.pi, math.pi))
    mirror = -1. if rng.random() < .5 else 1.
    tx, ty = rng.uniform(-3., 3., size=2)
    transform = lambda p: (tx + p[0]*math.cos(angle)-mirror*p[1]*math.sin(angle),
                           ty + p[0]*math.sin(angle)+mirror*p[1]*math.cos(angle))
    starts, goals = [transform(p) for p in starts], [transform(p) for p in goals]
    poses = tuple(VesselPose(float(x), float(y), math.atan2(gy-y, gx-x))
                  for (x, y), (gx, gy) in zip(starts, goals))
    scenario = Scenario(split, seed, poses, tuple(tuple(map(float, p)) for p in goals))
    validate_scenario(scenario, TaskConfig(agent_count=4, action_mode="high_level"))
    return scenario


class ResidualCoordinatorEnv(NavigationParallelEnv):
    """Adapts the qualified parallel env to direct-to-goal residual commands."""
    def __init__(self, config=None, **kwargs):
        config = config or TaskConfig(agent_count=4, dynamics="bcod-reduced",
                                      action_mode="high_level", dt_s=.5,
                                      max_heading_offset_rad=math.pi,
                                      deadline_steps=600)
        if config.agent_count != 4 or config.action_mode != "high_level":
            raise ValueError("Residual coordinator requires four high-level vessels")
        super().__init__(config, **kwargs)
        self._previous_actions = {name: np.zeros(2, np.float32) for name in self.possible_agents}
        self._previous_positions = None
        self._previous_distances = None
        self._previous_reached = set()
        self.actor_observation_schema = {"version": OBS_VERSION, "fields": OBS_FIELDS}
        self.action_schema = ACT_VERSION

    def observation_space(self, agent):
        if agent not in self.possible_agents: raise KeyError(agent)
        return OBS_SPACE

    def action_space(self, agent):
        if agent not in self.possible_agents: raise KeyError(agent)
        return ACTION_SPACE

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self._previous_actions = {name: np.zeros(2, np.float32) for name in self.possible_agents}
        self._previous_positions = {n: (r.x_m, r.y_m) for n, r in self._frame.readings.items()}
        self._previous_distances = {n: math.dist((self._frame.readings[n].x_m,
            self._frame.readings[n].y_m), self.scenario.goals[i])
            for i, n in enumerate(self.possible_agents)}
        self._previous_reached = set()
        return self._encode(), info

    def _encode(self):
        readings = self._frame.readings
        result = {}
        for i, name in enumerate(self.possible_agents):
            ego = readings[name]; gx, gy = self.scenario.goals[i]
            gbx, gby = _rotate(gx-ego.x_m, gy-ego.y_m, ego.heading_rad)
            own_vx = (ego.x_m-self._previous_positions[name][0])/self.config.dt_s
            own_vy = (ego.y_m-self._previous_positions[name][1])/self.config.dt_s
            values = [gbx/(2*self.config.half_width_m), gby/(2*self.config.half_width_m),
                      ego.surge_mps/self.config.max_surge_mps, 0., ego.yaw_rps/self.config.max_yaw_rps,
                      float(self._previous_actions[name][0])*2-1, float(self._previous_actions[name][1]),
                      max(0., (self.config.deadline_steps-self.task.steps)/self.config.deadline_steps),
                      (i-1.5)/1.5]
            for j, other in enumerate(self.possible_agents):
                if other == name: continue
                ro = readings[other]; ogx, ogy = self.scenario.goals[j]
                rx, ry = ro.x_m-ego.x_m, ro.y_m-ego.y_m
                ovx = (ro.x_m-self._previous_positions[other][0])/self.config.dt_s
                ovy = (ro.y_m-self._previous_positions[other][1])/self.config.dt_s
                rbx, rby = _rotate(rx, ry, ego.heading_rad)
                rvbx, rvby = _rotate(ovx-own_vx, ovy-own_vy, ego.heading_rad)
                odx, ody = _rotate(ogx-ro.x_m, ogy-ro.y_m, ego.heading_rad)
                dcpa, tcpa = dcpa_tcpa(rx, ry, ovx-own_vx, ovy-own_vy)
                values.extend((rbx/(2*self.config.half_width_m), rby/(2*self.config.half_width_m),
                               rvbx/(2*self.config.max_surge_mps), rvby/(2*self.config.max_surge_mps),
                               odx/(2*self.config.half_width_m), ody/(2*self.config.half_width_m),
                               min(dcpa/(2*self.config.half_width_m), 1.), min(tcpa/60., 1.),
                               float(self.task.reached.get(other, False))))
            result[name] = np.clip(np.asarray(values, np.float32), -1, 1)
        return result

    def step(self, actions):
        if set(actions) != set(self.agents): raise ValueError("Expected one action per live vessel")
        decoded = {}
        for i, name in enumerate(self.possible_agents):
            action = np.asarray(actions[name], np.float32)
            if action.shape != (2,) or not np.isfinite(action).all() or action[0] < 0 or action[0] > 1 or abs(action[1]) > 1:
                raise ValueError("Coordinator action is [speed multiplier 0..1, heading residual -1..1]")
            reading = self._frame.readings[name]
            gx, gy = self.scenario.goals[i]
            direct = math.atan2(gy-reading.y_m, gx-reading.x_m)
            residual = float(action[1])*math.radians(35)
            current_relative = wrap_angle(direct + residual - reading.heading_rad)
            decoded[name] = np.array([float(action[0])*2.-1., current_relative/math.pi], np.float32)
        old_distances = self._previous_distances.copy()
        old_reached = set(self._previous_reached)
        old_actions = {n: a.copy() for n, a in self._previous_actions.items()}
        old_positions = self._previous_positions
        out = super().step(decoded)
        self._previous_actions.update({n: np.asarray(actions[n], np.float32).copy() for n in actions})
        new_positions = {n: (r.x_m, r.y_m) for n, r in self._frame.readings.items()}
        observations, rewards, terms, truncs, infos = out
        # One fleet reward replicated to every live policy row for shared-credit MAPPO.
        distances = {n: math.dist((self._frame.readings[n].x_m,
            self._frame.readings[n].y_m), self.scenario.goals[i])
            for i, n in enumerate(self.possible_agents)}
        progress = (sum(old_distances.values())-sum(distances.values())) / max(
            len(distances)*self.config.max_surge_mps*self.config.dt_s, 1e-6)
        new_arrivals = sum(bool(self.task.reached.get(n, False)) and n not in old_reached
                           for n in self.possible_agents)
        risk = 0.
        for i, name in enumerate(self.possible_agents):
            for other in self.possible_agents[i+1:]:
                a, b = self._frame.readings[name], self._frame.readings[other]
                rx, ry = b.x_m-a.x_m, b.y_m-a.y_m
                avx, avy = a.surge_mps*math.cos(a.heading_rad), a.surge_mps*math.sin(a.heading_rad)
                bvx, bvy = b.surge_mps*math.cos(b.heading_rad), b.surge_mps*math.sin(b.heading_rad)
                dcpa, tcpa = dcpa_tcpa(rx, ry, bvx-avx, bvy-avy)
                risk += max(0., (4.-dcpa)/4.) * max(0., (20.-tcpa)/20.)
        change = sum(float(np.square(np.asarray(actions[n])-old_actions[n]).sum()) for n in actions)
        reason = next(iter(infos.values()))["terminal_reason"]
        event = .25*new_arrivals + (5. if reason == "success" else
                    -5. if reason == "collision" else -1. if reason == "deadline" else 0.)
        components = {"fleet_progress": float(progress), "collision_risk": -.2*risk,
                      "time": -.001, "action_change": -.01*change, "events": event}
        fleet_reward = float(sum(components.values()))
        rewards = {n: fleet_reward for n in rewards}
        for name in infos:
            infos[name]["shared_fleet_reward_components"] = components
            infos[name]["fleet_reward"] = fleet_reward
        self._previous_distances = distances
        self._previous_reached = {n for n in self.possible_agents if self.task.reached.get(n, False)}
        self._previous_positions = old_positions
        encoded = self._encode()
        self._previous_positions = new_positions
        return encoded, rewards, terms, truncs, infos


class GeometricCoordinator:
    """Simple right-of-way teacher in the exact normalized coordinator action space."""
    def actions(self, env):
        readings = env._frame.readings
        actions = {}
        for i, name in enumerate(env.possible_agents):
            ego = readings[name]; gx, gy = env.scenario.goals[i]
            direct = math.atan2(gy-ego.y_m, gx-ego.x_m)
            residual, speed = 0., .8
            for j, other in enumerate(env.possible_agents):
                if other == name or env.task.reached.get(other, False): continue
                ro = readings[other]
                rx, ry = ro.x_m-ego.x_m, ro.y_m-ego.y_m
                evx, evy = ego.surge_mps*math.cos(ego.heading_rad), ego.surge_mps*math.sin(ego.heading_rad)
                ovx, ovy = ro.surge_mps*math.cos(ro.heading_rad), ro.surge_mps*math.sin(ro.heading_rad)
                dcpa, tcpa = dcpa_tcpa(rx, ry, ovx-evx, ovy-evy)
                bearing = wrap_angle(math.atan2(ry, rx)-ego.heading_rad)
                if dcpa < 3.5 and tcpa < 15. and math.hypot(rx, ry) < 18.:
                    # Deterministic vessel index priority: later index yields.
                    if abs(bearing) < 1.25:
                        residual += -.35 if j < i else .35
                        speed = min(speed, .45 if j < i else .25)
            residual = float(np.clip(residual, -1., 1.))
            actions[name] = np.array([speed, residual], np.float32)
        return actions
