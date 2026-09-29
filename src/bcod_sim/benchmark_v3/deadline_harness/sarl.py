"""Independent low-level speed/heading tracking task with direct twin thrust."""
from __future__ import annotations

import math

import numpy as np

from bcod_sim.benchmark_v2.scenarios import Scenario, VesselPose
from bcod_sim.actuators.thruster import ThrustCommand
from bcod_sim.core.lifecycle import DirectAction
from ..config import TaskConfig
from ..control.high_level import wrap_angle


def common_differential_to_thrusters(common, differential):
    values = np.asarray([common, differential], dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(np.abs(values) > 1):
        raise ValueError("Common and differential thrust must be finite and normalized")
    return float(np.clip(common-differential, -1, 1)), float(np.clip(common+differential, -1, 1))


def tracking_observation(*, desired_speed, desired_heading, reading, speed_integral,
                         heading_integral, previous_action, sway=0., max_speed=2.):
    speed_error = desired_speed-reading.surge_mps
    heading_error = wrap_angle(desired_heading-reading.heading_rad)
    return np.clip(np.asarray([speed_error/max_speed, math.sin(heading_error),
        math.cos(heading_error), reading.surge_mps/max_speed, sway/max_speed,
        reading.yaw_rps/.25, speed_integral, heading_integral,
        previous_action[0], previous_action[1]], dtype=np.float32), -1., 1.)


class PIDThrustTeacher:
    """The current Otter PI controller represented in common/differential thrust."""
    def __init__(self):
        pass

    def predict(self, obs, *, env, deterministic=True):
        state = env.backend.engine.states["vessel_0"]
        reading = env.reading
        target_speed, target_heading = env.desired_speed, env.desired_heading
        speed_error = target_speed-float(state.nu_body[0])
        heading_error = wrap_angle(target_heading-reading.heading_rad)
        target_yaw = float(np.clip(.5*heading_error,-.25,.25))
        yaw_error = target_yaw-reading.yaw_rps
        params = env.backend._params
        speed_pi, yaw_pi = env.backend.controllers["vessel_0"]
        requested = (speed_pi.request(speed_error, env.config.dt_s,
                      params["linear_damping"][0]*target_speed),
                     yaw_pi.request(yaw_error, env.config.dt_s,
                     params["linear_damping"][5]*target_yaw))
        thrusters = env.backend.engine.vessels["vessel_0"].actuators
        from bcod_sim.actuators.allocation import allocate_fixed_thrusters
        allocated = allocate_fixed_thrusters(thrusters, surge_n=requested[0], yaw_nm=-requested[1])
        bounds = [t.config.thrust_bounds_n for t in thrusters]
        normalized = [2*(allocated[t.config.instance_id].thrust_n-b.minimum)/(b.maximum-b.minimum)-1
                      for t, b in zip(thrusters, bounds)]
        saturated = any(value <= -1. or value >= 1. for value in normalized)
        common = float(np.clip((normalized[0]+normalized[1])/2, -1, 1))
        differential = float(np.clip((normalized[1]-normalized[0])/2, -1, 1))
        return np.asarray([common, differential], np.float32), None


class SARLTrackingEnv:
    """One vessel, tracked commands only, direct actuator control; no navigation."""
    observation_size = 10
    action_size = 2

    def __init__(self, *, config=None, backend=None, episode_steps=1000):
        self.config = config or TaskConfig(agent_count=1, dynamics="bcod-reduced", dt_s=.1,
                                           deadline_steps=episode_steps)
        if backend is None:
            from bcod_sim.benchmark_v2.adapters.bcod import BCODBackend
            backend = BCODBackend(self.config)
        self.backend = backend
        self.backend.reduced_fidelity = True
        self.episode_steps = episode_steps
        self.closed = False
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, command_family="combined", disturbance="nominal"):
        self.rng = np.random.default_rng(seed)
        scenario = Scenario("sarl-tracking", int(seed or 0),
            (VesselPose(0., 0., float(self.rng.uniform(-math.pi, math.pi))),), ((0., 0.),))
        self._frame = self.backend.reset(scenario, scenario.seed)
        from bcod_sim.benchmark_v2.adapters.bcod import _otter_parameters
        self.backend._params = _otter_parameters()
        self.disturbance=disturbance
        world=self.backend.engine.world
        if disturbance=="current":
            world.current.vector_ned_mps=(0.25,0.,0.)
        elif disturbance=="wind":
            from bcod_sim.dynamics.environmental import RelativeWindLoads,WindCoefficientPoint
            world.wind.vector_ned_mps=(0.,3.,0.)
            coeff=tuple(WindCoefficientPoint(a,1.,0.,0.,0.) for a in (-math.pi,-math.pi/2,0.,math.pi/2,math.pi))
            object.__setattr__(self.backend.engine.vessels["vessel_0"],"wind_loads",
                               RelativeWindLoads(4.,8.,3.,coeff))
        elif disturbance=="parameter":
            plant=self.backend.engine.vessels["vessel_0"].plant
            plant.damping.linear.mul_(1.05); plant.damping.quadratic.mul_(1.05)
        elif disturbance!="nominal":
            raise ValueError(f"Unknown SARL disturbance: {disturbance}")
        self.steps = 0; self.speed_integral = 0.; self.heading_integral = 0.
        self.previous_action = np.zeros(2, np.float32)
        self.command_family = command_family
        self.desired_speed = float(self.rng.uniform(.35, 1.8))
        self.reading = self._frame.readings["vessel_0"]
        self.desired_heading = float(self.reading.heading_rad)
        self.last_command_time = 0
        return self._obs(), {"command_family": command_family}

    def _obs(self):
        speed_pi, heading_pi = self.backend.controllers["vessel_0"]
        speed_scale=max(5.*abs(speed_pi.ki),1.)
        heading_scale=max(5.*abs(heading_pi.ki),1.)
        return tracking_observation(desired_speed=self.desired_speed,
            desired_heading=self.desired_heading, reading=self.reading,
            speed_integral=float(np.clip(speed_pi.integral/speed_scale,-1.,1.)),
            heading_integral=float(np.clip(heading_pi.integral/heading_scale,-1.,1.)),
            previous_action=self.previous_action,
            sway=float(self.backend.engine.states["vessel_0"].nu_body[1]))

    def _prepare_pid_integrators(self):
        """Maintain the teacher's exposed PI memory for cloned-policy observations."""
        state=self.backend.engine.states["vessel_0"]
        speed_pi,yaw_pi=self.backend.controllers["vessel_0"]
        params=self.backend._params
        speed_error=self.desired_speed-float(state.nu_body[0])
        heading_error=wrap_angle(self.desired_heading-self.reading.heading_rad)
        target_yaw=float(np.clip(.5*heading_error,-.25,.25))
        speed_pi.request(speed_error,self.config.dt_s,params["linear_damping"][0]*self.desired_speed)
        yaw_pi.request(target_yaw-self.reading.yaw_rps,self.config.dt_s,
                       params["linear_damping"][5]*target_yaw)

    def _maybe_new_command(self):
        if self.steps and self.steps % 100 == 0:
            family = self.command_family
            if family in ("speed", "combined"):
                self.desired_speed = float(self.rng.uniform(.25, 1.9))
            if family in ("heading", "combined"):
                degrees = float(self.rng.choice([15., 45., 90.])) * self.rng.choice([-1., 1.])
                self.desired_heading = wrap_angle(self.desired_heading+math.radians(degrees))

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError("SARL action must be common/differential thrust in [-1,1]")
        command_used=(self.desired_speed,self.desired_heading)
        left, right = common_differential_to_thrusters(*action)
        self._prepare_pid_integrators()
        thrusters = self.backend.engine.vessels["vessel_0"].actuators
        commands = {}
        for thruster, value in zip(thrusters, (left, right)):
            bounds = thruster.config.thrust_bounds_n
            thrust = bounds.minimum + (value+1.)*.5*(bounds.maximum-bounds.minimum)
            commands[thruster.config.instance_id] = ThrustCommand(float(thrust))
        frame = self.backend.engine.step({"vessel_0": DirectAction(tuple(commands.items()))})
        readings, _ = self.backend._read(frame)
        self.reading = readings["vessel_0"]
        self.steps += 1
        speed_error = self.desired_speed-self.reading.surge_mps
        heading_error = wrap_angle(self.desired_heading-self.reading.heading_rad)
        self.speed_integral = float(np.clip(self.speed_integral+speed_error*self.config.dt_s, -5., 5.))
        self.heading_integral = float(np.clip(self.heading_integral+heading_error*self.config.dt_s, -5., 5.))
        sway = float(self.backend.engine.states["vessel_0"].nu_body[1])
        reward_components = {"speed": -(speed_error/2.)**2,
            "heading": -2.*(1.-math.cos(heading_error)), "sway": -.05*(sway/2.)**2,
            "effort": -.01*float(np.dot(action, action)),
            "slew": -.05*float(np.dot(action-self.previous_action, action-self.previous_action))}
        reward = float(sum(reward_components.values()))
        saturated = {k: abs(v) >= .999 for k, v in zip(commands, (left, right))}
        any_saturated=any(saturated.values())
        speed_pi,yaw_pi=self.backend.controllers["vessel_0"]
        speed_pi.commit(any_saturated); yaw_pi.commit(any_saturated)
        self.previous_action = action.copy()
        self._maybe_new_command()
        return self._obs(), reward, False, self.steps >= self.episode_steps, {
            "command": command_used,"next_command": (self.desired_speed,self.desired_heading),
            "actual_speed": self.reading.surge_mps,"speed_error": command_used[0]-self.reading.surge_mps,
            "heading_error": wrap_angle(command_used[1]-self.reading.heading_rad),
            "time_s":self.steps*self.config.dt_s,
            "thrust": (left, right), "saturation": saturated,
            "reward_components": reward_components}

    def close(self):
        self.backend.close(); self.closed = True
