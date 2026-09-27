"""JSON-lines worker for the official Python-3.10-only Pyquaticus simulator.

Run with an isolated interpreter. Native capture-the-flag outcomes are disabled;
Pyquaticus still supplies its own vessel dynamics and obstacle interactions.
"""

import contextlib
import io
import json
import math
import sys

from pyquaticus.config import get_std_config
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.utils.utils import detect_collision


class NavigationEnv(PyQuaticusEnv):
    def _move_agents(self, action_dict):
        # Native obstacle response tests the previous position before moving.
        # Record that event separately from post-step geometric overlap.
        self.benchmark_contacts = [f"vessel_{i}" for i, player in enumerate(self.players.values())
            if detect_collision(player.pos, self.agent_radius[i], self.obstacle_geoms)]
        return super()._move_agents(action_dict)

    def _check_flag_pickups(self):
        pass

    def _check_agent_made_tag(self):
        pass

    def _check_flag_captures(self):
        pass

    def _check_untag(self):
        pass

    def _set_dones(self):
        self.dones = {"blue": False, "red": False, "__all__": False}


def readings(env, scenario, prior_heading, dt):
    positions = env.state["agent_position"]
    headings = env.state["agent_heading"]
    speed = env.state["agent_speed"]
    obstacles = scenario["obstacles"]
    visibility = scenario["visibility_m"]
    half = scenario["half_width_m"]
    sensor = {}
    truth = {}
    for i in range(4):
        name = f"vessel_{i}"
        x, y = float(positions[i, 0] - half), float(positions[i, 1] - half)
        heading = math.pi / 2 - math.radians(float(headings[i]))
        agents = [[float(positions[j, 0] - half), float(positions[j, 1] - half)]
                  for j in range(4) if j != i and
                  math.dist(positions[i], positions[j]) <= visibility]
        detected_obstacles = []
        for o in obstacles:
            if math.dist((x, y), (o["x_m"], o["y_m"])) <= visibility:
                detected_obstacles.append([o["x_m"], o["y_m"], o["radius_m"]])
        delta = math.atan2(math.sin(heading - prior_heading[i]), math.cos(heading - prior_heading[i]))
        sensor[name] = {"x_m": x, "y_m": y, "heading_rad": heading,
                        "surge_mps": float(speed[i]), "yaw_rps": delta / dt,
                        "nearby_agents": agents, "nearby_obstacles": detected_obstacles}
        truth[name] = {"x_m": x, "y_m": y, "heading_rad": heading}
    return {"readings": sensor, "truth": truth, "diagnostics": {
        "sim_time_s": env.current_time,
        "saturated": {f"vessel_{i}": bool(abs(player.state.get("rudder", 0)) >= player.max_rudder or
                      abs(player.state.get("thrust", 0)) >= player.max_thrust)
                      for i, player in enumerate(env.players.values())},
        "native_contacts": getattr(env, "benchmark_contacts", []),
        "native_overlap": [f"vessel_{i}" for i, player in enumerate(env.players.values())
                            if detect_collision(player.pos, env.agent_radius[i], env.obstacle_geoms)],
        "native_agent_contacts": bool(env.active_collisions.any())}}


def main():
    env = None
    scenario = None
    heading = [0.0] * 4
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "close":
                if env is not None:
                    env.close()
                print(json.dumps({"ok": True}), flush=True)
                return
            if op == "reset":
                if env is not None:
                    env.close()
                scenario = request["scenario"]
                half = scenario["half_width_m"]
                dt = scenario["dt_s"]
                config = get_std_config()
                config.update({"env_bounds": [2 * half, 2 * half], "agent_radius": scenario["vessel_radius_m"],
                               "flag_keepout": 0.0, "catch_radius": 0.01, "tag_on_oob": False,
                               "tau": dt, "sim_speedup_factor": 1, "tag_on_collision": False,
                               "max_time": scenario["max_steps"] * dt + 1,
                               "obstacles": {"circle": [(o["radius_m"], (o["x_m"] + half, o["y_m"] + half))
                                                        for o in scenario["obstacles"]] or
                                                       [(1.0, (1e6, 1e6))]}})
                with contextlib.redirect_stdout(io.StringIO()):
                    env = NavigationEnv(team_size=2, action_space="continuous", config_dict=config)
                    starts = scenario["starts"]
                    env.reset(seed=request["seed"], options={"init_dict": {
                        "agent_position": [[s["x_m"] + half, s["y_m"] + half] for s in starts],
                        "agent_heading": [math.degrees(math.pi / 2 - s["heading_rad"]) for s in starts],
                        "agent_speed": [0.0] * 4}})
                heading = [s["heading_rad"] for s in starts]
                result = readings(env, scenario, heading, dt)
                heading = [result["truth"][f"vessel_{i}"]["heading_rad"] for i in range(4)]
            elif op == "step":
                if env is None:
                    raise RuntimeError("Reset required")
                dt = scenario["dt_s"]
                native = {}
                for i, player in enumerate(env.players.values()):
                    action = request["actions"][f"vessel_{i}"]
                    target = action[1] * scenario["max_yaw_rps"]
                    measured = result["readings"][f"vessel_{i}"]["yaw_rps"]
                    # Native Heron yaw = rudder * turn_rate/100 * thrust/50
                    # (degrees/s). Invert that steady gain, then reject residual
                    # error with measured-rate feedback. Keep native heading PID.
                    gain = (player.turn_rate / 100) * max(5., player.state["thrust"]) / 50
                    desired_rudder = -math.degrees(target + .2 * (target - measured)) / gain
                    # Invert the pinned native PID without advancing/mutating it.
                    # Ignoring its derivative term creates an unstable outer loop.
                    pid = player._pid_controllers["heading"]
                    def output(error):
                        derivative = 0. if pid._prev_error is None else (error-pid._prev_error)/pid._dt
                        integral = max(-pid._integral_limit, min(pid._integral_limit,
                                       pid._integral + pid._ki*error*pid._dt))
                        return pid._kp*error + pid._kd*derivative + integral
                    lo, hi = -180., 180.
                    for _ in range(40):
                        mid = (lo+hi)/2
                        if output(mid) < desired_rudder: lo=mid
                        else: hi=mid
                    heading_error = (lo+hi)/2
                    native[f"agent_{i}"] = [action[0] * scenario["max_surge_mps"],
                                            max(-180., min(180., heading_error))]
                with contextlib.redirect_stdout(io.StringIO()):
                    env.step(native)
                result = readings(env, scenario, heading, dt)
                heading = [result["truth"][f"vessel_{i}"]["heading_rad"] for i in range(4)]
            else:
                raise ValueError(f"Unknown operation: {op}")
            print(json.dumps(result, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
