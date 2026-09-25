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


class NavigationEnv(PyQuaticusEnv):
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
    return {"readings": sensor, "truth": truth}


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
                               "tau": dt, "max_time": scenario["max_steps"] * dt + 1,
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
                # Heron consumes heading error (degrees), while the common action
                # requests yaw rate. Its heading PID and rudder model attenuate a
                # one-tick error by roughly 20x at benchmark cruising thrust.
                native = {f"agent_{i}": [max(0.0, request["actions"][f"vessel_{i}"][0]) * scenario["max_surge_mps"],
                                         max(-180.0, min(180.0, -20.0 * math.degrees(
                                             request["actions"][f"vessel_{i}"][1] * scenario["max_yaw_rps"] * dt)))]
                          for i in range(4)}
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
