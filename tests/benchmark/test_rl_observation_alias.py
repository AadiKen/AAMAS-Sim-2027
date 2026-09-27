"""Direct alias test for another vessel's unobserved motion."""
import math

import numpy as np

from bcod_sim.benchmark.core import BenchmarkConfig, VesselReading, observation


def test_same_relative_position_aliases_four_other_vessel_velocities():
    other_velocities = {
        "approaching": (-1.0, 0.0),
        "receding": (1.0, 0.0),
        "crossing_left": (0.0, 1.0),
        "crossing_right": (0.0, -1.0),
    }
    policy_inputs = {}
    separations_after_5s = {}
    for motion, (vx, vy) in other_velocities.items():
        # The other vessel is at (10, 0) in every world state. Its velocity
        # cannot be passed through the current nearby_agents position API.
        own_reading = VesselReading(0, 0, 0, 0, 0, nearby_agents=((10, 0),))
        policy_inputs[motion] = observation(own_reading, (30, 0), BenchmarkConfig())
        separations_after_5s[motion] = math.hypot(10 + 5 * vx, 5 * vy)
    assert all(np.array_equal(policy_inputs["approaching"], value)
               for value in policy_inputs.values())
    assert separations_after_5s["approaching"] == 5
    assert separations_after_5s["receding"] == 15
    assert separations_after_5s["crossing_left"] == separations_after_5s["crossing_right"]
