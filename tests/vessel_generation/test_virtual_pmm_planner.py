from bcod_sim.vessel_generation.virtual_pmm import initial_states, select_next_state


def test_seed_design_contains_symmetric_drift_yaw_and_coupling():
    states = initial_states()
    assert len(states) == 9
    assert len(set(states)) == 9
    assert all(type(state)(-state.v_over_u, -state.r_l_over_u) in states for state in states)
    assert any(state.v_over_u and state.r_l_over_u for state in states)


def test_adaptive_selection_is_deterministic_and_skips_completed():
    states = initial_states()
    uncertainty = {state: 1.0 for state in states}
    assert select_next_state(uncertainty, set()) == sorted(states)[0]
    assert select_next_state(uncertainty, {sorted(states)[0]}) == sorted(states)[1]
