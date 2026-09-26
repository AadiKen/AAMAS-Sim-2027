"""Output-error coefficient fitting from field maneuvers.

The caller supplies a Plant6 trajectory simulator with fixed added mass and
fixed actuator model. One maneuver is held out explicitly.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np


def fit_output_error(prior: Sequence[float], maneuvers: Sequence[dict], *,
                     simulator: Callable[[np.ndarray, dict], np.ndarray],
                     holdout_index: int, ridge_weight: float = 1.) -> dict:
    parameters = np.asarray(prior, float)
    if (parameters.ndim != 1 or not len(parameters) or not np.isfinite(parameters).all() or
        not 0 <= holdout_index < len(maneuvers) or len(maneuvers) < 2 or
        not np.isfinite(ridge_weight) or ridge_weight < 0):
        raise ValueError("Invalid system-ID prior, maneuvers, holdout, or ridge")
    from scipy.optimize import least_squares
    scales = np.maximum(np.abs(parameters), 1e-8)
    train = [case for i, case in enumerate(maneuvers) if i != holdout_index]
    def residual(normalized):
        proposed = normalized*scales
        errors = []
        for case in train:
            observed = np.asarray(case["velocity_uvw_r"], float)
            predicted = np.asarray(simulator(proposed, case), float)
            if (observed.shape != predicted.shape or observed.ndim != 2 or observed.shape[1] != 3 or
                not np.isfinite(observed).all() or not np.isfinite(predicted).all()):
                raise ValueError("Simulator must reproduce u,v,r observations")
            errors.extend((predicted-observed).ravel())
        errors.extend(np.sqrt(ridge_weight)*(normalized-parameters/scales))
        return np.asarray(errors)
    result = least_squares(residual, parameters/scales)
    fitted = result.x*scales
    held = maneuvers[holdout_index]
    prediction = np.asarray(simulator(fitted, held), float)
    observed = np.asarray(held["velocity_uvw_r"], float)
    if prediction.shape != observed.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid held-out simulator trajectory")
    return {"coefficients": fitted.tolist(), "prior": parameters.tolist(),
            "relative_change": ((fitted-parameters)/scales).tolist(),
            "large_change_indices": np.flatnonzero(np.abs(fitted)>2*np.maximum(np.abs(parameters),1e-8)).tolist(),
            "train_residual_norm": float(np.linalg.norm(residual(result.x))),
            "held_out_rmse": float(np.sqrt(np.mean((prediction-observed)**2))),
            "success": bool(result.success), "message": result.message}


def plant6_simulator(plant_factory):
    """Adapt a fixed-mass/actuator Plant6 factory to the output-error API.

    Logs contain explicit timestamps, initial VesselState, observed u/v/r, and
    physical actuation wrenches evaluated by the independently fixed actuator
    model. This adapter does not identify thrust and hull coefficients jointly.
    """
    import torch
    from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS

    def simulate(parameters, maneuver):
        if not maneuver.get("fixed_actuator_model_id"):
            raise ValueError("System ID requires an independently fixed actuator model")
        plant = plant_factory(parameters)
        if plant.maneuvering_surface is None or plant.added_mass_coriolis_enabled:
            raise ValueError("System-ID Plant6 must obey the force-surface load contract")
        state = maneuver["initial_state"]
        times = np.asarray(maneuver["time_s"], float)
        loads = np.asarray(maneuver["actuation_wrench_frd"], float)
        if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0) or
            loads.shape != (len(times), 6) or not np.isfinite(times).all() or
            not np.isfinite(loads).all()):
            raise ValueError("Invalid timestamped fixed-actuator input")
        velocity = [state.nu_body[[0, 1, 5]].detach().cpu().numpy()]
        for i, dt in enumerate(np.diff(times)):
            external = {name: torch.zeros_like(state.nu_body) for name in EXTERNAL_TERMS}
            external["propulsion"] = state.nu_body.new_tensor(loads[i])
            remaining = float(dt)
            while remaining > 1e-12:
                step = min(remaining, plant.envelope.max_substep_s)
                state = plant.step(state, external, step).state
                remaining -= step
            velocity.append(state.nu_body[[0, 1, 5]].detach().cpu().numpy())
        return np.asarray(velocity)
    return simulate
