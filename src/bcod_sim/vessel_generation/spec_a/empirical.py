"""Ship-like linear-derivative comparison for Spec A.

The equations match Thor I. Fossen's published `clarke83` reference
implementation (PythonVehicleSimulator/lib/models.py, lines 60--70). The 1983
original is not bundled here; applicability is restricted to ship-like hulls.
All derivatives are physical FRD Y/N versus positive FRD v/r.
"""
from __future__ import annotations

import math
from typing import Mapping
import numpy as np


SOURCE = "https://raw.githubusercontent.com/cybergalactic/PythonVehicleSimulator/master/src/python_vehicle_simulator/lib/models.py"


def clarke_linear(length_m: float, beam_m: float, draft_m: float, block_coefficient: float) -> dict[str, float]:
    if (min(length_m, beam_m, draft_m) <= 0 or not 0 < block_coefficient <= 1 or
        not all(math.isfinite(z) for z in (length_m, beam_m, draft_m, block_coefficient))):
        raise ValueError("Invalid ship geometry")
    l, b, t, cb = length_m, beam_m, draft_m, block_coefficient
    s = math.pi*(t/l)**2
    return {"Y_v": -s*(1+.4*cb*b/t),
            "Y_r": -s*(-.5+2.2*b/l-.08*b/t),
            "N_v": -s*(.5+2.4*t/l),
            "N_r": -s*(.25+.039*b/t-.56*b/l)}


def compare_v5(v5: Mapping[str, float], *, length_m: float, beam_m: float,
               draft_m: float, block_coefficient: float,
               max_beta_deg: float, max_abs_r_prime: float,
               hull_kind: str = "displacement_monohull") -> dict:
    empirical = clarke_linear(length_m, beam_m, draft_m, block_coefficient)
    values = {}
    for name, reference in empirical.items():
        actual = float(v5[name])
        if not math.isfinite(actual):
            raise ValueError("Nonfinite V5 derivative")
        values[name] = {"v5": actual, "empirical": reference,
                        "relative_error": abs(actual-reference)/abs(reference),
                        "sign_match": actual*reference > 0}
    ship_like = (hull_kind == "displacement_monohull" and length_m/beam_m >= 4
                 and block_coefficient >= .45)
    narrow_envelope = max_beta_deg <= 10 and max_abs_r_prime <= .3
    use_v5 = ship_like and narrow_envelope and all(
        row["sign_match"] and row["relative_error"] <= .30 for row in values.values())
    return {"reference": SOURCE, "terms": values, "ship_like": ship_like,
            "narrow_envelope": narrow_envelope,
            "decision": "ship_v5" if use_v5 else "cfd_or_sysid",
            "requires_review": True,
            "original_1983_paper_directly_verified": False}


def linear_derivatives(physical_wrench, *, speed_mps: float, length_m: float,
                       density_kg_m3: float = 1025., epsilon: float = 1e-5) -> dict:
    """Differentiate a physical hull-wrench callback (u,v,r)->(X,Y,N).

    Do not pass resisting damping alone or include rigid-body Coriolis. Sway
    dimensional derivative scales with rho L² U; yaw-rate derivative scales
    with rho L³ U (and rho L⁴ U for N_r), due to r'=rL/U.
    """
    if not all(math.isfinite(z) and z > 0 for z in
               (speed_mps, length_m, density_kg_m3, epsilon)):
        raise ValueError("Invalid derivative reference")
    force_scale = .5*density_kg_m3*length_m**2*speed_mps**2
    result = {}
    for axis, key in ((1, "v"), (2, "r")):
        plus = np.array([speed_mps, 0., 0.])
        step = epsilon*speed_mps/(length_m if key == "r" else 1.)
        plus[axis] = step
        minus = plus.copy(); minus[axis] = -step
        gradient = (np.asarray(physical_wrench(*plus), float)-
                    np.asarray(physical_wrench(*minus), float))/(2*epsilon)
        if gradient.shape != (3,) or not np.isfinite(gradient).all():
            raise ValueError("Invalid physical wrench callback")
        result[f"Y_{key}"] = float(gradient[1]/force_scale)
        result[f"N_{key}"] = float(gradient[2]/(force_scale*length_m))
    return result


def v5_wrench_from_payload(payload: dict):
    """Read the existing passive runtime package without regenerating BEM."""
    import torch
    from bcod_sim.dynamics.coriolis import coriolis_wrench
    from bcod_sim.dynamics.crossflow import SectionalCrossflow
    from bcod_sim.dynamics.damping import Damping, CoupledDampingTerm
    from bcod_sim.state.vessel_state import VesselState
    tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    item = payload.get('crossflow', {})
    if item.get('model') != 'sectional_stations':
        raise ValueError('Expected frozen sectional V5 runtime payload')
    flow = SectionalCrossflow.from_stations(item['stations'],
        density=item.get('water_density_kg_m3',1025.),cd_scale=item.get('cd_scale',1.),dtype=torch.float64)
    resistance = payload.get('surge_resistance')
    curve = (tensor(resistance['speed_mps']),tensor(resistance['force_x_n'])) if resistance else None
    matrix = payload.get('linear_damping_matrix')
    damping = Damping(tensor(payload['linear_damping']), tensor(payload['quadratic_damping']),
        tensor(matrix) if matrix is not None else None,
        tuple(CoupledDampingTerm(**row) for row in payload.get('coupled_damping_terms',())),
        surge_resistance_curve=curve)
    added = tensor(payload['added_mass_kg'])
    def physical(u,v,r):
        nu = tensor([u,v,0,0,0,r])
        state = VesselState(tensor([0,0,0]),tensor([1,0,0,0]),nu)
        linear, nonlinear = damping.components(nu)
        total = linear+nonlinear+flow.evaluate(state,None).tau_body-coriolis_wrench(added,nu)
        return total[[0,1,5]].numpy()
    return physical
