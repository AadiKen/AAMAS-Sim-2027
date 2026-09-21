import math

import pytest
import torch

from bcod_sim.core.errors import NonFiniteStateError, OperatingEnvelopeError, PhysicalValidationError
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import PlanarEquilibrium, Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.state.tables import VesselTable
from bcod_sim.state.vessel_state import VesselState


DTYPE = torch.float64


def v(values):
    return torch.tensor(values, dtype=DTYPE)


def zero_external():
    return {name: torch.zeros(6, dtype=DTYPE) for name in EXTERNAL_TERMS}


def state(*, p=(0, 0, 0), q=(1, 0, 0, 0), nu=(0, 0, 0, 0, 0, 0)):
    return VesselState(v(p), v(q), v(nu))


def plant(*, mass=10.0, added=None, linear=None, quadratic=None, mode="full6", equilibrium=None, max_nu=None):
    properties = MassProperties(mass, v((0, 0, 0)), torch.diag(v((4, 5, 6))),
                                added if added is not None else torch.zeros((6, 6), dtype=DTYPE))
    damping = Damping(v(linear or (0,)*6), v(quadratic or (0,)*6))
    hydro = Hydrostatics(mass * 9.80665, v((0, 0, 0)))
    envelope = OperatingEnvelope(v(max_nu or (1e6,)*6), max_substep_s=1.0)
    return Plant6(properties, damping, hydro, envelope, mode=mode, planar_equilibrium=equilibrium)


def test_equilibrium_zero_motion_and_balanced_ledger():
    model = plant()
    initial = state()
    result = model.step(initial, zero_external(), 0.02)
    assert result.state.position_ned.tolist() == pytest.approx([0, 0, 0], abs=1e-12)
    assert result.state.nu_body.tolist() == pytest.approx([0]*6, abs=1e-12)
    assert set(result.diagnostics.terms) == set(EXTERNAL_TERMS) | {"rigid_coriolis", "added_mass_coriolis", "linear_damping", "nonlinear_damping", "restoring", "crossflow"}
    result.diagnostics.assert_balanced()


def test_analytic_force_and_damping():
    model = plant(linear=(2, 0, 0, 0, 0, 0))
    initial = state(nu=(3, 0, 0, 0, 0, 0))
    result = initial
    for _ in range(100):
        result = model.step(result, zero_external(), 0.01).state
    assert result.nu_body[0].item() == pytest.approx(3 * math.exp(-0.2), rel=1e-9)
    assert result.position_ned[0].item() == pytest.approx(15 * (1 - math.exp(-0.2)), rel=1e-9)


def test_added_mass_is_in_solved_matrix():
    added = torch.diag(v((10, 0, 0, 0, 0, 0)))
    model = plant(added=added)
    external = zero_external()
    external["propulsion"][0] = 100
    assert model.acceleration(state(), external)[0].item() == pytest.approx(5)
    assert model.total_mass[0, 0].item() == pytest.approx(20)
    assert model.step(state(), external, 0.1).state.nu_body[0].item() == pytest.approx(0.5)


def test_no_damping_energy_conservation():
    model = plant()
    current = state(nu=(1.5, -0.7, 0.2, 0.3, -0.2, 0.4))
    energy0 = 0.5 * current.nu_body @ model.total_mass @ current.nu_body
    for _ in range(100):
        current = model.step(current, zero_external(), 0.005).state
    energy1 = 0.5 * current.nu_body @ model.total_mass @ current.nu_body
    assert energy1.item() == pytest.approx(energy0.item(), rel=1e-9)


def test_quaternion_is_normalized_and_large_correction_is_reported():
    model = plant()
    result = model.step(state(nu=(0, 0, 0, 0, 0, 4)), zero_external(), 0.5)
    assert torch.linalg.vector_norm(result.state.q_body_to_ned).item() == pytest.approx(1.0)
    assert result.normalization_diagnostic is not None
    assert result.normalization_diagnostic.norm_error > result.normalization_diagnostic.threshold


def test_timestep_convergence():
    model = plant(linear=(2, 0, 0, 0, 0, 0))
    def integrate(dt):
        current = state(nu=(3, 0, 0, 0, 0, 0))
        for _ in range(round(1 / dt)):
            current = model.step(current, zero_external(), dt).state
        return current.nu_body[0].item()
    exact = 3 * math.exp(-0.2)
    coarse = abs(integrate(0.2) - exact)
    fine = abs(integrate(0.1) - exact)
    assert fine < coarse / 10


def test_translation_rotation_and_force_mirror_symmetry():
    model = plant()
    east = state(p=(0, 10, 0), q=(math.sqrt(0.5), 0, 0, math.sqrt(0.5)), nu=(1, 0, 0, 0, 0, 0))
    north = state(p=(10, 0, 0), nu=(1, 0, 0, 0, 0, 0))
    east_next = model.step(east, zero_external(), 0.1).state
    north_next = model.step(north, zero_external(), 0.1).state
    assert east_next.position_ned.tolist() == pytest.approx((0, 10.1, 0), abs=1e-12)
    assert north_next.position_ned.tolist() == pytest.approx((10.1, 0, 0), abs=1e-12)
    for sign in (1, -1):
        external = zero_external()
        external["propulsion"][1] = sign * 20
        result = model.step(state(), external, 0.1).state
        assert result.nu_body[1].item() == pytest.approx(sign * 0.2, abs=1e-12)


def test_planar_projection_uses_active_mass_and_fixes_inactive_state():
    added = torch.zeros((6, 6), dtype=DTYPE)
    added[0, 2] = added[2, 0] = 2.0
    model = plant(added=added, mode="planar3", equilibrium=PlanarEquilibrium(1.2, 0, 0.0))
    external = zero_external()
    external["propulsion"] = v((10, 0, 20, 0, 0, 0))
    initial = state(p=(0, 0, 9), nu=(0, 0, 2, 1, 1, 0))
    assert model.acceleration(state(), external)[0].item() == pytest.approx(1.0)
    result = model.step(initial, external, 0.1).state
    assert result.position_ned[2].item() == pytest.approx(1.2)
    assert result.nu_body[[2, 3, 4]].tolist() == [0, 0, 0]
    assert result.nu_body[0].item() == pytest.approx(0.1)


@pytest.mark.parametrize("case", ["negative_mass", "nonpositive_total", "nan_added", "invalid_inertia"])
def test_invalid_mass_rejected(case):
    properties = MassProperties(10, v((0, 0, 0)), torch.diag(v((4, 5, 6))), torch.zeros((6, 6), dtype=DTYPE))
    if case == "negative_mass":
        properties = MassProperties(-1, properties.cg_frd_m, properties.inertia_cg_kg_m2, properties.added_mass_kg)
    elif case == "nonpositive_total":
        properties = MassProperties(10, properties.cg_frd_m, properties.inertia_cg_kg_m2, torch.diag(v((-20, 0, 0, 0, 0, 0))))
    elif case == "nan_added":
        properties = MassProperties(10, properties.cg_frd_m, properties.inertia_cg_kg_m2, torch.full((6, 6), float("nan"), dtype=DTYPE))
    else:
        properties = MassProperties(10, properties.cg_frd_m, torch.diag(v((1, 1, 4))), properties.added_mass_kg)
    with pytest.raises(PhysicalValidationError):
        properties.matrices()


def test_envelope_and_nonfinite_state_rejected():
    model = plant(max_nu=(1,)*6)
    with pytest.raises(OperatingEnvelopeError):
        model.step(state(nu=(2, 0, 0, 0, 0, 0)), zero_external(), 0.1)
    with pytest.raises(NonFiniteStateError):
        state(nu=(float("nan"), 0, 0, 0, 0, 0))


def test_flattened_vessel_table():
    table = VesselTable(torch.tensor([0, 1]), torch.tensor([0, 0]), v(((0, 0, 0), (1, 2, 3))),
                        v(((1, 0, 0, 0), (1, 0, 0, 0))), v(((0,)*6, (0,)*6)),
                        torch.tensor([True, False]), torch.tensor([True, True]), (None, None))
    assert table.position_ned.shape == (2, 3)
