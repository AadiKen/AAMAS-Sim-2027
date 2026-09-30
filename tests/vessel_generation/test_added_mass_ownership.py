import copy
from pathlib import Path

import pytest
import torch
import yaml

from bcod_sim.dynamics.coriolis import coriolis_wrench
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.vessel_generation.coefficient_package import _canonical_hash, _runtime_plant, load_coefficient_package


ROOT = Path(__file__).resolve().parents[2]
REV4_PACKAGE = ROOT / "docs/passive_model_revision_4/resistance/kvlcc2m/package_frozen/coefficient_package.yaml"


def ownership_package(tmp_path, interpretation, owner):
    package = copy.deepcopy(yaml.safe_load(REV4_PACKAGE.read_text()))
    package.pop("canonical_sha256")
    package["maneuvering"]["interpretation"] = interpretation
    package["added_mass"]["inertia_owner"] = "bem"
    package["added_mass"]["steady_coriolis_owner"] = owner
    runtime = package["runtime_payload"]
    runtime["maneuvering_interpretation"] = interpretation
    runtime["steady_coriolis_owner"] = owner
    runtime["added_mass_coriolis_enabled"] = owner == "bem"
    package["canonical_sha256"] = _canonical_hash(package)
    path = tmp_path / "coefficient_package.yaml"
    path.write_text(yaml.safe_dump(package, sort_keys=True))
    return path


def state(nu):
    return VesselState(torch.zeros(3, dtype=torch.float64), torch.tensor([1., 0., 0., 0.], dtype=torch.float64),
                       torch.tensor(nu, dtype=torch.float64))


def test_total_steady_load_mode_keeps_added_mass_inertia_but_disables_steady_coriolis(tmp_path):
    pkg = load_coefficient_package(ownership_package(tmp_path, "total_steady_hull_load", "maneuvering_model"))
    plant = _runtime_plant(pkg)
    assert plant.added_mass_coriolis_enabled is False
    assert torch.count_nonzero(plant.added_mass).item() > 0
    assert torch.allclose(plant.total_mass, plant.rigid_mass + plant.added_mass)

    external = {name: torch.zeros(6, dtype=torch.float64) for name in EXTERNAL_TERMS}
    external["propulsion"][0] = 100.
    expected = 100. * torch.linalg.inv(plant.total_mass)[0, 0]
    assert plant.acceleration(state([0., 0., 0., 0., 0., 0.]), external)[0].item() == pytest.approx(expected.item())

    moving = plant.diagnostics(state([0.994, -0.1039013, 0., 0., 0., 0.]),
                               {name: torch.zeros(6, dtype=torch.float64) for name in EXTERNAL_TERMS})
    assert torch.count_nonzero(moving.terms["added_mass_coriolis"]).item() == 0


def test_residual_viscous_mode_explicitly_uses_bem_coriolis(tmp_path):
    pkg = load_coefficient_package(ownership_package(tmp_path, "residual_viscous", "bem"))
    plant = _runtime_plant(pkg)
    assert plant.added_mass_coriolis_enabled is True
    moving = state([0.994, -0.1039013, 0., 0., 0., 0.])
    ledger = plant.diagnostics(moving,
                               {name: torch.zeros(6, dtype=torch.float64) for name in EXTERNAL_TERMS})
    expected = -coriolis_wrench(plant.added_mass, moving.nu_body)
    assert torch.allclose(ledger.terms["added_mass_coriolis"], expected)


@pytest.mark.parametrize(("interpretation", "owner"), [
    ("total_steady_hull_load", "bem"),
    ("residual_viscous", "maneuvering_model"),
])
def test_conflicting_ownership_fails_at_package_load(tmp_path, interpretation, owner):
    path = ownership_package(tmp_path, interpretation, owner)
    # Rehash the intentionally conflicting package so the ownership validator,
    # rather than the integrity check, is what rejects it.
    package = yaml.safe_load(path.read_text())
    package.pop("canonical_sha256")
    package["runtime_payload"]["steady_coriolis_owner"] = owner
    package["runtime_payload"]["maneuvering_interpretation"] = interpretation
    package["runtime_payload"]["added_mass_coriolis_enabled"] = owner == "bem"
    package["canonical_sha256"] = _canonical_hash(package)
    path.write_text(yaml.safe_dump(package, sort_keys=True))
    with pytest.raises(ValueError, match="ownership|Coriolis"):
        load_coefficient_package(path)
