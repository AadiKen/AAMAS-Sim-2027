"""Keep HMRI hydrodynamic loads separate from BCOD resisting loads."""
from __future__ import annotations

from dataclasses import dataclass

from .frame_contract import foam_wrench_to_body


@dataclass(frozen=True)
class PhysicalFluidWrench:
    force_frd_n: tuple[float, float, float]
    moment_frd_nm: tuple[float, float, float]


@dataclass(frozen=True)
class BCODResistingWrench:
    force_frd_n: tuple[float, float, float]
    moment_frd_nm: tuple[float, float, float]


def separate_openfoam_wrench(force, moment, *, foam_reference, body_reference,
                             waterline_z_m) -> tuple[PhysicalFluidWrench, BCODResistingWrench]:
    """Convert a fluid-on-hull OpenFOAM sample into explicitly tagged forms."""
    kwargs = dict(foam_reference=foam_reference, body_reference=body_reference,
                  waterline_z_m=waterline_z_m)
    physical = foam_wrench_to_body(force, moment, resisting=False, **kwargs)
    resisting = foam_wrench_to_body(force, moment, resisting=True, **kwargs)
    return PhysicalFluidWrench(*physical), BCODResistingWrench(*resisting)


def hmri_y_n_error(observed: PhysicalFluidWrench, *, target_y_n: float,
                   target_n_nm: float) -> tuple[float, float]:
    """HMRI EFD coefficients describe physical loads, never resisting loads."""
    if not isinstance(observed, PhysicalFluidWrench):
        raise TypeError("HMRI comparison requires a PhysicalFluidWrench")
    return (observed.force_frd_n[1] - target_y_n,
            observed.moment_frd_nm[2] - target_n_nm)
