"""Versioned source/solver to BCOD body-frame contract.

Source and OpenFOAM axes are X forward, Y port, Z up. BCOD uses FRD.
The diagonal matrix is a proper 180-degree rotation about X (determinant +1).
"""
from __future__ import annotations

import numpy as np

CONVENTION = "bcod-openfoam-frd-v1"
R = np.diag((1.0, -1.0, -1.0))


def vector_to_foam(body_vector):
    value = np.asarray(body_vector, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("expected finite three-vector")
    return tuple(R @ value)


def vector_to_body(foam_vector):
    return vector_to_foam(foam_vector)


def point_to_foam(body_point, waterline_z_m):
    point = np.asarray(vector_to_foam(body_point))
    if not np.isfinite(waterline_z_m):
        raise ValueError("waterline must be finite")
    point[2] -= waterline_z_m
    return tuple(point)


def body_velocity_to_fixed_hull_inlet(body_velocity):
    return tuple(-np.asarray(vector_to_foam(body_velocity)))


def foam_wrench_to_body(force, moment, *, foam_reference, body_reference,
                        waterline_z_m, resisting=True):
    """Shift a fluid-on-body wrench from A to B, then convert axes.

    r_BA is the vector from B to A. M_B = M_A + r_BA cross F.
    ``resisting`` negates the physical load for the positive-coefficient fitter.
    """
    force = np.asarray(force, float)
    moment = np.asarray(moment, float)
    if force.shape != (3,) or moment.shape != (3,) or not all(
            np.isfinite(x).all() for x in (force, moment)):
        raise ValueError("invalid OpenFOAM wrench")
    a = np.asarray(foam_reference, float)
    b = np.asarray(point_to_foam(body_reference, waterline_z_m))
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("invalid OpenFOAM moment center")
    shifted = moment + np.cross(a - b, force)
    sign = -1 if resisting else 1
    return tuple(sign * R @ force), tuple(sign * R @ shifted)
