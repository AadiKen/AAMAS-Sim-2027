"""Ship central-plane horseshoe vortices and finite-segment Biot--Savart.

The longitudinal positions and weights are Ayubi (2021), eqs. 4.28, 4.34.
Wake topologies follow thesis Fig. 4.10. This component is experimental and
requires a coupled source/vortex qualification before any benchmark use.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def segment_velocity(points: np.ndarray, start: np.ndarray, end: np.ndarray,
                     *, core_radius: float = 0.) -> np.ndarray:
    """Finite-segment induced velocity for unit circulation, BCOD FRD."""
    points = np.asarray(points, dtype=float)
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    r1, r2 = points-start, points-end
    line = end-start
    cross = np.cross(r1, r2)
    line2 = float(line @ line)
    if line2 <= 0 or core_radius < 0:
        raise ValueError("Invalid vortex segment/core")
    denominator = np.sum(cross*cross, axis=-1) + core_radius**2*line2
    magnitude = np.sum(line * (r1/np.maximum(np.linalg.norm(r1, axis=-1, keepdims=True), 1e-15)
                               - r2/np.maximum(np.linalg.norm(r2, axis=-1, keepdims=True), 1e-15)), axis=-1)
    return cross*(magnitude/(4*math.pi*np.maximum(denominator, 1e-30)))[..., None]


def qcm_longitudinal_positions(x_stern: float, x_bow: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev positions and Lan quadrature weights, ordered stern to bow."""
    if count < 2 or x_stern >= x_bow:
        raise ValueError("Invalid QCM chord")
    nu = np.arange(1, count+1, dtype=float)
    phase = (2*nu-1)*math.pi/(2*count)
    # Thesis x is aft-positive; BCOD x is forward-positive.
    x = x_stern + (x_bow-x_stern)*(1-np.cos(phase))/2
    weights = (math.pi*(x_bow-x_stern)/(2*count))*np.sin(2*phase)
    return x, weights


def qcm_control_positions(x_stern: float, x_bow: float, count: int) -> np.ndarray:
    """Lan-style interleaved collocation downstream of each BCOD bound segment."""
    if count < 2 or x_stern >= x_bow:
        raise ValueError("Invalid QCM chord")
    nu = np.arange(count, dtype=float)
    return x_stern + (x_bow-x_stern)*(1-np.cos(nu*math.pi/count))/2


@dataclass(frozen=True)
class Horseshoe:
    bound_start: np.ndarray
    bound_end: np.ndarray
    leg_start: np.ndarray  # nodes from bound start to downstream end
    leg_end: np.ndarray    # nodes from bound end to downstream end
    weight: float
    hull_id: int
    model: int
    leg_start_free_index: int
    leg_end_free_index: int

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        # A far end -> bound A -> bound B -> B far end is one closed horseshoe
        starts = np.concatenate((self.leg_start[:0:-1], self.bound_start[None], self.leg_end[:-1]), axis=0)
        ends = np.concatenate((self.leg_start[-2::-1], self.bound_end[None], self.leg_end[1:]), axis=0)
        return starts, ends


def make_horseshoes(*, x_stern: float, x_bow: float, z_waterline: float,
                    z_keel: float, y_center: float = 0., hull_id: int = 0,
                    longitudinal: int = 30, vertical: int = 5,
                    wake_elements: int = 50, wake_length: float | None = None,
                    drift_angle: float = 0., model: int = 1) -> list[Horseshoe]:
    """Two published initial wake topologies; x-forward means wake toward -x."""
    if model not in (1, 2) or vertical < 2 or wake_elements < 1 or z_keel <= z_waterline:
        raise ValueError("Invalid ship horseshoe configuration")
    length = x_bow-x_stern
    wake_length = length*3 if wake_length is None else wake_length
    x, weights = qcm_longitudinal_positions(x_stern, x_bow, longitudinal)
    z = np.linspace(z_waterline, z_keel, vertical+1)
    horseshoes = []
    for j in range(vertical):
        for i, xi in enumerate(x):
            a = np.array([xi, y_center, z[j]])
            b = np.array([xi, y_center, z[j+1]])
            legs = []
            free_indices=[]
            for endpoint, bottom in ((a, False), (b, j == vertical-1)):
                start = endpoint.copy()
                # Model 2 releases the lower leg of a bottom-row horsehoe
                # directly from the keel; all other legs reach the stern.
                if not (model == 2 and bottom):
                    start[0] = x_stern
                direction = np.array([-math.cos(drift_angle), math.sin(drift_angle), 0.])
                downstream = start[None] + np.linspace(0., wake_length, wake_elements+1)[:, None]*direction
                if np.linalg.norm(start-endpoint)>1e-12:
                    # Bound-to-stern attachment is fixed; free nodes begin
                    # *after* the stern. It is not one giant free segment.
                    legs.append(np.vstack((endpoint,downstream)))
                    free_indices.append(2)
                else:
                    legs.append(downstream)
                    free_indices.append(1)
            horseshoes.append(Horseshoe(a, b, legs[0], legs[1], float(weights[i]),
                                         hull_id, model, *free_indices))
    return horseshoes


def horseshoe_velocity(points: np.ndarray, horseshoes: list[Horseshoe],
                       *, core_radius: float = 0.) -> np.ndarray:
    """Induced velocity at P points per unit QCM strength, shape P×N×3."""
    points = np.asarray(points, dtype=float)
    result = np.zeros((len(points), len(horseshoes), 3))
    for j, horse in enumerate(horseshoes):
        starts, ends = horse.segments()
        value = np.zeros((len(points), 3))
        for a, b in zip(starts, ends):
            value += segment_velocity(points, a, b, core_radius=core_radius)
        result[:, j] = horse.weight*value
    return result
