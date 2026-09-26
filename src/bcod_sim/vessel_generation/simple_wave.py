"""Guarded Michell thin-ship wave resistance for slender displacement monohulls.

The three-integral form and integration-by-parts offset representation follow
E. O. Tuck, *Wave Resistance of Thin Ships and Catamarans*, §4. It is a
potential-flow wave term, separate from ITTC-1957 skin friction. It does not
estimate viscous pressure, trim, planing, or shallow-water effects.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

from .simple_sections import section_properties


def thin_ship_applicable(*, length_beam: float, block_coefficient: float,
                         max_froude: float, hull_count: int) -> bool:
    """Conservative thin/fine domain; Tuck's compared Taylor hull has L/B=10.1.

    This guard prevents applying first-order thin-ship theory to fuller ship
    forms where bulb, transom, and nonlinear waves can dominate.
    """
    return (hull_count == 1 and length_beam >= 10. and
            0. < block_coefficient <= .55 and max_froude <= .45)


def _width_at_depth(loops: list[np.ndarray], z: float) -> float:
    crossings = []
    for loop in loops:
        points = np.asarray(loop, dtype=float)
        if len(points) > 1 and np.linalg.norm(points[0]-points[-1]) < 1e-9:
            points = points[:-1]
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            za, zb = a[2], b[2]
            if min(za, zb) <= z <= max(za, zb) and abs(zb-za) > 1e-12:
                crossings.append(float(a[1] + (z-za)*(b[1]-a[1])/(zb-za)))
    return 0. if len(crossings) < 2 else max(0., (max(crossings)-min(crossings))/2)


def wave_offset_grid(mesh: trimesh.Trimesh, waterline: float,
                     *, x_intervals: int = 80, z_intervals: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, positive depth, half breadth, including zero-width end stations."""
    if x_intervals < 16 or z_intervals < 8:
        raise ValueError("Thin-ship offset grid is too coarse")
    x = np.linspace(float(mesh.bounds[0, 0]), float(mesh.bounds[1, 0]), x_intervals+1)
    depth = float(mesh.bounds[1, 2]-waterline)
    if depth <= 0:
        raise ValueError("No submerged depth for thin-ship resistance")
    z = np.linspace(0., depth, z_intervals+1)
    width = np.zeros((len(x), len(z)))
    for i in range(1, len(x)-1):
        loops = section_properties(mesh, 0, float(x[i]))["loops"]
        width[i] = [_width_at_depth(loops, float(waterline+d)) for d in z]
    if not np.isfinite(width).all() or np.max(width) <= 0:
        raise ValueError("Thin-ship offsets unavailable")
    return x, z, width


def _vertical_integral(width: np.ndarray, depth: np.ndarray, a: float) -> np.ndarray:
    h = np.diff(depth)
    kh = a*h
    exp0 = np.exp(-a*depth[:-1])
    j0 = exp0 * (-np.expm1(-kh))/a
    j1 = exp0 * (-np.expm1(-kh)-kh*np.exp(-kh))/a**2
    return np.sum(width[:, :-1]*j0 + np.diff(width, axis=1)/h*j1, axis=1)


def _longitudinal_integral(x: np.ndarray, f: np.ndarray, k: float) -> complex:
    h = np.diff(x)
    kh = k*h
    i0 = h*np.exp(0.5j*kh)*np.sinc(kh/(2*np.pi))
    i1 = np.empty_like(i0)
    small = np.abs(kh) < 0.01
    i1[small] = h[small]**2*(.5 + 1j*kh[small]/3 - kh[small]**2/8)
    i1[~small] = (h[~small]*np.exp(1j*kh[~small])-i0[~small])/(1j*k)
    return complex(np.sum(np.exp(1j*k*x[:-1])*(f[:-1]*i0 + np.diff(f)/h*i1)))


def michell_wave_resistance(x: np.ndarray, depth: np.ndarray, half_width: np.ndarray,
                            speed: float, density: float, *, theta_limit: float = 1.52,
                            theta_nodes: int = 64) -> float:
    """Evaluate Rw in newtons; theta truncation is checked against another limit."""
    if speed <= 0 or density <= 0 or not 1.4 <= theta_limit < math.pi/2:
        raise ValueError("Invalid thin-ship wave-resistance condition")
    nodes, weights = np.polynomial.legendre.leggauss(theta_nodes)
    theta = theta_limit*(nodes+1)/2
    total = 0.
    kappa = 9.80665/speed**2
    for angle, weight in zip(theta, weights):
        sec = 1/math.cos(float(angle))
        f = _vertical_integral(half_width, depth, kappa*sec**2)
        pq = _longitudinal_integral(x, f, kappa*sec)
        total += float(weight) * abs(pq)**2 * sec**5
    resistance = 4*density*9.80665**4/(math.pi*speed**6) * theta_limit/2 * total
    if not math.isfinite(resistance) or resistance < 0:
        raise ValueError("Invalid thin-ship wave resistance")
    return resistance


def guarded_wave_curve(mesh: trimesh.Trimesh, waterline: float, speeds: np.ndarray,
                       density: float) -> dict:
    x, z, width = wave_offset_grid(mesh, waterline)
    values, checks = [], []
    cache = {}
    for u in speeds:
        speed = abs(float(u))
        if speed < 1e-8:
            values.append(0.)
            continue
        if speed not in cache:
            cache[speed] = (michell_wave_resistance(x, z, width, speed, density),
                            michell_wave_resistance(x, z, width, speed, density, theta_limit=1.50))
        main, check = cache[speed]
        change = abs(main-check)/max(main, check, 1e-12)
        if change > .15:
            raise ValueError(f"Thin-ship angular truncation sensitivity {change:.1%}")
        values.append(main)
        checks.append(change)
    return {"wave_resistance_n": values, "maximum_theta_sensitivity": max(checks, default=0.),
            "method": "michell_thin_ship_v1", "source": "Tuck, Wave Resistance of Thin Ships and Catamarans, section 4",
            "applicability": "slender displacement monohull only; deep calm water; fixed trim",
            "confidence": "low", "grid": {"x_intervals": 80, "z_intervals": 32, "theta_nodes": 64}}
