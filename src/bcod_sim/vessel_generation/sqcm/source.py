"""Constant-strength quadrilateral source panels for SQCM qualification.

Hess--Smith source velocity is the surface integral of r/(4*pi*|r|**3).
The exterior collocation limit contributes +1/2 in the panel's normal
direction. This module deliberately does not call a source-only result SQCM.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import lu_factor, lu_solve


@dataclass(frozen=True)
class SourcePanels:
    corners: np.ndarray  # (panel, 4, xyz), consecutive perimeter order
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray

    @classmethod
    def from_corners(cls, corners: np.ndarray) -> "SourcePanels":
        corners = np.asarray(corners, dtype=np.float64)
        if corners.ndim != 3 or corners.shape[1:] != (4, 3) or not np.isfinite(corners).all():
            raise ValueError("Expected finite (N,4,3) quadrilateral corners")
        a, b, c, d = np.moveaxis(corners, 1, 0)
        vector_area = (np.cross(b-a, c-a) + np.cross(c-a, d-a))/2
        areas = np.linalg.norm(vector_area, axis=1)
        if np.min(areas) <= 0:
            raise ValueError("Degenerate source panel")
        normals = vector_area/areas[:, None]
        centers = np.mean(corners, axis=1)
        return cls(corners, centers, normals, areas)


def quadrature(panels: SourcePanels, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear quadrilateral Gauss points and physical area weights."""
    abscissa, weights = np.polynomial.legendre.leggauss(order)
    q, w = [], []
    a, b, c, d = np.moveaxis(panels.corners, 1, 0)
    for xi, wx in zip(abscissa, weights):
        for eta, wy in zip(abscissa, weights):
            point = ((1-xi)*(1-eta)*a + (1+xi)*(1-eta)*b +
                     (1+xi)*(1+eta)*c + (1-xi)*(1+eta)*d)/4
            dx = (-(1-eta)*a + (1-eta)*b + (1+eta)*c - (1+eta)*d)/4
            dy = (-(1-xi)*a - (1+xi)*b + (1+xi)*c + (1-xi)*d)/4
            jacobian = np.linalg.norm(np.cross(dx, dy), axis=1)
            q.append(point)
            w.append(wx*wy*jacobian)
    return np.stack(q, axis=1), np.stack(w, axis=1)


def source_velocity(points: np.ndarray, panels: SourcePanels, *, order: int = 4,
                    surface_self: bool = False, batch: int = 32) -> np.ndarray:
    """Velocity at targets per unit panel source strength, shape (P,N,3)."""
    points = np.asarray(points, dtype=np.float64)
    q, w = quadrature(panels, order)
    n = len(panels.corners)
    result = np.zeros((len(points), n, 3))
    for start in range(0, len(points), batch):
        stop = min(start+batch, len(points))
        r = points[start:stop, None, None, :] - q[None, :, :, :]
        r2 = np.sum(r*r, axis=-1)
        kernel = r / (4*math.pi*np.maximum(r2, 1e-28)[..., None]**1.5)
        result[start:stop] = np.einsum("pnqk,nq->pnk", kernel, w)
    if surface_self:
        if len(points) != n or not np.allclose(points, panels.centers, atol=1e-12):
            raise ValueError("Self limit only defined at matching panel collocation points")
        result[np.arange(n), np.arange(n)] = .5*panels.normals
    return result


def solve_source(panels: SourcePanels, flow: np.ndarray, *, order: int = 4,
                 condition_limit: float = 1e9) -> dict:
    """Solve the non-lifting source boundary condition and retain diagnostics."""
    flow = np.asarray(flow, dtype=np.float64)
    if flow.shape != (3,) or not np.isfinite(flow).all():
        raise ValueError("Flow must be a finite 3-vector")
    influence = source_velocity(panels.centers, panels, order=order, surface_self=True)
    matrix = np.einsum("ijk,ik->ij", influence, panels.normals)
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > condition_limit:
        raise ValueError(f"Ill-conditioned source matrix: {condition:g}")
    strength = lu_solve(lu_factor(matrix), -panels.normals @ flow)
    velocity = flow + np.einsum("ijk,j->ik", influence, strength)
    residual = np.einsum("ik,ik->i", velocity, panels.normals)
    speed2 = float(flow @ flow)
    cp = np.ones(len(panels.corners)) - np.sum(velocity*velocity, axis=1)/speed2 if speed2 else None
    return {"strength": strength, "surface_velocity": velocity, "cp": cp,
            "normal_residual": residual, "condition_number": condition,
            "influence": influence}


def cube_sphere_quads(radius: float, divisions: int) -> SourcePanels:
    """Deterministic six-face cubed sphere for source-panel analytic tests."""
    if radius <= 0 or divisions < 2:
        raise ValueError("Invalid sphere resolution")
    panels = []
    grid = np.linspace(-1., 1., divisions+1)
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for sign in (-1., 1.):
            for i in range(divisions):
                for j in range(divisions):
                    coords = [(grid[i], grid[j]), (grid[i+1], grid[j]),
                              (grid[i+1], grid[j+1]), (grid[i], grid[j+1])]
                    quad = np.zeros((4, 3))
                    quad[:, axis] = sign
                    quad[:, others[0]] = [v[0] for v in coords]
                    quad[:, others[1]] = [v[1] for v in coords]
                    quad *= radius/np.linalg.norm(quad, axis=1)[:, None]
                    if np.dot(np.cross(quad[1]-quad[0], quad[2]-quad[0]), quad.mean(axis=0)) < 0:
                        quad = quad[::-1]
                    panels.append(quad)
    return SourcePanels.from_corners(np.asarray(panels))
