"""Fail-closed diagonal 6-DOF hydrodynamic coefficient identification."""

from dataclasses import dataclass
import math

import numpy as np


class CalibrationError(ValueError): pass


@dataclass(frozen=True)
class ForceMomentDataset:
    velocity: np.ndarray       # [samples, 6]
    acceleration: np.ndarray   # [samples, 6]
    wrench: np.ndarray         # positive resisting force magnitude [samples, 6]
    units: tuple[str, str, str] = ("m/s,rad/s", "m/s2,rad/s2", "N,Nm")
    seed: int = 0

    def validate(self) -> None:
        if self.velocity.shape != self.acceleration.shape or self.velocity.shape != self.wrench.shape:
            raise CalibrationError("velocity, acceleration, and wrench shapes must match")
        if self.velocity.ndim != 2 or self.velocity.shape[1] != 6 or self.velocity.shape[0] < 3:
            raise CalibrationError("6-DOF observations require at least three samples")
        if not self.units or any(not unit for unit in self.units): raise CalibrationError("units are required")
        if not all(np.isfinite(x).all() for x in (self.velocity, self.acceleration, self.wrench)):
            raise CalibrationError("observations must be finite")


@dataclass(frozen=True)
class FitResult:
    added_mass: tuple[float, ...]
    linear_damping: tuple[float, ...]
    quadratic_damping: tuple[float, ...]
    condition_numbers: tuple[float, ...]
    residual_rms: tuple[float, ...]
    standard_errors: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class DampingFitResult:
    axis: int
    linear_damping: float
    quadratic_damping: float
    rank: int
    condition_number: float
    residual_rms: float
    design_matrix: tuple[tuple[float, float], ...]


class CoefficientFitter:
    def __init__(self, *, max_condition: float = 1e8, bounds: tuple[float, float] = (0, math.inf)):
        self.max_condition, self.bounds = max_condition, bounds

    def fit(self, dataset: ForceMomentDataset, *, axes: tuple[int, ...] = tuple(range(6))) -> FitResult:
        dataset.validate(); results = [None] * 6; conditions=[]; residuals=[]; errors=[]
        for axis in range(6):
            x = np.column_stack((dataset.acceleration[:,axis], dataset.velocity[:,axis],
                                 np.abs(dataset.velocity[:,axis])*dataset.velocity[:,axis]))
            if axis not in axes:
                results[axis] = (0.,0.,0.); conditions.append(1.); residuals.append(0.); errors.append((0.,0.,0.)); continue
            rank = np.linalg.matrix_rank(x); condition = float(np.linalg.cond(x))
            if rank < 3 or not math.isfinite(condition) or condition > self.max_condition:
                raise CalibrationError("requested parameters are not identifiable from supplied maneuvers")
            beta, _, _, _ = np.linalg.lstsq(x, dataset.wrench[:,axis], rcond=None)
            lo, hi = self.bounds
            if (beta < lo).any() or (beta > hi).any():
                raise CalibrationError("coefficient fit lies outside declared bounds")
            residual = dataset.wrench[:,axis]-x@beta; dof=max(1,len(x)-3)
            covariance = (residual@residual/dof)*np.linalg.inv(x.T@x)
            results[axis]=tuple(float(v) for v in beta); conditions.append(condition)
            residuals.append(float(np.sqrt(np.mean(residual**2)))); errors.append(tuple(np.sqrt(np.diag(covariance))))
        return FitResult(tuple(r[0] for r in results), tuple(r[1] for r in results), tuple(r[2] for r in results),
                         tuple(conditions), tuple(residuals), tuple(tuple(float(v) for v in e) for e in errors))

    def fit_damping_axis(self, dataset: ForceMomentDataset, *, axis: int) -> DampingFitResult:
        dataset.validate()
        if axis not in range(6): raise CalibrationError("axis must be in [0, 5]")
        velocity=dataset.velocity[:,axis]
        design=np.column_stack((velocity,np.abs(velocity)*velocity))
        rank=int(np.linalg.matrix_rank(design)); condition=float(np.linalg.cond(design))
        if rank<2 or not math.isfinite(condition) or condition>self.max_condition:
            raise CalibrationError("requested parameters are not identifiable from supplied maneuvers")
        values=np.linalg.lstsq(design,dataset.wrench[:,axis],rcond=None)[0]
        if (values<self.bounds[0]).any() or (values>self.bounds[1]).any():
            raise CalibrationError("coefficient fit lies outside declared bounds")
        residual=dataset.wrench[:,axis]-design@values
        return DampingFitResult(axis,float(values[0]),float(values[1]),rank,condition,
            float(np.sqrt(np.mean(residual**2))),tuple(tuple(float(x) for x in row) for row in design))


def synthetic_dataset(added_mass, linear, quadratic, *, samples: int=96, seed: int=0,
                      noise_std: float=0, active_axes: tuple[int,...]=tuple(range(6))) -> ForceMomentDataset:
    rng=np.random.default_rng(seed); velocity=np.zeros((samples,6)); acceleration=np.zeros((samples,6))
    for axis in active_axes:
        velocity[:,axis]=rng.uniform(-2,2,samples); acceleration[:,axis]=rng.uniform(-1,1,samples)
    wrench=acceleration*np.asarray(added_mass)+velocity*np.asarray(linear)+np.abs(velocity)*velocity*np.asarray(quadratic)
    if noise_std: wrench += rng.normal(0,noise_std,wrench.shape)
    return ForceMomentDataset(velocity,acceleration,wrench,seed=seed)
