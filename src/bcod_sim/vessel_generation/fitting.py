"""Fail-closed diagonal 6-DOF hydrodynamic coefficient identification."""

from dataclasses import dataclass
import math

import numpy as np

from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties


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


@dataclass(frozen=True)
class CoupledTermSpec:
    output_axis: int
    factors: tuple[int, ...]
    absolute_factors: tuple[int, ...] = ()


@dataclass(frozen=True)
class Plant6ModelForm:
    """Capabilities discovered from the production model rather than invented by CFD."""
    added_mass_matrix: bool
    linear_damping_matrix: bool
    diagonal_quadratic: bool
    coupled_terms: tuple[CoupledTermSpec, ...] = ()

    @classmethod
    def discover(cls, coupled_terms=()) -> "Plant6ModelForm":
        mass_fields=MassProperties.__dataclass_fields__; damping_fields=Damping.__dataclass_fields__
        return cls("added_mass_kg" in mass_fields,"linear_matrix" in damping_fields,
            "quadratic" in damping_fields,tuple(coupled_terms) if "coupled_terms" in damping_fields else ())


@dataclass(frozen=True)
class CoefficientEstimate:
    name: str
    value: float
    units: str
    uncertainty: float
    fit_quality_r2: float
    source_cases: tuple[str, ...]


@dataclass(frozen=True)
class GeneralFitResult:
    added_mass: tuple[tuple[float, ...], ...]
    linear_damping: tuple[tuple[float, ...], ...]
    quadratic_damping: tuple[float, ...]
    coupled_damping: tuple[float, ...]
    estimates: tuple[CoefficientEstimate, ...]
    rank: tuple[int, ...]
    condition_numbers: tuple[float, ...]
    residual_rms: tuple[float, ...]


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

    def fit_plant6(self, dataset: ForceMomentDataset, *, model_form: Plant6ModelForm | None=None,
                   source_cases: tuple[str,...]=()) -> GeneralFitResult:
        """Fit each output wrench using every Plant6-supported input basis.

        The sign convention matches ``ForceMomentDataset``: observations are the
        positive inertial/resisting wrench magnitudes consumed as coefficients.
        """
        dataset.validate(); form=model_form or Plant6ModelForm.discover()
        n=len(dataset.velocity); added=np.zeros((6,6)); linear=np.zeros((6,6)); quadratic=np.zeros(6)
        coupled=np.zeros(len(form.coupled_terms)); estimates=[]; ranks=[]; conditions=[]; residuals=[]
        for output in range(6):
            columns=[]; names=[]; units=[]
            if form.added_mass_matrix:
                for axis in range(6):
                    columns.append(dataset.acceleration[:,axis]); names.append(("added",axis));
                    units.append("kg" if output<3 and axis<3 else "kg*m^2" if output>=3 and axis>=3 else "kg*m")
            if form.linear_damping_matrix:
                for axis in range(6):
                    columns.append(dataset.velocity[:,axis]); names.append(("linear",axis)); units.append("N/(m/s)" if output<3 else "N*m/(rad/s)")
            if form.diagonal_quadratic:
                columns.append(np.abs(dataset.velocity[:,output])*dataset.velocity[:,output]); names.append(("quadratic",output)); units.append("N/(m/s)^2" if output<3 else "N*m/(rad/s)^2")
            for index,term in enumerate(form.coupled_terms):
                if term.output_axis != output: continue
                value=np.ones(n)
                for axis in term.factors: value*=dataset.velocity[:,axis]
                for axis in term.absolute_factors: value*=np.abs(dataset.velocity[:,axis])
                columns.append(value); names.append(("coupled",index)); units.append("derived SI")
            design=np.column_stack(columns); rank=int(np.linalg.matrix_rank(design)); condition=float(np.linalg.cond(design))
            if rank<len(columns) or not math.isfinite(condition) or condition>self.max_condition:
                raise CalibrationError(f"output axis {output} coefficients are not identifiable from supplied maneuvers")
            beta=np.linalg.lstsq(design,dataset.wrench[:,output],rcond=None)[0]
            residual=dataset.wrench[:,output]-design@beta; dof=max(1,n-len(columns)); mse=float(residual@residual/dof)
            covariance=mse*np.linalg.pinv(design.T@design); stderr=np.sqrt(np.maximum(0,np.diag(covariance)))
            total=float(np.sum((dataset.wrench[:,output]-dataset.wrench[:,output].mean())**2))
            r2=1-float(residual@residual)/total if total>0 else float(residual@residual)==0
            for value,error,name,unit in zip(beta,stderr,names,units):
                kind,index=name
                if kind=="added": added[output,index]=value
                elif kind=="linear": linear[output,index]=value
                elif kind=="quadratic": quadratic[output]=value
                else: coupled[index]=value
                estimates.append(CoefficientEstimate(f"{kind}[{output},{index}]",float(value),unit,float(error),float(r2),source_cases))
            ranks.append(rank);conditions.append(condition);residuals.append(float(np.sqrt(np.mean(residual**2))))
        # Plant6 requires symmetric positive-semidefinite added mass. CFD noise can
        # break reciprocity slightly, so symmetrize but fail rather than clip an invalid model.
        added=(added+added.T)/2
        if np.linalg.eigvalsh(added).min() < -1e-9: raise CalibrationError("fitted added-mass matrix is not positive semidefinite")
        if np.linalg.eigvalsh((linear+linear.T)/2).min() < -1e-9: raise CalibrationError("fitted linear damping is not dissipative")
        if (quadratic < -1e-9).any(): raise CalibrationError("fitted quadratic damping is not dissipative")
        return GeneralFitResult(tuple(map(tuple,added)),tuple(map(tuple,linear)),tuple(map(float,quadratic)),
            tuple(map(float,coupled)),tuple(estimates),tuple(ranks),tuple(conditions),tuple(residuals))


def synthetic_dataset(added_mass, linear, quadratic, *, samples: int=96, seed: int=0,
                      noise_std: float=0, active_axes: tuple[int,...]=tuple(range(6))) -> ForceMomentDataset:
    rng=np.random.default_rng(seed); velocity=np.zeros((samples,6)); acceleration=np.zeros((samples,6))
    for axis in active_axes:
        velocity[:,axis]=rng.uniform(-2,2,samples); acceleration[:,axis]=rng.uniform(-1,1,samples)
    wrench=acceleration*np.asarray(added_mass)+velocity*np.asarray(linear)+np.abs(velocity)*velocity*np.asarray(quadratic)
    if noise_std: wrench += rng.normal(0,noise_std,wrench.shape)
    return ForceMomentDataset(velocity,acceleration,wrench,seed=seed)
