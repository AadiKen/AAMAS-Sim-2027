"""Synthetic short-run calibration with maneuver-family holdout."""

from dataclasses import dataclass
import numpy as np

from .fitting import CalibrationError
from .models import CanonicalVessel, ParameterLineage


@dataclass(frozen=True)
class CalibrationLog:
    family: str
    command_wrench: np.ndarray  # [steps, 6]
    velocity: np.ndarray        # [steps + 1, 6]
    dt_s: float

    def validate(self):
        if self.command_wrench.ndim != 2 or self.command_wrench.shape[1] != 6 or self.velocity.shape != (len(self.command_wrench)+1,6):
            raise CalibrationError("calibration command/state log shape mismatch")
        if self.dt_s <= 0 or not np.isfinite(self.command_wrench).all() or not np.isfinite(self.velocity).all():
            raise CalibrationError("calibration log must be finite with positive timestep")


def simulate_maneuver(family: str, *, mass: float, added_mass, linear, quadratic, dt=.05, steps=160) -> CalibrationLog:
    patterns={
        "straight_acceleration":lambda t: np.array([80 if t<steps*.6 else 0,0,0,0,0,0]),
        "coast_down":lambda t: np.array([100 if t<steps*.2 else 0,0,0,0,0,0]),
        "constant_thrust":lambda t: np.array([55,0,0,0,0,0]),
        "differential_thrust":lambda t: np.array([25,0,0,0,0,18*np.sin(t*.13)]),
        "turning":lambda t: np.array([45,0,0,0,0,20]),
        "combined_excitation":lambda t: np.array([55+20*np.sin(t*.17),15*np.cos(t*.11),0,0,0,22*np.sin(t*.07)])}
    if family not in patterns: raise CalibrationError("unknown maneuver family")
    am=np.asarray(added_mass,float); lin=np.asarray(linear,float); quad=np.asarray(quadratic,float)
    effective=am+mass; velocity=np.zeros((steps+1,6)); command=np.zeros((steps,6))
    for t in range(steps):
        command[t]=patterns[family](t)
        acceleration=(command[t]-lin*velocity[t]-quad*np.abs(velocity[t])*velocity[t])/effective
        velocity[t+1]=velocity[t]+dt*acceleration
    return CalibrationLog(family,command,velocity,dt)


@dataclass(frozen=True)
class CalibrationResult:
    vessel: CanonicalVessel
    train_families: tuple[str,...]
    held_out_family: str
    pre_rmse: float
    post_rmse: float
    recovered_added_mass: tuple[float,...]
    recovered_linear: tuple[float,...]
    recovered_quadratic: tuple[float,...]


def _fit_logs(logs: tuple[CalibrationLog,...], initial: CanonicalVessel):
    mass=initial.mass_kg
    matrices=[]; targets=[]
    for log in logs:
        log.validate(); v=log.velocity[:-1]; acceleration=np.diff(log.velocity,axis=0)/log.dt_s
        for axis in range(6):
            x=np.column_stack((acceleration[:,axis],v[:,axis],np.abs(v[:,axis])*v[:,axis]))
            active=np.linalg.norm(x,axis=0)>1e-10
            if active.sum() and np.linalg.matrix_rank(x[:,active])<active.sum():
                raise CalibrationError("requested parameters are not identifiable from supplied maneuvers")
        matrices.append((v,acceleration)); targets.append(log.command_wrench)
    values=[]
    v=np.concatenate([x[0] for x in matrices]); a=np.concatenate([x[1] for x in matrices]); y=np.concatenate(targets)
    initial_added=np.diag(initial.added_mass_kg)
    for axis in range(6):
        x=np.column_stack((a[:,axis],v[:,axis],np.abs(v[:,axis])*v[:,axis])); active=np.linalg.norm(x,axis=0)>1e-10
        beta=np.array((mass+initial_added[axis],initial.linear_damping[axis],initial.quadratic_damping[axis]),float)
        if active.any():
            xa=x[:,active]
            if np.linalg.matrix_rank(xa)<active.sum(): raise CalibrationError("requested parameters are not identifiable from supplied maneuvers")
            beta[active]=np.linalg.lstsq(xa,y[:,axis],rcond=None)[0]
        values.append(beta)
    values=np.asarray(values)
    if (values< -1e-8).any(): raise CalibrationError("calibration produced physically invalid coefficients")
    return tuple(values[:,0]-mass),tuple(values[:,1]),tuple(values[:,2])


def _rmse(log: CalibrationLog, mass, added, linear, quadratic):
    predicted=simulate_commands(log.command_wrench,mass,added,linear,quadratic,log.dt_s)
    return float(np.sqrt(np.mean((predicted-log.velocity)**2)))


def simulate_commands(commands,mass,added,linear,quadratic,dt):
    commands=np.asarray(commands); v=np.zeros((len(commands)+1,6)); effective=mass+np.asarray(added)
    for t in range(len(commands)):
        v[t+1]=v[t]+dt*(commands[t]-np.asarray(linear)*v[t]-np.asarray(quadratic)*np.abs(v[t])*v[t])/effective
    return v


def calibrate(initial: CanonicalVessel, train: tuple[CalibrationLog,...], held_out: CalibrationLog,
              *, run_id: str) -> CalibrationResult:
    if held_out.family in {x.family for x in train}: raise CalibrationError("held-out maneuver must use a separate family")
    added,linear,quadratic=_fit_logs(train,initial)
    before=_rmse(held_out,initial.mass_kg,np.diag(initial.added_mass_kg),initial.linear_damping,initial.quadratic_damping)
    after=_rmse(held_out,initial.mass_kg,added,linear,quadratic)
    provenance=dict(initial.provenance)
    for name,old,new in (("added_mass_kg",float(np.mean(np.diag(initial.added_mass_kg))),float(np.mean(added))),
                         ("linear_damping",float(np.mean(initial.linear_damping)),float(np.mean(linear))),
                         ("quadratic_damping",float(np.mean(initial.quadratic_damping)),float(np.mean(quadratic)))):
        provenance[name]=ParameterLineage(source_kind="calibration-adjusted",source_id=run_id,
            original_source=initial.provenance[name].source_id,original_value=old,uncertainty=abs(new-old))
    matrix=tuple(tuple(added[i] if i==j else 0. for j in range(6)) for i in range(6))
    vessel=initial.model_copy(update={"added_mass_kg":matrix,"linear_damping":linear,
        "quadratic_damping":quadratic,"provenance":provenance})
    return CalibrationResult(vessel,tuple(x.family for x in train),held_out.family,before,after,added,linear,quadratic)
