"""Artifact-producing public workflows for identification and calibration."""

from pathlib import Path
import numpy as np

from .artifacts import write_run_artifact
from .calibration import CalibrationLog, CalibrationResult, calibrate
from .cfd import CFDAdapter, deterministic_test_matrix
from .fitting import CoefficientFitter
from .generation import VesselFactory
from .models import CanonicalVessel


def identify_from_cfd(vessel: CanonicalVessel, adapter: CFDAdapter, *, run_id: str,
                      artifact_root: str|Path, seed: int=0, fitter: CoefficientFitter|None=None):
    case=adapter.generate_case(vessel.geometry.content_hash,deterministic_test_matrix(seed=seed))
    result=adapter.execute(case).validated(); fit=(fitter or CoefficientFitter()).fit(result.dataset)
    matrix=tuple(tuple(fit.added_mass[i] if i==j else 0. for j in range(6)) for i in range(6))
    uncertainty=float(np.mean([sum(x)/3 for x in fit.standard_errors]))
    fitted=VesselFactory.from_cfd(vessel,matrix,fit.linear_damping,fit.quadratic_damping,
                                  run_id=run_id,uncertainty=uncertainty)
    artifact=write_run_artifact(artifact_root,run_id=run_id,vessel=fitted,cfd_result=result,
                                fit_result=fit,seed=seed)
    return fitted,fit,artifact


def calibrate_from_logs(vessel: CanonicalVessel, train: tuple[CalibrationLog,...], held_out: CalibrationLog,
                        *, run_id: str, artifact_root: str|Path, seed: int=0) -> tuple[CalibrationResult,Path]:
    result=calibrate(vessel,train,held_out,run_id=run_id)
    artifact=write_run_artifact(artifact_root,run_id=run_id,vessel=result.vessel,
                                calibration_result=result,seed=seed)
    return result,artifact
