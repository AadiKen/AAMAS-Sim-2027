"""Compare an interrupted forced-surge tiny case with its uninterrupted peer."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter
from bcod_sim.vessel_generation.identification import (
    DOF, DomainSettings, FluidModel, IdentificationCase, MeshSettings,
    MotionType, SolverSettings, TurbulenceModel, TurbulenceSettings)
from bcod_sim.vessel_generation.qualified_observations import qualify_case, verify_artifact
import hashlib


def main():
    geometry=Path('/tmp/bcod-frame-qualification/box.stl')
    existing=Path('/tmp/bcod-stage3a-qualification')
    output=Path('/tmp/bcod-stage3a-restart');output.mkdir(exist_ok=True)
    digest=hashlib.sha256(geometry.read_bytes()).hexdigest()
    case=IdentificationCase(geometry_hash=digest,motion_type=MotionType.FORCED_TRANSLATION,
        dof=DOF.SURGE,amplitude=.01,frequency_rad_s=20.,fluid_model=FluidModel.FREE_SURFACE,
        turbulence_settings=TurbulenceSettings(model=TurbulenceModel.LAMINAR),
        mesh_settings=MeshSettings(.15,(1,1),0),
        domain_settings=DomainSettings((-1.,-.5,-.3),(1.5,.5,.3)),
        solver_settings=SolverSettings(end_time_s=1.5,timestep_s=.004,
            initial_timestep_s=.001,write_interval_s=.05))
    adapter=OpenFOAMAdapter(output/'cases');root=adapter.work_root/case.case_id
    if not root.exists():
        adapter.generate_identification_case(SimpleNamespace(path=str(geometry),content_hash=digest),case)
        adapter.mesh_case(root)
    control=root/'system/controlDict'
    if not (root/'solver.log').exists():
        text=control.read_text().replace('endTime 1.5;','endTime 0.75;')
        control.write_text(text)
        adapter.solve_case(root)
        control.write_text(control.read_text().replace('startFrom startTime;','startFrom latestTime;').replace('endTime 0.75;','endTime 1.5;'))
        adapter.solve_case(root,resume=True)
    uninterrupted=existing/'cases'/case.case_id
    a=np.asarray(OpenFOAMAdapter.force_history(uninterrupted));b=np.asarray(OpenFOAMAdapter.force_history(root))
    if a.shape!=b.shape or not np.allclose(a[:,0],b[:,0],rtol=0,atol=1e-9):
        raise RuntimeError(f'restart sample times differ: {a.shape} vs {b.shape}')
    absolute=float(np.max(abs(a[:,1:]-b[:,1:])))
    relative=float(np.max(abs(a[:,1:]-b[:,1:]))/max(np.max(abs(a[:,1:])),1.))
    artifact=qualify_case(root,existing/'cases'/'9f074fe18be9a11648855ab9092f69d322edb1162794d024743b457718bfd56f',output/'forced_surge_restart.json')
    verify_artifact(artifact)
    result={'sample_count':len(a),'max_absolute_wrench_difference':absolute,
            'max_relative_wrench_difference':relative,'artifact':str(artifact),
            'accepted':bool(relative<.02)}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['accepted']: raise RuntimeError('restart force equivalence failed')


if __name__=='__main__':main()
