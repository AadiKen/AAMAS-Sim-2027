"""Run four bounded tiny OpenFOAM cases and attempt Stage 3A observations."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter
from bcod_sim.vessel_generation.identification import (
    DOF, DomainSettings, FluidModel, IdentificationCase, MeshSettings,
    MotionType, SolverSettings, TurbulenceModel, TurbulenceSettings)
from bcod_sim.vessel_generation.qualified_observations import qualify_case


def main() -> int:
    output=Path(sys.argv[1]).resolve()
    geometry=Path(sys.argv[2]).resolve()
    output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(geometry.read_bytes()).hexdigest()
    common=dict(geometry_hash=digest,fluid_model=FluidModel.FREE_SURFACE,
        turbulence_settings=TurbulenceSettings(model=TurbulenceModel.LAMINAR),
        mesh_settings=MeshSettings(.15,(1,1),0),
        domain_settings=DomainSettings((-1.,-.5,-.3),(1.5,.5,.3)),
        reference_point_frd_m=(0.,0.,0.),cg_frd_m=(0.,0.,0.),waterline_z_m=0.)
    steady_solver=SolverSettings(end_time_s=.5,timestep_s=.004,initial_timestep_s=.001,
        write_interval_s=.05)
    forced_solver=replace(steady_solver,end_time_s=1.5)
    cases={
        "static":IdentificationCase(motion_type=MotionType.STEADY_VELOCITY,dof=DOF.SURGE,
            magnitude=1e-9,solver_settings=steady_solver,**common),
        "steady_surge":IdentificationCase(motion_type=MotionType.STEADY_VELOCITY,dof=DOF.SURGE,
            magnitude=.2,solver_settings=steady_solver,**common),
        "forced_surge":IdentificationCase(motion_type=MotionType.FORCED_TRANSLATION,dof=DOF.SURGE,
            amplitude=.01,frequency_rad_s=20.,solver_settings=forced_solver,**common),
        "forced_yaw":IdentificationCase(motion_type=MotionType.FORCED_ROTATION,dof=DOF.YAW,
            amplitude=.05,frequency_rad_s=20.,solver_settings=replace(forced_solver,end_time_s=2.5),**common),
        "steady_yaw":IdentificationCase(motion_type=MotionType.STEADY_ROTATION,dof=DOF.YAW,
            magnitude=20.,solver_settings=replace(forced_solver,end_time_s=1.25),**common)}
    adapter=OpenFOAMAdapter(output/"cases")
    reference=SimpleNamespace(path=str(geometry),content_hash=digest)
    result={}
    for name,case in cases.items():
        root=adapter.work_root/case.case_id
        try:
            if not root.exists(): adapter.generate_identification_case(reference,case)
            if not (root/"constant/polyMesh/points").exists(): adapter.mesh_case(root)
            if not (root/"solver.log").exists():
                run=adapter.solve_case(root)
            else: run={"seconds":None,"converged":True}
            result[name]={"case_root":str(root),"cell_count":None,"simulated_duration_s":case.solver_settings.end_time_s,
                          "solver_wall_seconds":run["seconds"],"run":"completed"}
            if name!="static":
                artifact=output/"observations"/f"{name}.json"
                qualify_case(root,adapter.work_root/cases["static"].case_id,artifact)
                result[name]["observation"]=str(artifact)
        except Exception as exc:
            result[name]={"case_root":str(root),"error":f"{type(exc).__name__}: {exc}"}
        (output/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
