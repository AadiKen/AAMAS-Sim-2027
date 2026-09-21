"""End-to-end genuine OpenFOAM integration smoke driver."""

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from bcod_sim.config.hashing import canonical_json, content_hash
from bcod_sim.config.resolver import resolve
from bcod_sim.core.lifecycle import DirectAction
from bcod_sim.actuators.thruster import ThrustCommand
from bcod_sim.vessel_generation.artifacts import write_hash_manifest, verify_hash_manifest
from bcod_sim.vessel_generation.cfd import CFDExecutionError, OpenFOAMAdapter
from bcod_sim.vessel_generation.fitting import CalibrationError, CoefficientFitter, ForceMomentDataset
from bcod_sim.vessel_generation.generation import VesselFactory
from bcod_sim.vessel_generation.geometry import import_ascii_stl
from bcod_sim.vessel_generation.models import GeometryReference, ParameterLineage
from bcod_sim.web.runtime_factory import build_engine
from bcod_sim.web.service import build_registry, serialize_frame
from tools.cfd_smoke.generate_smoke_hull import generate


DISCLAIMER="This smoke test validates software plumbing only. It does not constitute real-vessel hydrodynamic validation."
SPEEDS=(.5,1.,1.5)


def _plain(value):
    if hasattr(value,"tolist"): return value.tolist()
    if hasattr(value,"__dataclass_fields__"): return {k:_plain(v) for k,v in asdict(value).items()}
    if isinstance(value,dict): return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_plain(v) for v in value]
    return value


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(_plain(value),sort_keys=True,indent=2,allow_nan=False))


def _repository(root: Path):
    def git(*args):
        result=subprocess.run(["git",*args],cwd=root,capture_output=True,text=True)
        return result.stdout.strip() if result.returncode==0 else "UNBORN"
    status=git("status","--short")
    text="\n".join((str(root),git("rev-parse","--show-toplevel"),git("rev-parse","HEAD"),status,
                     git("branch","--show-current")))+"\n"
    return {"commit":git("rev-parse","HEAD"),"dirty":bool(status),"branch":git("branch","--show-current"),"text":text}


def _environment_evidence(adapter, openfoam):
    native=subprocess.run(["bash","-lc",'''set -o pipefail
echo "=== foamVersion ==="; foamVersion 2>&1 || true
echo "=== WM_PROJECT ==="; printf "%s\\n" "${WM_PROJECT:-UNSET}"
echo "=== WM_PROJECT_VERSION ==="; printf "%s\\n" "${WM_PROJECT_VERSION:-UNSET}"
echo "=== native binaries ==="
for x in blockMesh snappyHexMesh checkMesh simpleFoam interFoam potentialFoam foamRun; do printf "%-24s" "$x"; command -v "$x" || true; done'''],capture_output=True,text=True)
    inside=adapter._container(["foamVersion",";","for","x","in","blockMesh","snappyHexMesh","checkMesh","simpleFoam","foamRun",";","do","command","-v","$x",";","done",";","foamRun","-help"],capture=True)
    docker=subprocess.run(["docker","--version"],capture_output=True,text=True)
    return native.stdout+native.stderr+"\n=== Docker ===\n"+docker.stdout+docker.stderr+json.dumps(openfoam,sort_keys=True,indent=2)+"\n=== container binaries and solver help ===\n"+inside.stdout+inside.stderr


def _seed(geometry) :
    vessel=VesselFactory.procedural("cfd-smoke-vessel",seed=1200,length_m=2,beam_m=.4,mass_kg=120,
                                    actuator_strength_n=30)
    manual=ParameterLineage(source_kind="manual",source_id="manual_smoke_fixture",uncertainty=None)
    provenance={name:manual for name in vessel.provenance}
    return vessel.model_copy(update={"version":"openfoam-smoke-1",
        "geometry":GeometryReference(format="stl",content_hash=geometry.content_hash,length_m=2,beam_m=.4,draft_m=.15),
        "provenance":provenance,"validation_claim":"synthetic/reference validation only"})


def _configuration(vessel):
    definitions=vessel.simulator_definitions()+[{"kind":"task","id":"smoke-waypoint","version":"1","source":"cfd-smoke",
        "payload":{"kind":"waypoint","agent_id":"agent","target_ned_m":[1000,0,0],"radius_m":.1}}]
    config={"schema_version":1,"experiment":{"id":"cfd-smoke","seed":1200,"numerical_profile":"validation"},
        "simulation":{"dynamics_mode":"full6","master_dt_s":.05,"dynamics_substeps":1,
            "policy_every_n_master_steps":1,"max_master_steps":100},
        "world":{"source":{"kind":"parametric"},"environment":{"current":{"kind":"uniform","ned_mps":[0,0,0]},
            "wind":{"kind":"uniform","ned_mps":[0,0,0]},"waves":{"kind":"calm"},"visibility_m":1000}},
        "vessels":[{"instance_id":"agent","definition":f"{vessel.id}@{vessel.version}",
            "actuators":["port@1","starboard@1"],"controller":{"mode":"direct_actuator"},
            "spawn":{"ned_m":[0,0,0],"rpy_rad":[0,0,0]}}],
        "task":{"type":"smoke-waypoint@1","reward":{"individual_weight":1,"team_weight":0},
            "disabled_agent_behavior":"deactivate_keep_physical"},
        "logging":{"states":True,"sensor_payloads":False,"queue_capacity":32,"backpressure":"block"}}
    return config,definitions


def _rollout(vessel, output: Path, run_name: str):
    config,definitions=_configuration(vessel); resolved=resolve(config,build_registry(definitions)); engine=build_engine(resolved)
    vessel_definition=next(x for x in resolved.definitions if x.kind=="vessel")
    expected_payload=next(x for x in definitions if x["kind"]=="vessel")["payload"]
    if vessel_definition.id!=vessel.id or vessel_definition.content_hash!=content_hash(expected_payload):
        raise CFDExecutionError("normal loader identity or fingerprint mismatch")
    frame=engine.reset(seed=1200); trajectory=[serialize_frame(frame)]; max_state=0.; max_diagnostic=0.
    action={"agent":DirectAction((("port",ThrustCommand(5.)),("starboard",ThrustCommand(5.))))}
    for _ in range(100):
        frame=engine.step(action); state=engine.states["agent"]; plant=engine.vessels["agent"].plant
        external=engine._external(engine.vessels["agent"],state,frame.sim_time_s,engine.last_propulsion["agent"])
        acceleration=plant.acceleration(state,external); ledger=plant.diagnostics(state,external); ledger.assert_balanced()
        tensors=(state.position_ned,state.q_body_to_ned,state.nu_body,acceleration,ledger.total,*ledger.terms.values())
        if not all(torch.isfinite(x).all().item() for x in tensors): raise CFDExecutionError("nonfinite runtime diagnostic")
        if abs(torch.linalg.vector_norm(state.q_body_to_ned).item()-1)>1e-9: raise CFDExecutionError("invalid orientation")
        max_state=max(max_state,max(float(torch.max(torch.abs(x)).item()) for x in (state.position_ned,state.q_body_to_ned,state.nu_body)))
        max_diagnostic=max(max_diagnostic,max(float(torch.max(torch.abs(x)).item()) for x in (acceleration,ledger.total,*ledger.terms.values())))
        trajectory.append(serialize_frame(frame))
    final=engine.states["agent"].position_ned.tolist()
    if final[0]<=0 or abs(final[1])>.05 or abs(final[2])>.05:
        raise CFDExecutionError("runtime motion sanity check failed")
    _write_json(output/run_name/"trajectory.json",trajectory)
    _write_json(output/run_name/"manifest.json",{"config_hash":resolved.content_hash,"seed":1200,
        "steps":100,"duration_s":5.,"vessel_definition_hash":vessel_definition.content_hash})
    return {"trajectory":trajectory,"duration_s":5.,"final_displacement_m":final,"max_state_magnitude":max_state,
            "max_diagnostic_magnitude":max_diagnostic,
            "definition_hash":vessel_definition.content_hash,"config_hash":resolved.content_hash}


def _negative_tests(adapter, cases, dataset, output):
    results={}
    failure=output/"negative"/"solver_failure"; shutil.copytree(cases[0],failure)
    fv=failure/"system"/"fvSolution"; fv.write_text(fv.read_text().replace("solver GAMG","solver definitelyInvalidSolver",1))
    try: adapter.solve_case(failure); results["solver_failure"]="FAIL"
    except CFDExecutionError: results["solver_failure"]="PASS"
    raw=next(cases[0].glob("postProcessing/forces/*/forces.dat")); hidden=raw.with_suffix(".missing"); raw.rename(hidden)
    try:
        try: adapter.parse_case(cases[0]); results["missing_force_output"]="FAIL"
        except CFDExecutionError: results["missing_force_output"]="PASS"
    finally: hidden.rename(raw)
    bad=ForceMomentDataset(dataset.velocity.copy(),dataset.acceleration.copy(),dataset.wrench.copy(),dataset.units); bad.wrench[0,0]=np.nan
    try: CoefficientFitter().fit_damping_axis(bad,axis=0); results["nonfinite_input"]="FAIL"
    except CalibrationError: results["nonfinite_input"]="PASS"
    rank_bad=ForceMomentDataset(np.tile(dataset.velocity[:1],(3,1)),np.zeros((3,6)),np.tile(dataset.wrench[:1],(3,1)))
    try: CoefficientFitter().fit_damping_axis(rank_bad,axis=0); results["unidentifiable_fit"]="FAIL"
    except CalibrationError: results["unidentifiable_fit"]="PASS"
    return results


def run(output: Path, *, keep_cases: bool) -> int:
    project=Path(__file__).parents[3]; output=output.resolve()
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True); repository=_repository(project); (output/"repository_state.txt").write_text(repository.pop("text"))
    shutil.copyfile(project/"docs"/"cfd_smoke_discovery.md",output/"discovery.md")
    adapter=OpenFOAMAdapter(output/"cases")
    try: openfoam=adapter.verify_runtime()
    except CFDExecutionError as exc:
        report={"overall":"BLOCKED","reason":str(exc),"repository":repository,"openfoam":None}
        _write_json(output/"report.json",report); (output/"REPORT.md").write_text(f"# CFD smoke: BLOCKED\n\n{DISCLAIMER}\n\n{exc}\n")
        return 2
    (output/"openfoam_environment.txt").write_text(_environment_evidence(adapter,openfoam))
    try:
        source=generate(output/"input"/"smoke_hull_source.stl")
        geometry=import_ascii_stl(source,output/"input"/"smoke_hull.stl",source_frame="body_FRD_m")
        vessel_seed=_seed(geometry); _write_json(output/"vessel"/"seed.json",vessel_seed.model_dump(mode="json"))
        case_roots=[]; parsed=[]; case_rows=[]; generated_case_files=[]; raw_outputs=[]
        for speed in SPEEDS:
            case_id=f"surge_{str(speed).replace('.','p')}"; root=adapter.generate_operating_case(geometry,case_id=case_id,surge_mps=speed)
            for path in sorted(x for x in root.rglob("*") if x.is_file()):
                data=path.read_bytes(); generated_case_files.append({"case":case_id,"path":path.relative_to(root).as_posix(),
                    "bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
            mesh=adapter.mesh_case(root); solve=adapter.solve_case(root); result=adapter.parse_case(root)
            if abs(result.force_body_frd_n[0])<=0 or abs(result.force_body_frd_n[1])>.15*abs(result.force_body_frd_n[0]) or abs(result.moment_body_frd_nm[2])>.1*abs(result.force_body_frd_n[0])*2:
                raise CFDExecutionError(f"force/moment sanity check failed for {case_id}")
            _write_json(output/"results"/f"{case_id}.json",result); case_roots.append(root); parsed.append(result)
            raw=next(root.glob("postProcessing/forces/*/forces.dat")); lines=raw.read_text().splitlines()
            raw_outputs.append({"case":case_id,"path":str(raw),"bytes":raw.stat().st_size,
                "sha256":hashlib.sha256(raw.read_bytes()).hexdigest(),"first_lines":lines[:5],"last_lines":lines[-5:]})
            case_rows.append({"id":case_id,"mesh":"PASS","solver":"PASS","converged":True,"result":"PASS",
                              "cell_count":mesh["cell_count"],"mesh_quality_findings":mesh["accepted_quality_findings"],
                              "mesh_seconds":sum(v["seconds"] for v in mesh.values() if isinstance(v,dict)),
                              "solver_seconds":solve["seconds"]})
        _write_json(output/"case_files.json",generated_case_files); _write_json(output/"raw_force_outputs.json",raw_outputs)
        shutil.copyfile(output/"cases"/"surge_1p0"/"checkMesh.log",output/"checkMesh.log")
        summary=[]
        for root in case_roots:
            summary.extend(line for line in (root/"solver.log").read_text().splitlines()
                           if any(term in line for term in ("OpenFOAM","Version","Application","ExecutionTime","SIMPLE solution converged","End")))
        (output/"solver_summary.txt").write_text("\n".join(summary)+"\n")
        velocity=np.zeros((3,6)); velocity[:,0]=SPEEDS; acceleration=np.zeros((3,6)); wrench=np.zeros((3,6))
        wrench[:,0]=[abs(x.force_body_frd_n[0]) for x in parsed]
        dataset=ForceMomentDataset(velocity,acceleration,wrench,seed=1200); fit=CoefficientFitter().fit_damping_axis(dataset,axis=0)
        linear=list(vessel_seed.linear_damping); quadratic=list(vessel_seed.quadratic_damping)
        linear[0]=fit.linear_damping; quadratic[0]=fit.quadratic_damping
        provenance=dict(vessel_seed.provenance); cfd_lineage=ParameterLineage(source_kind="CFD-derived",source_id="openfoam-smoke-surge",
            uncertainty=fit.residual_rms)
        provenance["linear_damping"]=cfd_lineage; provenance["quadratic_damping"]=cfd_lineage
        vessel=vessel_seed.model_copy(update={"linear_damping":tuple(linear),"quadratic_damping":tuple(quadratic),
                                               "provenance":provenance})
        vessel=type(vessel).model_validate(vessel.model_dump(mode="json")); vessel_fingerprint=hashlib.sha256(vessel.model_dump_json().encode()).hexdigest()
        fit_document={**_plain(fit),"parameters_requested":["linear_damping[0]","quadratic_damping[0]"],
            "parameters_fitted":["linear_damping[0]","quadratic_damping[0]"],
            "unchanged_seed_parameters":["added_mass_kg","linear_damping[1:6]","quadratic_damping[1:6]"],
            "not_identifiable":["sway","heave","roll","pitch","yaw"],"input_cases":[x.case_id for x in parsed],
            "input_hashes":[x.raw_output_sha256 for x in parsed]}
        _write_json(output/"fit"/"fit_result.json",fit_document); _write_json(output/"vessel"/"vessel.json",vessel.model_dump(mode="json"))
        cfd_manifest={"geometry_sha256":geometry.content_hash,"case_matrix_sha256":hashlib.sha256(canonical_json([.5,1.,1.5]).encode()).hexdigest(),
            "raw_result_hashes":[x.raw_output_sha256 for x in parsed],"openfoam":openfoam,"solver":"foamRun -solver incompressibleFluid",
            "meshing_settings":"blockMesh + snappyHexMesh v11 dictionaries","adapter_version":adapter.adapter_version,
            "fitter":"CoefficientFitter.fit_damping_axis","fitted_coefficients":fit_document,"validation_state":"unvalidated",
            "vessel_fingerprint":vessel_fingerprint}
        _write_json(output/"vessel"/"cfd_manifest.json",cfd_manifest)
        run_a=_rollout(vessel,output,"run_a"); run_b=_rollout(vessel,output,"run_b")
        max_difference=0.
        for a,b in zip(run_a["trajectory"],run_b["trajectory"]):
            av=np.asarray(a["states"]["agent"]["position_ned_m"]+a["states"]["agent"]["nu_body"])
            bv=np.asarray(b["states"]["agent"]["position_ned_m"]+b["states"]["agent"]["nu_body"])
            max_difference=max(max_difference,float(np.max(np.abs(av-bv))))
        if max_difference!=0: raise CFDExecutionError("deterministic repeatability failed")
        _write_json(output/"repeatability.json",{"status":"PASS","max_difference":max_difference})
        load_report={"requested_vessel_id":vessel.id,"loaded_vessel_id":vessel.id,"vessel_fingerprint":vessel_fingerprint,
            "loaded_fingerprint_matches":True,"definition_hash":run_a["definition_hash"],"validation_state":"unvalidated",
            "fallback_used":False,"legacy_preset_used":False}
        _write_json(output/"vessel"/"load_report.json",load_report)
        negative=_negative_tests(adapter,case_roots,dataset,output)
        if any(x!="PASS" for x in negative.values()): raise CFDExecutionError("mandatory negative test failed")
        manifest=write_hash_manifest(output,output/"integrity_manifest.json",exclude=("integrity_manifest.json","report.json","REPORT.md"))
        tamper=output/"results"/"surge_0p5.json"; original=tamper.read_bytes(); tamper.write_bytes(original+b" ")
        try:
            try: verify_hash_manifest(output,manifest); negative["provenance_tampering"]="FAIL"
            except ValueError: negative["provenance_tampering"]="PASS"
        finally: tamper.write_bytes(original)
        verify_hash_manifest(output,manifest)
        report={"overall":"PASS","repository":repository,"openfoam":openfoam,
            "geometry":{"path":str(geometry.path),"source":"deterministic symmetric ellipsoidal smoke hull",
                "dimensions_m":{"length":2.,"beam":.4,"draft":.15},"sha256":geometry.content_hash},"cases":case_rows,
            "coefficient_fit":{"status":"PASS","parameters_fitted":fit_document["parameters_fitted"],
                "identifiability":"PASS","rank":fit.rank,"condition_number":fit.condition_number,
                "linear_damping":fit.linear_damping,"quadratic_damping":fit.quadratic_damping,
                "residual_rms":fit.residual_rms},"vessel_generation":"PASS","vessel_validation":"PASS",
            "simulator_load":"PASS","runtime_rollout":"PASS","repeatability":"PASS","runtime":{k:v for k,v in run_a.items() if k!="trajectory"},
            "repeatability_metrics":{"max_difference":max_difference},"vessel":{"artifact":str(output/"vessel"/"vessel.json"),
                "fingerprint":vessel_fingerprint,"validation_state":"unvalidated"},"negative_tests":negative,"keep_cases":keep_cases}
        _write_json(output/"report.json",report); _write_report(output/"REPORT.md",report,parsed)
        return 0
    except Exception as exc:
        report={"overall":"FAIL","reason":f"{type(exc).__name__}: {exc}","repository":repository,"openfoam":openfoam}
        _write_json(output/"report.json",report); (output/"REPORT.md").write_text(f"# CFD smoke: FAIL\n\n{DISCLAIMER}\n\n{report['reason']}\n")
        return 1


def _write_report(path,report,parsed):
    fit=report["coefficient_fit"]; runtime=report["runtime"]
    lines=["# CFD smoke: PASS","",DISCLAIMER,"",f"Git commit: {report['repository']['commit']} (dirty={report['repository']['dirty']})",
        f"OpenFOAM: {report['openfoam']['version']} via Docker; solver `{report['openfoam']['solver']}`",
        f"Geometry: {report['geometry']['source']}, 2.0 × 0.4 × 0.15 m, `{report['geometry']['sha256']}`","",
        "## Mesh and cases"]
    for case,result in zip(report["cases"],parsed):
        lines.append(f"- {case['id']}: {case['cell_count']} cells; mesh {case['mesh']}; solver converged in {case['solver_seconds']:.3f}s; "
                     f"[X,Y,Z,K,M,N] = {_plain(result.force_body_frd_n+result.moment_body_frd_nm)}")
    lines += ["","## Coefficient fit",f"Rank {fit['rank']}; condition {fit['condition_number']:.6g}; residual RMS {fit['residual_rms']:.6g} N.",
        f"Surge linear damping {fit['linear_damping']:.9g}; surge quadratic damping {fit['quadratic_damping']:.9g}.","",
        "## Vessel and runtime",f"Artifact: `{report['vessel']['artifact']}`",f"Fingerprint: `{report['vessel']['fingerprint']}`",
        f"Validation state: {report['vessel']['validation_state']}",f"5 s final displacement: {runtime['final_displacement_m']}",
        f"Maximum recorded state/diagnostic magnitude: {runtime['max_state_magnitude']}",
        f"Repeatability maximum difference: {report['repeatability_metrics']['max_difference']}","","## Negative tests"]
    lines += [f"- {name}: {status}" for name,status in report["negative_tests"].items()]
    lines += ["","## Code changes needed","Added production STL validation, OpenFOAM v11 case generation, real Docker execution, mesh/convergence checks, native force parsing, subset damping identification, integrity verification, and the smoke orchestrator.","",
        "## Remaining limitations","This coarse steady, laminar, single-phase sweep validates integration only. Mesh concavity findings from strict `checkMesh -allGeometry` are recorded; topology, volumes, non-orthogonality, and skewness pass the production smoke criteria. No real-vessel accuracy or full 6-DOF identification is claimed."]
    path.write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("artifacts/cfd-smoke/latest")); parser.add_argument("--keep-cases",action="store_true")
    args=parser.parse_args(); raise SystemExit(run(args.output,keep_cases=args.keep_cases))


if __name__=="__main__": main()
