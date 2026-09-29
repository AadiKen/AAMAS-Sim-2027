"""BCOD command line entry point."""

import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter
from bcod_sim.vessel_generation.hydrostatics import derive_hydrostatics, load_ascii_stl
from bcod_sim.vessel_generation.identification import DOF, FluidModel, MotionType, Sweep
from bcod_sim.vessel_generation.workflow import identify_package, read_observations
from bcod_sim.vessel_generation.qualified_observations import qualify_case
from bcod_sim.vessel_generation.simple_pipeline import generate_simple_vessel
from bcod_sim.vessel_generation.coefficient_package import validate_coefficients
from bcod_sim.vessel_generation.input_contract import generate_from_config
from bcod_sim.rl.training_control import TrainingRun


DEFAULT_SWEEP=(-1.5,-1.0,-.5,-.25,.25,.5,1.0,1.5)


def _identify(args) -> int:
    geometry=Path(args.geometry);content=geometry.read_bytes();digest=hashlib.sha256(content).hexdigest()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    cg=tuple(map(float,args.cg.split(",")))
    if len(cg)!=3: raise ValueError("--cg must contain x,y,z")
    if args.observations:
        identify_package(name=args.name or output.name,geometry_path=geometry,mass_kg=args.mass,cg_frd_m=cg,
            observations=read_observations(args.observations,unsafe_debug=args.unsafe_debug_observations),output=output,
            solver={"name":"OpenFOAM","version":args.openfoam_version},
            unsafe_debug_observations=args.unsafe_debug_observations)
        return 0
    fluid=FluidModel(args.fluid_model);cases=[]
    waterline=0.
    if fluid==FluidModel.FREE_SURFACE:
        points,faces=load_ascii_stl(geometry)
        waterline=derive_hydrostatics(points,faces,mass_kg=args.mass,cg_frd_m=cg).equilibrium_waterline_z_m
    settings={"fluid_model":fluid,"openfoam_version":args.openfoam_version,"waterline_z_m":waterline,
        "cg_frd_m":cg,"reference_point_frd_m":cg}
    cases.extend(Sweep(MotionType.STEADY_VELOCITY,(DOF.SURGE,DOF.SWAY,DOF.HEAVE),DEFAULT_SWEEP).cases(digest,**settings))
    cases.extend(Sweep(MotionType.STEADY_ROTATION,(DOF.ROLL,DOF.PITCH,DOF.YAW),DEFAULT_SWEEP).cases(digest,**settings))
    if not args.no_forced_motion:
        cases.extend(Sweep(MotionType.FORCED_TRANSLATION,(DOF.SURGE,DOF.SWAY,DOF.HEAVE),amplitudes=(.05,.1),frequencies_rad_s=(.5,1.)).cases(digest,**settings))
        cases.extend(Sweep(MotionType.FORCED_ROTATION,(DOF.ROLL,DOF.PITCH,DOF.YAW),amplitudes=(.05,.1),frequencies_rad_s=(.5,1.)).cases(digest,**settings))
    adapter=OpenFOAMAdapter(output/"cases",version=args.openfoam_version)
    reference=SimpleNamespace(path=str(geometry),content_hash=digest)
    roots=[adapter.generate_identification_case(reference,case) for case in cases]
    manifest={"schema_version":1,"geometry_sha256":digest,"mass_kg":args.mass,"cg_frd_m":cg,
        "fluid_model":fluid.value,"source_waterline_z_m":waterline,"case_count":len(cases),"cases":[{"case_id":case.case_id,"path":str(root),
        "definition":asdict(case)} for case,root in zip(cases,roots)]}
    (output/"identification_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=lambda x:x.value))
    return 0


def _qualify(args) -> int:
    if args.resume and not args.solve: raise ValueError("--resume requires --solve")
    adapter=OpenFOAMAdapter(args.case.parent)
    if args.solve:
        if not args.resume and not (args.case/"constant/polyMesh/points").exists():
            adapter.mesh_case(args.case)
        adapter.solve_case(args.case,resume=args.resume)
    print(qualify_case(args.case,args.reference,args.output))
    return 0


def _generate(args) -> int:
    if args.config is not None:
        if args.geometry is not None or args.mass is not None or args.cg is not None:
            raise ValueError("--config cannot be combined with --geometry/--mass/--cg")
        root=generate_from_config(args.config, args.output)
        print(json.dumps({"package":str(root),"validation":validate_coefficients(root/"coefficient_package.yaml")},indent=2))
        return 0
    if args.geometry is None or args.mass is None or args.cg is None or (args.units is None and args.known_length is None):
        raise ValueError("direct generation requires --geometry, --mass, --cg and --units or --known-length")
    cg = tuple(float(x) for x in args.cg)
    inertia = None
    if args.inertia:
        import numpy as np
        inertia = np.asarray([float(x) for x in args.inertia]).reshape(3, 3)
    result = generate_simple_vessel(
        geometry=args.geometry, output=args.output, mass_kg=args.mass,
        cg_frd_m=cg, units=args.units, known_length_m=args.known_length,
        source_frame=args.source_frame, draft_m=args.draft,
        geometry_mode=args.geometry_mode, reference_draft_m=args.reference_draft,
        reference_length_m=args.reference_length,
        water_kinematic_viscosity_m2_s=args.water_viscosity,
        water_density_kg_m3=args.water_density, inertia_cg_kg_m2=inertia,
        speed_range_mps=tuple(args.speed_range), classification=args.classification,
        disable_bem=args.disable_bem, lut_samples=args.lut_samples,
        confidence_policy=args.confidence_policy, bem_panel_target=args.bem_panel_target)
    print(json.dumps({"package":str(result),"validation":validate_coefficients(result/"coefficient_package.yaml")},indent=2))
    return 0


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(prog="bcod");commands=parser.add_subparsers(dest="command",required=True)
    benchmark=commands.add_parser("benchmark",help="check and run paper benchmark recipes")
    benchmark.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    benchmark.set_defaults(handler=lambda args: __import__("bcod_sim.training.benchmark", fromlist=["main"]).main(args.benchmark_args))
    train=commands.add_parser('train',help='inspect or control an active training run')
    train.add_argument('action',choices=['status','pause','resume','stop','kill','checkpoint','evaluate','set'])
    train.add_argument('run',type=Path)
    train.add_argument('parameter',nargs='?')
    train.add_argument('value',nargs='?')
    train.add_argument('--episodes',type=int,default=20)
    def train_handler(args):
        run=TrainingRun(args.run)
        if args.action=='status': print(json.dumps(run.state(),indent=2)); return 0
        if args.action=='kill': run.kill(); return 0
        if args.action=='set':
            if args.parameter!='learning_rate' or args.value is None:
                parser.error('only set learning_rate VALUE is supported')
            identifier=run.set_learning_rate(float(args.value))
        elif args.action=='evaluate': identifier=run.evaluate(args.episodes)
        else: identifier=getattr(run,args.action)()
        print(identifier); return 0
    train.set_defaults(handler=train_handler)
    vessel=commands.add_parser("vessel");vessel_commands=vessel.add_subparsers(dest="vessel_command",required=True)
    generate=vessel_commands.add_parser("generate",help="generate passive vessel dynamics from geometry without CFD")
    generate.add_argument("--config",type=Path,help="YAML vessel input; supports mass or draft loading")
    generate.add_argument("--geometry",type=Path)
    generate.add_argument("--mass",type=float)
    generate.add_argument("--cg",nargs=3,type=float,metavar=("X","Y","Z"))
    generate.add_argument("--output",required=True,type=Path)
    scale=generate.add_mutually_exclusive_group()
    scale.add_argument("--units",choices=("m","cm","mm","ft","in"))
    scale.add_argument("--known-length",type=float)
    generate.add_argument("--source-frame",choices=("FRD","FPU"),default="FRD")
    generate.add_argument("--draft",type=float)
    generate.add_argument("--geometry-mode",choices=("full_hull","design_waterline_submerged_hull"),default="full_hull")
    generate.add_argument("--reference-draft",type=float)
    generate.add_argument("--reference-length",type=float)
    generate.add_argument("--water-density",type=float,default=1025.)
    generate.add_argument("--water-viscosity",type=float,default=1.05e-6)
    generate.add_argument("--inertia",nargs=9,type=float,metavar="I")
    generate.add_argument("--speed-range",nargs=2,type=float,default=(0.,3.),metavar=("MIN","MAX"))
    generate.add_argument("--classification")
    generate.add_argument("--disable-bem",action="store_true")
    generate.add_argument("--bem-panel-target",type=int,default=900)
    generate.add_argument("--confidence-policy",choices=("allow_low","strict"),default="allow_low")
    generate.add_argument("--lut-samples",type=int,default=9)
    generate.set_defaults(handler=_generate)
    validate=vessel_commands.add_parser("validate-coefficients",help="validate coefficient package against Plant6")
    validate.add_argument("package",type=Path)
    validate.add_argument("--grid-size",type=int,default=7)
    def validate_handler(args):
        package=args.package if args.package.is_file() else args.package/"coefficient_package.yaml"
        print(json.dumps(validate_coefficients(package,grid_size=args.grid_size),indent=2))
        return 0
    validate.set_defaults(handler=validate_handler)
    identify=vessel_commands.add_parser("identify",help="prepare or fit an offline 6-DOF CFD identification campaign")
    identify.add_argument("--geometry",required=True);identify.add_argument("--mass",required=True,type=float)
    identify.add_argument("--cg",required=True,help="body-FRD x,y,z in metres");identify.add_argument("--output",required=True)
    identify.add_argument("--name");identify.add_argument("--fluid-model",choices=[x.value for x in FluidModel],default="free_surface")
    identify.add_argument("--openfoam-version",default="11");identify.add_argument("--observations")
    identify.add_argument("--unsafe-debug-observations",action="store_true")
    identify.add_argument("--no-forced-motion",action="store_true");identify.set_defaults(handler=_identify)
    qualify=vessel_commands.add_parser("qualify-case",help="qualify one solved OpenFOAM case against a static reference")
    qualify.add_argument("--case",required=True,type=Path)
    qualify.add_argument("--reference",required=True,type=Path)
    qualify.add_argument("--output",required=True,type=Path)
    qualify.add_argument("--solve",action="store_true",help="run the case before qualification")
    qualify.add_argument("--resume",action="store_true",help="resume a case with --solve")
    qualify.set_defaults(handler=_qualify)
    args=parser.parse_args(argv);return args.handler(args)


if __name__=="__main__": raise SystemExit(main())
