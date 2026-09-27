"""Minimal explicit CLI for Spec A case generation, execution, extraction, fit.

Input vessel.yaml holds the fields of VesselCaseSpec. No command runs CFD
unless the user invokes `mesh` or `run`. Example:

  python tools/spec_a_pipeline.py make output/ vessel.yaml
  python tools/spec_a_pipeline.py mesh output/
  python tools/spec_a_pipeline.py run output/
  python tools/spec_a_pipeline.py extract output/
  python tools/spec_a_pipeline.py fit output/ vessel.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from bcod_sim.vessel_generation.spec_a.check import sanity_checks, write_fit_report, plot_fit
from bcod_sim.vessel_generation.spec_a.empirical import compare_v5, linear_derivatives, v5_wrench_from_payload
from bcod_sim.vessel_generation.spec_a.extract_forces import (
    parse_force_history, physical_frd_row, qualify_force_tail, write_cases_csv)
from bcod_sim.vessel_generation.spec_a.fit import fit_cases, write_coefficients_yaml
from bcod_sim.vessel_generation.spec_a.make_case import VesselCaseSpec, write_case
from bcod_sim.vessel_generation.spec_a.matrix import case_matrix, froude_gate, CaseState
from bcod_sim.vessel_generation.spec_a.selection import prepare_sentinels, run_sentinels, materialize_matrix
from bcod_sim.vessel_generation.spec_a.run_matrix import mesh_once, run_case, run_matrix


def vessel(path: Path) -> tuple[VesselCaseSpec, dict]:
    data = yaml.safe_load(path.read_text())
    case = dict(data["case"])
    hull = Path(case["hull_path"])
    case["hull_path"] = hull if hull.is_absolute() else path.parent/hull
    case["cg_frd_m"] = tuple(case["cg_frd_m"])
    return VesselCaseSpec(**case), data


def case_dirs(root: Path) -> list[Path]:
    cases = sorted(root.glob("[0-9][0-9]_*"))
    if not cases:
        raise ValueError("No generated cases")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("make", "mesh", "run", "extract", "fit", "empirical", "select"))
    parser.add_argument("root", type=Path)
    parser.add_argument("vessel", type=Path, nargs="?")
    parser.add_argument("--quality", choices=("auto", "fast", "standard", "reference"), default="auto")
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()
    if args.cores < 1:
        parser.error("--cores must be positive")
    root = args.root
    if args.command in ("make", "fit", "empirical", "select") and args.vessel is None:
        parser.error("This command requires vessel.yaml after the output directory")
    if args.command == "make":
        spec, config = vessel(args.vessel)
        root.mkdir(parents=True, exist_ok=True)
        maximum = config.get("operating_envelope", {})
        states = case_matrix(max_beta_deg=maximum.get("max_beta_deg", 16.),
                             max_abs_r_prime=maximum.get("max_abs_r_prime", .6))
        if args.quality == "auto":
            if not any(s.beta_deg == 8 and s.r_prime == 0 for s in states):
                raise ValueError("Auto requires the +8 degree sentinel in the operating envelope")
            prepare_sentinels(root, spec, mesh_family=config.get("hull_family", "displacement_monohull"))
        for index, state in enumerate(states) if args.quality != "auto" else []:
            name = f"{index:02d}_{state.family}_{state.beta_deg:g}_{state.r_prime:+g}"
            write_case(root/name, spec, state, mesh_family=config.get("hull_family", "displacement_monohull"), mesh_profile=args.quality)
        (root/"matrix.json").write_text(json.dumps([state.__dict__ for state in states], indent=2)+"\n")
        print("Generated FAST/STANDARD sentinels; run select before the matrix" if args.quality == "auto" else f"Generated {len(states)} cases")
    elif args.command == "select":
        spec, config = vessel(args.vessel)
        selection = run_sentinels(root, core_count=args.cores)
        states = [CaseState(**row) for row in json.loads((root/"matrix.json").read_text())]
        materialize_matrix(root, spec, states, mesh_family=config.get("hull_family", "displacement_monohull"), selection=selection)
        print(json.dumps(selection, indent=2))
        print("Selection complete. Remaining matrix has NOT been run.")
    elif args.command == "mesh":
        print(json.dumps(mesh_once(case_dirs(root)), indent=2))
    elif args.command == "run":
        results = run_matrix(case_dirs(root), core_count=args.cores)
        (root/"runtime.json").write_text(json.dumps(results, indent=2)+"\n")
        print(f"Ran {len(results)} cases")
    elif args.command == "extract":
        rows = []
        for case in case_dirs(root):
            config = json.loads((case/"case_config.json").read_text())
            result = json.loads((case/"run_result.json").read_text())
            qualified = result["qualification"]
            u, v, r = config["body_velocity_frd"]
            row = physical_frd_row(case_id=case.name, u=u, v=v, r=r,
                                   qualified=qualified, foam_reference=tuple(config["moment_reference_foam"]),
                                   body_reference=tuple(config["case_spec"]["cg_frd_m"]),
                                   waterline_z_m=-config["case_spec"]["waterline_frd_z_m"])
            row.update(mesh_profile=config.get("mesh_profile"), **{k: result.get(k) for k in ("cell_count", "wall_seconds", "core_count", "core_hours", "iterations")})
            rows.append(row)
        write_cases_csv(root/"cases.csv", rows)
        print(f"Extracted {len(rows)} cases")
    elif args.command == "fit":
        spec, config = vessel(args.vessel)
        with (root/"cases.csv").open(newline="") as stream:
            cases = list(csv.DictReader(stream))
        failed = [row for row in cases if row["convergence_status"] == "failed"]
        if len(failed) > 2:
            raise ValueError("More than two failed cases; fix the template")
        cases = [row for row in cases if row["convergence_status"] in ("converged", "oscillatory")]
        surface, diagnostics = fit_cases(cases, length_m=spec.length_m, draft_m=spec.draft_m,
                                         reference_speed_mps=spec.speed_mps,
                                         density_kg_m3=spec.density_kg_m3,
                                         moment_reference_frd_m=spec.cg_frd_m)
        write_coefficients_yaml(root/"coefficients.yaml", surface,
                                froude=froude_gate(spec.speed_mps, spec.length_m),
                                mesh_selection=json.loads((root/"mesh_selection.json").read_text()) if (root/"mesh_selection.json").exists() else None)
        checks = sanity_checks(surface, diagnostics, beam_m=spec.beam_m,
                               block_coefficient=spec.block_coefficient,
                               mass_kg=float(config["mass_kg"]), cg_x_m=spec.cg_frd_m[0],
                               cases=cases)
        runtime_path = root/"runtime.json"
        core_hours = (sum(row.get("core_hours", row["core_hours_serial"]) for row in json.loads(runtime_path.read_text()))
                      if runtime_path.exists() else None)
        write_fit_report(root/"fit_report.md", checks, measured_core_hours=core_hours)
        if (root/"mesh_selection.md").exists():
            with (root/"fit_report.md").open("a") as stream:
                stream.write("\n"+(root/"mesh_selection.md").read_text())
        try:
            baseline = None
            if config.get('v5_runtime_payload'):
                baseline_path = Path(config['v5_runtime_payload'])
                if not baseline_path.is_absolute():
                    baseline_path = args.vessel.parent/baseline_path
                baseline = v5_wrench_from_payload(json.loads(baseline_path.read_text()))
            plot_fit(root/"fit_comparison.png", surface, cases, v5_surface=baseline)
        except ImportError:
            with (root/"fit_report.md").open("a") as stream:
                stream.write("\nPlot unavailable: install the validation extra (matplotlib).\n")
        (root/"fit_diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
        print(f"Y basis {surface.y_basis}; N basis {surface.n_basis}; review {checks['requires_review']}")
    elif args.command == "empirical":
        spec, config = vessel(args.vessel)
        envelope = config.get("operating_envelope", {})
        derivatives = config.get('v5_linear_derivatives')
        if derivatives is None:
            payload_path = Path(config['v5_runtime_payload'])
            if not payload_path.is_absolute():
                payload_path = args.vessel.parent/payload_path
            derivatives = linear_derivatives(v5_wrench_from_payload(json.loads(payload_path.read_text())),
                speed_mps=spec.speed_mps,length_m=spec.length_m,density_kg_m3=spec.density_kg_m3)
        comparison = compare_v5(derivatives, length_m=spec.length_m,
            beam_m=spec.beam_m, draft_m=spec.draft_m, block_coefficient=spec.block_coefficient,
            max_beta_deg=envelope.get("max_beta_deg", 16.),
            max_abs_r_prime=envelope.get("max_abs_r_prime", .6),
            hull_kind=config.get("hull_family", "displacement_monohull"))
        root.mkdir(parents=True,exist_ok=True)
        (root/'empirical_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
