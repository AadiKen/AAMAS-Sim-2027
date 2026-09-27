"""Coarse-first orchestration within the existing Spec A pipeline."""
import json
import shutil
from pathlib import Path
from .make_case import write_case
from .matrix import CaseState
from .mesh_profiles import select_profile
from .run_matrix import mesh_once, run_case


def prepare_sentinels(root, spec, *, mesh_family):
    if list(root.glob("[0-9][0-9]_*")):
        raise FileExistsError("Generate auto sentinels in a fresh campaign directory")
    for profile in ("fast", "standard"):
        write_case(root/"sentinels"/profile, spec, CaseState(8., 0., "drift"),
                   mesh_family=mesh_family, mesh_profile=profile)


def run_sentinels(root, *, core_count=1):
    results = {}
    for profile in ("fast", "standard"):
        case = root/"sentinels"/profile
        if not (case/"log.checkMesh").exists():
            mesh = mesh_once([case])
            (case/"mesh_runtime.json").write_text(json.dumps(mesh, indent=2)+"\n")
        saved = case/"run_result.json"
        results[profile] = json.loads(saved.read_text()) if saved.exists() else run_case(case, core_count=core_count)
        result = results[profile]
        if (result.get("mesh_profile") != profile or result["state"]["beta_deg"] != 8
                or result["state"]["r_prime"] != 0):
            raise ValueError("Sentinel result does not match its mesh profile and +8 degree state")
    config = json.loads((root/"sentinels/fast/case_config.json").read_text())["case_spec"]
    # 1e-4 coefficient floor, geometry/speed only; no experimental force access.
    scale = .5*config["density_kg_m3"]*config["length_m"]**2*config["speed_mps"]**2
    selection = select_profile(results["fast"], results["standard"],
                               force_floor=1e-4*scale, moment_floor=1e-4*scale*config["length_m"])
    existing_path = root/"mesh_selection.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if existing.get("selection_basis") == "user_selected_after_provisional_fast_standard_comparison":
            selection["automatic_profile_indicated"] = selection["mesh_profile_selected"]
            selection["mesh_profile_selected"] = "fast"
            selection["selection_basis"] = existing["selection_basis"]
    selection["sentinels"] = results
    lines = ["# Spec A +8 degree mesh selection", "",
             "| Profile | Cells | Solver seconds | Cores | Core-hours | Iterations | Status |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for profile, result in results.items():
        lines.append(f"| {profile} | {result['cell_count']} | {result['wall_seconds']:.2f} | {result['core_count']} | {result['core_hours']:.4f} | {result['iterations']} | {result['status']} |")
    lines += ["", f"Y difference: {selection['sentinel_Y_difference']:.3%}",
              f"N difference: {selection['sentinel_N_difference']:.3%}",
              f"Selected: **{selection['mesh_profile_selected']}** (threshold 10%).",
              "", "No experimental loads were used. Selection does not launch the remaining matrix."]
    (root/"mesh_selection.md").write_text("\n".join(lines)+"\n")
    (root/"mesh_selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    return selection


def materialize_matrix(root, spec, states, *, mesh_family, selection):
    profile = selection["mesh_profile_selected"]
    source = root/"sentinels"/profile
    for index, state in enumerate(states):
        case = root/f"{index:02d}_{state.family}_{state.beta_deg:g}_{state.r_prime:+g}"
        if case.exists():
            recorded = json.loads((case/"case_config.json").read_text())
            if recorded.get("mesh_profile") != profile or recorded.get("state") != state.__dict__:
                raise FileExistsError("Existing production case conflicts with selected profile")
            continue
        if state.beta_deg == 8 and state.r_prime == 0:
            shutil.copytree(source, case)
            result = json.loads((case/"run_result.json").read_text())
            result.update(case=str(case.resolve()), sentinel_reused=True)
            (case/"run_result.json").write_text(json.dumps(result, indent=2)+"\n")
        else:
            write_case(case, spec, state, mesh_family=mesh_family, mesh_profile=profile)
            shutil.copytree(source/"constant/polyMesh", case/"constant/polyMesh")
            shutil.copy2(source/"log.checkMesh", case/"log.checkMesh")

        # Production meshes are the checked sentinel snapshot, even if profile
        # defaults have since changed. Keep rebuild dictionaries/metadata aligned.
        for name in ("blockMeshDict", "snappyHexMeshDict"):
            original = source/"system"/name
            if original.exists():
                shutil.copy2(original, case/"system"/name)
        source_config = json.loads((source/"case_config.json").read_text())
        case_config = json.loads((case/"case_config.json").read_text())
        for key in ("mesh_profile", "mesh_level", "mesh_budget", "background_spacing_m",
                    "cell_count_background", "target_y_plus", "first_cell_height_m",
                    "first_layer_thickness_m"):
            if key in source_config:
                case_config[key] = source_config[key]
        (case/"case_config.json").write_text(json.dumps(case_config, indent=2)+"\n")
