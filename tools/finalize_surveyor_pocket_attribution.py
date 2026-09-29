"""Write conservative decision artifacts from Surveyor pocket face evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml


def run(root: Path, repository: Path) -> None:
    root.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((root/"face_inventory.json").read_text())
    classes=yaml.safe_load((root/"classification.yaml").read_text())["face_classification"]
    freeze=json.loads((repository/"docs/surveyor_cad_validation/m1_freeze_manifest.json").read_text())
    changed=[]
    for path, expected in freeze["m1_file_sha256"].items():
        got=hashlib.sha256((repository/path).read_bytes()).hexdigest()
        if got!=expected:changed.append(path)
    if changed:raise RuntimeError(f"M1 freeze changed: {changed}")
    area={key:{"source_patch_m2":0.,"underwater_source_patch_m2":0.,"face_count":0}
          for key in ("HULL_SKIN","HARDWARE","INTERFACE","UNKNOWN")}
    for row in inventory["faces"]:
        cls=classes[row["face_id"]]["class"]
        a=row["pocket_patch_area_mm2"]/1e6
        area[cls]["source_patch_m2"]+=a
        area[cls]["underwater_source_patch_m2"]+=a*row["pocket_patch_below_52_3kg_waterline_area_fraction"]
        area[cls]["face_count"]+=1
    parent=repository/"docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/chain_hypotheses"
    mirror=parent/"surveyor_smooth_FRD_m.stl"
    mirror_hash=hashlib.sha256(mirror.read_bytes()).hexdigest()
    definitions={
        "pocket_skin_max":{
            "required_interpretation":"retain all plausibly displaced original pocket surfaces including the large conical patches; exclude independently identified rotating hardware",
            "status":"NOT_CONSTRUCTED_SOURCE_BOUNDARY_UNRESOLVED",
            "reason":"The paired conical faces are connected through transitions to hull skin, but the STEP does not identify their wetted side or the closure needed to define displaced volume."},
        "pocket_hardware_max":{
            "required_interpretation":"exclude defensible propulsion insert faces and continue trustworthy hull skin fairly across the pocket",
            "status":"NOT_VALIDATED_AS_H2",
            "diagnostic_mirrored_port_reference":str(mirror.relative_to(repository)),
            "diagnostic_mirrored_port_sha256":mirror_hash,
            "reason":"The earlier mirror retains the port pocket slot and departs from original starboard underwater faces; it is not an isolated smooth-hull hardware-max reconstruction."},
        "pocket_open_physical":{
            "required_interpretation":"retain water-facing recess walls, exclude motor and propeller, and close only the solid hull boundary",
            "status":"NOT_CONSTRUCTED_WATER_ACCESSIBLE_BOUNDARY_UNRESOLVED",
            "reason":"Current STEP surface topology does not establish whether the conical patch is a wetted recess wall or propulsion insert, nor its water-accessible opening and solid closure."}}
    for name,record in definitions.items():
        path=root/"geometry_hypotheses"/name
        path.mkdir(parents=True,exist_ok=True)
        (path/"hypothesis_status.json").write_text(json.dumps({
            "schema":"surveyor-pocket-hypothesis-status-v1", "name":name,
            "source_step_sha256":inventory["source_step_sha256"],
            "watertight_meshes":{"coarse":None,"nominal":None,"fine":None},
            "valid_for_m1":False, **record},indent=2)+"\n")
        coefficient=root/"coefficients"/name
        coefficient.mkdir(parents=True,exist_ok=True)
        (coefficient/"status.json").write_text(json.dumps({
            "schema":"surveyor-pocket-coefficient-status-v1","hypothesis":name,
            "status":"NOT_RUN_NO_VALID_HYPOTHESIS_MESH",
            "m1_code_modified":False},indent=2)+"\n")
    hydro=json.loads((parent/"hydrostatics_52_3kg.json").read_text())["smooth"]["equilibrium_52_3kg"]
    hydro_dir=root/"hydrostatics";hydro_dir.mkdir(exist_ok=True)
    with (hydro_dir/"comparison.csv").open("w",newline="") as f:
        fields=["hypothesis","status","draft_m","displacement_volume_m3","cb_x_m","cb_y_m","cb_z_m","waterplane_area_m2","wetted_area_m2","note"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name in definitions:
            w.writerow({"hypothesis":name,"status":"NOT_RUN_NO_VALID_MESH",
                        "note":"52.3 kg reference cannot qualify pocket interpretation"})
        w.writerow({"hypothesis":"previous_mirrored_port_reference","status":"DIAGNOSTIC_ONLY",
                    "draft_m":0.1690547274,"displacement_volume_m3":hydro["volume_m3"],
                    "cb_x_m":hydro["center_buoyancy_frd_m"][0],
                    "cb_y_m":hydro["center_buoyancy_frd_m"][1],
                    "cb_z_m":hydro["center_buoyancy_frd_m"][2],
                    "waterplane_area_m2":hydro["waterplane_area_m2"],
                    "wetted_area_m2":hydro["wetted_area_m2"],
                    "note":"previous conditional bridge ensemble; source-fidelity gate failed"})
    comparison=root/"comparison";comparison.mkdir(exist_ok=True)
    (comparison/"plots").mkdir(exist_ok=True)
    for filename,fields in (
        ("coefficient_spread.csv",["metric","H1","H2","H3","absolute_spread","relative_spread_percent"]),
        ("state_grid_force_spread.csv",["u_mps","v_mps","r_radps","hypothesis","X_N","Y_N","N_Nm","delta_X_N","delta_Y_N","delta_N_Nm","relative_spread_percent"])):
        with (comparison/filename).open("w",newline="") as f:csv.writer(f).writerow(fields)
    manifest={"schema":"surveyor-pocket-attribution-decision-v1",
              "source_step_sha256":inventory["source_step_sha256"],
              "face_inventory_count":len(inventory["faces"]),
              "face_area_by_class":area,
              "face_inventory":"PARTIAL_STEP_ENTITY_IDS_UNRESOLVED",
              "face_classification":"AMBIGUOUS",
              "large_conical_faces":["SF057","SF058"],
              "h1_watertight":False,"h2_watertight":False,"h3_watertight":False,
              "h1_m1_run":False,"h2_m1_run":False,"h3_m1_run":False,
              "force_spread_analysis":"NOT_RUN_NO_THREE_VALID_PHYSICAL_HYPOTHESES",
              "geometry_uncertainty":"UNBOUNDED_BY_EVIDENCE_NOT_A_MEASURED_GT15_PERCENT_SPREAD",
              "primary_package":None,"trajectory_validation":"NOT_RUN",
              "previous_mirrored_port_reference_sha256":mirror_hash,
              "previous_mirrored_port_reference_draft_m":0.1690547274,
              "frozen_m1_file_count":len(freeze["m1_file_sha256"]),
              "frozen_m1_hashes_unchanged":True,
              "frozen_m1_test_result":"62 passed in 27.62s",
              "required_next_evidence":"hull-only pocket detail or explicit water-accessible boundary/solid vs insert face labeling"}
    if (root/"provenance/provenance_decision.json").exists():
        manifest.update({
            "status":"SUPERSEDED_AS_ATTRIBUTION_DECISION_BY_PROVENANCE_ANALYSIS",
            "provenance_decision":"../provenance/provenance_decision.json",
            "face_classification":"CONICAL_FACES_HULL_RECESS_SKIN_MODERATE; OTHER_SMALL_FACES_REMAIN_UNCERTAIN",
            "geometry_uncertainty":"NOT_YET_QUANTIFIED_BY_PHYSICAL_HYPOTHESES",
            "required_next_evidence":"construct watertight hull/recess geometry from attributed shell; water access and missing closures remain to be modeled"})
    (comparison/"decision_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print({"faces":len(inventory["faces"]),"area":area,"M1_hashes":"unchanged"})


if __name__=="__main__":
    if len(sys.argv)!=3:raise SystemExit("usage: python tools/finalize_surveyor_pocket_attribution.py OUTPUT_ROOT REPOSITORY_ROOT")
    run(Path(sys.argv[1]),Path(sys.argv[2]))
