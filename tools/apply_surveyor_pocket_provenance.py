"""Attach verified AP214 shell provenance to conservative face classifications."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import yaml


def run(root: Path) -> None:
    path=root/"classification.yaml"
    data=yaml.safe_load(path.read_text())
    rows={r["face_id"]:r for r in csv.DictReader((root/"provenance/face_to_component.csv").open())}
    for sid,record in data["face_classification"].items():
        row=rows[sid]
        for key in ("component_product","component_instance","representation_id",
                    "shell_id","assembly_path","style","layer","instance_transform"):
            record[key]=row[key]
        record["step_face_id"]=int(row["step_face_id"]) if row["step_face_id"] else None
        record["provenance_method"]=row["step_face_match_method"]
    for sid,stepid in (("SF057",376853),("SF058",712907)):
        record=data["face_classification"][sid]
        record["class"]="HULL_SKIN"
        record["confidence"]="moderate"
        record["evidence"]=[
            f"raw STEP ADVANCED_FACE #{stepid} is in OPEN_SHELL #817172, directly matched from original cone boundary",
            "same raw shell contains matched broad pontoon sidewall faces #75086 and #901868",
            "no AP214 assembly occurrence or separate conical component instance exists",
            "port-side cone within the same shell has different radius and angle, so no exact repeated thruster component explains this face",
            "physical water accessibility and final closure still require later geometry work"]
    data["schema"]="surveyor-pocket-face-classification-with-ap214-provenance-v2"
    data["status"]="CONE_FACES_ATTRIBUTED_TO_HULL_RECESS_SKIN_MODERATE; OTHER_SMALL_FACES_RETAIN_LOCAL_UNCERTAINTY"
    data["provenance_report"]="provenance/provenance_report.md"
    path.write_text(yaml.safe_dump(data,sort_keys=False))


if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: python tools/apply_surveyor_pocket_provenance.py POCKET_ROOT")
    run(Path(sys.argv[1]))
