"""Reproducible Phase 12 run artifacts."""

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import subprocess

from pydantic import BaseModel

from .models import CanonicalVessel


def write_run_artifact(root: str|Path, *, run_id: str, vessel: CanonicalVessel, cfd_result=None,
                       fit_result=None, calibration_result=None, seed: int=0) -> Path:
    root=Path(root)/f"vessel_run_{run_id}"
    if root.exists(): raise ValueError("artifact run identity already exists")
    root.mkdir(parents=True)
    commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or "uncommitted"
    def plain(value):
        if isinstance(value,BaseModel): value=value.model_dump(mode="json")
        if is_dataclass(value): value=asdict(value)
        if hasattr(value,"tolist"): return value.tolist()
        if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [plain(v) for v in value]
        return value
    (root/"resolved_vessel.json").write_text(vessel.model_dump_json(indent=2))
    documents={"cfd.json":cfd_result,"fit.json":fit_result,"calibration.json":calibration_result}
    for name,value in documents.items():
        if value is not None: (root/name).write_text(json.dumps(plain(value),sort_keys=True,indent=2))
    if calibration_result is not None:
        (root/"calibration_dataset_manifest.json").write_text(json.dumps({
            "training_maneuver_families":list(calibration_result.train_families),
            "held_out_maneuver_family":calibration_result.held_out_family,"synthetic":True},sort_keys=True,indent=2))
        (root/"held_out_validation.json").write_text(json.dumps({"pre_calibration_rmse":calibration_result.pre_rmse,
            "post_calibration_rmse":calibration_result.post_rmse},sort_keys=True,indent=2))
    manifest={"schema_version":1,"run_id":run_id,"geometry_hash":vessel.geometry.content_hash,
        "code_commit":commit,"random_seed":seed,"validation_claim":vessel.validation_claim,
        "files":sorted([p.name for p in root.iterdir()])}
    manifest["files"].append("manifest.json")
    (root/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2))
    return root


def write_hash_manifest(root: str|Path, output: str|Path, *, exclude: tuple[str,...]=()) -> Path:
    root,output=Path(root),Path(output); excluded={Path(x).as_posix() for x in exclude}
    rows=[]
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        relative=path.relative_to(root).as_posix()
        if relative in excluded or path.resolve()==output.resolve(): continue
        rows.append({"path":relative,"bytes":path.stat().st_size,
                     "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(rows,sort_keys=True,indent=2))
    return output


def verify_hash_manifest(root: str|Path, manifest: str|Path) -> None:
    root=Path(root); rows=json.loads(Path(manifest).read_text())
    for row in rows:
        path=root/row["path"]
        if (not path.is_file() or path.stat().st_size!=row["bytes"] or
                hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]):
            raise ValueError(f"artifact integrity failure: {row['path']}")
