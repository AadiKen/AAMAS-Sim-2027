"""BCOD vessel package writer with coefficient-level provenance."""

from dataclasses import asdict
from pathlib import Path
import json
import yaml

from .fitting import GeneralFitResult
from .models import CanonicalVessel


def write_vessel_package(output: str|Path, vessel: CanonicalVessel, fit: GeneralFitResult, *,
                         geometry_details: dict, case_ids: tuple[str,...], solver: dict,
                         fit_method="ordinary least squares") -> Path:
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    coefficients={"added_mass":fit.added_mass,"linear_damping_matrix":fit.linear_damping,
        "quadratic_damping":fit.quadratic_damping,"coupled_damping":fit.coupled_damping,
        "estimates":[asdict(item) for item in fit.estimates],"residual_rms":fit.residual_rms,
        "condition_numbers":fit.condition_numbers}
    provenance={"geometry_hash":vessel.geometry.content_hash,"cfd_case_ids":case_ids,
        "solver":solver,"fit_method":fit_method,"coefficient_uncertainty":{
            item.name:item.uncertainty for item in fit.estimates}}
    (root/"vessel.yaml").write_text(yaml.safe_dump(vessel.model_dump(mode="json"),sort_keys=False))
    (root/"coefficients.json").write_text(json.dumps(coefficients,indent=2,sort_keys=True))
    (root/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True))
    (root/"geometry.json").write_text(json.dumps(geometry_details,indent=2,sort_keys=True))
    lines=[f"# {vessel.id} coefficient fit","",f"Geometry SHA-256: `{vessel.geometry.content_hash}`",
        f"Solver: `{solver.get('name','unknown')} {solver.get('version','unknown')}`",f"Fit: {fit_method}","",
        "Only quality-accepted source cases were included.","",f"Maximum condition number: {max(fit.condition_numbers):.6g}",
        f"Maximum residual RMS: {max(fit.residual_rms):.6g}","",f"Source cases: {', '.join(case_ids)}"]
    (root/"fit_report.md").write_text("\n".join(lines)+"\n")
    return root
