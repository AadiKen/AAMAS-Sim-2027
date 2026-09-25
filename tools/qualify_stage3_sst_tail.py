"""Freeze the existing 57–80 s SST baseline after read-only requalification.

This postprocessor never invokes OpenFOAM or changes CFD inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter
from bcod_sim.vessel_generation.quality import (
    STATIONARITY_GATE_VERSION, numerical_history_gate, wrench_stationarity,
)

ROOT = Path("stage3_results/wamv-stage3-generalized")
CASE = ROOT / "cfd/sst-baseline/ef3e2555a9e8824f36a7c5f3e5ead4145f29bd536ac036f46cfc69e1fd9e356e"
WINDOW = (57.0, 80.0)
FORCE_SCALE = 1431.0
MOMENT_SCALE = 7057.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    history = np.asarray(OpenFOAMAdapter.force_history(CASE), dtype="<f8")
    if history.shape[1] != 7 or np.any(np.diff(history[:, 0]) <= 0) or abs(history[-1, 0] - 80) > 1e-6:
        raise ValueError("Incomplete or discontinuous 0–80 s six-axis wrench history")
    selected = history[(history[:, 0] >= WINDOW[0]) & (history[:, 0] <= WINDOW[1])]
    if selected[0, 0] > 57.001 or selected[-1, 0] < 79.999:
        raise ValueError("Incomplete 57–80 s fitting window")
    def check(end: float, duration: float):
        subset = selected[selected[:, 0] <= end + 1e-8]
        return wrench_stationarity(subset[:, 0], subset[:, 1:],
            force_scale_n=FORCE_SCALE, moment_scale_nm=MOMENT_SCALE, window_s=duration)
    whole = check(80., 23.)
    rolling = []
    for duration in (8., 10., 15., 20.):
        for end in np.arange(WINDOW[0] + duration, 80.0001, .25):
            gate = check(float(end), duration)
            rolling.append({"start_s": round(float(end-duration), 3), "end_s": round(float(end), 3),
                            "window_s": duration, "accepted": gate.accepted,
                            "channels": [channel.accepted for channel in gate.channels]})
    diagnostics = OpenFOAMAdapter.diagnose_case(CASE)
    numerical = numerical_history_gate(diagnostics, expected_start_s=0., expected_end_s=80.,
        max_courant=.5, max_alpha_courant=.5, max_timestep_s=.0006)
    if not (whole.accepted and all(item["accepted"] for item in rolling) and numerical["accepted"]):
        raise RuntimeError(json.dumps({"full_window": whole.accepted,
            "failed_rolling": [item for item in rolling if not item["accepted"]],
            "numerical": numerical}, indent=2))
    metadata = json.loads((CASE / "case_metadata.json").read_text())
    case_files = ("case_metadata.json", "system/controlDict", "system/fvSchemes",
                  "system/fvSolution", "system/blockMeshDict", "constant/momentumTransport")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    qualified = {
        "status": "CFD_QUALIFIED", "case_id": metadata["case_id"],
        "fitting_window_s": list(WINDOW), "selected_sample_count": len(selected),
        "selected_samples_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "mean_six_axis_wrench": selected[:, 1:].mean(axis=0).tolist(),
        "units": ["N", "N", "N", "N m", "N m", "N m"],
        "stationarity_gate_version": STATIONARITY_GATE_VERSION,
        "unchanged_material_relative_mean_and_slope_limit": .03,
        "physical_scales": {"force_n": FORCE_SCALE, "moment_nm": MOMENT_SCALE},
        "all_six_full_window_accepted": whole.accepted,
        "all_six_rolling_accepted": True, "rolling_window_count": len(rolling),
        "rolling_durations_s": [8, 10, 15, 20], "rolling_endpoint_step_s": .25,
        "numerical_gate": numerical,
        "source_geometry_sha256": metadata["geometry_sha256"],
        "openfoam_image_identity": manifest.get("openfoam", {}).get("identity"),
        "adapter_version": metadata["adapter_version"],
        "case_file_sha256": {name: sha(CASE / name) for name in case_files},
        "gate_source_sha256": sha(Path("src/bcod_sim/vessel_generation/quality.py")),
        "parser_source_sha256": sha(Path("src/bcod_sim/vessel_generation/cfd.py")),
    }
    # Parser checks both the sample hash and recomputed mean before yielding an
    # observation, so no earlier transient samples can enter a fit by accident.
    target = CASE / "qualified_fitting_window.json"
    target.write_text(json.dumps(qualified, indent=2) + "\n")
    parsed = OpenFOAMAdapter(CASE.parent).parse_case(CASE,debug_last_sample=True)
    assert parsed.fitting_window_s == WINDOW
    output = CASE.parent / "qualification_57_80"
    output.mkdir(exist_ok=True)
    (output / "rolling_gate.json").write_text(json.dumps(rolling, indent=2) + "\n")
    summary = {"status": qualified["status"], "case_id": metadata["case_id"],
        "fitting_window_s": list(WINDOW), "mean_six_axis_wrench": qualified["mean_six_axis_wrench"],
        "channel_results": [{"name": name, "regime": channel.regime,
                             "mean": channel.mean, "normalized_mean_change": channel.normalized_mean_change,
                             "normalized_slope": channel.normalized_slope,
                             "normalized_std": channel.normalized_standard_deviation,
                             "accepted": channel.accepted}
                            for name, channel in zip(("Fx", "Fy", "Fz", "Mx", "My", "Mz"), whole.channels)],
        "rolling_windows_passed": len(rolling), "numerical_gate": numerical,
        "parser_selected_window": list(parsed.fitting_window_s or ())}
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
