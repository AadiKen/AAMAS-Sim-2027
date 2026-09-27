"""Explicit local or Slurm execution of a frozen Spec A case directory."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time
import re
import os
import shlex

import numpy as np

from .extract_forces import case_force_history, qualify_force_tail


def foam_command(command, case):
    """Optional Foundation 11 container execution (serial or MPI)."""
    if os.environ.get("SPEC_A_DOCKER") != "1":
        return command
    command = ["/case" if x == str(case) else x for x in command]
    return ["docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "/bin/bash",
            "-v", f"{case.resolve()}:/case", "-w", "/case", "openfoam/openfoam11-paraview510:11",
            "-lc", "source /opt/openfoam11/etc/bashrc; " + shlex.join(command)]


def cell_count(case):
    match = re.search(r"^\s*cells:\s*(\d+)", (case/"log.checkMesh").read_text(), re.M)
    return int(match[1]) if match else None


def _run(command: list[str], case: Path, log: str) -> float:
    start = time.monotonic()
    with (case/log).open("w") as stream:
        result = subprocess.run(foam_command(command, case), cwd=case, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed in {case}; see {log}")
    return time.monotonic()-start


def mesh_once(cases: list[Path], *, mesh_index: int = 0) -> dict:
    if not cases:
        raise ValueError("No cases")
    profiles = {json.loads((p/"case_config.json").read_text()).get("mesh_profile") for p in cases}
    if len(profiles) != 1:
        raise ValueError("Cannot share a mesh across different profiles")
    source = cases[mesh_index]
    elapsed = _run(["blockMesh"], source, "log.blockMesh")
    elapsed += _run(["snappyHexMesh", "-overwrite"], source, "log.snappyHexMesh")
    elapsed += _run(["checkMesh"], source, "log.checkMesh")
    check = (source/"log.checkMesh").read_text()
    if "Mesh OK." not in check:
        raise RuntimeError("Mesh check failed; CFD is prohibited")
    for case in cases:
        if case == source:
            continue
        if (case/"constant/polyMesh").exists():
            raise FileExistsError(f"Existing mesh in {case}")
        shutil.copytree(source/"constant/polyMesh", case/"constant/polyMesh")
        shutil.copy2(source/"log.checkMesh", case/"log.checkMesh")
    return {"source": str(source), "mesh_seconds": elapsed,
            "case_count": len(cases)}


def _edit_control(case: Path, *, restart: bool, end_time: int) -> None:
    path = case/"system/controlDict"
    source = path.read_text()
    source = re.sub(r"\bstartFrom\s+\w+\s*;",
                    "startFrom latestTime;" if restart else "startFrom startTime;", source)
    source = re.sub(r"\bendTime\s+[-+0-9.eE]+\s*;", f"endTime {end_time};", source)
    path.write_text(source)


def _halve_relaxation(case: Path) -> None:
    path = case/"system/fvSolution"
    source = path.read_text()
    start = source.index("relaxationFactors")
    before, tail = source[:start], source[start:]
    tail = re.sub(r"\b(p|U|k|omega)\s+([0-9.]+)\s*;",
                  lambda m: f"{m[1]} {float(m[2])/2:.12g};", tail)
    path.write_text(before+tail)


def run_case(case: Path, *, minimum_iterations: int = 3000, core_count: int = 1,
             reference_y_n: tuple[float, float] | None = None) -> dict:
    """One initial solve and at most one under-relaxed continuation."""
    case = Path(case).resolve()
    config = json.loads((case/"case_config.json").read_text())
    if config.get("retired"):
        raise RuntimeError("Pre-audit case is retired; regenerate from the current template")
    if (not (case/"constant/polyMesh").exists() or not (case/"log.checkMesh").exists() or
        "Mesh OK." not in (case/"log.checkMesh").read_text()):
        raise RuntimeError("Case has no checked mesh")
    if minimum_iterations < 3000:
        raise ValueError("Spec A requires at least 3000 iterations")
    if (case/"run_result.json").exists():
        raise FileExistsError("Case already executed; inspect its saved result before rerunning")
    if isinstance(core_count, bool) or not isinstance(core_count, int) or core_count < 1:
        raise ValueError("core_count must be a positive integer")
    setup_seconds = 0.
    if core_count > 1:
        if any(case.glob("processor[0-9]*")):
            raise FileExistsError("Existing decomposition; inspect before starting a new run")
        (case/"system/decomposeParDict").write_text(
            'FoamFile {version 2.0; format ascii; class dictionary; object decomposeParDict;}\n'
            f'numberOfSubdomains {core_count}; method scotch;\n')
        setup_seconds += _run(["decomposePar", "-case", str(case)], case, "log.decomposePar")
    elapsed = 0.
    prior_path = case/"interrupted_runs.json"
    segments = json.loads(prior_path.read_text()) if prior_path.exists() else []
    attempts = []
    solver_logs = []
    for attempt in range(2):
        if attempt:
            checkpoints = [float(p.name) for p in case.iterdir()
                           if p.is_dir() and re.fullmatch(r"[0-9.eE+-]+", p.name)]
            if not checkpoints or max(checkpoints) <= 0:
                break
            _halve_relaxation(case)
            end_time = int(max(checkpoints)) + minimum_iterations
        else:
            end_time = minimum_iterations
        _edit_control(case, restart=bool(attempt), end_time=end_time)
        log = "log.simpleFoam" if attempt == 0 else "log.simpleFoam.retry"
        started = time.monotonic()
        command = ["simpleFoam", "-case", str(case)]
        if core_count > 1:
            command = ["mpirun", "--allow-run-as-root", "-np", str(core_count), *command, "-parallel"]
        with (case/log).open("w") as stream:
            process = subprocess.run(foam_command(command, case), cwd=case,
                                     stdout=stream, stderr=subprocess.STDOUT)
        duration = time.monotonic()-started
        elapsed += duration
        solver_logs.append((case/log).read_text())
        segments.append({"log": log, "wall_seconds": duration, "core_count": core_count,
                         "core_hours": duration*core_count/3600,
                         "iterations": len(re.findall(r"^Time =", solver_logs[-1], re.M))})
        if core_count > 1 and process.returncode == 0:
            setup_seconds += _run(["reconstructPar", "-case", str(case), "-latestTime"],
                                  case, "log.reconstructPar" if not attempt else "log.reconstructPar.retry")
        try:
            history = case_force_history(case)
            qualified = qualify_force_tail(history, residual_log=solver_logs[-1],
                                            reference_y_n=reference_y_n)
        except ValueError as error:
            qualified = {"status": "failed", "reason": str(error)}
        if process.returncode or not re.search(r"^End\s*$", solver_logs[-1], re.M):
            qualified = {"status": "failed", "reason": "solver_failed", "exit_code": process.returncode}
        attempts.append({"attempt": attempt, **qualified})
        if qualified["status"] != "failed":
            break
    qualified = attempts[-1]
    output = {"case": str(case), "state": config["state"],
              "wall_seconds": sum(s["wall_seconds"] for s in segments) + setup_seconds,
              "core_hours_serial": sum(s["wall_seconds"] for s in segments if s["core_count"] == 1)/3600,
              "core_hours": sum(s["core_hours"] for s in segments) + setup_seconds/3600,
              "core_count": core_count, "cell_count": cell_count(case),
              "mesh_profile": config.get("mesh_profile"), "execution_segments": segments,
              "decomposition_reconstruction_seconds": setup_seconds,
              "iterations": sum(s["iterations"] for s in segments),
              "attempts": attempts, "qualification": qualified, "status": qualified["status"]}
    (case/"run_result.json").write_text(json.dumps(output, indent=2)+"\n")
    return output


def run_or_reuse(case: Path, *, reference_y_n=None, core_count=1) -> dict:
    """Reuse only the selected, qualified sentinel, including Slurm execution."""
    saved = case/"run_result.json"
    if not saved.exists():
        return run_case(case, reference_y_n=reference_y_n, core_count=core_count)
    result = json.loads(saved.read_text())
    config = json.loads((case/"case_config.json").read_text())
    same_case = Path(result.get("case", "")).resolve() == case.resolve()
    matching = (result.get("state") == config["state"] and
                result.get("mesh_profile") == config.get("mesh_profile") and
                result.get("cell_count") == cell_count(case) and
                result.get("status") in ("converged", "oscillatory") and
                result.get("qualification", {}).get("mean_foam") is not None)
    sentinel = (result.get("sentinel_reused") and config["state"]["beta_deg"] == 8
                and config["state"]["r_prime"] == 0)
    if not matching or not (sentinel or same_case):
        raise ValueError("Saved result does not match the selected checked case")
    return result


def run_matrix(cases: list[Path], *, core_count=1) -> list[dict]:
    """Run the 8° drift first to normalize near-zero straight loads."""
    if not cases:
        raise ValueError("No cases")
    profiles = {json.loads((p/"case_config.json").read_text()).get("mesh_profile") for p in cases}
    if len(profiles) != 1:
        raise ValueError("Production matrix cannot mix mesh profiles")
    def anchor(case):
        state = json.loads((case/"case_config.json").read_text())["state"]
        return state["beta_deg"] == 8 and state["r_prime"] == 0
    ordered = sorted(cases, key=lambda p: (not anchor(p), str(p)))
    results, reference = [], None
    for case in ordered:
        result = run_or_reuse(case, reference_y_n=reference, core_count=core_count)
        results.append(result)
        if anchor(case) and result["status"] != "failed":
            mean = result["qualification"]["mean_foam"]
            reference = (mean[1], mean[5])
        (case.parent/"runtime.json").write_text(json.dumps(results, indent=2)+"\n")
        if sum(row["status"] == "failed" for row in results) > 2:
            raise RuntimeError("More than two CFD cases failed; fix the template before continuing")
    return results


def write_slurm_array(path: Path, cases: list[Path], *, partition: str | None = None,
                      time_limit: str = "24:00:00") -> None:
    if not cases:
        raise ValueError("No cases")
    # Serial array prevents simultaneous jobs from bypassing the three-failure
    # campaign stop. Independent clusters can choose a reviewed broader mode.
    lines = ["#!/bin/bash", f"#SBATCH --array=0-{len(cases)-1}%1",
             f"#SBATCH --time={time_limit}", "#SBATCH --cpus-per-task=1"]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    ordered = sorted(cases, key=lambda p: (
        json.loads((p/"case_config.json").read_text())["state"]["family"] != "drift" or
        json.loads((p/"case_config.json").read_text())["state"]["beta_deg"] != 8, str(p)))
    quoted = " ".join("'"+str(p.resolve()).replace("'", "'\\''")+"'" for p in ordered)
    lines += ["set -euo pipefail", f"cases=({quoted})",
              'case_dir="${cases[$SLURM_ARRAY_TASK_ID]}"',
              'python -m bcod_sim.vessel_generation.spec_a.run_matrix "$case_dir"']
    Path(path).write_text("\n".join(lines)+"\n")


def write_family_mesh_check(path: Path, *, family: str,
                            observations: dict[str, dict[str, tuple[float, float]]]) -> bool:
    """Write one family-level three-mesh comparison for β=8° and r'=0.4."""
    expected = {"drift_8", "yaw_0.4"}
    if set(observations) != expected or any(set(row) != {"coarse", "medium", "fine"}
                                           for row in observations.values()):
        raise ValueError("Expected coarse/medium/fine at β=8° and r'=0.4")
    lines = [f"# {family} mesh check", "", "| State | Coarse Y/N | Medium Y/N | Fine Y/N | Medium/fine max error |",
             "|---|---|---|---|---:|"]
    passed = True
    for name in ("drift_8", "yaw_0.4"):
        row = observations[name]
        if any(np.shape(pair) != (2,) or not np.isfinite(pair).all() for pair in row.values()):
            raise ValueError("Finite Y/N mesh observations required")
        medium, fine = row["medium"], row["fine"]
        errors = [abs(a-b)/max(abs(b), 1e-12) for a, b in zip(medium, fine)]
        passed &= max(errors) <= .05
        lines.append(f"| {name} | {row['coarse']} | {medium} | {fine} | {max(errors):.2%} |")
    lines += ["", f"Medium template frozen: **{passed}**", "",
              "The case template must be developer-reviewed if this gate fails.", ""]
    Path(path).write_text("\n".join(lines))
    return passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    siblings = [json.loads(p.read_text()) for p in args.case.parent.glob("*/run_result.json")]
    if sum(row["status"] == "failed" for row in siblings) > 2:
        raise SystemExit("More than two failed cases; template requires developer review")
    anchor = next((row for row in siblings if row["state"]["beta_deg"] == 8 and
                   row["state"]["r_prime"] == 0 and row["status"] != "failed"), None)
    reference = ((anchor["qualification"]["mean_foam"][1], anchor["qualification"]["mean_foam"][5])
                 if anchor else None)
    print(json.dumps(run_or_reuse(args.case, reference_y_n=reference), indent=2))
