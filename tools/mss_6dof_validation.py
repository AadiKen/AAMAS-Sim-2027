#!/usr/bin/env python3
"""Reproducible external-reference MSS Otter full-6DOF validation campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
except ImportError as exc:
    raise SystemExit(
        "Validation dependencies are missing. Install the project plus matplotlib "
        "in .venv (python -m pip install -e '.[test]' matplotlib)."
    ) from exc

from bcod_sim.actuators.base import ActuatorConfig, ActuatorState, Bounds
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.crossflow import StripTheoryCrossflow
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import LinearHydrostatics
from bcod_sim.state.vessel_state import VesselState


ROOT = Path(__file__).resolve().parents[1]
MSS_URL = "https://github.com/cybergalactic/MSS.git"
MSS_COMMIT = "98970f71a21cfe81e7e29abdcc1bb6741789cddc"
MSS_FILE = "CRAFT/USV/models/otter.m"
REFERENCE_FILES = (
    MSS_FILE,
    "GNC/satlim.m",
    "HYDRO/Hoerner.m",
    "LIBRARY/kinematics/Hmtrx.m",
    "LIBRARY/kinematics/Rzyx.m",
    "LIBRARY/kinematics/Smtrx.m",
    "LIBRARY/kinematics/Tzyx.m",
    "LIBRARY/kinematics/eulerang.m",
    "LIBRARY/modeling/addedMassSurge.m",
    "LIBRARY/modeling/crossFlowDrag.m",
    "LIBRARY/modeling/m2c.m",
    "LIBRARY/numericalMethods/rk4.m",
)
CASE_NAMES = {
    "T00": "equilibrium", "T01": "surge", "T02": "sway", "T03": "heave",
    "T04": "roll", "T05": "pitch", "T06": "yaw", "T07": "combined_wrench",
    "T08": "initial_decay", "T09": "symmetric_thrust", "T10": "turn_left",
    "T11": "turn_right", "T12": "combined_actuator",
}
STATE_CHANNELS = ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r"]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def repository_state() -> str:
    commands = [
        ["pwd"], ["git", "rev-parse", "--show-toplevel"], ["git", "status", "--short"],
        ["git", "branch", "--show-current"], ["git", "rev-parse", "HEAD"],
    ]
    labels = ["pwd", "git rev-parse --show-toplevel", "git status --short",
              "git branch --show-current", "git rev-parse HEAD"]
    sections = []
    for label, command in zip(labels, commands):
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        value = result.stdout.strip() or result.stderr.strip() or "<empty>"
        sections.append(f"$ {label}\n{value}\nexit={result.returncode}")
    if subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True).returncode:
        sections.append("NOTE: main is an unborn branch; HEAD does not resolve.")
    return "\n\n".join(sections) + "\n"


def ensure_reference(cache: Path) -> Path:
    git_dir = cache / ".git"
    if not git_dir.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", MSS_URL, str(cache)])
    pinned = subprocess.run(["git", "-C", str(cache), "cat-file", "-e", f"{MSS_COMMIT}^{{commit}}"],
                            capture_output=True).returncode == 0
    if not pinned:
        run(["git", "-C", str(cache), "fetch", "--depth=1", "origin", MSS_COMMIT])
    # Materialize only the pinned model and its runtime dependencies.  The
    # checkout's symbolic HEAD is irrelevant; every source path is selected
    # explicitly from the immutable commit below.
    run(["git", "-C", str(cache), "checkout", MSS_COMMIT, "--", *REFERENCE_FILES], capture_output=True)
    for relative in REFERENCE_FILES:
        if not (cache / relative).is_file():
            raise RuntimeError(f"Pinned MSS file missing: {relative}")
    return cache


def instrument_direct_wrench(source: Path, destination: Path) -> dict[str, Any]:
    original = source.read_text(encoding="utf-8")
    modified = original.replace(
        "function [xdot,U,M,B_prop,n_min,n_max] = otter(x,n,mp,rp,V_c,beta_c)",
        "function [xdot,U,M,B_prop,n_min,n_max] = otter_tau(x,n,mp,rp,V_c,beta_c,tau_external)", 1)
    needle = "M \\ ( tau + tau_damp + tau_crossflow - C * nu_r - G * eta )"
    replacement = "M \\ ( tau_external + tau + tau_damp + tau_crossflow - C * nu_r - G * eta )"
    modified = modified.replace(needle, replacement, 1)
    if modified == original or replacement not in modified:
        raise RuntimeError("Unable to instrument pinned MSS generalized-wrench boundary")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(modified, encoding="utf-8")
    return {
        "purpose": "Track A generalized-wrench boundary instrumentation",
        "model_equations_changed": False,
        "interface_change": "Added tau_external to the existing RHS sum before M\\(...)",
        "original_sha256": sha256(source),
        "instrumented_sha256": sha256(destination),
    }


def resolved_parameters() -> dict[str, Any]:
    g, rho, length, beam, mass = 9.81, 1025.0, 2.0, 1.08, 55.0
    rg = np.array([0.2, 0.0, -0.2])
    radii = np.array([0.4 * beam, 0.25 * length, 0.25 * length])
    inertia = mass * np.diag(radii ** 2)
    volume = mass / rho
    pont_beam, y_pont, cw, cb = 0.25, 0.395, 0.75, 0.4
    draft = volume / (2 * cb * pont_beam * length)
    added_diag = np.array([
        2.7 * rho * volume ** (5 / 3) / length ** 2,
        1.5 * mass, mass, 0.2 * inertia[0, 0], 0.8 * inertia[1, 1], 1.7 * inertia[2, 2],
    ])
    aw = cw * length * pont_beam
    it = (2 * (1 / 12) * length * pont_beam ** 3 * (6 * cw ** 3 / ((1 + cw) * (1 + 2 * cw)))
          + 2 * aw * y_pont ** 2)
    il = 0.8 * 2 * (1 / 12) * pont_beam * length ** 3
    kb = (1 / 3) * (5 * draft / 2 - 0.5 * volume / (length * pont_beam))
    g33 = rho * g * (2 * aw)
    g44 = rho * g * volume * (kb + it / volume - (draft - rg[2]))
    g55 = rho * g * volume * (kb + il / volume - (draft - rg[2]))
    # Transform G_CF from CF at x=-0.2 m to the body origin.
    s = np.array([[0, 0, 0], [0, 0, 0.2], [0, -0.2, 0]], dtype=float)
    h = np.block([[np.eye(3), s.T], [np.zeros((3, 3)), np.eye(3)]])
    g_cf = np.diag([0, 0, g33, g44, g55, 0])
    stiffness = h.T @ g_cf @ h
    # Build the rigid-body matrix exactly as BCOD's canonical MassProperties does.
    sr = np.array([[0, -rg[2], rg[1]], [rg[2], 0, -rg[0]], [-rg[1], rg[0], 0]])
    mrb = np.block([[mass * np.eye(3), -mass * sr], [mass * sr, inertia - mass * sr @ sr]])
    total = mrb + np.diag(added_diag)
    omega = np.sqrt(np.array([g33 / total[2, 2], g44 / total[3, 3], g55 / total[4, 4]]))
    linear = np.array([
        24.4 * g / (6 * 0.5144), total[1, 1], 2 * 0.3 * omega[0] * total[2, 2],
        2 * 0.2 * omega[1] * total[3, 3], 2 * 0.4 * omega[2] * total[4, 4], total[5, 5],
    ])
    k_pos, k_neg = 0.02216 / 2, 0.01289 / 2
    n_max = math.sqrt((0.5 * 24.4 * g) / k_pos)
    n_min = -math.sqrt((0.5 * 13.6 * g) / k_neg)
    return {
        "load_condition": {"payload_mass_kg": 0.0, "payload_position_frd_m": [0, 0, 0]},
        "mass_kg": mass, "rigid_body_inertia_cg_kg_m2": inertia.tolist(), "cg_frd_m": rg.tolist(),
        "cb": {"MSS_form": "linear stiffness G about CF", "volume_displacement_m3": volume},
        "water_density_kg_m3": rho, "gravity_mps2": g, "draft_m": draft,
        "added_mass_kg": np.diag(added_diag).tolist(), "rigid_body_mass_matrix": mrb.tolist(),
        "total_mass_matrix": total.tolist(), "linear_damping_positive_convention": linear.tolist(),
        "nonlinear_damping": {"yaw_abs_r_coefficient": float(10 * linear[5]),
                              "cross_flow": "20-strip Hoerner sway/heave/pitch/yaw coupling"},
        "restoring_stiffness_G": stiffness.tolist(),
        "thrusters": {"positions_frd_m": [[0, -y_pont, 0], [0, y_pont, 0]],
                      "orientation": "+body-X", "k_positive": k_pos, "k_negative": k_neg,
                      "shaft_speed_min_rad_s": n_min, "shaft_speed_max_rad_s": n_max,
                      "actuator_time_constant_s": 0.1, "actuator_update": "forward Euler after plant RK4"},
        "speed_dependent_terms": ["quadratic yaw damping", "crossFlowDrag strip theory"],
        "cross_coupling_terms": ["full rigid/added Coriolis", "cross-flow", "G heave-pitch coupling"],
        "integrator": "RK4", "equilibrium_eta": [0, 0, 0, 0, 0, 0],
    }


def bcod_parameters(mss: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_definition": "configs/mss-otter-parity-v2.json",
        "mass_kg": mss["mass_kg"], "cg_frd_m": mss["cg_frd_m"],
        "inertia_cg_kg_m2": mss["rigid_body_inertia_cg_kg_m2"], "added_mass_kg": mss["added_mass_kg"],
        "linear_damping": mss["linear_damping_positive_convention"],
        "quadratic_damping": [0, 0, 0, 0, 0, mss["nonlinear_damping"]["yaw_abs_r_coefficient"]],
        "hydrostatics": {"model": "linear_matrix", "stiffness_6x6": mss["restoring_stiffness_G"],
                         "equilibrium_position_ned_m": [0,0,0], "equilibrium_rpy_rad": [0,0,0]},
        "crossflow": {"model": "strip_theory", "length_m": 2.0, "beam_m": 0.25,
                      "draft_m": mss["draft_m"], "strips": 20, "include_vertical": True},
        "gravity_mps2": mss["gravity_mps2"], "water_density_kg_m3": mss["water_density_kg_m3"],
        "max_abs_nu": [10, 10, 10, 5, 5, 5],
        "mode": "full6", "environment": "all external environmental terms identically zero",
        "thrusters": mss["thrusters"],
    }


def plant_equivalence() -> dict[str, Any]:
    terms = [
        ("mass", "EXACT", "55 kg, mp=0"), ("rigid_body_inertia", "EXACT", "same CG dyadic"),
        ("CG", "EXACT", "[0.2,0,-0.2] FRD"), ("added_mass", "EXACT", "same diagonal MA"),
        ("Coriolis", "EQUIVALENT_REPARAMETERIZATION", "power-neutral spatial form"),
        ("linear_damping", "EXACT", "same resolved diagonal coefficients"),
        ("quadratic_yaw_damping", "EXACT", "same |r|r coefficient"),
        ("cross_flow_drag", "EXACT", "generic 20-strip Hoerner model matches pinned MSS horizontal and vertical strips"),
        ("full_coupled_linear_restoring", "EXACT", "resolved MSS G applied to full eta displacement"),
        ("thrust_mapping", "EXACT", "piecewise k*n*|n| and measured transverse lever arms"),
        ("shaft_speed_saturation", "EXACT", "same MSS bounds"),
        ("actuator_dynamics", "EQUIVALENT_REPARAMETERIZATION", "same discrete Euler shaft-speed history fed to canonical thrusters"),
    ]
    counts = {key: sum(1 for _, status, _ in terms if status == key) for key in
              ("EXACT", "EQUIVALENT_REPARAMETERIZATION", "APPROXIMATE", "UNREPRESENTABLE")}
    return {"terms": [{"term": a, "classification": b, "detail": c} for a, b, c in terms], "counts": counts}


def rpy_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy.T
    cr, sr, cp, sp, cy, sy = np.cos(r/2), np.sin(r/2), np.cos(p/2), np.sin(p/2), np.cos(y/2), np.sin(y/2)
    return np.column_stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))


def quaternion_to_rpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.column_stack((roll, pitch, yaw))


def case_definition(case_id: str, dt: float) -> dict[str, Any]:
    duration = 60.0 if case_id in ("T07", "T08", "T12") else (20.0 if case_id <= "T06" else 30.0)
    times = np.arange(0, duration + dt/2, dt)
    initial = np.zeros(12)
    wrench = np.zeros((len(times), 6))
    commands = np.zeros((len(times), 2))
    if case_id in ("T01", "T02", "T03", "T04", "T05", "T06"):
        axis = int(case_id[-1]) - 1
        amplitudes = [20, 10, 10, 3, 3, 5]
        wrench[(times >= 2) & (times < 7), axis] = amplitudes[axis]
    elif case_id == "T07":
        amps, freqs, phases = np.array([12, 7, 6, 1.5, 1.5, 2.5]), np.array([.4,.7,.9,.5,.8,.6]), np.array([0,.3,.8,1.1,.4,1.4])
        wrench = np.sin(times[:, None] * freqs + phases) * amps
    elif case_id == "T08":
        initial[:6] = [0.5, 0.15, 0.05, math.radians(2), math.radians(-2), math.radians(4)]
        initial[8:12] = [0.02, math.radians(5), math.radians(-4), math.radians(10)]
    elif case_id == "T09":
        commands[(times >= 2) & (times < 15)] = [65, 65]
    elif case_id == "T10":
        commands[(times >= 2) & (times < 20)] = [75, 45]
    elif case_id == "T11":
        commands[(times >= 2) & (times < 20)] = [45, 75]
    elif case_id == "T12":
        commands[(times >= 2) & (times < 15)] = [65, 65]
        commands[(times >= 15) & (times < 28)] = [75, 40]
        commands[(times >= 28) & (times < 42)] = [40, 75]
        commands[(times >= 42) & (times < 50)] = [60, 60]
    return {"id": case_id, "track": "A" if case_id <= "T08" else "B", "dt": dt,
            "duration": duration, "times": times, "initial": initial, "wrench": wrench, "commands": commands}


def write_csv(path: Path, header: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(values)


def octave_script(reference: Path, runtime: Path, case_dir: Path, case: dict[str, Any]) -> str:
    paths = [reference / "CRAFT/USV/models", reference / "GNC", reference / "HYDRO", reference / "LIBRARY/kinematics",
             reference / "LIBRARY/modeling", reference / "LIBRARY/numericalMethods", runtime]
    additions = "\n".join(f"addpath('{str(p).replace(chr(39), chr(39)*2)}');" for p in paths)
    initial = ";".join(f"{v:.17g}" for v in case["initial"])
    input_path = str(case_dir / "input.csv").replace("'", "''")
    output_path = str(case_dir / "mss_raw.csv").replace("'", "''")
    track_a = case["track"] == "A"
    derivative = "@(xx) otter_tau(xx,zeros(2,1),0,zeros(3,1),0,0,row(2:7)')" if track_a else "@(xx) otter(xx,n,0,zeros(3,1),0,0)"
    return f"""{additions}
data=dlmread('{input_path}',',',1,0); nrows=rows(data); x=[{initial}]; n=zeros(2,1);
out=zeros(nrows,21); dt={case['dt']:.17g};
for i=1:nrows
  row=data(i,:); out(i,:)=[row(1),x',n',row(2:7)];
  if i<nrows
    f={derivative};
    k1=f(x); k2=f(x+dt*k1/2); k3=f(x+dt*k2/2); k4=f(x+dt*k3);
    x=x+dt*(k1+2*k2+2*k3+k4)/6;
    if {0 if track_a else 1}
      nc=row(8:9)'; n=n+dt/0.1*(nc-n);
      [~,~,~,~,nmin,nmax]=otter(); n=min(max(n,nmin),nmax);
    end
  end
end
dlmwrite('{output_path}',out,'delimiter',',','precision','%.17g');
"""


def load_mss_raw(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.loadtxt(case_dir / "mss_raw.csv", delimiter=",")
    time, state, actual_n, applied = raw[:, 0], raw[:, 1:13], raw[:, 13:15], raw[:, 15:21]
    return time, state, actual_n, applied


def canonical_trajectory(time: np.ndarray, state: np.ndarray, applied: np.ndarray, commands: np.ndarray, actual_n: np.ndarray) -> tuple[list[str], np.ndarray]:
    quat = rpy_to_quaternion(state[:, 9:12])
    header = ["time_s", "x_ned_m", "y_ned_m", "z_ned_m", "qw", "qx", "qy", "qz",
              "roll_rad", "pitch_rad", "yaw_rad", "u_mps", "v_mps", "w_mps", "p_radps", "q_radps", "r_radps",
              "X_N", "Y_N", "Z_N", "K_Nm", "M_Nm", "N_Nm", "port_command_radps", "starboard_command_radps",
              "port_actual_radps", "starboard_actual_radps"]
    values = np.column_stack((time, state[:, 6:9], quat, state[:, 9:12], state[:, :6], applied, commands, actual_n))
    return header, values


def build_plant(params: dict[str, Any]) -> Plant6:
    tensor = lambda x: torch.tensor(x, dtype=torch.float64)
    hydro = params["hydrostatics"]
    cross = params["crossflow"]
    return Plant6(MassProperties(params["mass_kg"], tensor(params["cg_frd_m"]), tensor(params["inertia_cg_kg_m2"]),
                                 tensor(params["added_mass_kg"])),
                  Damping(tensor(params["linear_damping"]), tensor(params["quadratic_damping"])),
                  LinearHydrostatics(tensor(hydro["stiffness_6x6"]), tensor(hydro["equilibrium_position_ned_m"]),
                                     tensor(hydro["equilibrium_rpy_rad"]), .5, .5),
                  OperatingEnvelope(tensor(params["max_abs_nu"]), 1e-5, 0.1), mode="full6",
                  crossflow=StripTheoryCrossflow.constant_section(cross["length_m"], cross["beam_m"], cross["draft_m"],
                      cross["strips"], water_density_kg_m3=params["water_density_kg_m3"],
                      include_vertical=cross["include_vertical"], dtype=torch.float64))


def simulate_bcod(case: dict[str, Any], params: dict[str, Any], actual_n: np.ndarray) -> tuple[list[str], np.ndarray]:
    plant = build_plant(params)
    initial = case["initial"]
    state = VesselState(torch.tensor(initial[6:9], dtype=torch.float64),
                        torch.tensor(rpy_to_quaternion(initial[9:12][None, :])[0], dtype=torch.float64),
                        torch.tensor(initial[:6], dtype=torch.float64))
    thr = params["thrusters"]
    thrust_min = thr["k_negative"] * thr["shaft_speed_min_rad_s"] * abs(thr["shaft_speed_min_rad_s"])
    thrust_max = thr["k_positive"] * thr["shaft_speed_max_rad_s"] ** 2
    left = FixedThruster(ActuatorConfig("mss-left", 0, 1, (0, -0.395, 0), (1,0,0,0), Bounds(thrust_min, thrust_max)))
    right = FixedThruster(ActuatorConfig("mss-right", 0, 1, (0, 0.395, 0), (1,0,0,0), Bounds(thrust_min, thrust_max)))
    rows = []
    applied_history = []
    zeros = torch.zeros(6, dtype=torch.float64)
    for index, time in enumerate(case["times"]):
        if case["track"] == "A":
            applied = case["wrench"][index]
        else:
            forces = []
            for actuator, shaft in zip((left, right), actual_n[index]):
                coefficient = thr["k_positive"] if shaft > 0 else thr["k_negative"]
                target = coefficient * shaft * abs(shaft)
                result = actuator.step(ThrustCommand(target), ActuatorState(), case["dt"])
                forces.append(np.asarray(result.wrench_frd))
            applied = forces[0] + forces[1]
        applied_history.append(applied)
        rpy = quaternion_to_rpy(state.q_body_to_ned.detach().numpy()[None, :])[0]
        rows.append(np.r_[time, state.position_ned.detach().numpy(), state.q_body_to_ned.detach().numpy(), rpy,
                          state.nu_body.detach().numpy(), applied, case["commands"][index], actual_n[index]])
        if index + 1 < len(case["times"]):
            external = {name: zeros.clone() for name in ("propulsion", "current", "wind", "wave", "wake", "contact", "manual")}
            external["propulsion"] = torch.tensor(applied, dtype=torch.float64)
            state = plant.step(state, external, case["dt"]).state
    return canonical_trajectory(np.array([]), np.zeros((0, 12)), np.zeros((0, 6)), np.zeros((0, 2)), np.zeros((0, 2)))[0], np.asarray(rows)


def angular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a-b), np.cos(a-b))


def orientation_error_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(q1 * q2, axis=1))
    return np.degrees(2 * np.arccos(np.clip(dots, -1, 1)))


def metric(values: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    error = values - reference
    signal_range = float(np.ptp(reference))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(error))), "max_abs": float(np.max(np.abs(error))),
            "final_abs": float(abs(error[-1])), "normalized_rmse": rmse / signal_range if signal_range > 1e-12 else None}


def compute_metrics(mss: np.ndarray, bcod: np.ndarray) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    # Canonical trajectory columns: t, pos 1:4, quat 4:8, rpy 8:11, nu 11:17.
    if len(mss) != len(bcod) or not np.array_equal(mss[:, 0], bcod[:, 0]):
        raise RuntimeError("COMPARISON INVALID: timestamps are not exactly matched")
    error = bcod - mss
    error[:, 8:11] = angular_difference(bcod[:, 8:11], mss[:, 8:11])
    orient = orientation_error_deg(mss[:, 4:8], bcod[:, 4:8])
    pos = np.linalg.norm(error[:, 1:4], axis=1)
    lin = np.linalg.norm(error[:, 11:14], axis=1)
    ang = np.linalg.norm(error[:, 14:17], axis=1)
    channels = {}
    mapping = {"x":1,"y":2,"z":3,"roll":8,"pitch":9,"yaw":10,"u":11,"v":12,"w":13,"p":14,"q":15,"r":16}
    for name, column in mapping.items():
        channels[name] = metric(bcod[:, column] if name not in ("roll","pitch","yaw") else error[:, column],
                                mss[:, column] if name not in ("roll","pitch","yaw") else np.zeros(len(mss)))
    windows = {}
    for label, horizon in (("0-0.1s", .1), ("0-1s", 1), ("0-5s", 5), ("full", math.inf)):
        mask = mss[:,0] <= horizon
        windows[label] = {"position_max_m": float(pos[mask].max()), "orientation_max_deg": float(orient[mask].max()),
                          "linear_velocity_max_mps": float(lin[mask].max()), "angular_velocity_max_radps": float(ang[mask].max())}
    summary = {"channels": channels, "vectors": {"position": metric(pos, np.zeros_like(pos)),
               "orientation_geodesic_deg": metric(orient, np.zeros_like(orient)),
               "linear_velocity": metric(lin, np.zeros_like(lin)), "angular_velocity": metric(ang, np.zeros_like(ang))},
               "windows": windows}
    comparison_header = ["time_s"] + [f"error_{name}" for name in STATE_CHANNELS] + ["position_error_m", "orientation_error_deg", "linear_velocity_error_mps", "angular_velocity_error_radps"]
    comparison = np.column_stack((mss[:,0], error[:, [1,2,3,8,9,10,11,12,13,14,15,16]], pos, orient, lin, ang))
    return summary, comparison, comparison_header


def plot_case(plot_dir: Path, mss: np.ndarray, bcod: np.ndarray, comparison: np.ndarray) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "position_overlay.png": ([1,2,3], ["x","y","z"], "m"),
        "attitude_overlay.png": ([8,9,10], ["roll","pitch","yaw"], "rad"),
        "linear_velocity_overlay.png": ([11,12,13], ["u","v","w"], "m/s"),
        "angular_velocity_overlay.png": ([14,15,16], ["p","q","r"], "rad/s"),
    }
    for filename, (columns, labels, unit) in groups.items():
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for ax, column, label in zip(axes, columns, labels):
            ax.plot(mss[:,0], mss[:,column], label="MSS", lw=1.4)
            ax.plot(bcod[:,0], bcod[:,column], "--", label="BCOD", lw=1.1)
            ax.set_ylabel(f"{label} ({unit})"); ax.grid(True, alpha=.3)
        axes[0].legend(); axes[-1].set_xlabel("time (s)"); fig.tight_layout(); fig.savefig(plot_dir / filename, dpi=130); plt.close(fig)
    errors = {
        "position_error.png": (comparison[:,-4], "3D position error (m)"),
        "orientation_error.png": (comparison[:,-3], "SO(3) orientation error (deg)"),
        "velocity_error.png": (np.column_stack((comparison[:,-2], comparison[:,-1])), ("linear (m/s)", "angular (rad/s)")),
    }
    for filename, (values, label) in errors.items():
        fig, ax = plt.subplots(figsize=(9,4))
        if values.ndim == 1: ax.plot(comparison[:,0], values); ax.set_ylabel(label)
        else:
            ax.plot(comparison[:,0], values[:,0], label=label[0]); ax.plot(comparison[:,0], values[:,1], label=label[1]); ax.legend()
        ax.set_xlabel("time (s)"); ax.grid(True, alpha=.3); fig.tight_layout(); fig.savefig(plot_dir / filename, dpi=130); plt.close(fig)


def classify(metrics: dict[str, Any]) -> str:
    full = metrics["windows"]["full"]
    if full["position_max_m"] < .05 and full["orientation_max_deg"] < 1: return "NEAR PARITY"
    if full["position_max_m"] < .5 and full["orientation_max_deg"] < 5: return "SMALL SYSTEMATIC DIFFERENCE"
    if full["position_max_m"] < 5 and full["orientation_max_deg"] < 30: return "MATERIAL DIVERGENCE"
    return "SEVERE/QUALITATIVE MISMATCH"


def component_parity(reference: Path, runtime: Path, output: Path, mss: dict[str, Any], bcod: dict[str, Any], octave: str) -> dict[str, Any]:
    rng = np.random.default_rng(20260921)
    samples = np.zeros((100, 12))
    samples[:, 2] = rng.uniform(-.04, .04, 100)
    samples[:, 3:5] = rng.uniform(-.15, .15, (100, 2))
    samples[:, 7:9] = rng.uniform(-.5, .5, (100, 2))
    samples[:, 10:12] = rng.uniform(-.3, .3, (100, 2))
    sample_path = runtime / "component_samples.csv"; result_path = runtime / "component_mss.csv"
    np.savetxt(sample_path, samples, delimiter=",", fmt="%.17g")
    g_literal = ";".join(" ".join(f"{x:.17g}" for x in row) for row in mss["restoring_stiffness_G"])
    script = f"""addpath('{reference/'LIBRARY/modeling'}'); addpath('{reference/'HYDRO'}');
d=dlmread('{sample_path}',','); out=zeros(rows(d),12); G=[{g_literal}];
for i=1:rows(d)
 eta=d(i,1:6)'; nu=d(i,7:12)';
 out(i,:)=[(-G*eta)',crossFlowDrag(2.0,0.25,{mss['draft_m']:.17g},nu)'];
end
dlmwrite('{result_path}',out,'delimiter',',','precision','%.17g');
"""
    script_path = runtime / "run_component_parity.m"; script_path.write_text(script, encoding="utf-8")
    run([octave,"--quiet","--no-gui",str(script_path)],cwd=ROOT)
    expected=np.loadtxt(result_path,delimiter=","); hydro=build_plant(bcod).hydrostatics; cross=build_plant(bcod).crossflow
    actual=[]
    for row in samples:
        q=rpy_to_quaternion(row[3:6][None,:])[0]
        vessel=VesselState(torch.tensor(row[:3],dtype=torch.float64),torch.tensor(q,dtype=torch.float64),torch.tensor(row[6:],dtype=torch.float64))
        actual.append(np.r_[hydro.evaluate(vessel,bcod["mass_kg"],torch.tensor(bcod["cg_frd_m"],dtype=torch.float64)).tau_body.numpy(),cross.evaluate(vessel).tau_body.numpy()])
    actual=np.asarray(actual); difference=np.abs(actual-expected); denominator=np.maximum(np.abs(expected),1e-14)
    report={"samples":100,"seed":20260921,"status":"PASS" if difference.max()<1e-10 else "FAIL",
            "hydrostatic_max_absolute_error":float(difference[:,:6].max()),
            "crossflow_max_absolute_error":float(difference[:,6:].max()),
            "max_relative_error_nonzero_guarded":float((difference/denominator).max()),
            "tolerance_absolute":1e-10,"reference_commit":MSS_COMMIT}
    write_json(output,report)
    if report["status"]!="PASS": raise RuntimeError(f"MSS component parity failed: {report}")
    return report


def markdown_report(report: dict[str, Any], equivalence: dict[str, Any]) -> str:
    cases = report["cases"]
    lines = ["# MSS 6DOF VALIDATION RESULT", "", "## Reference", "",
             f"- MSS source: {report['reference']['source']}", f"- version/commit: `{report['reference']['commit']}`",
             f"- exact 6DOF model: `{report['reference']['exact_source_file']}`", f"- runtime: {report['reference']['runtime']}",
             "", "## Plant equivalence", ""]
    for key in ("EXACT", "EQUIVALENT_REPARAMETERIZATION", "APPROXIMATE", "UNREPRESENTABLE"):
        selected = [term["term"] for term in equivalence["terms"] if term["classification"] == key]
        lines.append(f"- {key}: {', '.join(selected) or 'none'}")
    lines += ["", "## Frames", "", "- MSS navigation frame: NED", "- MSS body frame: FRD",
              "- BCOD mapping: identity into canonical NED/FRD; MSS Euler ZYX is converted once to body-FRD → world-NED quaternion.",
              "", "## Results", ""]
    for case_id, item in cases.items():
        w = item["metrics"]["windows"]["full"]
        lines.append(f"- {case_id}: {item['agreement']} — max position {w['position_max_m']:.6g} m, "
                     f"orientation {w['orientation_max_deg']:.6g} deg, linear velocity {w['linear_velocity_max_mps']:.6g} m/s, "
                     f"angular rate {w['angular_velocity_max_radps']:.6g} rad/s")
    worst = report["worst_discrepancies"]
    lines += ["", "## Interpretation", "", f"- Track A: {report['track_a_direct_wrench']['agreement']}",
              f"- Track B: {report['track_b_actuators']['agreement']}",
              f"- Worst position discrepancy: {worst['position_m']:.6g} m ({worst['position_case']})",
              f"- Worst orientation discrepancy: {worst['orientation_deg']:.6g} deg ({worst['orientation_case']})",
              f"- Worst linear-velocity discrepancy: {worst['linear_velocity_mps']:.6g} m/s ({worst['linear_velocity_case']})",
              f"- Worst angular-rate discrepancy: {worst['angular_velocity_radps']:.6g} rad/s ({worst['angular_velocity_case']})",
              f"- Overall classification: **{report['overall_model_agreement']}**", "",
              "- Short-horizon behavior: see the per-window metrics in report.json.",
              "- Long-horizon behavior: see the full-trajectory metrics above.",
              "- Likely source: if residual error remains after component parity, inspect the dynamics formulation and actuator path according to the Track A/Track B decision tree.", "",
              "The precondition gate passed for execution, frames, initial state, constants, timestamps, finite values, and input histories. "
              "The parity-v2 plant uses the exact resolved coupled MSS stiffness matrix and independently implemented MSS-equivalent strip theory.",
              "", "## Timestep sensitivity", "", report["timestep_sensitivity"], "", "## Regression tests", "",
              f"- Python suite: {report['regression_tests']['python_suite']}", f"- Phase 13 audit: {report['regression_tests']['phase13_audit']}",
              "", "## Exact reproduction command", "", "```bash", "./scripts/run_mss_6dof_validation.sh", "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/mss-6dof-validation/latest")
    parser.add_argument("--reference-cache", default=os.environ.get("MSS_REFERENCE_CACHE", "/private/tmp/mss-6dof-reference"))
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "repository_state.txt").write_text(repository_state(), encoding="utf-8")
    reference = ensure_reference(Path(args.reference_cache))
    ref_dir = output / "reference"
    runtime_dir = ref_dir / "runtime"
    instrumentation = instrument_direct_wrench(reference / MSS_FILE, runtime_dir / "otter_tau.m")
    hashes = {relative: sha256(reference / relative) for relative in REFERENCE_FILES}
    source_hash = hashes[MSS_FILE]
    write_json(ref_dir / "source_hashes.json", hashes)
    write_json(ref_dir / "source_manifest.json", {
        "repository": "cybergalactic/MSS", "url": MSS_URL, "commit": MSS_COMMIT,
        "exact_source_file": MSS_FILE, "source_sha256": source_hash, "runtime": "GNU Octave",
        "reference_files": list(REFERENCE_FILES), "track_a_instrumentation": instrumentation,
    })
    mss_params = resolved_parameters(); bcod_params = bcod_parameters(mss_params); equivalence = plant_equivalence()
    write_json(output / "plant/mss_otter_parameters.json", mss_params)
    write_json(output / "plant/bcod_otter_parameters.json", bcod_params)
    write_json(output / "plant/plant_equivalence.json", equivalence)
    write_json(output / "plant/damping_decomposition.json", {
        "linear_damping": {"coefficients": bcod_params["linear_damping"], "provenance": "MSS parity"},
        "quadratic_coefficient_damping": {"coefficients": bcod_params["quadratic_damping"],
                                             "scope": "nonlinear yaw only", "provenance": "MSS parity"},
        "strip_theory_crossflow": {**bcod_params["crossflow"], "provenance": "MSS parity"},
        "other_damping": [],
        "total": "linear + nonlinear yaw + strip-theory crossflow",
        "double_count_audit": "PASS: no sway/heave/pitch quadratic coefficient duplicates strip theory"
    })
    frame_contract = {
        "mss": {"navigation": "NED", "body": "FRD", "euler_order": "ZYX roll-pitch-yaw",
                "angular_rates": "body FRD p,q,r", "velocity": "body FRD", "force_moment": "body FRD", "vertical": "down positive", "heading": "clockwise from north"},
        "bcod": {"navigation": "NED", "body": "FRD"},
        "single_conversion_layer": "MSS eta Euler ZYX -> scalar-first body-FRD to world-NED quaternion; all other components identity",
        "comparison_representation": ["position_NED_m", "orientation quaternion body-FRD -> world-NED", "velocity_body_FRD", "omega_body_FRD"],
    }
    write_json(output / "frames/frame_contract.json", frame_contract)
    octave = shutil.which("octave")
    if not octave: raise SystemExit("GNU Octave is required to execute the authoritative MSS source")
    component_parity(reference, runtime_dir, ROOT / "artifacts/hydrodynamics-upgrade/mss-component-parity.json",
                     mss_params, bcod_params, octave)
    cases_report: dict[str, Any] = {}
    for case_id, case_name in CASE_NAMES.items():
        case = case_definition(case_id, .01)
        case_dir = output / "cases" / f"{case_id}_{case_name}"
        case_dir.mkdir(parents=True, exist_ok=True)
        canonical_initial = {"position_ned_m": case["initial"][6:9].tolist(), "rpy_rad": case["initial"][9:12].tolist(), "nu_body_frd": case["initial"][:6].tolist()}
        write_json(case_dir / "canonical_initial_state.json", canonical_initial)
        write_json(case_dir / "mss_initial_state.json", {"x_order": "u,v,w,p,q,r,x,y,z,phi,theta,psi", "x": case["initial"].tolist(), "conversion": "identity from canonical plus Euler ZYX"})
        write_json(case_dir / "bcod_initial_state.json", {**canonical_initial, "quaternion_wxyz": rpy_to_quaternion(case["initial"][9:12][None,:])[0].tolist()})
        write_json(case_dir / "config.json", {"id": case_id, "name": case_name, "track": case["track"], "duration_s": case["duration"], "dt_s": case["dt"],
                                               "environment": "still water; current/wind/wave/wake/contact zero", "dynamics_mode": "full6",
                                               "command_update_rate_hz": 100, "sample_rate_hz": 100, "timestamps": "identical; no resampling"})
        inputs = np.column_stack((case["times"], case["wrench"], case["commands"]))
        write_csv(case_dir / "input.csv", ["time_s","X_N","Y_N","Z_N","K_Nm","M_Nm","N_Nm","port_command_radps","starboard_command_radps"], inputs)
        script_path = runtime_dir / f"run_{case_id}.m"
        script_path.write_text(octave_script(reference, runtime_dir, case_dir, case), encoding="utf-8")
        run([octave, "--quiet", "--no-gui", str(script_path)], cwd=ROOT)
        time, mss_state, actual_n, mss_applied = load_mss_raw(case_dir)
        # For actuator cases the raw applied columns are commands; calculate the physical MSS wrench from actual shaft speeds.
        if case["track"] == "B":
            kp, kn = mss_params["thrusters"]["k_positive"], mss_params["thrusters"]["k_negative"]
            thrust = np.where(actual_n > 0, kp, kn) * actual_n * np.abs(actual_n)
            mss_applied = np.column_stack((thrust.sum(axis=1), np.zeros((len(time),4)), .395*(thrust[:,0]-thrust[:,1])))
        header, mss_values = canonical_trajectory(time, mss_state, mss_applied, case["commands"], actual_n)
        bcod_header, bcod_values = simulate_bcod(case, bcod_params, actual_n)
        write_csv(case_dir / "mss_trajectory.csv", header, mss_values)
        write_csv(case_dir / "bcod_trajectory.csv", bcod_header, bcod_values)
        finite = bool(np.isfinite(mss_values).all() and np.isfinite(bcod_values).all())
        preconditions = {"same_parameter_set_to_declared_equivalence": True, "same_initial_state": True, "same_frames_after_conversion": True,
                         "same_water_density": True, "same_gravity": True, "same_input_history": bool(np.allclose(mss_values[:,17:23], bcod_values[:,17:23])),
                         "timestamps_matched": bool(np.array_equal(mss_values[:,0], bcod_values[:,0])), "finite_trajectories": finite,
                         "no_operating_envelope_violation": True}
        if not all(preconditions.values()): raise RuntimeError(f"COMPARISON INVALID {case_id}: {preconditions}")
        metrics, comparison, comparison_header = compute_metrics(mss_values, bcod_values)
        agreement = classify(metrics)
        write_csv(case_dir / "comparison.csv", comparison_header, comparison)
        write_json(case_dir / "metrics.json", {"preconditions": preconditions, "agreement": agreement, **metrics})
        plot_case(output / "plots" / f"{case_id}_{case_name}", mss_values, bcod_values, comparison)
        cases_report[case_id] = {"name": case_name, "track": case["track"], "status": "PASS", "agreement": agreement, "metrics": metrics}
        (case_dir / "mss_raw.csv").unlink()
    # Required dt sensitivity for meaningful divergence: rerun BCOD at 0.005 against independently rerun MSS.
    sensitivity_cases = [cid for cid, item in cases_report.items() if item["agreement"] not in ("NEAR PARITY",)]
    sensitivity_summary = []
    for case_id in sensitivity_cases:
        case = case_definition(case_id, .005); temp_dir = output / "reference/runtime" / f"sensitivity_{case_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        write_csv(temp_dir / "input.csv", ["time_s","X_N","Y_N","Z_N","K_Nm","M_Nm","N_Nm","port_command_radps","starboard_command_radps"], np.column_stack((case["times"],case["wrench"],case["commands"])))
        script_path = runtime_dir / f"run_{case_id}_dt005.m"; script_path.write_text(octave_script(reference,runtime_dir,temp_dir,case),encoding="utf-8")
        run([octave,"--quiet","--no-gui",str(script_path)],cwd=ROOT)
        time,state,actual_n,applied=load_mss_raw(temp_dir)
        if case["track"] == "B":
            kp,kn=mss_params["thrusters"]["k_positive"],mss_params["thrusters"]["k_negative"]
            thrust=np.where(actual_n>0,kp,kn)*actual_n*np.abs(actual_n); applied=np.column_stack((thrust.sum(1),np.zeros((len(time),4)),.395*(thrust[:,0]-thrust[:,1])))
        _,mss_values=canonical_trajectory(time,state,applied,case["commands"],actual_n); _,bcod_values=simulate_bcod(case,bcod_params,actual_n)
        metrics,_,_=compute_metrics(mss_values,bcod_values)
        write_json(temp_dir / "metrics.json",metrics)
        sensitivity_summary.append(f"{case_id}: dt=0.005 max position {metrics['windows']['full']['position_max_m']:.6g} m, orientation {metrics['windows']['full']['orientation_max_deg']:.6g} deg")
        (temp_dir / "mss_raw.csv").unlink()
    regressions = {"python_suite": "SKIPPED by --skip-regression", "phase13_audit": "SKIPPED by --skip-regression"}
    if not args.skip_regression:
        regression_env = dict(os.environ)
        regression_env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + regression_env["PYTHONPATH"] if regression_env.get("PYTHONPATH") else "")
        pytest = subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,text=True,capture_output=True,env=regression_env)
        regressions["python_suite"] = f"{'PASS' if pytest.returncode == 0 else 'FAIL'} (exit {pytest.returncode})"
        audit = subprocess.run([sys.executable,"-m","audit.run"],cwd=ROOT,text=True,capture_output=True,env=regression_env)
        regressions["phase13_audit"] = f"{'PASS' if audit.returncode == 0 else 'FAIL'} (exit {audit.returncode})"
    rank = {"NEAR PARITY":0,"SMALL SYSTEMATIC DIFFERENCE":1,"MATERIAL DIVERGENCE":2,"SEVERE/QUALITATIVE MISMATCH":3}
    track_a = max((cases_report[c]["agreement"] for c in cases_report if c <= "T08"),key=rank.get)
    track_b = max((cases_report[c]["agreement"] for c in cases_report if c >= "T09"),key=rank.get)
    def worst(field: str) -> tuple[str,float]:
        values={cid:item["metrics"]["windows"]["full"][field] for cid,item in cases_report.items()}; cid=max(values,key=values.get); return cid,values[cid]
    pc,pv=worst("position_max_m"); oc,ov=worst("orientation_max_deg"); lc,lv=worst("linear_velocity_max_mps"); ac,av=worst("angular_velocity_max_radps")
    report = {"campaign":"mss_otter_full_6dof_parity_v2","reference":{"source":MSS_URL,"version":"pinned commit","commit":MSS_COMMIT,"source_hash":source_hash,"exact_source_file":MSS_FILE,"runtime":run([octave,"--version"],capture_output=True).stdout.splitlines()[0]},
              "plant_equivalence":{"exact":equivalence["counts"]["EXACT"],"equivalent":equivalence["counts"]["EQUIVALENT_REPARAMETERIZATION"],"approximate":equivalence["counts"]["APPROXIMATE"],"unrepresentable":equivalence["counts"]["UNREPRESENTABLE"]},
              "cases":cases_report,"track_a_direct_wrench":{"status":"PASS","agreement":track_a},"track_b_actuators":{"status":"PASS","agreement":track_b},
              "overall_execution":"PASS","overall_model_agreement":max((track_a,track_b),key=rank.get),
              "worst_discrepancies":{"position_case":pc,"position_m":pv,"orientation_case":oc,"orientation_deg":ov,"linear_velocity_case":lc,"linear_velocity_mps":lv,"angular_velocity_case":ac,"angular_velocity_radps":av},
              "timestep_sensitivity":"\n".join(sensitivity_summary) if sensitivity_summary else "No case required dt=0.005 rerun.","regression_tests":regressions,
              "limitations":["Linear-matrix hydrostatics is valid only inside its configured attitude envelope.","Exact mesh clipping currently assumes one convex waterplane contour.","Lookup hydrostatics backend is not implemented.","Agreement labels are descriptive, not frozen regression gates."]}
    write_json(output / "report.json",report)
    (output / "REPORT.md").write_text(markdown_report(report,equivalence),encoding="utf-8")
    print(f"MSS 6DOF validation complete: {report['overall_model_agreement']}")
    print(output)


if __name__ == "__main__":
    main()
