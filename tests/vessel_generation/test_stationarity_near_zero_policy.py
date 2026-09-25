"""General regression coverage for the continuous physical-scale load gate."""

import hashlib
import json

import numpy as np
import pytest

from bcod_sim.vessel_generation.cfd import CFDExecutionError, OpenFOAMAdapter
from bcod_sim.vessel_generation.quality import validate_run, wrench_stationarity


TIME = np.linspace(0., 20., 2001)
FORCE_SCALE = 1000.
MOMENT_SCALE = 5000.


def check(values: np.ndarray):
    return wrench_stationarity(TIME, values, force_scale_n=FORCE_SCALE,
                               moment_scale_nm=MOMENT_SCALE, window_s=10.)


def test_material_force_and_moment_keep_relative_stationarity() -> None:
    values = np.zeros((len(TIME), 6))
    values[:, 0] = 100. + .1 * np.sin(TIME * 3)
    values[:, 4] = 250. + .5 * np.sin(TIME * 3)
    result = check(values)
    assert result.accepted
    assert result.channels[0].regime == "significant"
    assert result.channels[4].regime == "significant"
    assert validate_run(values, [1e-7], time_s=TIME,
                        force_scale_n=FORCE_SCALE, moment_scale_nm=MOMENT_SCALE,
                        window_s=10.).accepted


def test_material_mean_drift_still_fails_in_force_and_moment_channels() -> None:
    for axis, base, change in ((0, 100., 15.), (4, 250., 40.)):
        values = np.zeros((len(TIME), 6))
        values[:, axis] = base + change * (TIME - 15.) / 10.
        result = check(values)
        assert not result.accepted
        assert result.channels[axis].regime == "significant"
        assert result.channels[axis].normalized_slope > .03


def test_exact_zero_and_numerically_perturbed_zero_remain_stable() -> None:
    rng = np.random.default_rng(144)
    noise = rng.normal(0., .2, len(TIME))
    values = np.zeros((len(TIME), 6))
    values[:, 1] = noise
    zero = check(values)
    values[:, 1] += 1e-8
    perturbed = check(values)
    assert zero.accepted and perturbed.accepted
    assert zero.channels[0].mean == 0. and zero.channels[0].accepted
    assert zero.channels[1].regime == perturbed.channels[1].regime == "near_zero"
    assert abs(zero.channels[1].normalized_slope -
               perturbed.channels[1].normalized_slope) < 1e-8


def test_tiny_noisy_cross_force_and_moment_use_physical_scale() -> None:
    rng = np.random.default_rng(220)
    values = np.zeros((len(TIME), 6))
    values[:, 1] = 2. + rng.normal(0., 1.2, len(TIME))
    values[:, 3] = 8. + rng.normal(0., 5., len(TIME))
    result = check(values)
    assert result.accepted
    assert all(result.channels[i].regime == "near_zero" for i in (1, 3))
    assert result.channels[1].normalized_standard_deviation < .10
    assert result.channels[3].normalized_standard_deviation < .10


def test_near_zero_channel_with_material_absolute_drift_fails() -> None:
    for axis, mean, total_change in ((1, 2., 10.), (3, 8., 80.)):
        values = np.zeros((len(TIME), 6))
        values[:, axis] = mean + total_change * (TIME - 15.) / 10.
        result = check(values)
        assert result.channels[axis].regime == "near_zero"
        assert result.channels[axis].normalized_slope > .03
        assert not result.accepted


def test_transition_is_continuous_and_material_load_cannot_hide() -> None:
    slopes = []
    for mean in (9.999, 10.001):  # Either side of the 1%-of-scale transition.
        values = np.zeros((len(TIME), 6))
        values[:, 1] = mean + .1 * (TIME - 15.) / 10.
        result = check(values)
        slopes.append(result.channels[1].normalized_slope)
        assert result.channels[1].accepted
    assert slopes[0] == pytest.approx(slopes[1], rel=.02)
    values = np.zeros((len(TIME), 6))
    values[:, 1] = 50. + 5. * (TIME - 15.) / 10.
    result = check(values)
    assert result.channels[1].regime == "significant"
    assert result.channels[1].normalized_slope > .03
    assert not result.accepted


def test_near_zero_periodic_mean_drift_is_not_hidden() -> None:
    values = np.zeros((len(TIME), 6))
    phase = 2 * np.pi * .8 * TIME
    values[:, 1] = 2. + 2. * np.sin(phase)
    assert check(values).channels[1].accepted
    values[:, 1] += 1.5 * (TIME - 15.)
    result = check(values)
    assert not result.channels[1].accepted


def test_qualified_case_parser_uses_only_frozen_window(tmp_path) -> None:
    output = tmp_path / "postProcessing/forces/0"
    output.mkdir(parents=True)
    rows = []
    cases = [(0., 1000.)] + list(zip((57., 60., 63., 66., 69., 72., 75., 78., 80.),
                                     range(8, 17)))
    for time, force in cases:
        rows.append(f"{time:g} (({force:g} 0 0) (0 0 0)) ((0 0 0) (0 0 0))\n")
    (output / "forces.dat").write_text(
        "# Time forces(pressure viscous) moments(pressure viscous)\n" + "".join(rows))
    (tmp_path / "case_metadata.json").write_text(json.dumps({
        "case_id": "case-a", "velocity_body_frd": [1, 0, 0, 0, 0, 0],
            "moment_reference_point_frd_m": [0, 0, 0], "solver_moment_reference_point_m": [0, 0, 0],
            "source_waterline_z_m": 0, "frame_contract": "bcod-openfoam-frd-v1",
            "units": {"length":"m","velocity":"m/s","angular_rate":"rad/s",
                      "force":"N","moment":"N*m"},
            "solver": "foamRun"}))
    (tmp_path / "solver.log").write_text("\nEnd\n")
    selected = np.asarray([[time, force, 0., 0., 0., 0., 0.]
                           for time, force in cases[1:]], dtype="<f8")
    manifest = {"status": "CFD_QUALIFIED", "case_id": "case-a",
                "fitting_window_s": [57., 80.],
                "selected_samples_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                "mean_six_axis_wrench": [12., 0., 0., 0., 0., 0.]}
    (tmp_path / "qualified_fitting_window.json").write_text(json.dumps(manifest))
    adapter = OpenFOAMAdapter(tmp_path)
    parsed = adapter.parse_case(tmp_path)
    assert parsed.force_body_frd_n[0] == -12.  # resisting load is opposite physical fluid force
    assert parsed.fitting_window_s == (57., 80.)
    manifest["selected_samples_sha256"] = "0" * 64
    (tmp_path / "qualified_fitting_window.json").write_text(json.dumps(manifest))
    with pytest.raises(CFDExecutionError, match="sample hash mismatch"):
        adapter.parse_case(tmp_path)
