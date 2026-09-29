import pytest

from bcod_sim.vessel_generation.spec_a.benchmark_mapping import (
    assert_comparable, body_to_tow_force, draft_normalization_ratio,
    normalize_static_drift, shift_yaw_moment,
)
from bcod_sim.vessel_generation.spec_a.harmonic_postprocess import harmonic
from bcod_sim.vessel_generation.spec_a.matrix import CaseState


def _record(family):
    return {"comparison_family": family, "reference_point_m": [0., 0., 0.],
            "frame": "FRD", "force_frame": "body_frd",
            "sign_convention": "physical_fluid_on_hull",
            "normalization": "half_rho_L2_T_U02", "motion_definition": "pure_yaw_rprime_0.2",
            "load_definition": "physical_pressure_plus_viscous",
            "vessel_configuration": "bare_hull", "frequency_hz": .08,
            "amplitude_r_prime": .2,
            "phase_convention": "rate_cos_accel_minus_sin"}


@pytest.mark.parametrize("field", ["reference_point_m", "frame", "force_frame", "sign_convention",
                                     "normalization", "motion_definition", "load_definition",
                                     "frequency_hz", "phase_convention", "amplitude_r_prime"])
def test_reject_missing_harmonic_mapping_field(field):
    cfd, reference = _record("harmonic_pmm"), _record("harmonic_pmm")
    reference[field] = None
    with pytest.raises(ValueError):
        assert_comparable(cfd, reference, kind="harmonic_pmm")


def test_cmt_pmm_separation_and_reference_shift_guard():
    cfd, reference = _record("steady_cmt"), _record("harmonic_pmm")
    with pytest.raises(ValueError, match="not steady_cmt"):
        assert_comparable(cfd, reference, kind="steady_cmt")
    reference = _record("steady_cmt")
    reference["reference_point_m"] = [0.1, 0, 0]
    with pytest.raises(ValueError, match="reference point"):
        assert_comparable(cfd, reference, kind="steady_cmt")


@pytest.mark.parametrize("field", ["reference_point_m", "frame", "force_frame", "sign_convention",
                                     "normalization", "motion_definition", "load_definition"])
def test_reject_missing_steady_mapping_field(field):
    cfd, reference = _record("steady_cmt"), _record("steady_cmt")
    reference[field] = None
    with pytest.raises(ValueError, match="missing"):
        assert_comparable(cfd, reference, kind="steady_cmt")


def test_reject_blank_or_nonfinite_reference_metadata():
    cfd, reference = _record("steady_cmt"), _record("steady_cmt")
    reference["normalization"] = ""
    with pytest.raises(ValueError, match="missing"):
        assert_comparable(cfd, reference, kind="steady_cmt")
    reference["normalization"] = cfd["normalization"]
    reference["reference_point_m"] = [float("nan"), 0, 0]
    with pytest.raises(ValueError, match="finite"):
        assert_comparable(cfd, reference, kind="steady_cmt")


def test_draft_normalization_sensitivity():
    # N' = N/(0.5 rho L² T U²), so the same N differs by this ratio.
    assert draft_normalization_ratio(.207, .270) == pytest.approx(.270/.207)
    assert draft_normalization_ratio(.270, .207) == pytest.approx(.207/.270)


def test_body_and_tow_force_frames_at_signed_drift():
    import math
    x, y, beta = -17.883325758, 29.3910009332, 6.
    xt, yt = body_to_tow_force(x, y, beta)
    assert yt == pytest.approx(math.sin(math.radians(beta))*x + math.cos(math.radians(beta))*y)
    assert yt == pytest.approx(27.3606773953)
    assert yt != pytest.approx(y)
    assert body_to_tow_force(x, y, -beta)[1] == pytest.approx(
        -math.sin(math.radians(beta))*x + math.cos(math.radians(beta))*y)


def test_positive_and_negative_nmri_drift_motion_convention():
    import math
    plus = CaseState(6., 0., "drift").body_velocity(.994, 4.970)
    minus = CaseState(-6., 0., "drift").body_velocity(.994, 4.970)
    assert plus[0] == pytest.approx(.994*math.cos(math.radians(6.)))
    assert plus[1] == pytest.approx(-.994*math.sin(math.radians(6.)))
    assert minus[0] == pytest.approx(plus[0])
    assert minus[1] == pytest.approx(-plus[1])
    assert math.degrees(math.atan2(-plus[1], plus[0])) == pytest.approx(6.)
    assert math.degrees(math.atan2(-minus[1], minus[0])) == pytest.approx(-6.)


def test_independent_nmri_normalization_and_midship_shift():
    rho, speed, length, draft = 1000., .994, 4.970, .323
    y_ref = .02560 * (.5*rho*speed**2*length*draft)
    n_ref = .01392 * (.5*rho*speed**2*length**2*draft)
    assert y_ref == pytest.approx(20.302132110848)
    assert n_ref == pytest.approx(54.86524314631)
    assert normalize_static_drift(y_ref, n_ref, rho=rho, speed=speed,
                                  length=length, draft=draft) == pytest.approx((.02560, .01392))
    assert shift_yaw_moment(n_ref, -17., y_ref, (0., 0.), (0., 0.)) == pytest.approx(n_ref)
    assert shift_yaw_moment(n_ref, -17., y_ref, (.10, -.05), (0., 0.)) == pytest.approx(
        n_ref + .10*y_ref - (-.05)*(-17.))


def test_reject_unspecified_or_mismatched_force_frame():
    cfd, reference = _record("steady_cmt"), _record("steady_cmt")
    reference.pop("force_frame")
    with pytest.raises(ValueError, match="force_frame"):
        assert_comparable(cfd, reference, kind="steady_cmt")
    reference["force_frame"] = "tow_track"
    with pytest.raises(ValueError, match="force_frame"):
        assert_comparable(cfd, reference, kind="steady_cmt")


def test_signed_phase_and_frequency_guard():
    import numpy as np
    t = np.linspace(0., 12.5, 3001)
    rate = -7*np.cos(2*np.pi*.08*t) + 2*np.sin(2*np.pi*.08*t)
    h = harmonic(t, rate)
    assert h["rate_1"] == pytest.approx(-7, abs=1e-8)
    assert h["accel_1"] == pytest.approx(-2, abs=1e-8)
    a, b = _record("harmonic_pmm"), _record("harmonic_pmm")
    b["phase_convention"] = "rate_minus_cos"
    with pytest.raises(ValueError, match="phase"):
        assert_comparable(a, b, kind="harmonic_pmm")
    b["phase_convention"] = a["phase_convention"]
    b["frequency_hz"] = .065
    with pytest.raises(ValueError, match="frequency"):
        assert_comparable(a, b, kind="harmonic_pmm")
    b["frequency_hz"] = a["frequency_hz"]
    b["amplitude_r_prime"] = .4
    with pytest.raises(ValueError, match="amplitude"):
        assert_comparable(a, b, kind="harmonic_pmm")
