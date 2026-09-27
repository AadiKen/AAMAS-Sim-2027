import pytest

from bcod_sim.vessel_generation.spec_a.benchmark_mapping import (
    assert_comparable, draft_normalization_ratio,
)
from bcod_sim.vessel_generation.spec_a.harmonic_postprocess import harmonic


def _record(family):
    return {"comparison_family": family, "reference_point_m": [0., 0., 0.],
            "frame": "FRD", "sign_convention": "physical_fluid_on_hull",
            "normalization": "half_rho_L2_T_U02", "motion_definition": "pure_yaw_rprime_0.2",
            "load_definition": "physical_pressure_plus_viscous",
            "vessel_configuration": "bare_hull", "frequency_hz": .08,
            "amplitude_r_prime": .2,
            "phase_convention": "rate_cos_accel_minus_sin"}


@pytest.mark.parametrize("field", ["reference_point_m", "frame", "sign_convention",
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


@pytest.mark.parametrize("field", ["reference_point_m", "frame", "sign_convention",
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
