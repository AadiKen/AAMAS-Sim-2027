import pytest

from bcod_sim.vessel_generation.hmri_wrench import (
    BCODResistingWrench, PhysicalFluidWrench, hmri_y_n_error,
    separate_openfoam_wrench,
)


def test_hmri_comparison_rejects_bcod_resisting_wrench():
    physical, resisting = separate_openfoam_wrench(
        (0, -86, 0), (0, 0, -258),
        foam_reference=(0, 0, 0), body_reference=(0, 0, 0),
        waterline_z_m=0,
    )
    assert isinstance(physical, PhysicalFluidWrench)
    assert isinstance(resisting, BCODResistingWrench)
    assert physical.force_frd_n[1] == 86
    assert physical.moment_frd_nm[2] == 258
    assert resisting.force_frd_n[1] == -86
    assert resisting.moment_frd_nm[2] == -258
    assert hmri_y_n_error(physical, target_y_n=86, target_n_nm=258) == (0, 0)
    with pytest.raises(TypeError, match="PhysicalFluidWrench"):
        hmri_y_n_error(resisting, target_y_n=86, target_n_nm=258)
