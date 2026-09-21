import math

import pytest
from hypothesis import given, strategies as st

from bcod_sim.core.errors import FrameConversionError
from bcod_sim.frames.transforms import (
    bathymetry_to_ned_z, body_to_world, enu_to_ned, flu_to_frd,
    frd_to_flu, moment_from_force, ned_to_enu, ned_to_web,
    normalize_quaternion, rotate_wrench, web_to_ned, world_to_body, rpy_to_quaternion,
)
from bcod_sim.frames.geodesy import geodetic_to_ned, ned_to_geodetic
from bcod_sim.frames.units import to_si

finite = st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False)
vectors = st.tuples(finite, finite, finite)


@given(vectors)
def test_axis_round_trips(v):
    assert enu_to_ned(ned_to_enu(v)) == v
    assert flu_to_frd(frd_to_flu(v)) == v
    assert web_to_ned(ned_to_web(v)) == v


@given(vectors, st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False))
def test_vector_and_wrench_round_trip(v, yaw):
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    recovered = world_to_body(body_to_world(v, q), q)
    assert recovered == pytest.approx(v, abs=1e-9)
    force, moment = rotate_wrench(rotate_wrench((v, v), q), q, to_world=False)
    assert force == pytest.approx(v, abs=1e-9)
    assert moment == pytest.approx(v, abs=1e-9)


def test_quaternion_normalization_and_yaw_sign():
    assert normalize_quaternion((2, 0, 0, 0)) == (1, 0, 0, 0)
    with pytest.raises(FrameConversionError):
        normalize_quaternion((0, 0, 0, 0))
    # Starboard offset (+y) with forward thrust (+x) creates negative yaw (+z down).
    assert moment_from_force((0, 2, 0), (10, 0, 0)) == (0, 0, -20)
    assert body_to_world((1, 0, 0), rpy_to_quaternion(0, 0, math.pi / 2)) == pytest.approx((0, 1, 0), abs=1e-12)


def test_bathymetry_sign_and_datum():
    assert bathymetry_to_ned_z(-10, convention="elevation_positive_up", datum="MSL") == 10
    assert bathymetry_to_ned_z(10, convention="depth_positive_down", datum="MSL") == 10
    assert bathymetry_to_ned_z(3, convention="elevation_positive_up", datum="MSL") == -3
    with pytest.raises(FrameConversionError):
        bathymetry_to_ned_z(3, convention="depth_positive_down", datum="")


def test_units_fail_closed():
    assert to_si(36, dimension="speed", unit="km/h") == pytest.approx(10)
    assert to_si(180, dimension="angle", unit="deg") == pytest.approx(math.pi)
    with pytest.raises(FrameConversionError):
        to_si(1, dimension="speed", unit="")


@given(st.floats(min_value=-1.2, max_value=1.2), st.floats(min_value=-3, max_value=3), vectors)
def test_geodetic_ned_round_trip(lat, lon, offset):
    origin = (lat, lon, 50.0)
    geodetic = ned_to_geodetic(offset, origin)
    assert geodetic_to_ned(*geodetic, origin) == pytest.approx(offset, abs=1e-6)
