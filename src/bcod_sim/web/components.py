"""Inspector descriptions derived from registered component kinds and runtime fields."""

from bcod_sim.sensors.registry import runtime_sensor_registry


_SENSOR_PARAMETERS = {
    "gps": ("origin_wgs84_rad_m", "rate_hz", "mount_frd_m", "mount_q_to_frd"),
    "imu": ("rate_hz", "mount_frd_m", "mount_q_to_frd"),
    "sonar": ("min_range_m", "max_range_m", "fov_rad", "beam_count", "rate_hz",
              "mount_frd_m", "mount_q_to_frd"),
    "lidar": ("min_range_m", "max_range_m", "fov_rad", "ray_count", "rate_hz",
              "mount_frd_m", "mount_q_to_frd"),
}


def sensor_types() -> list[dict]:
    from bcod_sim.web.runtime_factory import SensorRuntime
    schema = SensorRuntime.model_json_schema()["properties"]
    return [{"type": kind, "display_name": kind.replace("_", " ").title(),
             "parameters": {name: schema[name] for name in _SENSOR_PARAMETERS.get(kind, ())},
             "pose_supported": True, "output_schema": {},
             "visualization": "cone" if kind in {"sonar", "lidar"} else "marker"}
            for kind in runtime_sensor_registry.registered_types()]


def actuator_types() -> list[dict]:
    from bcod_sim.web.runtime_factory import FixedThrusterRuntime
    schema = FixedThrusterRuntime.model_json_schema()["properties"]
    return [{"type": "fixed_thruster", "display_name": "Fixed thruster",
             "parameters": {key: schema[key] for key in
                            ("mount_frd_m", "mount_q_to_frd", "thrust_bounds_n")},
             "pose_supported": True, "output_schema": {"force_frd_n": [3], "moment_frd_nm": [3]},
             "visualization": "arrow"}]
