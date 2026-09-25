# Sensor SDK

Sensors receive a `SensorContext`. Its `environment` property is the public physical-domain API. A sensor declares a frozen set of capability names and samples them at its mounted NED position and simulation time:

```python
class CurrentMagnitudeSensor:
    kind = "physical"
    requirements = frozenset({"water.current"})

    def sample(self, context, *, sample_step):
        mount = mount_kinematics(self.config, context)
        context.environment.require(self.requirements)
        current = context.environment.sample(
            "water.current", mount.position_ned_m, context.sim_time_s
        ).value[0]
        ...
```

`has`, `require`, `describe`, and `sample` are stable operations. Descriptors state units, frame, valid media, spatial/temporal semantics, and provenance. A query in an invalid medium raises `InvalidMediumError`; callers that need data-quality propagation may pass `invalid="status"`. Missing capabilities raise `MissingCapabilityError`. Providers register through `CapabilityRegistry`, including future `optical.*`, `acoustic.*`, `magnetic.*`, and `rf.*` providers; no `WorldSample` change is required.

External runtime sensor types register once with:

```python
from bcod_sim.sensors.registry import register_sensor_type
register_sensor_type("my_sensor", MySensorFactory)
```

The resolved sensor definition uses common fields plus a `parameters` object owned by the plugin. The factory receives `SensorConfig`, the complete definition, and the parameters. Its sensor then passes unchanged through the vessel, scheduler, engine, observations, and recorder. Required capabilities are validated when the engine is built. A SHA-256 fingerprint binds definition payload, plugin identity/version/source, environment ID, and owner vessel ID; the engine rejects divergence.

## Compatibility and migration

- Existing `SensorConfig`, `SensorPacket`, `SensorScheduler`, mounts, deterministic noise, and sensor kinds remain valid.
- `noise_std` remains as a compatibility default. New GPS and IMU definitions should place unit-specific error fields in `parameters`; `GPSErrorModel` and `IMUErrorModel` are the typed runtime models.
- `sonar` is specifically an idealized downward bathymetric range sonar. It uses `geometry.raycast` over the canonical bathymetry surface and does not claim acoustic propagation, scattering, or multipath.
- `local_water_depth_m` now means `bottom_ned_z_m - surface_ned_z_m`. Code that treated it as bottom NED Z must migrate to `bottom_ned_z_m`.
- Real-world bathymetry and water levels must use the same vertical datum until an explicit datum transform provider is configured. Mismatches fail before simulation.
- Large tensor sensor fields are stored as `.npy` artifacts referenced from packet JSON. Small tensors and structured detections remain lossless JSON structures; arbitrary object stringification is rejected.

The base scheduler supports integer master-clock rates, integer-step latency, warmup, deterministic dropout, multiple instances, and independent RNG streams keyed by sensor identity and sample step.

## Freshness and stale values

The observation pipeline retains the latest **delivered** packet between sensor updates. It never substitutes a zero measurement. Every packet preserves `sample_time_s` and `delivery_time_s`; `SensorPacket.freshness(current_time_s, max_age_s)` reports `age_s`, `stale`, and `validity`. `EpisodeFrame.observation_freshness` exposes the same metadata beside assembled observation values.

When `SensorConfig.max_age_s` is absent, the latest delivered packet remains valid indefinitely while its age continues to increase. When configured, age greater than `max_age_s` changes freshness validity to `stale`; the original measurement and timestamps remain available. Latency does not reset age because age is measured from the original sample time.

## IMU random walk

Accelerometer and gyro random walks are independent, persistent three-axis bias states. At each sensor sample they update by `sigma_rw * sqrt(1/rate_hz) * epsilon`, using deterministic per-instance streams. The measurement is truth plus fixed bias plus cumulative random-walk state plus white noise. Constructing a new scheduler (including episode reset) resets the states and reproduces the same sequence for the same seed.
