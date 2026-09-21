"""Phase 1 config schema. Unimplemented component definitions resolve through Registry."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Vec3 = tuple[Finite, Finite, Finite]
Mat3 = tuple[Vec3, Vec3, Vec3]


class StrictModel(BaseModel):
    # JSON/YAML arrays must be accepted and frozen as tuples after validation.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Experiment(StrictModel):
    id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    numerical_profile: Literal["validation", "training"] = "validation"


class Simulation(StrictModel):
    dynamics_mode: Literal["full6", "planar3"]
    master_dt_s: Positive
    dynamics_substeps: int = Field(ge=1)
    policy_every_n_master_steps: int = Field(ge=1)
    deterministic: bool = True
    max_master_steps: int = Field(default=1000, ge=1)


class UniformVector(StrictModel):
    kind: Literal["uniform"]
    ned_mps: Vec3


class LinearVector(StrictModel):
    kind: Literal["linear"]
    origin_ned_m: Vec3
    base_ned_mps: Vec3
    gradient_per_s: Mat3

class SinusoidalVector(StrictModel):
    kind: Literal["sinusoidal"]
    mean_ned_mps: Vec3
    amplitude_ned_mps: Vec3
    period_s: Positive
    phase_rad: Finite = 0.0


class ShoalingConfig(StrictModel):
    enabled: bool = False
    reference_depth_m: Positive | None = None


class BreakingConfig(StrictModel):
    enabled: bool = False
    gamma: Positive = 0.78


class CurrentInteractionConfig(StrictModel):
    enabled: bool = False


class WaveDepthOptions(StrictModel):
    depth_model: Literal["deep_water", "finite_depth"] = "deep_water"
    shoaling: ShoalingConfig = ShoalingConfig()
    breaking: BreakingConfig = BreakingConfig()
    current_interaction: CurrentInteractionConfig = CurrentInteractionConfig()
    minimum_wet_depth_m: Positive = 0.05
    refraction: Literal["disabled"] = "disabled"


class RegularWaves(WaveDepthOptions):
    kind: Literal["regular"]
    height_m: Nonnegative
    period_s: Positive
    direction_rad: Finite
    phase_rad: Finite = 0.0


class CalmWaves(WaveDepthOptions):
    kind: Literal["calm"]

class IrregularWaves(WaveDepthOptions):
    kind: Literal["irregular"]
    spectrum: Literal["jonswap", "pierson_moskowitz"]
    significant_height_m: Positive
    peak_period_s: Positive
    direction_rad: Finite
    component_count: int = Field(default=64, ge=8, le=512)
    seed: int = Field(ge=0)
    gamma: Positive = 3.3


class Environment(StrictModel):
    current: UniformVector | LinearVector | SinusoidalVector = Field(discriminator="kind")
    wind: UniformVector | LinearVector | SinusoidalVector = Field(discriminator="kind")
    waves: RegularWaves | CalmWaves | IrregularWaves = Field(discriminator="kind")
    visibility_m: Positive
    rain_rate_mps: Nonnegative = 0.0
    fog_extinction_per_m: Nonnegative = 0.0


class BathymetryCollision(StrictModel):
    enabled: bool = False
    resolution_m: Positive = 1.0
    tile_size_m: Positive = 100.0
    friction: Nonnegative = 0.6
    restitution: Nonnegative = 0.0
    position_correction_fraction: Nonnegative = 1.0

    @model_validator(mode="after")
    def valid_contact(self):
        if self.restitution > 1 or self.position_correction_fraction > 1:
            raise ValueError("Bathymetry restitution and correction must be in [0,1]")
        return self


class FlatBathymetry(StrictModel):
    kind: Literal["flat"]
    bottom_ned_z_m: Finite
    vertical_datum: str = Field(min_length=1)
    collision: BathymetryCollision = BathymetryCollision()


class SlopedBathymetry(StrictModel):
    kind: Literal["plane"]
    origin_ned_m: Vec3
    bottom_at_origin_ned_z_m: Finite
    north_slope: Finite
    east_slope: Finite
    vertical_datum: str = Field(min_length=1)
    collision: BathymetryCollision = BathymetryCollision()


class SphereShape(StrictModel):
    kind: Literal["sphere"]
    radius_m: Positive


class BoxShape(StrictModel):
    kind: Literal["box"]
    half_extents_m: Vec3

    @model_validator(mode="after")
    def positive_extents(self):
        if any(x <= 0 for x in self.half_extents_m):
            raise ValueError("Box half extents must be positive")
        return self


class StaticEntity(StrictModel):
    id: str = Field(min_length=1)
    position_ned_m: Vec3
    shape: SphereShape | BoxShape = Field(discriminator="kind")
    collision_enabled: bool = True


class ScriptedEntity(StaticEntity):
    velocity_ned_mps: Vec3
    start_time_s: Nonnegative = 0.0


class Region(StrictModel):
    id: str = Field(min_length=1)
    min_ned_m: Vec3
    max_ned_m: Vec3

    @model_validator(mode="after")
    def ordered(self):
        if any(a >= b for a, b in zip(self.min_ned_m, self.max_ned_m)):
            raise ValueError("Region extents must be ordered")
        return self


class WorldSource(StrictModel):
    kind: Literal["parametric", "real_world"]
    data_product: str | None = None

    @model_validator(mode="after")
    def check_product(self):
        if self.kind == "real_world" and not self.data_product:
            raise ValueError("real_world requires data_product reference")
        if self.kind == "parametric" and self.data_product is not None:
            raise ValueError("parametric world cannot name a data_product")
        return self


class World(StrictModel):
    source: WorldSource
    environment: Environment | None = None
    obstacles: tuple[str, ...] = ()
    bathymetry: FlatBathymetry | SlopedBathymetry | None = Field(default=None, discriminator="kind")
    static_entities: tuple[StaticEntity, ...] = ()
    scripted_entities: tuple[ScriptedEntity, ...] = ()
    spawn_regions: tuple[Region, ...] = ()
    boundary: Region | None = None
    seabed_collision: BathymetryCollision = BathymetryCollision()

    @model_validator(mode="after")
    def check_environment(self):
        if self.source.kind == "parametric" and self.environment is None:
            raise ValueError("parametric world requires environment")
        if self.source.kind == "real_world" and (self.environment is not None or self.bathymetry is not None or
                self.obstacles or self.static_entities or self.scripted_entities):
            raise ValueError("real_world fields must come from the resolved bundle")
        ids = [x.id for x in (*self.static_entities, *self.scripted_entities)]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate world entity ID")
        region_ids = [x.id for x in self.spawn_regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("Duplicate spawn region ID")
        return self


class Controller(StrictModel):
    mode: Literal["direct_actuator", "high_level", "scripted"]


class Spawn(StrictModel):
    ned_m: Vec3
    rpy_rad: Vec3


class Vessel(StrictModel):
    instance_id: str = Field(min_length=1)
    definition: str
    controller: Controller
    spawn: Spawn
    dynamics: str | None = None
    actuators: tuple[str, ...] = ()
    sensors: tuple[str, ...] = ()
    policy: str | None = None

    @model_validator(mode="after")
    def unique_components(self):
        if len(self.actuators) != len(set(self.actuators)):
            raise ValueError("Duplicate actuator reference")
        if len(self.sensors) != len(set(self.sensors)):
            raise ValueError("Duplicate sensor reference")
        return self


class Reward(StrictModel):
    individual_weight: Nonnegative
    team_weight: Nonnegative


class Task(StrictModel):
    type: str = Field(min_length=1)
    reward: Reward
    disabled_agent_behavior: Literal[
        "deactivate_keep_physical", "deactivate_remove_physical", "continue_scripted",
        "terminate_episode"
    ]


class Logging(StrictModel):
    metrics: tuple[str, ...] = ()
    states: bool = True
    sensor_payloads: bool = False
    queue_capacity: int = Field(default=1024, ge=1)
    backpressure: Literal["block", "drop_noncritical_with_event", "terminate"] = "block"


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    experiment: Experiment
    simulation: Simulation
    world: World
    vessels: tuple[Vessel, ...] = Field(min_length=1)
    task: Task
    logging: Logging = Logging()
    scenario_template: str | None = None

    @model_validator(mode="after")
    def unique_instances(self):
        names = [v.instance_id for v in self.vessels]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate vessel instance_id")
        return self
