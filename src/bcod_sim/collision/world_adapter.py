"""Phase 4 entity geometry into Phase 6 collision-only proxies."""

from bcod_sim.collision.broadphase import CollisionBody
from bcod_sim.collision.shapes import Box, Sphere
from bcod_sim.collision.solver import ContactMaterial
from bcod_sim.config.models import BoxShape, SphereShape
from bcod_sim.world.world import ParametricWorld


def world_collision_bodies(world: ParametricWorld, *, sim_time_s: float) -> tuple[CollisionBody, ...]:
    result = []
    for entity in world.entities(sim_time_s=sim_time_s, env_id=world.env_id):
        if not entity.collision_enabled:
            continue
        shape = (Sphere(entity.shape.radius_m) if isinstance(entity.shape, SphereShape)
                 else Box(entity.shape.half_extents_m))
        result.append(CollisionBody(world.env_id, entity.id, shape,
                                    static_position_ned_m=entity.position_ned_m,
                                    kinematic_velocity_ned_mps=entity.velocity_ned_mps))
    return tuple(result)


def seabed_collision_bodies(world: ParametricWorld, *, vessel_positions: tuple[tuple[float,float,float],...]) -> tuple[CollisionBody,...]:
    bathymetry=getattr(world,"bathymetry",None)
    collision=getattr(getattr(bathymetry,"spec",None),"collision",None)
    if collision is None: collision=getattr(getattr(world,"spec",None),"seabed_collision",None)
    if bathymetry is None or collision is None or not collision.enabled:
        return ()
    tiles={bathymetry.collision_tile(position[0],position[1]) for position in vessel_positions}
    return tuple(CollisionBody(world.env_id,f"world:seabed:{tile.tile_index[0]}:{tile.tile_index[1]}",tile,
                               static_position_ned_m=(tile.tile_index[0]*tile.tile_size_m,
                                                      tile.tile_index[1]*tile.tile_size_m,0.),
                               contact_material=ContactMaterial(
                                   collision.restitution,collision.friction,collision.position_correction_fraction))
                 for tile in sorted(tiles,key=lambda item:item.tile_index))
