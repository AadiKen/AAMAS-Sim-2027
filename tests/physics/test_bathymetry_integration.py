import math
import pytest
import torch

from bcod_sim.collision.broadphase import CollisionBody
from bcod_sim.collision.shapes import Box, SeabedSurface, Sphere
from bcod_sim.collision.solver import ContactMaterial, resolve_contacts
from bcod_sim.config.models import FlatBathymetry, IrregularWaves, RegularWaves, SlopedBathymetry
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.world.bathymetry import Bathymetry, RasterBathymetry
from bcod_sim.world.waves import G, WaveField, finite_depth_dispersion
from bcod_sim.world.world import ParametricWorld
from bcod_sim.config.models import World

DTYPE=torch.float64
def t(value): return torch.tensor(value,dtype=DTYPE)

def plant():
    zero=torch.zeros(6,dtype=DTYPE)
    return Plant6(MassProperties(10,t((0,0,0)),torch.diag(t((4,5,6))),torch.zeros((6,6),dtype=DTYPE)),
                  Damping(zero,zero),Hydrostatics(10*G,t((0,0,0))),OperatingEnvelope(t((1e6,)*6)))

def body(shape,position,velocity=(0,0,0,0,0,0)):
    return CollisionBody(0,"vessel",shape,VesselState(t(position),t((1,0,0,0)),t(velocity)),plant())

def test_shared_bathymetry_flat_xy_slopes_and_unique_raster():
    flat=Bathymetry(FlatBathymetry(kind="flat",bottom_ned_z_m=10,vertical_datum="MSL"))
    x=Bathymetry(SlopedBathymetry(kind="plane",origin_ned_m=(0,0,0),bottom_at_origin_ned_z_m=5,north_slope=.1,east_slope=0,vertical_datum="MSL"))
    y=Bathymetry(SlopedBathymetry(kind="plane",origin_ned_m=(0,0,0),bottom_at_origin_ned_z_m=5,north_slope=0,east_slope=.1,vertical_datum="MSL"))
    assert flat.depth_at(8,-3)==10
    assert x.depth_at(10,7)==pytest.approx(6)
    assert y.depth_at(10,7)==pytest.approx(5.7)
    raster=RasterBathymetry(t(((1,2,4),(10,20,40),(100,200,400))),(0,0),(2,3),"MSL")
    assert raster.depth_at(2,3)==20
    assert raster.depth_at(1,1.5)==pytest.approx(8.25)
    assert raster.bounds()==(0,4,0,6)

def independent_k(omega,depth):
    lo,hi=1e-12,max(1.,omega**2/G*20)
    for _ in range(200):
        mid=(lo+hi)/2
        if G*mid*math.tanh(mid*depth)<omega**2: lo=mid
        else: hi=mid
    return (lo+hi)/2

@pytest.mark.parametrize("depth",(1000.,20.,2.))
def test_finite_depth_dispersion_matches_independent_bisection(depth):
    omega=2*math.pi/5; k,c,cg=finite_depth_dispersion(omega,depth)
    expected=independent_k(omega,depth)
    assert k==pytest.approx(expected,rel=1e-10)
    assert c==pytest.approx(omega/k)
    assert 0<cg<=c

def test_finite_depth_orbit_vanishes_at_bottom_and_deep_mode_is_unchanged():
    finite=WaveField(RegularWaves(kind="regular",height_m=1,period_s=5,direction_rad=0,depth_model="finite_depth"))
    p=t(((0,0,10),)); _,velocity,_=finite.sample_kinematics(p,.7,local_depth_m=t((10,)))
    assert abs(float(velocity[0,2]))<1e-12
    legacy=WaveField(RegularWaves(kind="regular",height_m=1,period_s=5,direction_rad=0))
    surface,_,_=legacy.sample_kinematics(t(((0,0,0),)),0)
    assert float(surface[0])==pytest.approx(-.5)

def test_shoaling_breaking_and_irregular_component_depth_determinism():
    spec=RegularWaves(kind="regular",height_m=4,period_s=6,direction_rad=0,depth_model="finite_depth",
        shoaling={"enabled":True,"reference_depth_m":50},breaking={"enabled":True,"gamma":.5})
    field=WaveField(spec); surface,_,_,diag=field.sample_kinematics(t(((0,0,0),)),0,local_depth_m=t((2,)),diagnostics=True)
    assert bool(diag["breaking_active"][0]); assert abs(float(surface[0]))<=.5+1e-12
    irregular=WaveField(IrregularWaves(kind="irregular",spectrum="jonswap",significant_height_m=1,peak_period_s=5,
        direction_rad=0,component_count=16,seed=3,depth_model="finite_depth"))
    assert irregular.component_diagnostics(3)==irregular.component_diagnostics(3)
    assert len({round(row[0],8) for row in irregular.component_diagnostics(3)})>1

def test_uniform_current_wave_interaction_has_correct_projection_sign():
    omega=2*math.pi/5
    still=finite_depth_dispersion(omega,10,0)[0]
    following=finite_depth_dispersion(omega,10,.5)[0]
    opposing=finite_depth_dispersion(omega,10,-.5)[0]
    assert following<still<opposing
    field=WaveField(RegularWaves(kind="regular",height_m=1,period_s=5,direction_rad=0,
        depth_model="finite_depth",current_interaction={"enabled":True}))
    _,_,_,along=field.sample_kinematics(t(((0,0,0),)),0,local_depth_m=t((10,)),current_ned_mps=t(((.5,0,0),)),diagnostics=True)
    _,_,_,cross=field.sample_kinematics(t(((0,0,0),)),0,local_depth_m=t((10,)),current_ned_mps=t(((0,.5,0),)),diagnostics=True)
    assert float(along["wave_number_per_m"][0])<float(cross["wave_number_per_m"][0])

def test_current_validity_above_at_and_below_seabed_without_attenuation():
    world=ParametricWorld(World.model_validate({"source":{"kind":"parametric"},"environment":{
        "current":{"kind":"uniform","ned_mps":[1,2,0]},"wind":{"kind":"uniform","ned_mps":[0,0,0]},
        "waves":{"kind":"calm"},"visibility_m":1000},"bathymetry":{"kind":"flat","bottom_ned_z_m":10,"vertical_datum":"MSL"}}),0)
    sample=world.sample(t(((0,0,9),(0,0,10),(0,0,11))),sim_time_s=0,env_id=0)
    assert sample.current_valid.tolist()==[True,True,False]
    assert sample.current_ned_mps.tolist()==[[1,2,0]]*3
    assert sample.current_depth_semantics=="source_2d_vertically_uniform"

def test_flat_and_sloped_seabed_contact_uses_point_normal_and_torque():
    flat=Bathymetry(FlatBathymetry(kind="flat",bottom_ned_z_m=2,vertical_datum="MSL"))
    seabed=CollisionBody(0,"world:seabed",SeabedSurface(flat,(0,0),100,1),static_position_ned_m=(0,0,0),contact_material=ContactMaterial(0,.6,1))
    vessel=body(Box((1,1,1)),(0,0,1.2),(0,0,1,0,0,0))
    result=resolve_contacts((vessel,seabed),material=ContactMaterial(),dt_s=.02)
    assert len(result.events)==1; event=result.events[0]
    assert event.normal_a_to_b_ned.tolist()==pytest.approx([0,0,1])
    assert event.penetration_m==pytest.approx(.2)
    assert float(result.states[(0,"vessel")].position_ned[2])==pytest.approx(1.)
    slope=Bathymetry(SlopedBathymetry(kind="plane",origin_ned_m=(0,0,0),bottom_at_origin_ned_z_m=2,north_slope=.2,east_slope=.1,vertical_datum="MSL"))
    sloped=CollisionBody(0,"world:seabed",SeabedSurface(slope,(0,0),100,1),static_position_ned_m=(0,0,0))
    offcenter=body(Sphere(.5),(1,0,1.8),(0,0,1,0,0,0))
    grounded=resolve_contacts((offcenter,sloped),material=ContactMaterial(),dt_s=.02)
    assert grounded.events and grounded.events[0].normal_a_to_b_ned[0]<0
    assert torch.isfinite(grounded.events[0].body_a_equivalent_wrench_frd).all()

def test_cached_collision_tile_is_reused():
    bath=Bathymetry(FlatBathymetry(kind="flat",bottom_ned_z_m=10,vertical_datum="MSL",collision={"enabled":True,"tile_size_m":20,"resolution_m":2}))
    assert bath.collision_tile(1,1) is bath.collision_tile(19,19)
    assert bath.collision_tile(21,1) is not bath.collision_tile(1,1)
