"""Single canonical NED bathymetry API for environment and collision physics."""
from dataclasses import dataclass
import math
from typing import Protocol
import torch
from bcod_sim.config.models import FlatBathymetry, SlopedBathymetry
from bcod_sim.core.errors import ExternalDataCoverageError, PhysicalValidationError

class BathymetrySurface(Protocol):
    vertical_datum: str
    def bottom_ned_z_m(self, positions_ned_m: torch.Tensor) -> torch.Tensor: ...
    def seabed_position(self, north_m: float, east_m: float) -> tuple[float,float,float]: ...
    def seabed_normal(self, north_m: float, east_m: float) -> tuple[float,float,float]: ...
    def depth_at(self, north_m: float, east_m: float, *, surface_ned_z_m: float=0.) -> float: ...
    def is_wet(self, north_m: float, east_m: float, *, surface_ned_z_m: float=0.) -> bool: ...
    def bounds(self) -> tuple[float,float,float,float] | None: ...

class Bathymetry:
    """Analytic synthetic bathymetry in world NED; positive Z/depth is down."""
    def __init__(self, spec: FlatBathymetry | SlopedBathymetry) -> None:
        self.spec, self.vertical_datum = spec, spec.vertical_datum
        self._collision_tiles = {}

    def bottom_ned_z_m(self, positions_ned_m: torch.Tensor) -> torch.Tensor:
        if positions_ned_m.ndim != 2 or positions_ned_m.shape[1] != 3:
            raise PhysicalValidationError("Bathymetry queries require [N,3] NED positions")
        if isinstance(self.spec, FlatBathymetry):
            return positions_ned_m.new_full((positions_ned_m.shape[0],), self.spec.bottom_ned_z_m)
        s=self.spec
        return s.bottom_at_origin_ned_z_m+s.north_slope*(positions_ned_m[:,0]-s.origin_ned_m[0])+s.east_slope*(positions_ned_m[:,1]-s.origin_ned_m[1])

    def depth_at(self,north_m:float,east_m:float,*,surface_ned_z_m:float=0.)->float:
        p=torch.tensor(((north_m,east_m,0.),),dtype=torch.float64)
        return float(self.bottom_ned_z_m(p)[0])-surface_ned_z_m
    def seabed_position(self,north_m:float,east_m:float)->tuple[float,float,float]: return north_m,east_m,self.depth_at(north_m,east_m)
    def seabed_normal(self,north_m:float,east_m:float)->tuple[float,float,float]:
        raw=(0.,0.,1.) if isinstance(self.spec,FlatBathymetry) else (-self.spec.north_slope,-self.spec.east_slope,1.)
        length=math.sqrt(sum(x*x for x in raw)); return tuple(x/length for x in raw)
    def is_wet(self,north_m:float,east_m:float,*,surface_ned_z_m:float=0.)->bool: return self.depth_at(north_m,east_m,surface_ned_z_m=surface_ned_z_m)>0
    def bounds(self): return None
    def sample_grid(self,bounds:tuple[float,float,float,float],resolution_m:float)->torch.Tensor:
        if resolution_m<=0: raise PhysicalValidationError("Bathymetry grid resolution must be positive")
        n0,n1,e0,e1=bounds; n=torch.arange(n0,n1+resolution_m*.5,resolution_m,dtype=torch.float64); e=torch.arange(e0,e1+resolution_m*.5,resolution_m,dtype=torch.float64)
        nn,ee=torch.meshgrid(n,e,indexing="ij"); p=torch.stack((nn.flatten(),ee.flatten(),torch.zeros_like(nn).flatten()),1)
        return torch.stack((p[:,0],p[:,1],self.bottom_ned_z_m(p)),1).reshape(len(n),len(e),3)

    def collision_tile(self,north_m:float,east_m:float):
        from bcod_sim.collision.shapes import SeabedSurface
        config=self.spec.collision; key=(math.floor(north_m/config.tile_size_m),math.floor(east_m/config.tile_size_m))
        if key not in self._collision_tiles:
            self._collision_tiles[key]=SeabedSurface(self,key,config.tile_size_m,config.resolution_m)
        return self._collision_tiles[key]

@dataclass(frozen=True)
class RasterBathymetry:
    """Bilinear NED raster; rows increase north and columns increase east."""
    values_ned_z_m: torch.Tensor
    origin_north_east_m: tuple[float,float]
    spacing_north_east_m: tuple[float,float]
    vertical_datum: str
    def __post_init__(self):
        if self.values_ned_z_m.ndim!=2 or min(self.values_ned_z_m.shape)<2 or not torch.isfinite(self.values_ned_z_m).all().item(): raise PhysicalValidationError("Raster bathymetry requires finite 2D grid")
        if any(x<=0 or not math.isfinite(x) for x in self.spacing_north_east_m): raise PhysicalValidationError("Raster spacing must be positive")
    def bounds(self):
        n,e=self.origin_north_east_m; dn,de=self.spacing_north_east_m
        return n,n+dn*(self.values_ned_z_m.shape[0]-1),e,e+de*(self.values_ned_z_m.shape[1]-1)
    def bottom_ned_z_m(self,p:torch.Tensor)->torch.Tensor:
        n0,n1,e0,e1=self.bounds()
        if ((p[:,0]<n0)|(p[:,0]>n1)|(p[:,1]<e0)|(p[:,1]>e1)).any().item(): raise ExternalDataCoverageError("Bathymetry query outside raster coverage")
        dn,de=self.spacing_north_east_m; fn=(p[:,0]-n0)/dn; fe=(p[:,1]-e0)/de; i=torch.floor(fn).long().clamp(max=self.values_ned_z_m.shape[0]-2); j=torch.floor(fe).long().clamp(max=self.values_ned_z_m.shape[1]-2); an,ae=fn-i,fe-j; v=self.values_ned_z_m.to(p)
        return (1-an)*(1-ae)*v[i,j]+an*(1-ae)*v[i+1,j]+(1-an)*ae*v[i,j+1]+an*ae*v[i+1,j+1]
    def depth_at(self,n:float,e:float,*,surface_ned_z_m:float=0.): return float(self.bottom_ned_z_m(torch.tensor(((n,e,0.),),dtype=self.values_ned_z_m.dtype))[0])-surface_ned_z_m
    def seabed_position(self,n:float,e:float): return n,e,self.depth_at(n,e)
    def seabed_normal(self,n:float,e:float):
        dn,de=self.spacing_north_east_m; n0,n1,e0,e1=self.bounds(); a,b=max(n0,n-dn*.25),min(n1,n+dn*.25); c,d=max(e0,e-de*.25),min(e1,e+de*.25)
        raw=(-(self.depth_at(b,e)-self.depth_at(a,e))/(b-a),-(self.depth_at(n,d)-self.depth_at(n,c))/(d-c),1.); length=math.sqrt(sum(x*x for x in raw)); return tuple(x/length for x in raw)
    def is_wet(self,n:float,e:float,*,surface_ned_z_m:float=0.): return self.depth_at(n,e,surface_ned_z_m=surface_ned_z_m)>0
