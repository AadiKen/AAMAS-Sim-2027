"""Deterministic contacts for explicit supported shape pairs."""

from dataclasses import dataclass
import math

import torch

from bcod_sim.collision.broadphase import CollisionBody, broadphase_pairs
from bcod_sim.collision.shapes import Box, Capsule, CollisionShape, Compound, ConvexHull, SeabedSurface, Sphere, TriangleMesh
from bcod_sim.core.errors import CollisionSolverError
from bcod_sim.frames.tensor import rotate_body_to_world, rotate_world_to_body


@dataclass(frozen=True)
class Contact:
    id: str
    env_id: int
    body_a: str
    body_b: str
    point_ned_m: torch.Tensor
    normal_a_to_b_ned: torch.Tensor
    penetration_m: float


@dataclass(frozen=True)
class _Instance:
    body: CollisionBody
    suffix: str
    shape: CollisionShape
    center: torch.Tensor
    q: torch.Tensor


def _instances(body: CollisionBody, *, dtype: torch.dtype, device: torch.device) -> tuple[_Instance, ...]:
    center = body.position().to(dtype=dtype, device=device)
    orientation = body.orientation().to(dtype=dtype, device=device)
    if isinstance(body.shape, Compound):
        return tuple(_Instance(body, str(i), child.shape,
                               center + rotate_body_to_world(center.new_tensor(child.offset_m), orientation),
                               orientation) for i, child in enumerate(body.shape.children))
    return (_Instance(body, "0", body.shape, center, orientation),)


def _unit(delta: torch.Tensor) -> tuple[torch.Tensor, float]:
    length = torch.linalg.vector_norm(delta).item()
    if length < 1e-12:
        direction = delta.new_tensor((1.0, 0.0, 0.0))
        return direction, 0.0
    return delta / length, length


def _sphere_sphere(a: _Instance, b: _Instance):
    normal, distance = _unit(b.center - a.center)
    penetration = a.shape.radius_m + b.shape.radius_m - distance
    if penetration <= 0:
        return None
    point = a.center + normal * (a.shape.radius_m - penetration / 2)
    return point, normal, penetration


def _sphere_box(sphere: _Instance, box: _Instance):
    local = rotate_world_to_body(sphere.center - box.center, box.q)
    extents = local.new_tensor(box.shape.half_extents_m)
    closest = torch.minimum(torch.maximum(local, -extents), extents)
    delta = closest - local
    normal_local, distance = _unit(delta)
    if distance < 1e-12:
        gaps = extents - local.abs()
        axis = int(torch.argmin(gaps).item())
        normal_local = torch.zeros_like(local)
        normal_local[axis] = -1.0 if local[axis].item() >= 0 else 1.0
        closest[axis] = extents[axis] if local[axis].item() >= 0 else -extents[axis]
        distance = -gaps[axis].item()
    penetration = sphere.shape.radius_m - distance
    if penetration <= 0:
        return None
    point = box.center + rotate_body_to_world(closest, box.q)
    normal = rotate_body_to_world(normal_local, box.q)
    return point, normal, penetration


def _box_axes(box: _Instance) -> tuple[torch.Tensor, ...]:
    return tuple(rotate_body_to_world(box.center.new_tensor(axis), box.q) for axis in
                 ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))


def _box_support(box: _Instance, direction: torch.Tensor) -> torch.Tensor:
    result = box.center.clone()
    for axis, extent in zip(_box_axes(box), box.shape.half_extents_m):
        result = result + axis * (extent if torch.dot(axis, direction).item() >= 0 else -extent)
    return result


def _box_box(a: _Instance, b: _Instance):
    aa, bb = _box_axes(a), _box_axes(b)
    delta = b.center - a.center
    axes = (*aa, *bb, *(torch.linalg.cross(x, y) for x in aa for y in bb))
    best_overlap, best_normal = math.inf, None
    for axis in axes:
        normal, length = _unit(axis)
        if length < 1e-10:
            continue
        ra = sum(extent * abs(torch.dot(normal, direction).item()) for extent, direction in zip(a.shape.half_extents_m, aa))
        rb = sum(extent * abs(torch.dot(normal, direction).item()) for extent, direction in zip(b.shape.half_extents_m, bb))
        separation = abs(torch.dot(delta, normal).item())
        overlap = ra + rb - separation
        if overlap <= 0:
            return None
        if overlap < best_overlap:
            best_overlap = overlap
            best_normal = normal if torch.dot(delta, normal).item() >= 0 else -normal
    assert best_normal is not None
    point = (_box_support(a, best_normal) + _box_support(b, -best_normal)) / 2
    return point, best_normal, best_overlap


def _capsule_endpoints(capsule: _Instance) -> tuple[torch.Tensor, torch.Tensor]:
    axis = rotate_body_to_world(capsule.center.new_tensor((0., 0., 1.)), capsule.q)
    half = capsule.shape.half_segment_m
    return capsule.center - half * axis, capsule.center + half * axis


def _segment_closest(point: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = b - a
    denominator = torch.dot(ab, ab).item()
    t = 0.0 if denominator < 1e-15 else max(0.0, min(1.0, torch.dot(point-a, ab).item() / denominator))
    return a + t * ab


def _sphere_capsule(sphere: _Instance, capsule: _Instance):
    c0, c1 = _capsule_endpoints(capsule)
    closest = _segment_closest(sphere.center, c0, c1)
    normal, distance = _unit(closest - sphere.center)
    penetration = sphere.shape.radius_m + capsule.shape.radius_m - distance
    if penetration <= 0:
        return None
    point = sphere.center + normal * (sphere.shape.radius_m - penetration / 2)
    return point, normal, penetration


def _capsule_capsule(a: _Instance, b: _Instance):
    a0, a1 = _capsule_endpoints(a)
    b0, b1 = _capsule_endpoints(b)
    u, v, w = a1-a0, b1-b0, a0-b0
    aa, bb, cc = torch.dot(u, u).item(), torch.dot(u, v).item(), torch.dot(v, v).item()
    dd, ee = torch.dot(u, w).item(), torch.dot(v, w).item()
    denominator = aa*cc-bb*bb
    if denominator < 1e-14:
        s = 0.0
    else:
        s = max(0.0, min(1.0, (bb*ee-cc*dd)/denominator))
    t = max(0.0, min(1.0, (bb*s+ee)/cc)) if cc > 1e-14 else 0.0
    s = max(0.0, min(1.0, (bb*t-dd)/aa)) if aa > 1e-14 else 0.0
    closest_a, closest_b = a0+s*u, b0+t*v
    normal, distance = _unit(closest_b-closest_a)
    penetration = a.shape.radius_m+b.shape.radius_m-distance
    if penetration <= 0:
        return None
    point = closest_a + normal*(a.shape.radius_m-penetration/2)
    return point, normal, penetration


def _closest_triangle(point: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # Real-Time Collision Detection closest-point barycentric regions.
    ab, ac, ap = b-a, c-a, point-a
    d1, d2 = torch.dot(ab, ap).item(), torch.dot(ac, ap).item()
    if d1 <= 0 and d2 <= 0:
        return a
    bp = point-b
    d3, d4 = torch.dot(ab, bp).item(), torch.dot(ac, bp).item()
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1*d4 - d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1/(d1-d3))*ab
    cp = point-c
    d5, d6 = torch.dot(ab, cp).item(), torch.dot(ac, cp).item()
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2/(d2-d6))*ac
    va = d3*d6 - d5*d4
    if va <= 0 and (d4-d3) >= 0 and (d5-d6) >= 0:
        return b + ((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
    denominator = 1/(va+vb+vc)
    return a + vb*denominator*ab + vc*denominator*ac


def _sphere_surface(sphere: _Instance, surface: _Instance):
    shape = surface.shape
    faces = shape.faces if isinstance(shape, ConvexHull) else shape.triangles
    vertices = tuple(surface.center + rotate_body_to_world(surface.center.new_tensor(v), surface.q)
                     for v in shape.vertices_m)
    choices = []
    for index, (i, j, k) in enumerate(faces):
        closest = _closest_triangle(sphere.center, vertices[i], vertices[j], vertices[k])
        distance = torch.linalg.vector_norm(closest - sphere.center).item()
        choices.append((distance, index, closest))
    distance, index, closest = min(choices, key=lambda item: (item[0], item[1]))
    inside = False
    if isinstance(shape, ConvexHull):
        centroid = torch.stack(vertices).mean(dim=0)
        inside = True
        for i, j, k in faces:
            normal = torch.linalg.cross(vertices[j]-vertices[i], vertices[k]-vertices[i])
            if torch.dot(normal, vertices[i]-centroid).item() < 0:
                normal = -normal
            if torch.dot(normal, sphere.center-vertices[i]).item() > 1e-9:
                inside = False
                break
    penetration = sphere.shape.radius_m + distance if inside else sphere.shape.radius_m - distance
    if penetration <= 0:
        return None
    normal, _ = _unit(closest-sphere.center)
    if inside:
        normal = -normal
    return closest, normal, penetration


def _shape_seabed(body: _Instance, seabed: _Instance):
    """Contact against a canonical height surface; normal points into seabed."""
    surface = seabed.shape.bathymetry
    if isinstance(body.shape, Sphere):
        normal = body.center.new_tensor(surface.seabed_normal(float(body.center[0]), float(body.center[1])))
        candidates = (body.center + normal*body.shape.radius_m,)
    elif isinstance(body.shape, Box):
        axes = _box_axes(body); candidates = tuple(body.center + sum(
            (axes[i]*(signs[i]*body.shape.half_extents_m[i]) for i in range(3)),
            start=torch.zeros_like(body.center)) for signs in
            ((x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)))
    elif isinstance(body.shape, Capsule):
        endpoints = _capsule_endpoints(body); candidates=[]
        for endpoint in endpoints:
            normal=endpoint.new_tensor(surface.seabed_normal(float(endpoint[0]),float(endpoint[1])))
            candidates.append(endpoint+normal*body.shape.radius_m)
        candidates=tuple(candidates)
    else:
        candidates=tuple(body.center+rotate_body_to_world(body.center.new_tensor(v),body.q) for v in body.shape.vertices_m)
    choices=[]
    for candidate in candidates:
        n,e=float(candidate[0]),float(candidate[1]); bottom=surface.depth_at(n,e)
        penetration=float(candidate[2])-bottom
        choices.append((penetration,n,e,bottom,candidate))
    penetration,n,e,bottom,candidate=max(choices,key=lambda row: row[0])
    if penetration <= 0: return None
    normal=body.center.new_tensor(surface.seabed_normal(n,e))
    point=body.center.new_tensor((n,e,bottom))
    return point,normal,penetration


def _pair_contact(a: _Instance, b: _Instance):
    first, second = a.shape, b.shape
    if isinstance(first, Sphere) and isinstance(second, Sphere):
        return _sphere_sphere(a, b)
    if isinstance(first, Sphere) and isinstance(second, Box):
        return _sphere_box(a, b)
    if isinstance(first, Box) and isinstance(second, Sphere):
        found = _sphere_box(b, a)
        return (found[0], -found[1], found[2]) if found else None
    if isinstance(first, Box) and isinstance(second, Box):
        return _box_box(a, b)
    if isinstance(first, Sphere) and isinstance(second, Capsule):
        return _sphere_capsule(a, b)
    if isinstance(first, Capsule) and isinstance(second, Sphere):
        found = _sphere_capsule(b, a)
        return (found[0], -found[1], found[2]) if found else None
    if isinstance(first, Capsule) and isinstance(second, Capsule):
        return _capsule_capsule(a, b)
    if isinstance(first, Sphere) and isinstance(second, (ConvexHull, TriangleMesh)):
        return _sphere_surface(a, b)
    if isinstance(second, Sphere) and isinstance(first, (ConvexHull, TriangleMesh)):
        found = _sphere_surface(b, a)
        return (found[0], -found[1], found[2]) if found else None
    if isinstance(second, SeabedSurface) and isinstance(first, (Sphere, Box, Capsule, ConvexHull)):
        return _shape_seabed(a, b)
    if isinstance(first, SeabedSurface) and isinstance(second, (Sphere, Box, Capsule, ConvexHull)):
        found = _shape_seabed(b, a)
        return (found[0], -found[1], found[2]) if found else None
    raise CollisionSolverError(f"Unsupported collision shape pair: {type(first).__name__}/{type(second).__name__}")


def detect_contacts(bodies: tuple[CollisionBody, ...]) -> tuple[Contact, ...]:
    found = []
    for body_a, body_b in broadphase_pairs(bodies):
        dynamic = body_a if body_a.dynamic else body_b
        assert dynamic.state is not None
        dtype, device = dynamic.state.position_ned.dtype, dynamic.state.position_ned.device
        for a in _instances(body_a, dtype=dtype, device=device):
            for b in _instances(body_b, dtype=dtype, device=device):
                contact = _pair_contact(a, b)
                if contact is None:
                    continue
                point, normal, penetration = contact
                if not torch.isfinite(point).all().item() or not torch.isfinite(normal).all().item() or not math.isfinite(penetration):
                    raise CollisionSolverError("Nonfinite contact geometry")
                found.append(Contact(f"{body_a.env_id}:{body_a.id}:{a.suffix}|{body_b.id}:{b.suffix}",
                                     body_a.env_id, body_a.id, body_b.id, point, normal, penetration))
    return tuple(sorted(found, key=lambda item: item.id))
