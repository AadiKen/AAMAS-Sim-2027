"""Collision proxy geometry, separate from visual assets."""

from dataclasses import dataclass
import math
from typing import Any, TypeAlias

from bcod_sim.core.errors import PhysicalValidationError

Vec3 = tuple[float, float, float]


def _finite_vec3(value: Vec3) -> bool:
    return len(value) == 3 and all(math.isfinite(x) for x in value)


@dataclass(frozen=True)
class Sphere:
    radius_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise PhysicalValidationError("Sphere radius must be positive and finite")


@dataclass(frozen=True)
class Box:
    half_extents_m: Vec3

    def __post_init__(self) -> None:
        if not _finite_vec3(self.half_extents_m) or any(x <= 0 for x in self.half_extents_m):
            raise PhysicalValidationError("Box half extents must be positive and finite")


@dataclass(frozen=True)
class Capsule:
    radius_m: float
    half_segment_m: float  # local Z axis

    def __post_init__(self) -> None:
        if (not math.isfinite(self.radius_m) or self.radius_m <= 0 or
            not math.isfinite(self.half_segment_m) or self.half_segment_m < 0):
            raise PhysicalValidationError("Capsule dimensions must be finite and nonnegative")


@dataclass(frozen=True)
class ConvexHull:
    vertices_m: tuple[Vec3, ...]
    faces: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if len(self.vertices_m) < 4 or len(self.faces) < 4 or not all(_finite_vec3(v) for v in self.vertices_m):
            raise PhysicalValidationError("Convex hull requires finite vertices and triangular faces")
        if any(len(face) != 3 or len(set(face)) != 3 or any(i < 0 or i >= len(self.vertices_m) for i in face)
               for face in self.faces):
            raise PhysicalValidationError("Invalid convex hull face")
        edge_counts: dict[tuple[int, int], int] = {}
        centroid = tuple(sum(vertex[i] for vertex in self.vertices_m) / len(self.vertices_m) for i in range(3))
        for face in self.faces:
            a, b, c = (self.vertices_m[index] for index in face)
            ab, ac = tuple(b[i]-a[i] for i in range(3)), tuple(c[i]-a[i] for i in range(3))
            normal = (ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0])
            length = math.sqrt(sum(x*x for x in normal))
            if length < 1e-12:
                raise PhysicalValidationError("Degenerate convex hull face")
            if sum(normal[i]*(centroid[i]-a[i]) for i in range(3)) > 0:
                normal = tuple(-x for x in normal)
            if any(sum(normal[i]*(vertex[i]-a[i]) for i in range(3)) > 1e-8 * length
                   for vertex in self.vertices_m):
                raise PhysicalValidationError("Convex hull vertices lie outside a face")
            for i, j in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = (min(i, j), max(i, j))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        if any(count != 2 for count in edge_counts.values()):
            raise PhysicalValidationError("Convex hull must be closed")


@dataclass(frozen=True)
class TriangleMesh:
    vertices_m: tuple[Vec3, ...]
    triangles: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if len(self.vertices_m) < 3 or not self.triangles or not all(_finite_vec3(v) for v in self.vertices_m):
            raise PhysicalValidationError("Triangle mesh requires finite vertices and triangles")
        if any(len(face) != 3 or len(set(face)) != 3 or any(i < 0 or i >= len(self.vertices_m) for i in face)
               for face in self.triangles):
            raise PhysicalValidationError("Invalid triangle mesh triangle")
        for face in self.triangles:
            a, b, c = (self.vertices_m[index] for index in face)
            ab, ac = tuple(b[i]-a[i] for i in range(3)), tuple(c[i]-a[i] for i in range(3))
            cross = (ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0])
            if math.sqrt(sum(x*x for x in cross)) < 1e-12:
                raise PhysicalValidationError("Degenerate triangle mesh face")


@dataclass(frozen=True)
class SeabedSurface:
    """Cached native height surface tile backed by the canonical bathymetry API."""
    bathymetry: Any
    tile_index: tuple[int, int]
    tile_size_m: float
    resolution_m: float

    def __post_init__(self) -> None:
        if self.tile_size_m <= 0 or self.resolution_m <= 0:
            raise PhysicalValidationError("Seabed tile dimensions must be positive")


@dataclass(frozen=True)
class CompoundChild:
    shape: "PrimitiveShape"
    offset_m: Vec3

    def __post_init__(self) -> None:
        if not _finite_vec3(self.offset_m):
            raise PhysicalValidationError("Compound child offset must be finite")


PrimitiveShape: TypeAlias = Sphere | Box | Capsule | ConvexHull


@dataclass(frozen=True)
class Compound:
    children: tuple[CompoundChild, ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise PhysicalValidationError("Compound shape requires children")


CollisionShape: TypeAlias = PrimitiveShape | TriangleMesh | Compound | SeabedSurface


def bounding_radius(shape: CollisionShape) -> float:
    if isinstance(shape, SeabedSurface):
        return 1e12
    if isinstance(shape, Sphere):
        return shape.radius_m
    if isinstance(shape, Box):
        return math.sqrt(sum(x*x for x in shape.half_extents_m))
    if isinstance(shape, Capsule):
        return shape.radius_m + shape.half_segment_m
    if isinstance(shape, (ConvexHull, TriangleMesh)):
        return max(math.sqrt(sum(x*x for x in v)) for v in shape.vertices_m)
    return max(math.sqrt(sum(x*x for x in child.offset_m)) + bounding_radius(child.shape)
               for child in shape.children)
