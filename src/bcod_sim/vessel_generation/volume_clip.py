"""Exact volume and first moment below a horizontal plane for closed meshes.

The cap on the clipping plane has zero signed tetrahedron volume when the
reference origin lies on that plane, so no cap triangulation is needed.
"""
import numpy as np
import trimesh


def submerged_volume_centroid(mesh: trimesh.Trimesh, waterline_z: float) -> tuple[float, np.ndarray]:
    triangles = np.asarray(mesh.triangles, dtype=float)
    shifted = triangles.copy()
    shifted[:, :, 2] -= waterline_z
    wet = shifted[:, :, 2] >= 0
    count = wet.sum(axis=1)
    pieces = [shifted[count == 3]]
    for index in range(3):
        single = shifted[(count == 1) & wet[:, index]]
        if len(single):
            a = single[:, index]
            b = single[:, (index + 1) % 3]
            c = single[:, (index + 2) % 3]
            ab = a + (-a[:, 2] / (b[:, 2] - a[:, 2]))[:, None] * (b - a)
            ac = a + (-a[:, 2] / (c[:, 2] - a[:, 2]))[:, None] * (c - a)
            pieces.append(np.stack((a, ab, ac), axis=1))
        double = shifted[(count == 2) & ~wet[:, index]]
        if len(double):
            dry = double[:, index]
            b = double[:, (index + 1) % 3]
            c = double[:, (index + 2) % 3]
            ca = c + (-c[:, 2] / (dry[:, 2] - c[:, 2]))[:, None] * (dry - c)
            ab = dry + (-dry[:, 2] / (b[:, 2] - dry[:, 2]))[:, None] * (b - dry)
            pieces.append(np.stack((b, c, ca), axis=1))
            pieces.append(np.stack((b, ca, ab), axis=1))
    signed_volume = 0.
    first = np.zeros(3)
    for tri in pieces:
        if not len(tri):
            continue
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
        signed_volume += float(volumes.sum())
        first += np.einsum("i,ij->j", volumes, (a + b + c) / 4)
    if signed_volume <= 1e-12:
        raise ValueError("No positive submerged volume")
    cb = first / signed_volume + np.array((0., 0., waterline_z))
    return signed_volume, cb
