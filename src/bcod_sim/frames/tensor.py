"""Torch quaternion and NED/FRD transforms used by runtime components."""

import torch

from bcod_sim.core.errors import FrameConversionError


def q_normalize(q: torch.Tensor) -> torch.Tensor:
    if q.shape != (4,) or not torch.isfinite(q).all().item():
        raise FrameConversionError("Expected finite [w,x,y,z] quaternion")
    norm = torch.linalg.vector_norm(q)
    if norm.item() < 1e-12:
        raise FrameConversionError("Zero quaternion")
    return q / norm


def q_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind()
    bw, bx, by, bz = b.unbind()
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw))


def rotate_body_to_world(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    q = q_normalize(q)
    t = 2 * torch.linalg.cross(q[1:], v)
    return v + q[0] * t + torch.linalg.cross(q[1:], t)


def rotate_world_to_body(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    q = q_normalize(q)
    inverse = torch.cat((q[:1], -q[1:]))
    return rotate_body_to_world(v, inverse)
