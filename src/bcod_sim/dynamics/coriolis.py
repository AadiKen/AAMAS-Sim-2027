"""Power-neutral spatial Coriolis/centripetal action."""

import torch


def coriolis_wrench(mass_matrix: torch.Tensor, nu_body: torch.Tensor) -> torch.Tensor:
    linear_momentum, angular_momentum = (mass_matrix @ nu_body)[:3], (mass_matrix @ nu_body)[3:]
    velocity, omega = nu_body[:3], nu_body[3:]
    return torch.cat((torch.linalg.cross(omega, linear_momentum),
                      torch.linalg.cross(velocity, linear_momentum) + torch.linalg.cross(omega, angular_momentum)))
