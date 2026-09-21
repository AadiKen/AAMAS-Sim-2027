"""Rigid-body and total mass matrices with physical validation."""

from dataclasses import dataclass
import math

import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.dynamics.reference_points import transform_added_mass


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind()
    zero = torch.zeros_like(x)
    return torch.stack((torch.stack((zero, -z, y)),
                        torch.stack((z, zero, -x)),
                        torch.stack((-y, x, zero))))


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    cg_frd_m: torch.Tensor  # [3]
    inertia_cg_kg_m2: torch.Tensor  # [3,3]
    added_mass_kg: torch.Tensor  # [6,6], positive inertia convention
    max_condition_number: float = 1e10
    added_mass_reference_point_frd_m: tuple[float,float,float]=(0.0,0.0,0.0)
    added_mass_provenance: str="manual"
    added_mass_units: str="kg, kg*m, kg*m^2"

    @classmethod
    def with_added_mass_at_origin(cls,mass_kg:float,cg_frd_m:torch.Tensor,inertia_cg_kg_m2:torch.Tensor,
                                  added_mass_at_source:torch.Tensor,source_reference_point_frd_m:torch.Tensor,**kwargs):
        resolved=transform_added_mass(added_mass_at_source,source_reference_point_frd_m)
        return cls(mass_kg,cg_frd_m,inertia_cg_kg_m2,resolved,
                   added_mass_reference_point_frd_m=tuple(float(x) for x in source_reference_point_frd_m),**kwargs)

    def matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        cg, inertia, added = self.cg_frd_m, self.inertia_cg_kg_m2, self.added_mass_kg
        if cg.shape != (3,) or inertia.shape != (3, 3) or added.shape != (6, 6):
            raise PhysicalValidationError("Invalid mass property shapes")
        if not all(x.is_floating_point() and torch.isfinite(x).all().item() for x in (cg, inertia, added)):
            raise PhysicalValidationError("Mass properties must be finite floating tensors")
        if any(x.device != cg.device or x.dtype != cg.dtype for x in (inertia, added)):
            raise PhysicalValidationError("Mass properties must share device and dtype")
        if not (math.isfinite(self.mass_kg) and self.mass_kg > 0 and self.added_mass_provenance and self.added_mass_units and
                math.isfinite(self.max_condition_number) and self.max_condition_number > 1):
            raise PhysicalValidationError("Mass and condition limit must be positive")
        eye = torch.eye(3, dtype=cg.dtype, device=cg.device)
        cross = skew(cg)
        rigid = torch.cat((torch.cat((self.mass_kg * eye, -self.mass_kg * cross), dim=1),
                           torch.cat((self.mass_kg * cross, inertia - self.mass_kg * cross @ cross), dim=1)), dim=0)
        total = rigid + added
        for label, matrix in (("rigid inertia", inertia), ("added mass", added), ("total mass", total)):
            if not torch.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-10):
                raise PhysicalValidationError(f"{label} is not symmetric")
        inertia_eigenvalues = torch.linalg.eigvalsh(inertia)
        if inertia_eigenvalues[0].item() <= 0 or torch.linalg.eigvalsh(total)[0].item() <= 0:
            raise PhysicalValidationError("Rigid inertia and total mass must be positive definite")
        if inertia_eigenvalues[-1].item() > inertia_eigenvalues[0:2].sum().item() + 1e-9:
            raise PhysicalValidationError("Rigid inertia violates principal-moment triangle inequality")
        if torch.linalg.cond(total).item() > self.max_condition_number:
            raise PhysicalValidationError("Total mass matrix is ill-conditioned")
        return rigid, total

    def diagnostics(self)->dict[str,object]:
        rigid,total=self.matrices(); eig=torch.linalg.eigvalsh(total)
        return {"M_RB":rigid,"M_A":self.added_mass_kg,"M_total":total,"condition_number":float(torch.linalg.cond(total)),
                "eigenvalue_min":float(eig[0]),"eigenvalue_max":float(eig[-1]),"added_mass_source_reference_point_frd_m":self.added_mass_reference_point_frd_m,
                "added_mass_provenance":self.added_mass_provenance,"added_mass_units":self.added_mass_units}
