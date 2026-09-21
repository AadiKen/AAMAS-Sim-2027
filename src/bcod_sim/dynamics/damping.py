"""Configurable water-relative linear and nonlinear hydrodynamic damping."""
from dataclasses import dataclass
import math
import torch
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.dynamics.restoring import WrenchResult

@dataclass(frozen=True)
class CoupledDampingTerm:
    output_axis:int
    factors:tuple[int,...]
    coefficient:float
    absolute_factors:tuple[int,...]=()
    def validate(self)->None:
        if self.output_axis not in range(6) or not self.factors or any(i not in range(6) for i in (*self.factors,*self.absolute_factors)) or not math.isfinite(self.coefficient):
            raise PhysicalValidationError("Invalid coupled damping term")
    def value(self,nu:torch.Tensor)->torch.Tensor:
        result=nu.new_tensor(self.coefficient)
        for i in self.factors: result=result*nu[i]
        for i in self.absolute_factors: result=result*nu[i].abs()
        return result

@dataclass(frozen=True)
class Damping:
    """Legacy diagonal constructor plus optional full linear/coupled terms."""
    linear:torch.Tensor
    quadratic:torch.Tensor
    linear_matrix:torch.Tensor|None=None
    coupled_terms:tuple[CoupledDampingTerm,...]=()
    intended_dissipative:bool=True

    def validate(self,*,dtype:torch.dtype,device:torch.device)->None:
        for name,value in (("linear",self.linear),("quadratic",self.quadratic)):
            if value.shape!=(6,) or value.dtype!=dtype or value.device!=device or not torch.isfinite(value).all().item() or (value<0).any().item():
                raise PhysicalValidationError(f"{name} damping must be finite nonnegative and match plant")
        if self.linear_matrix is not None:
            matrix=self.linear_matrix
            if matrix.shape!=(6,6) or matrix.dtype!=dtype or matrix.device!=device or not torch.isfinite(matrix).all().item(): raise PhysicalValidationError("Linear damping matrix must be finite 6x6")
            if self.intended_dissipative:
                symmetric=(matrix+matrix.T)/2
                if torch.linalg.eigvalsh(symmetric)[0].item() < -1e-10*max(1.,torch.linalg.matrix_norm(matrix).item()):
                    raise PhysicalValidationError("Intended-dissipative linear damping is not positive semidefinite")
        for term in self.coupled_terms: term.validate()

    def components(self,nu_relative:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        matrix=torch.diag(self.linear) if self.linear_matrix is None else self.linear_matrix
        linear=-(matrix@nu_relative)
        nonlinear=-self.quadratic*nu_relative.abs()*nu_relative
        for term in self.coupled_terms:
            nonlinear[term.output_axis]-=term.value(nu_relative)
        return linear,nonlinear

    def evaluate(self,nu_relative:torch.Tensor)->WrenchResult:
        if nu_relative.shape!=(6,) or not torch.isfinite(nu_relative).all().item(): raise PhysicalValidationError("Damping velocity must be finite 6-vector")
        linear,nonlinear=self.components(nu_relative); tau=linear+nonlinear
        power=float(nu_relative@tau)
        if self.intended_dissipative and power>1e-9: raise PhysicalValidationError("Configured damping injects energy")
        return WrenchResult(tau,"hydrodynamic_damping",{"tau_linear_damping":linear,"tau_nonlinear_damping":nonlinear,"relative_velocity_body":nu_relative,"power_w":power})

    def wrench(self,nu_body:torch.Tensor)->torch.Tensor:
        return self.evaluate(nu_body).tau_body
