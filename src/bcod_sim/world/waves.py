"""Deterministic regular Airy and seeded irregular spectral wave fields."""
import math
import numpy as np
import torch
from bcod_sim.config.models import CalmWaves,RegularWaves,IrregularWaves

G=9.80665

def finite_depth_dispersion(omega:float,depth_m:float,current_along_mps:float=0.)->tuple[float,float,float]:
    """Deterministic safeguarded Newton solve of (omega-kU)^2=gk tanh(kh)."""
    if omega<=0 or depth_m<=0: raise ValueError("Positive frequency and wet depth required")
    k=max(omega**2/G,omega/math.sqrt(G*depth_m))
    for _ in range(40):
        sigma=omega-k*current_along_mps; kh=k*depth_m; tanh=math.tanh(kh)
        f=sigma*sigma-G*k*tanh
        derivative=-2*current_along_mps*sigma-G*(tanh+k*depth_m/(math.cosh(min(kh,350))**2))
        if abs(derivative)<1e-14: break
        candidate=k-f/derivative
        if not math.isfinite(candidate) or candidate<=0: candidate=k*.5
        if abs(candidate-k)<=1e-13*max(1.,k): k=candidate; break
        k=candidate
    sigma=omega-k*current_along_mps
    if sigma<=0 or abs(sigma*sigma-G*k*math.tanh(k*depth_m))>1e-8*max(1.,omega*omega):
        raise ValueError("No propagating finite-depth wave solution for current/depth")
    c=omega/k; kh=k*depth_m
    ratio=0. if kh>350 else 2*kh/math.sinh(2*kh)
    cg=.5*(1+ratio)*sigma/k+current_along_mps
    return k,c,cg

class WaveField:
    def __init__(self,spec:CalmWaves|RegularWaves|IrregularWaves)->None:
        self.spec=spec; self.components=None
        if isinstance(spec,IrregularWaves):
            rng=np.random.default_rng(spec.seed); wp=2*math.pi/spec.peak_period_s
            omega=np.linspace(.35*wp,3*wp,spec.component_count); dw=omega[1]-omega[0]
            alpha=8.1e-3; beta=.74
            spectrum=alpha*9.80665**2*omega**-5*np.exp(-beta*(wp/omega)**4)
            if spec.spectrum=="jonswap":
                sigma=np.where(omega<=wp,.07,.09); spectrum*=spec.gamma**np.exp(-((omega/wp-1)**2)/(2*sigma**2))
            amplitude=np.sqrt(2*spectrum*dw); amplitude*=spec.significant_height_m/(4*math.sqrt(np.sum(amplitude**2)/2))
            self.components=(omega,amplitude,rng.uniform(0,2*math.pi,spec.component_count))

    def component_diagnostics(self,depth_m:float,current_along_mps:float=0.):
        if isinstance(self.spec,CalmWaves): return ()
        components=([2*math.pi/self.spec.period_s],[self.spec.height_m/2],[self.spec.phase_rad]) if isinstance(self.spec,RegularWaves) else self.components
        return tuple(finite_depth_dispersion(float(w),depth_m,current_along_mps if self.spec.current_interaction.enabled else 0.) for w in components[0])

    def sample_kinematics(self,positions_ned_m:torch.Tensor,sim_time_s:float,*,local_depth_m:torch.Tensor|None=None,
                          current_ned_mps:torch.Tensor|None=None,diagnostics:bool=False):
        count=positions_ned_m.shape[0]
        if isinstance(self.spec,CalmWaves):
            result=(positions_ned_m.new_zeros(count),positions_ned_m.new_zeros((count,3)),positions_ned_m.new_zeros((count,3)))
            diag={"wave_number_per_m":positions_ned_m.new_zeros(count),"phase_velocity_mps":positions_ned_m.new_zeros(count),"group_velocity_mps":positions_ned_m.new_zeros(count),"breaking_active":positions_ned_m.new_zeros(count,dtype=torch.bool)}
            return (*result,diag) if diagnostics else result
        if isinstance(self.spec,RegularWaves):
            components=([2*math.pi/self.spec.period_s],[self.spec.height_m/2],[self.spec.phase_rad]); direction=self.spec.direction_rad
        else:
            assert self.components is not None; components=self.components; direction=self.spec.direction_rad
        surface=positions_ned_m.new_zeros(count); velocity=positions_ned_m.new_zeros((count,3)); acceleration=velocity.clone(); ks=positions_ned_m.new_zeros(count); cs=ks.clone(); cgs=ks.clone(); breaking=positions_ned_m.new_zeros(count,dtype=torch.bool)
        north,east=math.cos(direction),math.sin(direction)
        for omega,amplitude,phase0 in zip(*components):
            for index in range(count):
                finite=self.spec.depth_model=="finite_depth" and local_depth_m is not None
                depth=float(local_depth_m[index]) if finite else math.inf
                if finite and depth<=self.spec.minimum_wet_depth_m: continue
                projected_current=0. if current_ned_mps is None else north*float(current_ned_mps[index,0])+east*float(current_ned_mps[index,1])
                if finite: k,c,cg=finite_depth_dispersion(float(omega),depth,projected_current if self.spec.current_interaction.enabled else 0.)
                else: k=float(omega)**2/G; c=float(omega)/k; cg=.5*c
                local_amplitude=float(amplitude)
                if finite and self.spec.shoaling.enabled:
                    reference=self.spec.shoaling.reference_depth_m or max(depth,10*G/float(omega)**2)
                    _,_,cg_ref=finite_depth_dispersion(float(omega),reference,projected_current if self.spec.current_interaction.enabled else 0.)
                    local_amplitude*=math.sqrt(max(0.,cg_ref/cg))
                if finite and self.spec.breaking.enabled:
                    cap=.5*self.spec.breaking.gamma*depth
                    if local_amplitude>cap: local_amplitude=cap; breaking[index]=True
                phase=k*(north*float(positions_ned_m[index,0])+east*float(positions_ned_m[index,1]))-float(omega)*sim_time_s+float(phase0)
                surface[index]+=-local_amplitude*math.cos(phase)
                z=max(0.,min(depth,float(positions_ned_m[index,2]))) if finite else max(0.,float(positions_ned_m[index,2])-float(surface[index]))
                if finite:
                    kh=k*depth; denom=math.sinh(min(kh,350)); horizontal=math.cosh(min(k*(depth-z),350))/denom; vertical=math.sinh(min(k*(depth-z),350))/denom
                else: horizontal=vertical=math.exp(-k*z)
                velocity[index]+=positions_ned_m.new_tensor((north*local_amplitude*float(omega)*horizontal*math.cos(phase),east*local_amplitude*float(omega)*horizontal*math.cos(phase),-local_amplitude*float(omega)*vertical*math.sin(phase)))
                acceleration[index]+=positions_ned_m.new_tensor((north*local_amplitude*float(omega)**2*horizontal*math.sin(phase),east*local_amplitude*float(omega)**2*horizontal*math.sin(phase),local_amplitude*float(omega)**2*vertical*math.cos(phase)))
                ks[index]+=k/len(components[0]); cs[index]+=c/len(components[0]); cgs[index]+=cg/len(components[0])
        result=(surface,velocity,acceleration); diag={"wave_number_per_m":ks,"phase_velocity_mps":cs,"group_velocity_mps":cgs,"breaking_active":breaking}
        return (*result,diag) if diagnostics else result

    def sample(self,positions_ned_m:torch.Tensor,sim_time_s:float)->tuple[torch.Tensor,torch.Tensor]:
        surface,velocity,_=self.sample_kinematics(positions_ned_m,sim_time_s); return surface,velocity
