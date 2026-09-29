import math
import numpy as np
import pytest
import torch
from bcod_sim.dynamics.damping import Damping
from bcod_sim.vessel_generation.simple_models import inoue_linear_maneuvering, resistance_curve, strip_added_mass

def test_ittc57_friction_analytic_no_placeholder_form():
    rho,area,L,U,nu=1025.,6.58919,4.97,.994,1.2522636248415715e-6
    re=U*L/nu; cf=.075/(math.log10(re)-2)**2; expect=.5*rho*area*cf*U**2
    r=resistance_curve(np.array([0.,U]),length=L,wetted_area=area,density=rho,viscosity=nu,classification='displacement_monohull')
    assert r['method']=='ittc57_friction_line_v3'
    assert r['force_x_n'][1]==pytest.approx(-expect,rel=1e-12)
    c=r['components'][1]; assert c['form_factor']==1 and c['form_increment_n']==0 and c['ittc57_applicable']
    assert c['wave_n']==0 and c['other_corrections_n']==0

def test_resistance_wave_component_is_separate_and_gated_input():
    r=resistance_curve(np.array([-1.,0.,1.]),length=5.,wetted_area=10.,density=1000.,viscosity=1e-6,
      classification='displacement_monohull',wave={'wave_resistance_n':[2.,0.,3.]})
    c=r['components'][2]; assert r['force_x_n'][2]==pytest.approx(-(c['friction_n']+3.))
    assert c['form_factor']==1

def test_inoue_derivatives_matrix_units_passivity_and_gates():
    m,rho,L,T,cb=1200.7055,1025.,4.97,.323,.776
    q=inoue_linear_maneuvering(mass_kg=m,density=rho,length_m=L,draft_m=T,block_coefficient=cb)
    k=2*T/L; mp=m/(.5*rho*L**2*T)
    e={'Y_v':-.5*math.pi*k-.7*mp,'Y_r':.25*math.pi*k,'N_v':-k,'N_r':-.54*k+k*k}
    assert q['derivatives_prime']==pytest.approx(e)
    D=np.asarray(q['damping_matrix_per_mps']); assert np.linalg.eigvalsh((D+D.T)/2).min()>=-1e-10
    u=.994; eps=1e-6
    def tau(v,r): return -u*D@np.array([u,v,0,0,0,r])
    dv=(tau(eps,0)-tau(-eps,0))/(2*eps); dr=(tau(0,eps)-tau(0,-eps))/(2*eps)
    fy=.5*rho*L*T*u; mn=fy*L
    assert dv[1]/fy==pytest.approx(e['Y_v']); assert dr[1]/(fy*L)==pytest.approx(e['Y_r'])
    assert dv[5]/mn==pytest.approx(e['N_v']); assert dr[5]/(mn*L)==pytest.approx(e['N_r'])
    with pytest.raises(ValueError,match='L/T'): inoue_linear_maneuvering(mass_kg=100,density=rho,length_m=4,draft_m=.8,block_coefficient=.6)
    with pytest.raises(ValueError,match='block coefficient'): inoue_linear_maneuvering(mass_kg=100,density=rho,length_m=8,draft_m=.5,block_coefficient=.9)

def test_area_equivalent_ellipse_analytic_and_fullness_sensitive():
    L,B,T,rho,n=4.,2.,.5,1025.,1000; dx=L/n; A=math.pi*(B/2)*(T/2)
    xs=[{'x_m':-L/2+(i+.5)*dx,'beam_m':B,'draft_m':T,'section_area_m2':A,'dx_m':dx} for i in range(n)]
    M=np.array(strip_added_mass(xs,rho)['matrix_6x6']); sway=rho*math.pi*(T/2)**2*L; heave=rho*math.pi*(B/2)**2*L
    assert M[1,1]==pytest.approx(sway,rel=1e-12); assert M[2,2]==pytest.approx(heave,rel=1e-12)
    assert M[5,5]==pytest.approx(sway*L**2/12,rel=1e-5)
    thin=np.array(strip_added_mass([{**s,'section_area_m2':.7*A} for s in xs],rho)['matrix_6x6'])
    assert thin[1,1]==pytest.approx(.49*sway,rel=1e-12); assert np.linalg.eigvalsh(thin).min()>=-1e-9

def test_speed_dependent_linear_damping_runtime_is_passive():
    q=inoue_linear_maneuvering(mass_kg=1200.7,density=1025.,length_m=4.97,draft_m=.323,block_coefficient=.776)
    D=torch.tensor(q['damping_matrix_per_mps'],dtype=torch.float64); z=torch.zeros(6,dtype=torch.float64)
    damp=Damping(z,z,torch.zeros((6,6),dtype=torch.float64),speed_dependent_linear_matrix_per_mps=D)
    nu=torch.tensor([.5,.1,0.,0.,0.,.02],dtype=torch.float64); linear,nonlinear=damp.components(nu)
    assert torch.dot(nu,linear+nonlinear).item()<=1e-9 and linear[1].item()!=0 and linear[5].item()!=0
