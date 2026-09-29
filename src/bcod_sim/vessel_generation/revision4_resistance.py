"""Holtrop–Mennen (1982/84) bare-hull resistance, with an explicit regime gate.

Method source: Holtrop & Mennen, International Shipbuilding Progress 29 (335),
166–170 (1982), DOI 10.3233/ISP-1982-2933501; Holtrop, ISP 31 (363), 272–276
(1984). This is an empirical model for conventional displacement monohulls.
Unsupported bulb, appendage, transom and model-ship correlation terms are omitted.
"""
from __future__ import annotations
import math
import numpy as np

G=9.80665

def eligible(d: dict) -> tuple[bool,list[str]]:
    reasons=[]
    required=('lwl_m','beam_m','draft_m','volume_m3','wetted_area_m2','max_section_area_m2','waterplane_area_m2','lcb_from_midship_m')
    for key in required:
        if key not in d or not math.isfinite(float(d[key])) or float(d[key])<=0 and key not in ('lcb_from_midship_m',):reasons.append('missing_or_invalid_'+key)
    if reasons:return False,reasons
    L,B,T,V,S,Amax,Aw=(float(d[k]) for k in ('lwl_m','beam_m','draft_m','volume_m3','wetted_area_m2','max_section_area_m2','waterplane_area_m2'))
    Cb=V/(L*B*T);Cm=Amax/(B*T);Cp=Cb/Cm;Cwp=Aw/(L*B)
    if d.get('hull_count')!=1:reasons.append('requires_single_hull')
    if not 3.5<=L/B<=9.5:reasons.append('Lwl_over_B_outside_3.5_9.5')
    if not .5<=Cm<=1:reasons.append('Cm_outside_0.5_1.0')
    if not .4<=Cp<=.93:reasons.append('Cp_outside_0.40_0.93')
    if abs(float(d['lcb_from_midship_m']))/L>.05:reasons.append('LCB_outside_5_percent_Lwl')
    if not 0<Cwp<=1.05:reasons.append('Cwp_invalid')
    if d.get('planing_candidate') or d.get('catamaran'):reasons.append('unsupported_hull_class')
    return not reasons,reasons

def _waterline_entrance_deg(L,B,Cwp,Cp,lcb,LR,volume):
    # Holtrop's principal-particular estimate for the bow entrance angle iE.
    base=1-Cp-0.0225*lcb
    den=(LR/B)**0.34574*(100*volume/L**3)**0.16302
    if den<=0 or base<=0:raise ValueError('Invalid entrance-angle descriptor combination')
    exponent=-(L/B)**0.80856*(1-Cwp)**0.30484*base**0.6367/den
    return 1+89*math.exp(exponent)

def _holtrop_wave(speed,d,rho,nu):
    L,B,T,V,S,Amax,Aw=(float(d[k]) for k in ('lwl_m','beam_m','draft_m','volume_m3','wetted_area_m2','max_section_area_m2','waterplane_area_m2'))
    Cb=V/(L*B*T);Cm=Amax/(B*T);Cp=Cb/Cm;Cwp=Aw/(L*B);lcb=100*float(d['lcb_from_midship_m'])/L
    lr=L*(1-Cp+.06*Cp*lcb/(4*Cp-1))
    if lr<=0:raise ValueError('Holtrop run length is nonpositive')
    fn=speed/math.sqrt(G*L);re=speed*L/nu
    cf=.075/(math.log10(re)-2)**2
    c12=(T/L)**.2228446 if T/L>.05 else (48.20*(T/L-.02)**2.078+.479948 if T/L>.02 else .479948)
    c13=1.0 # normal stern; stern form was not robustly classified from this mesh
    form= c13*(.93 + c12*(B/L)**.92497*(.95-Cp)**(-.521448)*(1-Cp+.0225*lcb)**.6906)
    c7=B/L if B/L<.11 else (.5-.0625*L/B if B/L>=.25 else B/L)
    ie=_waterline_entrance_deg(L,B,Cwp,Cp,lcb,lr,V)
    c1=2223105*c7**3.78613*(T/B)**1.07961*(90-ie)**-1.37565
    # No bulb or transom is assigned absent positive geometry evidence.
    c2=1.0;c5=1.0
    l3v=L**3/V
    c16=(8.07981*Cp-13.8673*Cp**2+6.984388*Cp**3) if Cp<.8 else (1.73014-.7067*Cp)
    m1=.0140407*L/T-1.75254*V**(1/3)/L-4.79323*B/L-c16
    c15=(-1.69385 if l3v<512 else (0. if l3v>1727 else -1.69385*(1727-l3v)/(1727-512)))
    m2=c15*Cp**2*math.exp(-.1/max(fn**2,1e-12))
    lam=1.446*Cp-.03*L/B if L/B<12 else 1.446*Cp-.36
    d_exp=-.9
    rw=c1*c2*c5*rho*G*V*math.exp(m1*fn**d_exp+m2*math.cos(lam/max(fn**2,1e-12)))
    rf=.5*rho*speed**2*S*cf
    rform=rf*(form-1)
    if not all(math.isfinite(x) and x>=0 for x in (rf,rform,rw)):
        raise ValueError('Holtrop resistance produced invalid component')
    return {'friction_n':rf,'form_n':rform,'wave_residual_n':rw,'other_supported_n':0.,'total_n':rf+rform+rw,
      'coefficients':{'Cb':Cb,'Cp':Cp,'Cm':Cm,'Cwp':Cwp,'lcb_percent_forward_midship':lcb,'Lr_m':lr,'iE_deg':ie,'Fn':fn,'Re':re,'1_plus_k1':form,'c1':c1,'c2':c2,'c5':c5,'m1':m1,'m2':m2,'lambda':lam,'d':d_exp},
      'omitted_terms':['bulb_pressure_no_robust_bulb_descriptor','transom_pressure_no_robust_transom_descriptor','appendage_resistance_no_appendage_inventory','model_ship_allowance_model_scale_bare_hull']}

def curve(speeds,*,geometry,rho,nu):
    ok,reasons=eligible(geometry)
    if not ok:return {'provider':'ittc57_friction_only','applicable':False,'reason_codes':reasons,'confidence':'low'}
    components=[];forces=[]
    for u in speeds:
        if abs(float(u))<1e-12:
            components.append({'friction_n':0.,'form_n':0.,'wave_residual_n':0.,'other_supported_n':0.,'total_n':0.});forces.append(0.);continue
        c=_holtrop_wave(abs(float(u)),geometry,rho,nu)
        components.append(c);forces.append(-math.copysign(c['total_n'],float(u)))
    return {'provider':'holtrop_mennen_1982_84_bare_hull_v1','method':'holtrop_mennen_1982_84_bare_hull_v1',
      'applicable':True,'reason_codes':['ELIGIBLE_CONVENTIONAL_DISPLACEMENT_MONOHULL'],'confidence':'medium',
      'speed_mps':[float(x) for x in speeds],'components':components,'force_x_n':forces,
      'limitations':'No bulb, transom, appendage or model-ship correlation terms included without validated geometry descriptors.',
      'source_doi':['10.3233/ISP-1982-2933501'],'method_limitations':'No bulb, transom, appendage or model-ship correlation terms included without validated geometry descriptors.'}
