import math
import numpy as np
import pytest
from bcod_sim.vessel_generation.revision4_resistance import curve, eligible, _holtrop_wave


def descriptors(**update):
    d={'lwl_m':7.3,'beam_m':1.0,'draft_m':.34,'volume_m3':1.0,
       'wetted_area_m2':6.7,'max_section_area_m2':.32,'waterplane_area_m2':5.2,
       'lcb_from_midship_m':.04,'hull_count':1,'catamaran':False,'planing_candidate':False}
    d.update(update);return d


def test_holtrop_eligibility_is_displacement_monohull_gated():
    assert eligible(descriptors())[0]
    assert not eligible(descriptors(hull_count=2,catamaran=True))[0]
    assert not eligible(descriptors(lwl_m=2.0))[0]
    assert not eligible(descriptors(waterplane_area_m2=0))[0]


def test_holtrop_components_are_finite_nonnegative_and_signed_odd():
    d=descriptors();u=np.array([-2.2,0.,2.2])
    r=curve(u,geometry=d,rho=1025.,nu=1.19e-6)
    assert r['applicable'] and r['provider']=='holtrop_mennen_1982_84_bare_hull_v1'
    assert np.allclose(r['force_x_n'],[r['components'][2]['total_n'],0.,-r['components'][0]['total_n']])
    c=r['components'][2]
    assert all(math.isfinite(c[k]) and c[k]>=0 for k in ('friction_n','form_n','wave_residual_n','total_n'))
    assert c['total_n']==c['friction_n']+c['form_n']+c['wave_residual_n']
    assert c['coefficients']['Re']==pytest.approx(2.2*7.3/1.19e-6)


def test_holtrop_inapplicability_returns_reason_not_residual_force():
    r=curve([1.0],geometry=descriptors(hull_count=2,catamaran=True),rho=1025.,nu=1.19e-6)
    assert r['applicable'] is False and 'requires_single_hull' in r['reason_codes']
