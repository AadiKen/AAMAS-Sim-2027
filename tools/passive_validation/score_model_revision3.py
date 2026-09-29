from pathlib import Path
import sys,json,hashlib,math,csv,copy
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from bcod_sim.vessel_generation.coefficient_package import load_coefficient_package,validate_coefficients,_runtime_plant
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.state.vessel_state import VesselState
import torch
root=REPO/'docs/passive_hull_validation/benchmarks/kvlcc2m/model_revision_3'; bench=root.parent
manifest_path=root/'package_manifest.json'; mf=json.loads(manifest_path.read_text())
p=root/'package/coefficient_package.yaml'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(p)==mf['package_file_sha256']
for k,v in mf['generator_source_hashes_sha256'].items(): assert sha(REPO/k)==v
pkg=load_coefficient_package(p); assert pkg['canonical_sha256']==mf['canonical_package_sha256']
roundtrip=validate_coefficients(p,grid_size=7); assert roundtrip['passed']; (root/'results/runtime_roundtrip.json').write_text(json.dumps(roundtrip,indent=2)+'\n')
# freeze already exists before experimental reference reads below
mf['scoring_started']=True
(root/'package_manifest.json').write_text(json.dumps(mf,indent=2)+'\n')

def ledger(package,nu):
    plant=_runtime_plant(package); t=lambda x:torch.as_tensor(x,dtype=torch.float64)
    st=VesselState(t([0.,0.,0.]),t([1.,0.,0.,0.]),t(nu)); zeros={n:t([0.]*6) for n in EXTERNAL_TERMS}
    d=plant.diagnostics(st,zeros)
    return {k:v.detach().cpu().numpy() for k,v in d.terms.items()}
terms=('added_mass_coriolis','linear_damping','nonlinear_damping','crossflow')
# experiment files are opened only after freeze + runtime gate
ref=json.loads((bench/'reference/admitted_reference.json').read_text())
resref=json.loads((bench/'reference/resistance.json').read_text())
# source hashes recorded in manifest
mf['score_references_sha256']={q.name:sha(q) for q in (bench/'reference/admitted_reference.json',bench/'reference/resistance.json')}
(root/'package_manifest.json').write_text(json.dumps(mf,indent=2)+'\n')
RHO,U,L,T,AREA=1025.,.994,4.97,.323,6.58919
fs=.5*RHO*U**2*L*T; ms=fs*L
rows=[]; comps=[]
for item in ref['rows']:
    beta=float(item['beta_deg']); b=math.radians(beta); nu=np.array([U*math.cos(b),-U*math.sin(b),0,0,0,0.])
    ll=ledger(pkg,nu); total=sum((ll[k] for k in terms if k in ll),np.zeros(6))
    r={'beta_deg':beta,'experimental_CX':item['CX'],'manta_CX':total[0]/fs,'experimental_CY':item['CY'],'manta_CY':total[1]/fs,'experimental_CN':item['CN'],'manta_CN':total[5]/ms,'CY_uncertainty':item.get('CY_uncertainty'),'CN_uncertainty':item.get('CN_uncertainty')}
    for k in ('CX','CY','CN'):
        ev=r['experimental_'+k]; mv=r['manta_'+k]; r[k+'_abs_error']=abs(mv-ev); r[k+'_relative_error_percent']=None if abs(ev)<1e-3 else abs(mv-ev)/abs(ev)*100
    rows.append(r)
    cr={'beta_deg':beta}
    for nm,v in ll.items():
        cr[nm+'_X']=v[0]/fs; cr[nm+'_Y']=v[1]/fs; cr[nm+'_N']=v[5]/ms
    comps.append(cr)
def write_csv(path,data):
    fields=list(data[0]);
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
write_csv(root/'results/static_drift.csv',rows);write_csv(root/'results/force_components.csv',comps)
# resistance primary fixed tow, runtime interpolated at exact U
rp=pkg['runtime_payload']['surge_resistance']; pred=-float(np.interp(U,rp['speed_mps'],rp['force_x_n']))
ct=pred/(.5*RHO*U**2*AREA); efdct=float(resref['experimental']['CT']); efdR=efdct*.5*RHO*U**2*AREA
cnear=min((q for q in rp['components'] if 'reynolds_number' in q),key=lambda q:abs(q['reynolds_number']-U*L/1.2522636248415715e-6))
rb={'speed_mps':U,'manta_total_N':pred,'experimental_total_N':efdR,'predicted_CT':ct,'experimental_CT':efdct,'absolute_CT_error':abs(ct-efdct),'relative_error_percent':abs(ct-efdct)/efdct*100,'experimental_uncertainty_percent':resref['official_condition']['experimental_uncertainty_percent_of_CT'],'friction_N':pred,'form_factor':1.,'form_increment_N':0.,'wave_N':0.,'other_corrections_N':0.,'components_at_speed':cnear,'provider':rp['provider'],'reference_source_url':resref['source_url']}
(root/'results/resistance_breakdown.json').write_text(json.dumps(rb,indent=2)+'\n'); (root/'results/resistance.json').write_text(json.dumps(rb,indent=2)+'\n')
# Baseline package should come from frozen parent generated package if present? Preserve copied scored baseline's full evaluator by reconstructing terms from saved CSVs instead; available baseline contribution table in parent benchmark results.
base_csv=bench/'results/force_components.csv'
if not base_csv.exists(): raise FileNotFoundError('baseline contribution ledger missing; cannot claim exact attribution')
base=list(csv.DictReader(base_csv.open())); bidx={float(x['beta_deg']):x for x in base}
curidx={float(x['beta_deg']):x for x in comps}
# expected baseline header is inspected dynamically
print('baseline fields',list(base[0]))
# sensitivity only low-confidence strip selected matrix
mass=np.asarray(pkg['added_mass']['matrix_6x6'],float); sens=[]
for scale in (.5,1.,1.5):
    variant=copy.deepcopy(pkg);variant['runtime_payload']['added_mass_kg']=(mass*scale).tolist()
    for item in ref['rows']:
        beta=float(item['beta_deg']);b=math.radians(beta);nu=np.array([U*math.cos(b),-U*math.sin(b),0,0,0,0.])
        ll=ledger(variant,nu); tot=sum((ll[k] for k in terms if k in ll),np.zeros(6))
        sens.append({'added_mass_scale':scale,'beta_deg':beta,'CX':tot[0]/fs,'CY':tot[1]/fs,'CN':tot[5]/ms,'added_mass_coriolis_CN':ll['added_mass_coriolis'][5]/ms})
write_csv(root/'results/added_mass_sensitivity.csv',sens)
# summary and diagnostic table
summary={'package_canonical_sha256':pkg['canonical_sha256'],'runtime_roundtrip':roundtrip,'drift_points':len(rows),'resistance':rb,'model_revision':3,'added_mass_scales':[.5,1.,1.5]}
(root/'results/score_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
header='| beta | EFD CX | MANTA CX | abs err | rel err | EFD CY | MANTA CY | abs err | rel err | EFD CN | MANTA CN | abs err | rel err |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
lines=['# KVLCC2M revision 3 — complete 14 point table','',header.rstrip()]
for r in rows:
    vals=[r['beta_deg'],r['experimental_CX'],r['manta_CX'],r['CX_abs_error'],r['CX_relative_error_percent'],r['experimental_CY'],r['manta_CY'],r['CY_abs_error'],r['CY_relative_error_percent'],r['experimental_CN'],r['manta_CN'],r['CN_abs_error'],r['CN_relative_error_percent']]
    lines.append('| '+' | '.join(f'{v:.6g}' if isinstance(v,(int,float)) else '—' for v in vals)+' |')
(root/'results/diagnostic_14_point_table.md').write_text('\n'.join(lines)+'\n')
# Compare the new runtime terms with the previously frozen baseline ledger.
base_rows=list(csv.DictReader((bench/'results/force_components.csv').open()))
base_by_beta={float(x['beta_deg']):x for x in base_rows}
new_by_beta={float(x['beta_deg']):x for x in comps}
new_totals={float(x['beta_deg']):x for x in rows}
comparison=[]; ablations=[]
for beta in sorted(base_by_beta):
    old,cur,tot=base_by_beta[beta],new_by_beta[beta],new_totals[beta]
    row={'beta_deg':beta}
    for key,axis in zip(('CX','CY','CN'),('X','Y','N')):
        row[f'baseline_{key}']=float(old[f'manta_{key}'])
        row[f'revision3_{key}']=float(tot[f'manta_{key}'])
        row[f'delta_{key}']=row[f'revision3_{key}']-row[f'baseline_{key}']
        for term in ('nonlinear_damping','linear_damping','crossflow','added_mass_coriolis'):
            old_value=float(old.get(f'{term}_{key}',0) or 0)
            new_value=float(cur.get(f'{term}_{axis}',0) or 0)
            row[f'{term}_delta_{key}']=new_value-old_value
            ablations.append({'beta_deg':beta,'component':term,'coefficient':key,
                              'baseline':old_value,'revision3':new_value,'delta':new_value-old_value})
    comparison.append(row)
def write_rows(path,data):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
write_rows(root/'results/before_after_14_point.csv',comparison)
write_rows(root/'results/component_ablation.csv',ablations)
print(json.dumps(summary,indent=2))
