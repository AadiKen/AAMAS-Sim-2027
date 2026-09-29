"""Package final Surveyor v3 outputs with per-file SHA-256 provenance."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair'


def digest(path:Path)->str:
    result=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def run()->None:
    reproducibility=BASE/'reproducibility'
    reproducibility.mkdir(exist_ok=True)
    shutil.copy2(__file__,reproducibility/Path(__file__).name)
    files=[]
    for name in ('cad','meshes','frozen','coefficients','figures','reproducibility'):
        files.extend(path for path in (BASE/name).rglob('*') if path.is_file())
    files.extend(BASE/'provenance'/name for name in
                 ('input_snapshot.json','aft_source_contours_frd_mm.json'))
    files.extend((BASE/'trials').glob('*/rejection.json'))
    files.extend(path for path in BASE.iterdir() if path.is_file() and
                 path.suffix in ('.json','.md') and path.name!='final_artifact_manifest.json')
    files=sorted(set(files),key=lambda path:str(path.relative_to(BASE)))
    manifest={'schema':'surveyor-v3-final-artifact-package-1',
              'label':'SeaRobotics Surveyor-derived CAD — geometry-repair fallback',
              'excluded_preserved_items':[
                  'original provided STEP in user Downloads',
                  'pre_fallback_targeted_repair handoff ZIP and v2 exports in provenance/workspace',
                  'full rejected trial CAD and coefficients in trials/ workspace',
                  'independent nominal determinism rebuild in workspace'],
              'files':[{'path':str(path.relative_to(BASE)),'size_bytes':path.stat().st_size,
                        'sha256':digest(path)} for path in files]}
    manifest_path=BASE/'final_artifact_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(manifest_path)
    package=BASE/'Surveyor_v3_final_package.zip'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6,allowZip64=True) as archive:
        for index,path in enumerate(files,1):
            archive.write(path,'surveyor_v3/'+str(path.relative_to(BASE)))
            if index%50==0:
                print('zipped',index,'/',len(files),flush=True)
    (BASE/'final_package_sha256.txt').write_text(digest(package)+'  '+package.name+'\n')
    print('package',package,'bytes',package.stat().st_size,'sha256',digest(package),
          'files',len(files),flush=True)


if __name__=='__main__':run()
