#!/usr/bin/env python3
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
from hardware_manifest import manifest
p=argparse.ArgumentParser();p.add_argument('--source',default='paper_results/linux_gpu');p.add_argument('--output');a=p.parse_args()
src=Path(a.source);out=Path(a.output or src/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')));out.mkdir(parents=True,exist_ok=False)
index=[]
for f in src.rglob('*'):
 if not f.is_file() or out in f.parents:continue
 rel=f.relative_to(src)
 if f.suffix in ('.pt','.pth','.ckpt'):
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
  index.append({'path':str(f.resolve()),'sha256':h.hexdigest(),'bytes':f.stat().st_size})
 else:
  target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
(out/'checkpoint_index.json').write_text(json.dumps(index,indent=2)+'\n')
preflight=src/'preflight.json'
if preflight.is_file():
 hardware=json.loads(preflight.read_text())['hardware']
 hardware_source=str(preflight)
else:
 hardware=manifest()
 hardware_source='packaging_host'
(out/'hardware.json').write_text(json.dumps(hardware,indent=2,default=str)+'\n')
(out/'manifest.json').write_text(json.dumps({'source':str(src.resolve()),'hardware_source':hardware_source,'checkpoints_indexed':len(index),'checkpoints_copied':0},indent=2)+'\n')
print(out)
