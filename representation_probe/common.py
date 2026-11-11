"""Shared provenance and paths; importing never writes into upstream assets."""
import hashlib, json, os
from pathlib import Path
R = Path('/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/point-MAE/runs')
X = R/'mito_newmae_vicreg_reg02_cls005_20261105_01'
F = R/'mito_e16_reinfer_fillholes_lt200_pointmae_20261105_01'
D = R/'mito_embedding_dashboard_20261030_01'
CACHE = D/'cache/features_b9a9cbbcd15a81c16c36'
SEED = 20261111
def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)
def key(r): return json.dumps(r['key'],ensure_ascii=False,separators=(',',':'))
def load_rows(): return json.loads((X/'cohort.json').read_text())
def stat(p):
    s=Path(p).stat()
    return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
