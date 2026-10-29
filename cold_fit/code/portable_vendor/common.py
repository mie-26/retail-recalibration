import sys,json,hashlib,datetime,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/sr_enhanced_20261021'; DATA=HERE/'data'
SEEDS=[20261021,20261022,20261023]; TAUS=np.array([.9,.95])
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    s=json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
    p.with_suffix(p.suffix+'.tmp').write_text(s,encoding='utf8');p.with_suffix(p.suffix+'.tmp').replace(p)
def csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check_lock():
    plan=read(OUT/'freeze_manifest.json');assert plan['status']=='FROZEN'
    hashes=dict(plan['code_sha256'])
    for p in sorted(OUT.glob('amendment_*.json')):
        amendment=read(p);assert amendment['original_freeze_sha256']==sha(OUT/'freeze_manifest.json')
        hashes.update(amendment['replacement_code_sha256'])
    for name,digest in hashes.items():assert sha(HERE/name)==digest,('frozen code changed',name)
    for name,item in plan['selected'].items():assert sha(OUT/name)==item['sha256']
    return plan
def first_issued(source,H,family):
    s=read(OUT/'selection'/f'{source}_h{H}_{family}/selected.json')
    return int(s['validation_last_target'])+1
def torch_import():
    import torch
    torch.set_num_threads(4)
    return torch
def stable_sample(keys,labels,n_each):
    ix=[]
    for g in np.unique(labels):
        pool=np.flatnonzero(labels==g)
        order=sorted(pool,key=lambda i:hashlib.sha256(('enhanced20261021:'+str(keys[i])).encode()).digest())
        ix.extend(order[:n_each])
    return np.sort(ix)
def targets(y,origins,H):
    # t means t completed units; forecast y[t:t+H]. Missing target days remain missing.
    return np.array([np.sum(y[:,int(t):int(t)+H],axis=1) for t in origins],dtype='float32')

