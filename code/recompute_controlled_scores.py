"""Standalone review: python code/recompute_from_scores.py --root ."""
from pathlib import Path
import argparse,json,hashlib,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
def main(root,intervals=False):
    root=Path(root);manifest=root/'MANIFEST.json';files=0
    if manifest.exists():
        for r in json.loads(manifest.read_text(encoding='utf8'))['files']:
            p=root/r['path'];assert p.stat().st_size==r['bytes']
            with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
            files+=1
    effects=pd.read_csv(root/'tables/effects.csv');checked=0;summary_count=0;maxerror=0.;ci_checked=0
    ci=pd.read_csv(root/'tables/intervals.csv') if intervals else None
    weights={}
    fields=['source','H','U','family'];metrics=['scaled_pinball','pinball','weak','strict','equality','zero_tie']
    for kind in ['E1','E2','E3']:
        d=pd.read_parquet(root/kind/'daily.parquet');s=pd.read_csv(root/'tables'/f'{kind}_quantile_scores.csv')
        groups={k:g for k,g in d.groupby(fields+['method','quantile'])}
        for r in s.itertuples():
            g=groups[(r.source,r.H,r.U,r.family,r.method,r.quantile)]
            for m in metrics:
                value=np.nansum(g[m]*g.n)/g.n.sum();err=abs(value-getattr(r,m));maxerror=max(maxerror,err);assert err<1e-11
            summary_count+=1
        groups={k:g for k,g in d.groupby(fields+['method'])}
        for r in effects[effects.experiment==kind].itertuples():
            aa=groups[(r.source,r.H,r.U,r.family,r.candidate)].copy();bb=groups[(r.source,r.H,r.U,r.family,r.reference)].copy()
            if r.scope=='mature_phase':aa.loc[~aa.active_phase,'n']=0;bb.loc[~bb.active_phase,'n']=0
            av=np.nansum(aa.scaled_pinball*aa.n)/aa.n.sum();bv=np.nansum(bb.scaled_pinball*bb.n)/bb.n.sum();value=100*(av/bv-1)
            err=abs(value-r.relative_loss_pct);maxerror=max(maxerror,err);assert err<1e-9,(r,value)
            for tau,name in [(.9,'q90_weak_difference'),(.95,'q95_weak_difference')]:
                a=aa[aa['quantile']==tau];b=bb[bb['quantile']==tau];v=100*(np.nansum(a.weak*a.n)/a.n.sum()-np.nansum(b.weak*b.n)/b.n.sum());assert abs(v-getattr(r,name))<1e-9
            if intervals:
                c=ci[(ci.experiment==kind)&(ci.source==r.source)&(ci.H==r.H)&(ci.U==r.U)&(ci.family==r.family)&(ci.candidate==r.candidate)&(ci.reference==r.reference)&(ci.scope==r.scope)&(ci.metric=='relative_loss_pct')&ci.reference_block]
                if len(c):
                    assert len(c)==1
                    an=(aa.scaled_pinball*aa.n).fillna(0).groupby(aa.origin).sum().sort_index().to_numpy();bn=(bb.scaled_pinball*bb.n).fillna(0).groupby(bb.origin).sum().sort_index().to_numpy();T=len(an);L=int(c.block_length.iloc[0])
                    if (T,L) not in weights:
                        rng=np.random.default_rng(20261021);starts=rng.integers(0,T,(3000,(T+L-1)//L));idx=((starts[:,:,None]+np.arange(L))%T).reshape(3000,-1)[:,:T];w=np.zeros((3000,T))
                        for j,row in enumerate(idx):w[j]=np.bincount(row,minlength=T)
                        weights[T,L]=w
                    v=100*((weights[T,L]@an)/(weights[T,L]@bn)-1);bounds=np.quantile(v,[.025,.975]);np.testing.assert_allclose(bounds,c[['lower','upper']].iloc[0].to_numpy(),atol=1e-8,rtol=0);ci_checked+=1
            checked+=1
    print(json.dumps(dict(status='PASS',manifest_files=files,summary_rows=summary_count,paired_effects=checked,primary_joint_loss_intervals=ci_checked,max_absolute_recomputation_error=maxerror,scope='Score-level estimates and optional primary loss intervals; no base fits or individual-forecast reconstruction'),indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--intervals',action='store_true');a=ap.parse_args();main(a.root,a.intervals)
