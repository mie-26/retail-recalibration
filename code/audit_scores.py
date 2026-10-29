"""Recompute archived summaries and the original 3,000-draw reference intervals.
Run: python code/audit_scores.py
No fitting, parameter selection, new random seeds or missing-target imputation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['scaled_pinball','weak','strict','equality','zero_tie']

def stats(d):
    g=d.groupby(['origin','quantile'],as_index=False)[['n']+FIELDS].mean().sort_values(['origin','quantile'])
    o=np.sort(g.origin.unique());assert len(g)==2*len(o)
    n=g.n.to_numpy().reshape(-1,2);np.testing.assert_array_equal(n[:,0],n[:,1])
    return o,n[:,0],g.scaled_pinball.to_numpy().reshape(-1,2).mean(1)

def boot_index(T,L):
    rng=np.random.default_rng(20261021)
    start=rng.integers(0,T,(3000,(T+L-1)//L))
    return ((start[:,:,None]+np.arange(L))%T).reshape(3000,-1)[:,:T]

def contrast(a,b,r):
    o,n,av=stats(a);oo,nn,bv=stats(b)
    np.testing.assert_array_equal(o,oo);np.testing.assert_array_equal(n,nn)
    av=np.nan_to_num(av)*n;bv=np.nan_to_num(bv)*n
    point=100*(av.sum()/bv.sum()-1)
    ix=boot_index(len(o),2*int(r.H));bounds=np.quantile(100*(av[ix].sum(1)/bv[ix].sum(1)-1),[.025,.975])
    np.testing.assert_allclose([point,*bounds],[r.relative_loss_pct,r.lower,r.upper],atol=1e-9,rtol=1e-9)

def summaries(d,s,keys):
    ex=s.set_index(keys);count=0;worst=0.
    for k,g in d.groupby(keys):
        r=ex.loc[k];assert g.n.sum()==r.n
        for f in FIELDS:
            value=np.nansum(g[f]*g.n)/g.n.sum();err=abs(value-r[f]);assert err<1e-9
            worst=max(worst,err)
        count+=1
    return count,worst

def main():
    root=ROOT;out=root/'reproduced/checks';out.mkdir(parents=True,exist_ok=True)
    core=pd.read_parquet(root/'score_data/enhanced_daily.parquet')
    assert core['mode'].eq('core').all()
    saved=pd.read_csv(root/'score_data/enhanced_quantile_scores.csv').query("mode == 'core'")
    keys=['case','mode','method','delay','window_kind','window','quantile']
    count,worst=summaries(core,saved,keys)
    expected=saved.set_index(keys)
    for k,g in core.groupby(keys):
        np.testing.assert_allclose(np.nansum(g.pinball*g.n)/g.n.sum(),expected.loc[k,'pinball'],rtol=1e-10,atol=1e-10)
    counts={}
    for name,filename in [('all_methods','Data1_core_effects.csv'),('primary','Data1_primary_contrasts.csv')]:
        table=pd.read_csv(root/'supplementary_data'/filename)
        for r in table.itertuples():
            sub=core[(core.source==r.source)&(core.family==r.family)&(core.H==r.H)&(core.U==r.U)]
            contrast(sub[sub.method==r.candidate],sub[sub.method==r.reference],r)
        counts[name+'_reference_intervals']=len(table)
    d=pd.read_csv(root/'supplementary_data/Data4_decomposition.csv')
    np.testing.assert_allclose(d.miscalibration-d.discrimination+d.uncertainty,d.score,atol=1e-12)
    result=dict(passed=True,core_daily_rows=len(core),core_summary_rows=count,**counts,maximum_summary_error=worst,new_model_fits=0,scope='Core daily aggregation, original-unit summaries and 3000-draw reference-block loss intervals. Training-budget and non-core diagnostic summaries are retained but not reaggregated.')
    (out/'score_recomputation.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
