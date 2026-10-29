"""Release audit of saved Italian daily scores; no fitting or reselection.

Run beside unchanged common.py. The release wrapper supplies portable manifest
resolution. Intervals condition on fitted/selected paths, exactly as frozen.
"""
from pathlib import Path
import json,sys,functools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'italian/work/sr_external_italy_20261028'))
import numpy as np
import pandas as pd
from portable_review import portable_lock,manifest_path
import common
from recompute_review import aggregate,streams,blockweights

OUT=common.OUT;A=OUT/'analysis';DEST=OUT/'release_audit'
METRICS=['scaled_pinball','pinball','weak','strict','equality','zero_tie']
records=[]

def compare(name,got,keys,cols=None):
    expected=pd.read_csv(A/name)
    if name in ['scores.csv','seed_scores.csv','year_scores.csv']:
        expected=expected.merge(got[keys].drop_duplicates(),on=keys,validate='one_to_one')
    got=got.sort_values(keys).reset_index(drop=True);expected=expected.sort_values(keys).reset_index(drop=True)
    assert len(got)==len(expected),(name,len(got),len(expected))
    pd.testing.assert_frame_equal(got[keys],expected[keys],check_dtype=False)
    cols=cols or [c for c in got.columns if c not in keys]
    np.testing.assert_allclose(got[cols].to_numpy(float),expected[cols].to_numpy(float),rtol=1e-9,atol=1e-10,equal_nan=True)
    delta=np.abs(got[cols].to_numpy(float)-expected[cols].to_numpy(float))
    records.append(dict(file=name,rows=len(got),max_error=float(np.nanmax(delta))))
    got.to_csv(DEST/name,index=False)
    print('PASS',name,len(got),flush=True)

@functools.lru_cache(maxsize=4)
def weights(H):return blockweights(common.LAST-common.FIRST+1,int(H))

def main():
    DEST.mkdir(exist_ok=True)
    digest=portable_lock()
    # These are path-semantics tests; they are not a native Linux execution.
    p=manifest_path(Path('/safe'),r'selection\h7_lgbm\selected.json')
    assert p.parts[-3:]==('selection','h7_lgbm','selected.json')
    for bad in ['../secret','/absolute','C:\\absolute','a/../../x']:
        try:manifest_path(Path('/safe'),bad)
        except ValueError:pass
        else:raise AssertionError(bad)
    df=pd.read_parquet(OUT/'selected/daily.parquet')
    keys=['H','U','family','experiment','selection_length','method','quantile']
    compare('scores.csv',aggregate(df,keys),keys,METRICS)
    compare('seed_scores.csv',aggregate(df,keys+['seed']),keys+['seed'],METRICS)
    years=df.assign(year=pd.to_datetime(df.date).dt.year)
    compare('year_scores.csv',aggregate(years,keys+['year']),keys+['year'],METRICS)
    # Fixed-candidate daily arrays are excluded; retain their recorded summaries and test arithmetic only.
    fs=pd.read_csv(A/'fixed_grid_scores.csv')
    e3=pd.concat([pd.read_parquet(OUT/'e3'/c['case']/'daily.parquet') for c in common.configs()],ignore_index=True)
    ek=['H','U','family','method','quantile']
    compare('E3_scores.csv',aggregate(e3,ek),ek,METRICS)
    compare('E3_phase_scores.csv',aggregate(e3,ek+['phase']),ek+['phase'],METRICS)
    interactions=[];effects=[];offsets=[]
    for (H,U,fam,tau),g in df.groupby(['H','U','family','quantile']):
        main=g[(g.selection_length==84)|g.method.isin(['Raw','R2_reuse','R3_isolate'])]
        nums,n=streams(main,['Raw','QT_group','QT_series','WQ_group','WQ_series','R2_reuse','R3_isolate'])
        means={m:v.sum()/n.sum() for m,v in nums.items()}
        point=100*((means['QT_group']-means['QT_series'])-(means['WQ_group']-means['WQ_series']))/means['Raw']
        for block in sorted(set([H,2*H,4*H,56])):
            w=weights(block);den=w@n;loss={m:(w@v)/den for m,v in nums.items()}
            b=100*((loss['QT_group']-loss['QT_series'])-(loss['WQ_group']-loss['WQ_series']))/loss['Raw']
            q=np.quantile(b,[.0125,.025,.975,.9875])
            interactions.append(dict(H=H,U=U,family=fam,quantile=tau,block=int(block),interaction_percent=point,lower97_5=q[0],lower95=q[1],upper95=q[2],upper97_5=q[3]))
            for c,r in [('QT_group','QT_series'),('WQ_group','WQ_series'),('WQ_group','Raw'),('QT_group','Raw'),('R2_reuse','Raw'),('R3_isolate','Raw'),('R2_reuse','R3_isolate')]:
                lo,hi=np.quantile(100*(loss[c]/loss[r]-1),[.025,.975])
                effects.append(dict(H=H,U=U,family=fam,quantile=tau,block=int(block),candidate=c,reference=r,relative_loss_percent=100*(means[c]/means[r]-1),lower95=lo,upper95=hi))
        for offset in range(int(H)):
            ix=np.arange(offset,len(n),int(H));mm={m:v[ix].sum()/n[ix].sum() for m,v in nums.items()}
            offsets.append(dict(H=H,U=U,family=fam,quantile=tau,offset=offset,n=float(n[ix].sum()),interaction_percent=100*((mm['QT_group']-mm['QT_series'])-(mm['WQ_group']-mm['WQ_series']))/mm['Raw']))
    sk=['H','U','family','quantile'];compare('interaction_intervals.csv',pd.DataFrame(interactions),sk+['block'])
    compare('effect_intervals.csv',pd.DataFrame(effects),sk+['block','candidate','reference'])
    compare('nonoverlap_interactions.csv',pd.DataFrame(offsets),sk+['offset'])
    contrasts=[]
    for (H,U,fam,tau),g in e3.groupby(sk):
        nums,n=streams(g,['Raw','old_reference','new_reference']);w=weights(2*H);den=w@n;v={m:(w@a)/den for m,a in nums.items()}
        for c,r in [('new_reference','old_reference'),('new_reference','Raw'),('old_reference','Raw')]:
            lo,hi=np.quantile(100*(v[c]/v[r]-1),[.025,.975])
            contrasts.append(dict(H=H,U=U,family=fam,quantile=tau,candidate=c,reference=r,block=int(2*H),relative_loss_percent=100*(nums[c].sum()/nums[r].sum()-1),lower95=lo,upper95=hi))
    compare('E3_intervals.csv',pd.DataFrame(contrasts),sk+['candidate','reference'])
    # Explicitly verify the structural identity rather than count two replications.
    ik=['H','U','family','seed','origin','quantile'];cols=['n']+METRICS
    left=e3[e3.method=='new_reference'][ik+cols].sort_values(ik).reset_index(drop=True)
    right=df[df.method=='R3_isolate'][ik+cols].sort_values(ik).reset_index(drop=True)
    pd.testing.assert_frame_equal(left,right,check_dtype=False,check_exact=True)
    assert len(left)==34416
    # Post-hoc fixed-grid arithmetic, with no reselection and no new primary test.
    fr=[]
    for (H,U,f,q),g in fs.groupby(sk):
        def loss(method,parameter):return g[(g.method==method)&(g.parameter==parameter)].scaled_pinball.iloc[0]
        raw=loss('Raw',0)
        for window in [28,56,112]:
            for step in [.03,.1,.3]:
                val=100*((loss('QT_group',step)-loss('QT_series',step))-(loss('WQ_group',window)-loss('WQ_series',window)))/raw
                fr.append(dict(H=H,U=U,family=f,quantile=q,window=window,step=step,interaction_percent=val))
    fr=pd.DataFrame(fr);assert len(fr)==144 and (fr.interaction_percent<0).sum()==26
    assert set(fr.loc[fr.interaction_percent<0,'step'])=={.3}
    expected=pd.read_csv(OUT/'review_checks/fixed_parameter_interactions_posthoc.csv')
    kk=sk+['window','step'];pd.testing.assert_frame_equal(fr.sort_values(kk).reset_index(drop=True),expected.sort_values(kk).reset_index(drop=True)[fr.columns],check_dtype=False,check_exact=False,atol=1e-9,rtol=1e-9)
    fr.to_csv(DEST/'fixed_parameter_interactions_posthoc.csv',index=False)
    feedback=[]
    for c in common.configs():
        g=pd.read_parquet(OUT/'candidates'/c['case']/'feedback.parquet')
        g=g[(g.origin>=common.FIRST)&g.method.isin(['R2_reuse','R3_isolate'])]
        for m,f in g.groupby('method'):
            feedback.append(dict(**c,method=m,fallback_days=int(f.fallback_all.sum()),days=len(f),mean_valid_residual_targets=f.valid_residual_targets.mean(),median_active_series=f.active_series.median()))
    compare('feedback_availability.csv',pd.DataFrame(feedback),['H','U','family','seed','case','method'])
    # Confirm scientific files still match after audit.
    assert portable_lock()==digest
    result=dict(passed=True,freeze_sha256=digest,checks=records,selected_daily_rows=len(df),fixed_daily_rows_included=0,fixed_grid_verification='summary arithmetic only; no candidate daily aggregation',E3_daily_rows=len(e3),identity_rows=len(left),identity_max_error=0,fixed_parameter_cells=144,negative_fixed_interactions=26,new_model_fits=0,native_linux_executed=False,scope='Compact subset audit: selected E2 group/series, Raw, reuse/isolation and E3 daily scores; 5000-draw intervals at all saved block lengths. Fixed-grid results checked from saved summaries only; E1/global/sorted daily scores omitted. No model-bank reconstruction.')
    (DEST/'AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('ITALIAN_CORE_AUDIT_PASS',flush=True)

if __name__=='__main__':main()
