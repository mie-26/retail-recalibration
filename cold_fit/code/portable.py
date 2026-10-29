"""Portable raw-data reconstruction and representative cold-fit audit.

Uses user-supplied licensed raw archives; no private workstation path is required.
All reference manifests/cohort IDs are declared, never used as fitted predictors.
"""
import os,sys,json,hashlib,time,argparse,zipfile,io,gc,shutil,importlib.metadata,platform
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf8')
def sample(ids,labels,n):
    picked=[]
    for g in np.unique(labels):
        pool=np.flatnonzero(labels==g)
        picked.extend(sorted(pool,key=lambda i:hashlib.sha256(('enhanced20261021:'+str(ids[i])).encode()).digest())[:n])
    return np.sort(picked)

def m5(raw):
    with zipfile.ZipFile(raw['m5']) as z:
        with z.open('sales_train_evaluation.csv') as f:a=pd.read_csv(f)
        a=a[a.store_id.isin(['CA_1','TX_1','WI_1'])].reset_index(drop=True)
        meta=a.iloc[:,:5].copy();meta['series_id']=meta.item_id+'__'+meta.store_id;y=a.iloc[:,5:].to_numpy(dtype='float32');del a
        with z.open('calendar.csv') as f:cal=pd.read_csv(f,nrows=1941)
        weeks=pd.Index(cal.wm_yr_wk.drop_duplicates());wi=weeks.get_indexer(cal.wm_yr_wk)
        pairs=pd.MultiIndex.from_frame(meta[['store_id','item_id']]);wp=np.full((len(meta),len(weeks)),np.nan,dtype='float32')
        with z.open('sell_prices.csv') as f:
            for block in pd.read_csv(f,chunksize=200000):
                block=block[block.store_id.isin(['CA_1','TX_1','WI_1'])&block.wm_yr_wk.isin(weeks)]
                ri=pairs.get_indexer(pd.MultiIndex.from_frame(block[['store_id','item_id']]));ci=weeks.get_indexer(block.wm_yr_wk)
                wp[ri,ci]=block.sell_price.to_numpy(dtype='float32')
        price=wp[:,np.maximum(wi-1,0)];price[:,wi==0]=np.nan
    c=1492;freq=(y[:,c-365:c]>0).mean(1)
    keep=(y[:,:c-365]>0).any(1)&((y[:,c-365:c]>0).sum(1)>=20)&np.isfinite(price[:,c-365:c]).all(1)
    meta=meta[keep].reset_index(drop=True);y=y[keep]
    ix=sample(meta.series_id.to_numpy(),meta.store_id.to_numpy(),1000)
    return y[ix],meta.iloc[ix].reset_index(drop=True),cal.date.to_numpy()

def retail(raw):
    with zipfile.ZipFile(raw['retail']) as z:
        with z.open('online_retail_II.xlsx') as f:
            sheets=pd.read_excel(f,sheet_name=None,engine='openpyxl')
    a,b=list(sheets.values());a=a[a.InvoiceDate.lt('2010-12-01')];d=pd.concat([a,b],ignore_index=True)
    d['StockCode']=d.StockCode.astype(str);d['day']=pd.to_datetime(d.InvoiceDate).dt.normalize()
    good=d.Country.eq('United Kingdom')&d.StockCode.str.fullmatch(r'\d{5}[A-Za-z]*')&d.Price.gt(0)&d.Quantity.gt(0)&~d.Invoice.astype(str).str.upper().str.startswith('C')
    dates=pd.date_range('2009-12-01','2011-12-08');p=d[good].groupby(['StockCode','day']).Quantity.sum().unstack().reindex(columns=dates).fillna(0)
    y=p.to_numpy(dtype='float32');keep=(y[:,:185]>0).any(1)&((y[:,185:365]>0).sum(1)>=20)
    meta=pd.DataFrame(dict(series_id=p.index[keep].astype(str),store_id='UK',cat_id='retail'))
    return y[keep],meta,dates.astype(str).to_numpy()

def dominicks(raw):
    yy=[];mm=[]
    for cat in ['cer','frd','lnd','tti']:
        with zipfile.ZipFile(raw[cat]) as z,z.open(f'w{cat}.csv') as f:
            a=pd.read_csv(f,usecols=['STORE','UPC','WEEK','MOVE','OK'],dtype={'STORE':str,'UPC':str})
        assert not a.duplicated(['STORE','UPC','WEEK']).any()
        history=a[a.WEEK<296];keys=list(dict.fromkeys(zip(history.STORE,history.UPC)));mapping={k:i for i,k in enumerate(keys)}
        y=np.full((len(keys),399),np.nan,dtype='float32')
        indices=np.array([mapping.get(k,-1) for k in zip(a.STORE,a.UPC)]);keep=(indices>=0)&(a.OK.to_numpy()==1)&np.isfinite(a.MOVE.to_numpy())&(a.MOVE.to_numpy()>=0)
        y[indices[keep],a.WEEK.to_numpy()[keep]-1]=a.MOVE.to_numpy()[keep].astype('float32')
        h=y[:,158:262];elig=(np.isfinite(h).sum(1)>=94)&((h>0).sum(1)>=20)&np.isfinite(y[:,254:262]).any(1)
        candidates=np.flatnonzero(elig);ids=np.array([f'{cat}_{k[0]}_{k[1]}' for k in keys]);stores=np.array([k[0] for k in keys]);buckets=[]
        for store in sorted(set(stores[candidates])):
            ix=candidates[stores[candidates]==store];buckets.append(sorted(ix,key=lambda i:hashlib.sha256(('enhanced20261021:'+ids[i]).encode()).digest()))
        picked=[]
        for rank in range(max(map(len,buckets))):
            for bucket in buckets:
                if rank<len(bucket):picked.append(bucket[rank])
        ix=np.sort(picked[:500]);yy.append(y[ix]);mm += [dict(series_id=ids[i],store_id=stores[i],cat_id=cat,upc=keys[i][1]) for i in ix]
        print('RECONSTRUCT',cat,len(ix),flush=True);del a,y;gc.collect()
    return np.concatenate(yy),pd.DataFrame(mm),pd.date_range('1989-09-20',periods=399,freq='7D').astype(str).to_numpy()

def rebuild(source,raw,reference,output):
    dest=output/'panels'/source;done=dest/'reconstruction.json'
    if done.exists():return read(done)
    start=time.perf_counter();spec=read(reference/'panels'/source/'spec.json');y,meta,dates=globals()[source](raw)
    meta=meta.reset_index(drop=True);c=spec['initial_cutoff'];h=y[:,max(0,c-spec['history']):c]
    freq=np.nanmean(np.where(np.isfinite(h),h>0,np.nan),axis=1);meta['frequency']=freq;meta['group']=np.where(freq<=.2,0,np.where(freq<=.5,1,2))
    old=pd.read_csv(reference/'panels'/source/'cohort.csv',dtype={'series_id':str})
    assert set(meta.series_id.astype(str))==set(old.series_id),'historical cohort mismatch'
    # The same deterministic raw-row order is checked; no post-score reordering/subsetting.
    np.testing.assert_array_equal(meta.series_id.astype(str),old.series_id.astype(str))
    np.testing.assert_array_equal(meta.group,old.group);np.testing.assert_allclose(meta.frequency,old.frequency,rtol=1e-14)
    if (reference/'panels'/source/'y.npy').exists():
        expected=np.load(reference/'panels'/source/'y.npy',mmap_mode='r');np.testing.assert_array_equal(y,expected)
    np.testing.assert_array_equal(pd.to_datetime(dates),pd.to_datetime(pd.read_csv(reference/'panels'/source/'calendar.csv').date))
    dest.mkdir(parents=True,exist_ok=True);np.save(dest/'y.npy',y)
    assert sha(dest/'y.npy')==spec['sha256']['y.npy'],'raw panel content hash differs'
    meta.to_csv(dest/'cohort.csv',index=False);pd.DataFrame(dict(date=dates)).to_csv(dest/'calendar.csv',index=False)
    spec['sha256']={f:sha(dest/f) for f in ['y.npy','cohort.csv','calendar.csv']};write(dest/'spec.json',spec)
    result=dict(source=source,series=len(y),cells=int(y.size),finite_cells=int(np.isfinite(y).sum()),panel_bitwise_equal=True,cohort_and_group_equal=True,seconds=time.perf_counter()-start,created=pd.Timestamp.now().isoformat(),files=spec['sha256'])
    write(done,result);print('PANEL PASS',source,result['seconds'],flush=True);return result

def cold(source,family,reference,output):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    sys.path.insert(0,str(HERE/'portable_vendor'))
    import common,models
    common.OUT=output;models.OUT=output
    common.ROOT=output;models.ROOT=output
    # Standard installed torch only: no appending historical virtual-environment paths.
    def torch_import():
        import torch
        torch.set_num_threads(4);return torch
    common.torch_import=torch_import;models.torch_import=torch_import
    H=1 if source=='dominicks' else 7;seed=20261021;s=read(output/'panels'/source/'spec.json');cut=s['initial_cutoff']
    path=output/'cold_fits'/f'{source}_{family}_h{H}';path.mkdir(parents=True,exist_ok=True)
    cached=read(path/'complete.json') if (path/'complete.json').exists() else None
    if cached and 'legal_first_issue' in cached:return cached
    selection=read(reference/'selection'/f'{source}_h{H}_{family}/selected.json');cfg=selection['configuration']
    original=reference/'banks'/source/f'h{H}_{family}_s{seed}_hist{s["history"]}_delay0'/f'c{cut}'
    oldmeta=read(original/'complete.json');assert sha(original/'predictions.npz')==oldmeta['prediction_sha256']
    with np.load(original/'predictions.npz') as f:expected_origins=f['origins'];expected=f['base']
    oo=np.arange(cut+1,s['eval_end']+1) if oldmeta.get('anchor_reference') else expected_origins
    d=models.Design(source,H);x,y,train_o,audit=d.train_arrays(cut,s['history']);assert audit==oldmeta['training']
    if cached:
        with np.load(path/'predictions.npz') as f:pred=f['base']
        sec=cached['fit_seconds'];infer=cached['inference_seconds']
    else:
        if family=='lgbm':model,sec=models.lgb_fit(x,y,seed,cfg['rounds'],path)
        else:model,sec,_,_=models.mlp_fit(x,y,seed,cfg['width'],cfg['epochs'],path)
        before=time.perf_counter();pred=models.predict(model,d.predict_x(oo),family).reshape(len(oo),d.n,2)*d.scale[None,:,None];infer=time.perf_counter()-before
    pred=pred.astype('float32');comparison=pred[np.searchsorted(oo,expected_origins)]
    diff=float(np.max(np.abs(comparison-expected)));np.testing.assert_allclose(comparison,expected,rtol=1e-5,atol=1e-5)
    np.savez_compressed(path/'predictions.npz',origins=oo,base=pred)
    from calibration import replay,daily_scores
    first_issue=selection['validation_last_target']+1;legal=oo>=first_issue;issued_origins=oo[legal]
    assert len(issued_origins)>0
    truth=common.targets(d.y,issued_origins,H);versions=np.full(len(issued_origins),cut);rows=[]
    for method in ['R0_raw','R2_reuse','R3_isolate']:
        q,logs,elapsed=replay(truth,pred[legal],d.scale,d.meta.group.to_numpy(),issued_origins,versions,H,method,window=s['window'])
        assert all(r['latest_maturity']<=r['origin'] for r in logs);metrics=daily_scores(truth,q,d.scale)
        rows.append(dict(method=method,replay_seconds=elapsed,mean_joint_scaled=float(np.nanmean(metrics['scaled_pinball']))))
    result=dict(source=source,family=family,H=H,cutoff=cut,configuration=cfg,training=audit,fit_seconds=sec,inference_seconds=infer,logical_models=2 if family=='lgbm' else 1,reference_predictions_sha256=sha(original/'predictions.npz'),reference_scope='five-origin anchors' if oldmeta.get('anchor_reference') else 'complete initial-model cached predictions',max_absolute_prediction_error=diff,bitwise_equal=bool(np.array_equal(comparison,expected)),tolerance_pass=True,legal_first_issue=int(issued_origins[0]),replay_checks=rows,environment=sys.prefix)
    write(path/'complete.json',result);print('COLD PASS',source,family,diff,flush=True);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['panels','cold']);ap.add_argument('--raw-config',type=Path);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--sources',nargs='+',default=['m5','retail','dominicks']);ap.add_argument('--families',nargs='+',default=['lgbm','mlp']);a=ap.parse_args()
    if a.mode=='panels':
        raw=read(a.raw_config);rows=[rebuild(s,raw,a.reference,a.output) for s in a.sources];write(a.output/'panel_reconstruction.json',rows)
    else:
        rows=[cold(s,f,a.reference,a.output) for s in a.sources for f in a.families]
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','lightgbm','openpyxl']}
        if 'mlp' in a.families:versions['torch']=importlib.metadata.version('torch')
        write(a.output/'cold_reproduction.json',dict(checks=rows,python=sys.version,platform=platform.platform(),environment=sys.prefix,packages=versions,site_paths=sys.path,all_passed=True))
if __name__=='__main__':main()
