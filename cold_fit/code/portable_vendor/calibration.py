"""Frozen rolling calibrators and the published delayed lazy MultiQT recursion."""
import numpy as np,time
TAUS=np.array([.9,.95])
def ordered(b):return np.sort(np.maximum(b,0),axis=-1)
def project2(v):
    out=v.copy();cross=v[:,0]>v[:,1];out[cross]=v[cross].mean(1)[:,None];return out
def indices(origins,versions,t,H,d,window,kind,isolated):
    ix=np.flatnonzero(origins+H+d<=origins[t])
    if kind=='calendar':ix=ix[origins[ix]>=origins[t]-window]
    elif kind=='mature':ix=ix[-window:]
    else:raise ValueError(kind)
    if isolated:ix=ix[versions[ix]==versions[t]]
    return ix
def weighted_q(v,w,tau):
    keep=np.isfinite(v)&(w>0);v=v[keep];w=w[keep]
    if not len(v):return 0.
    if np.all(w==w[0]):return float(np.quantile(v,tau,method='higher'))
    ix=np.argsort(v,kind='stable');v=v[ix];w=w[ix];target=(tau+(1-tau)/len(v))*w.sum()
    return float(v[min(len(v)-1,np.searchsorted(np.cumsum(w),target,side='left'))])

def replay(y,b,sc,groups,origins,versions,H,method,d=0,window=56,kind='calendar',nonnegative=True):
    started=time.perf_counter();b=ordered(b.astype(float));T,N,K=b.shape
    q=np.empty_like(b);logs=[];hidden=np.zeros((N,2));allres=(y[:,:,None]-b)/sc[None,:,None]
    group_ids=np.unique(groups) if method!='R1_global' else np.array([0]);gm={g:groups==g for g in group_ids}
    if method=='R1_global':gm={0:np.ones(N,dtype=bool)}
    last=0
    for t in range(T):
        if t and versions[t]!=versions[t-1]:last=t
        ix=indices(origins,versions,t,H,d,window,kind,method=='R3_isolate')
        pre=b[t].copy();nscalar=0;active=0
        if method=='R5_multiqt_nonnegative':
            # Feedback arriving before current issue corresponds to D=H+d-1 in paper's after-issue convention.
            newly=np.flatnonzero(origins+H+d==origins[t])
            observed=np.flatnonzero(origins+H+d<=origins[t])[-50:]
            if len(newly):
                residual=np.abs(allres[observed]).transpose(1,0,2).reshape(N,-1)
                with np.errstate(all='ignore'):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning);eta=np.maximum(.1,.1*np.nanquantile(residual,.9,axis=1))
                eta=np.nan_to_num(eta,nan=.1)
                for s in newly:
                    valid=np.isfinite(y[s]);hidden[valid]+=eta[valid,None]*(TAUS-(y[s,valid,None]<=q[s,valid]))
            pre=project2(b[t]/sc[:,None]+hidden)*sc[:,None]
            ix=observed;nscalar=int(np.isfinite(y[ix]).sum())
            active=int(np.isfinite(y[ix]).any(0).sum()) if len(ix) else 0
        elif method!='R0_raw' and len(ix):
            for g,m in gm.items():
                rr=allres[ix][:,m,:]
                nscalar+=int(np.isfinite(rr[:,:,0]).sum())
                if np.isfinite(rr[:,:,0]).any():active+=int(m.sum())
                for k,tau in enumerate(TAUS):
                    v=rr[:,:,k].ravel();valid=np.isfinite(v)
                    if not valid.any():offset=0.
                    elif method=='R4_half':
                        w=np.broadcast_to(np.where(versions[ix]==versions[t],1.,.5)[:,None],rr.shape[:2]).ravel();offset=weighted_q(v,w,tau)
                    else:offset=float(np.quantile(v[valid],tau,method='higher'))
                    pre[m,k]+=offset*sc[m]
        q[t]=np.maximum(pre,0) if method=='R5_multiqt_nonnegative' and nonnegative else pre
        if method!='R5_multiqt_nonnegative':q[t]=ordered(q[t])
        logs.append(dict(origin=int(origins[t]),version=int(versions[t]),version_age=t-last,eligible_origins=len(ix),old_origins=int((versions[ix]!=versions[t]).sum()),valid_residual_targets=nscalar,active_series=active,fallback_all=bool(method!='R0_raw' and active==0),latest_released_origin=int(origins[ix[-1]]) if len(ix) else -1,latest_maturity=int(origins[ix[-1]]+H+d) if len(ix) else -1))
    assert np.isfinite(q).all() and (np.diff(q,axis=-1)>=-1e-12).all()
    return q,logs,time.perf_counter()-started

def daily_scores(y,q,sc):
    mask=np.isfinite(y);n=mask.sum(1);out={};e=y[:,:,None]-q;l=np.maximum(e*TAUS,e*(TAUS-1))
    out['n']=n
    for name,v in [('scaled_pinball',l/sc[None,:,None]),('pinball',l),('weak',(y[:,:,None]<=q).astype(float)),('strict',(y[:,:,None]<q).astype(float)),('equality',(y[:,:,None]==q).astype(float)),('zero_tie',((y[:,:,None]==0)&(q==0)).astype(float))]:
        out[name]=np.where(mask[:,:,None],v,np.nan)
        out[name]=np.nansum(out[name],axis=1)/np.maximum(n[:,None],1)
        out[name][n==0]=np.nan
    return out

