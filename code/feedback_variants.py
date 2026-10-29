"""Task-specific recursive feedback controls, extracted without numerical changes."""
import time,warnings,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cold_fit/code/portable_vendor"))
from calibration import replay,ordered,project2,TAUS

def eta_path(z,H):
    """Same finite-label adaptive scale as the archived implementation, computed once."""
    y=z['truth'];b=ordered(z['base'].astype(float));res=(y[:,:,None]-b)/z['scale'][None,:,None]
    oo=z['origins'];N=y.shape[1];v=np.ones(y.shape,dtype=float)
    for t,o in enumerate(oo):
        new=np.flatnonzero(oo+H==o)
        if not len(new):continue
        ix=np.flatnonzero(oo+H<=o)[-50:]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            q=np.nanquantile(np.abs(res[ix]).transpose(1,0,2).reshape(N,-1),.9,axis=1)
        v[t]=np.nan_to_num(q,nan=1.)
    return v


def multi(z,H,etas,rate=.1,played=True,clamp=True):
    start=time.perf_counter();b=ordered(z['base'].astype(float));q=np.empty_like(b);shadow=np.empty_like(b)
    theta=np.zeros_like(b[0]);oo=z['origins'];sc=z['scale'];y=z['truth']
    for t,o in enumerate(oo):
        for s in np.flatnonzero(oo+H==o):
            valid=np.isfinite(y[s]);eta=np.maximum(rate,rate*etas[t]);feedback=q[s] if played else shadow[s]
            theta[valid]+=eta[valid,None]*(TAUS-(y[s,valid,None]<=feedback[valid]))
        shadow[t]=project2(b[t]/sc[:,None]+theta)*sc[:,None]
        q[t]=np.maximum(shadow[t],0) if clamp else shadow[t]
    assert np.isfinite(q).all() and (np.diff(q,axis=-1)>=-1e-12).all()
    return q,time.perf_counter()-start
