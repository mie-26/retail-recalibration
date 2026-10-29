"""Two families with identical past-sales/static/calendar information and normalized targets."""
from common import *
import lightgbm as lgb, gc

class Design:
    def __init__(self,source,H):
        self.source=source;self.H=H;p=OUT/'panels'/source
        self.spec=read(p/'spec.json');self.y=np.load(p/'y.npy',mmap_mode='r');self.meta=pd.read_csv(p/'cohort.csv');self.n=len(self.meta)
        cut=self.spec['initial_cutoff'];hist=self.spec['history'];start=max(self.spec['lookback'],cut-hist-H+1)
        self.scale=np.maximum(1,np.nanmean(targets(self.y,np.arange(start,cut-H+1),H),axis=0)).astype('float32')
        self.scale=np.nan_to_num(self.scale,nan=1.)
        self.unit=self.spec['unit'];self.dates=pd.to_datetime(pd.read_csv(p/'calendar.csv').date)
        cats=pd.get_dummies(self.meta[['store_id','cat_id']].astype(str),dtype='float32').to_numpy()
        self.static=np.column_stack([cats,np.log1p(self.scale),self.meta.frequency]).astype('float32')
        self.start=max(self.spec['lookback'],cut-hist-H+1);self.end=self.spec['eval_end']
        self.X=None
    def make(self,origins,extra_delay=0):
        out=[];windows=[7,14,28,56] if self.unit=='day' else [1,2,4,8]
        lags=[0,1,2,6,13,27,55] if self.unit=='day' else [0,1,2,3,7]
        for origin in origins:
            t=int(origin)-extra_delay;assert t>=max(windows)
            cols=[self.static]
            denom=self.scale/self.H
            for lag in lags:
                v=self.y[:,t-1-lag];cols.extend([np.log1p(np.nan_to_num(v,nan=0)/denom)[:,None],np.isnan(v).astype('float32')[:,None]])
            for w in windows:
                v=self.y[:,t-w:t];valid=np.isfinite(v);n=valid.sum(1);a=np.nan_to_num(v,nan=0);mu=a.sum(1)/np.maximum(n,1)
                sd=np.sqrt(np.maximum(0,(a*a).sum(1)/np.maximum(n,1)-mu*mu))
                cols += [np.log1p(mu/denom)[:,None],np.log1p(sd/denom)[:,None],((v>0).sum(1)/np.maximum(n,1))[:,None],(n/w)[:,None]]
            date=self.dates.iloc[int(origin)-1];phase=date.dayofyear/365.25
            vals=[np.sin(2*np.pi*phase),np.cos(2*np.pi*phase),np.sin(2*np.pi*date.dayofweek/7),np.cos(2*np.pi*date.dayofweek/7)]
            cols.append(np.tile(vals,(self.n,1)));out.append(np.column_stack(cols).astype('float32'))
        return np.stack(out)
    def precompute(self):
        if self.X is None:
            self.X=self.make(np.arange(self.start,self.end+1));self.Y=targets(self.y,np.arange(self.start,self.end+1),self.H)/self.scale
    def train_arrays(self,cut,history,delay=0):
        lo=max(self.start,cut-delay-self.H-history+1);hi=cut-delay-self.H
        oo=np.arange(lo,hi+1);assert len(oo) and oo[-1]+self.H+delay<=cut
        if delay:
            # Whole-chain: feature origin has the same data delay, also at training examples.
            oo=oo[oo-delay>=self.spec['lookback']];x=self.make(oo,delay).reshape(-1,self.static.shape[1]+(len([0,1,2,6,13,27,55]) if self.unit=='day' else 5)*2+20)
            y=targets(self.y,oo,self.H)/self.scale
        else:
            self.precompute();x=self.X[oo-self.start].reshape(-1,self.X.shape[-1]);y=self.Y[oo-self.start]
        y=y.ravel();valid=np.isfinite(y);return x[valid],y[valid],oo,dict(train_rows=int(valid.sum()),candidate_rows=len(y),first_origin=int(oo[0]),last_origin=int(oo[-1]),last_target=int(oo[-1]+self.H),cutoff=cut,whole_chain_delay=delay)
    def predict_x(self,oo,delay=0):
        if delay:return self.make(oo,delay).reshape(-1,self.make(oo[:1],delay).shape[-1])
        self.precompute();return self.X[oo-self.start].reshape(-1,self.X.shape[-1])

def pinloss(y,q):
    e=y[:,None]-q;return np.nanmean(np.maximum(e*TAUS,e*(TAUS-1)))

def lgb_fit(x,y,seed,rounds,path=None):
    models=[];t=time.perf_counter()
    for tau in TAUS:
        params=dict(objective='quantile',alpha=float(tau),learning_rate=.05,num_leaves=31,min_data_in_leaf=100,max_bin=127,verbosity=-1,num_threads=8,seed=seed,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,deterministic=True,force_col_wise=True)
        m=lgb.train(params,lgb.Dataset(x,label=y,free_raw_data=False),num_boost_round=rounds);models.append(m)
        if path:m.save_model(str(path/f'q{int(tau*100)}.txt'))
    return models,time.perf_counter()-t

def network(torch,F,width):
    return torch.nn.Sequential(torch.nn.Linear(F,width),torch.nn.ReLU(),torch.nn.Linear(width,width),torch.nn.ReLU(),torch.nn.Linear(width,2),torch.nn.Softplus())

def mlp_fit(x,y,seed,width,epochs,path=None,val=None):
    torch=torch_import();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);device='cuda';assert torch.cuda.is_available()
    mean=x.mean(0,dtype='float64').astype('float32');sd=x.std(0,dtype='float64').astype('float32');sd=np.maximum(sd,.001)
    xx=torch.tensor(np.clip((x-mean)/sd,-20,20),device=device);yy=torch.tensor(y[:,None],device=device)
    net=network(torch,x.shape[1],width).to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    taus=torch.tensor(TAUS.astype('float32'),device=device);records=[];states={};start=time.perf_counter()
    batch=4096
    for epoch in range(1,epochs+1):
        net.train();order=torch.randperm(len(y),device=device)
        for begin in range(0,len(y),batch):
            ix=order[begin:begin+batch];q=net(xx[ix]);e=yy[ix]-q;loss=torch.maximum(e*taus,e*(taus-1)).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        if val is not None and epoch in [5,10,20]:
            pred=mlp_predict(net,mean,sd,val[0],torch);score=pinloss(val[1],np.sort(pred,axis=1))
            records.append(dict(width=width,epochs=epoch,validation_loss=score));states[epoch]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    if path:torch.save(dict(state=net.cpu().state_dict(),mean=mean,sd=sd,F=x.shape[1],width=width),path/'model.pt')
    del xx,yy;torch.cuda.empty_cache()
    return (net.to(device),mean,sd),seconds,records,states

def mlp_predict(net,mean,sd,x,torch=None):
    if torch is None:torch=torch_import()
    net.eval();result=[]
    with torch.no_grad():
        for i in range(0,len(x),32768):
            v=torch.tensor(np.clip((x[i:i+32768]-mean)/sd,-20,20),device='cuda');result.append(net(v).cpu().numpy())
    return np.concatenate(result)

def predict(model,x,family,rounds=None):
    if family=='lgbm':return np.column_stack([m.predict(x,num_iteration=rounds,num_threads=8) for m in model])
    return mlp_predict(*model,x)

def tune(source,H,family):
    p=OUT/'selection'/f'{source}_h{H}_{family}';p.mkdir(parents=True,exist_ok=True)
    if (p/'selected.json').exists():return read(p/'selected.json')
    d=Design(source,H);s=d.spec;c=s['initial_cutoff'];history=s['history'];x,y,_,audit=d.train_arrays(c,history)
    # One chronological validation interval, all targets mature before scored origins.
    length=56 if s['unit']=='day' else (8 if source=='m5_weekly' else 16);oo=np.arange(c+1,c+1+length)
    assert oo[-1]+H<s['eval_start'];vx=d.predict_x(oo);vy=targets(d.y,oo,H).ravel()/np.tile(d.scale,len(oo))
    valid=np.isfinite(vy);vx=vx[valid];vy=vy[valid];rows=[];cost=0
    if family=='lgbm':
        model,sec=lgb_fit(x,y,SEEDS[0],512,p);cost+=sec
        for rounds in [256,512]:rows.append(dict(rounds=rounds,validation_loss=pinloss(vy,np.sort(predict(model,vx,family,rounds),axis=1))))
    else:
        for width in [64,128]:
            _,sec,rr,_=mlp_fit(x,y,SEEDS[0],width,20,val=(vx,vy));cost+=sec;rows+=rr
    chosen=min(rows,key=lambda r:r['validation_loss']);csv(p/'candidates.csv',rows)
    result=dict(source=source,H=H,family=family,configuration={k:v for k,v in chosen.items() if k!='validation_loss'},selection_loss=chosen['validation_loss'],measured_fit_seconds=cost,validation_origins=[int(oo[0]),int(oo[-1])],validation_last_target=int(oo[-1]+H),training=audit,created=now(),feature_columns=int(x.shape[1]))
    write(p/'selected.json',result);print('SELECTED',result,flush=True);return result

def bank(source,H,family,seed,history=None,chain_delay=0):
    check_lock();selection=read(OUT/'selection'/f'{source}_h{H}_{family}/selected.json');d=Design(source,H);s=d.spec;history=history or s['history'];cfg=selection['configuration']
    # Dense minimum-period bank supports all coarser calendars and shifted phases.
    cuts=[s['initial_cutoff']]+list(range(s['eval_start'],s['eval_end']+1,min(s['Us'])))
    if history!=s['history'] or chain_delay:cuts=[s['initial_cutoff']]+list(range(s['eval_start'],s['eval_end']+1,s['Us'][1]))
    for ci,cut in enumerate(cuts):
        path=OUT/'banks'/source/f'h{H}_{family}_s{seed}_hist{history}_delay{chain_delay}'/f'c{cut}';path.mkdir(parents=True,exist_ok=True)
        if (path/'complete.json').exists():continue
        started=time.perf_counter();x,y,oo,audit=d.train_arrays(cut,history,chain_delay)
        first=cut+1 if ci==0 else cut
        last=s['eval_end'] if ci==0 else min(s['eval_end'],cut+max(s['Us'])-1)
        # Initial model retained over whole warmup; candidate banks never predict before fit.
        po=np.arange(first,last+1);px=d.predict_x(po,chain_delay)
        if family=='lgbm':model,sec=lgb_fit(x,y,seed,cfg['rounds'],path)
        else:model,sec,_,_=mlp_fit(x,y,seed,cfg['width'],cfg['epochs'],path)
        predstart=time.perf_counter();base=predict(model,px,family).reshape(len(po),d.n,2)*d.scale[None,:,None];infer=time.perf_counter()-predstart
        assert np.isfinite(base).all();np.savez_compressed(path/'predictions.npz',origins=po,base=base.astype('float32'))
        meta=dict(source=source,H=H,family=family,seed=seed,history=history,cutoff=cut,configuration=cfg,training=audit,fit_seconds=sec,inference_seconds=infer,total_seconds=time.perf_counter()-started,logical_quantile_models=2 if family=='lgbm' else 1,created=now(),prediction_sha256=sha(path/'predictions.npz'),model_sha256={p.name:sha(p) for p in path.iterdir() if p.suffix in ['.txt','.pt']})
        write(path/'complete.json',meta);print('FIT',source,H,family,seed,history,cut,round(sec,2),flush=True)
        del x,y,px,model,base;gc.collect()
        if family=='mlp':torch_import().cuda.empty_cache()

