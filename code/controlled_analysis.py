"""Restore, audit and display the controlled follow-up analyses. No model fitting."""
from pathlib import Path
import argparse,os,json,tarfile,subprocess,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
F=['source','H','U','family'];SOURCES=['m5','retail','dominicks'];NAMES={'m5':'M5','retail':'Retail','dominicks':"Dominick's"}
WEIGHTS={}
def weights(T,L):
    if (T,L) not in WEIGHTS:
        rng=np.random.default_rng(20261021);starts=rng.integers(0,T,(3000,(T+L-1)//L))
        ix=((starts[:,:,None]+np.arange(L))%T).reshape(3000,-1)[:,:T]
        WEIGHTS[T,L]=np.stack([np.bincount(x,minlength=T) for x in ix])
    return WEIGHTS[T,L]
def ordered(d):
    d=d.copy();d['source']=pd.Categorical(d.source,SOURCES,ordered=True)
    return d.sort_values(F).reset_index(drop=True)
def label(r):return f'{NAMES[str(r.source)]}  H{r.H}  '+('LGBM' if r.family=='lgbm' else 'MLP')
def save(fig,figdir,name):
    for ext in ['png','pdf','svg']:fig.savefig(figdir/(name+'.'+ext),dpi=320,bbox_inches='tight')
    plt.close(fig)
def panel(ax,labels,left=True):
    for lo,hi in [(-.5,3.5),(7.5,11.5)]:ax.axhspan(lo,hi,color='#F2F5F7',zorder=0)
    ax.axvline(0,c='#777777',lw=.8);ax.set_ylim(11.7,-.7)
    ax.set_yticks(range(12),labels if left else ['']*12);ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',color='#e6e6e6',lw=.5);ax.set_axisbelow(True)
def error(ax,g,y,color,marker='o'):
    ax.errorbar(g.point,y,xerr=[np.maximum(0,g.point-g.lower),np.maximum(0,g.upper-g.point)],fmt=marker,c=color,ms=4,capsize=2,lw=1)
def main(root,check=True):
    root=Path(root).resolve();p=root/'controlled';out=root/'reproduced/controlled';out.mkdir(parents=True,exist_ok=True)
    with tarfile.open(root/'controlled_analyses.tar.xz','r:xz') as t:
        for member in t.getmembers():
            target=(root/member.name).resolve()
            if not member.isfile() or not target.is_relative_to(p.resolve()):raise ValueError('Unexpected controlled archive member: '+member.name)
            data=t.extractfile(member).read()
            if target.exists():
                if target.read_bytes()!=data:raise ValueError('Conflicting existing controlled file: '+str(target))
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    if check:subprocess.run([sys.executable,str(root/'code/recompute_controlled_scores.py'),'--root',str(p),'--intervals'],check=True)
    d=pd.read_parquet(p/'E2/daily.parquet');d['num']=d.scaled_pinball*d.n
    rows=[];inter=[];maxerr=0.
    stored=pd.read_csv(p/'tables/E2_interaction_intervals.csv')
    for key,g in d.groupby(F):
        meta=dict(zip(F,key));H=meta['H']
        for tau in [.9,.95]:
            a=g[g['quantile']==tau].groupby(['origin','method'])[['num','n']].sum();v=a.num.unstack('method').sort_index();n=a.n.unstack('method');assert n.nunique(axis=1).eq(1).all()
            for rule in ['WQ','QT']:
                c=v[rule+'_group'].to_numpy();r=v[rule+'_series'].to_numpy();w=weights(len(v),2*H)
                lo,hi=np.quantile(100*((w@c)/(w@r)-1),[.025,.975])
                rows.append(dict(**meta,quantile=tau,rule=rule,point=100*(c.sum()/r.sum()-1),lower=lo,upper=hi))
        v=g.groupby(['origin','method']).num.sum().unstack('method').sort_index()
        for a,b in [('group','series'),('global','group')]:
            delta=(v['QT_'+a]-v['QT_'+b]-v['WQ_'+a]+v['WQ_'+b]).to_numpy();raw=v.Raw.to_numpy()
            mask=stored.sharing_contrast.eq(a+'-'+b)
            for k,x in meta.items():mask&=stored[k].eq(x)
            for row in stored[mask].itertuples():
                w=weights(len(v),row.block_length);lo,hi=np.quantile(100*(w@delta)/(w@raw),[.025,.975]);point=100*delta.sum()/raw.sum()
                maxerr=max(maxerr,abs(lo-row.lower),abs(hi-row.upper),abs(point-row.interaction_pct_of_Raw))
                inter.append(dict(**meta,sharing_contrast=a+'-'+b,point=point,lower=lo,upper=hi,primary=row.primary,block_length=row.block_length))
    assert maxerr<1e-9
    q=pd.DataFrame(rows);ii=pd.DataFrame(inter)
    q.to_csv(out/'E2_quantile_sharing_intervals.csv',index=False);ii.to_csv(out/'E2_interaction_intervals.csv',index=False)
    # Main figure uses the two prespecified scalar outcomes and the secondary joint interaction.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    figdir=root/'reproduced/figures';figdir.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,3,figsize=(10,5.6),gridspec_kw={'width_ratios':[1,1,1.1]})
    fig.subplots_adjust(left=.195,right=.985,top=.9,bottom=.19,wspace=.15)
    base=ordered(q[(q['quantile']==.9)&q.rule.eq('WQ')]);labs=[label(r) for r in base.itertuples()]
    limit=max(abs(q.lower.min()),abs(q.upper.max()));limits=(-max(21,limit*1.06),max(12,q.upper.max()*1.08))
    for ax,tau,letter in zip(axs[:2],[.9,.95],['a','b']):
        panel(ax,labs,letter=='a')
        for rule,col,mark,offset in [('WQ','#0072B2','o',-.12),('QT','#D55E00','s',.12)]:error(ax,ordered(q[(q['quantile']==tau)&q.rule.eq(rule)]),np.arange(12)+offset,col,mark)
        ax.set_xlim(*limits);ax.set_xlabel('Group vs series\nloss change (%)');ax.set_title(f'{letter}  q{int(tau*100)}',loc='left',weight='bold')
    ax=axs[2];panel(ax,labs,False);g=ordered(ii[ii.primary&ii.sharing_contrast.eq('group-series')]);error(ax,g,np.arange(12),'#4B5563','D');ax.set_title('c  Joint interaction',loc='left',weight='bold');ax.set_xlabel('Difference in sharing effects\n(% of Raw joint loss)')
    fig.legend(handles=[Line2D([],[],c='#0072B2',marker='o',ls='',label='Weighted residual quantile (WQ)'),Line2D([],[],c='#D55E00',marker='s',ls='',label='Constant-step scalar QT')],loc='lower center',bbox_to_anchor=(.57,.015),ncol=2,frameon=False,fontsize=10)
    save(fig,figdir,'Figure_4')
    e=pd.read_csv(p/'tables/effects.csv');e=e[e.scope.eq('all')]
    # E1 preserves structurally identical R3 as a marked control.
    methods=['R1_global','R2_reuse','R3_isolate','R4_half','R5_multiqt_nonnegative']
    a=e[e.experiment.eq('E1')&e.candidate.str.endswith('_TM')&e.reference.str.endswith('_TS')].copy();a['rule']=a.candidate.str[:-3]
    mat=a.pivot(index=F,columns='rule',values='relative_loss_pct').reset_index();mat=ordered(mat);arr=mat[methods].to_numpy()
    fig,ax=plt.subplots(figsize=(8.6,5.3));fig.subplots_adjust(left=.27,right=.90,bottom=.18,top=.92)
    lim=max(.35,np.max(abs(arr)));im=ax.imshow(arr,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_yticks(range(12),[label(r) for r in mat.itertuples()]);ax.set_xticks(range(5),['Global\nR1','Reuse\nR2','Isolation\nR3*','Half weight\nR4','Recursive\nR5'])
    for (i,j),value in np.ndenumerate(arr):ax.text(j,i,'=' if j==2 else f'{value:+.3f}',ha='center',va='center',fontsize=9,color='white' if abs(value)>.6*lim else '#202020')
    ax.set_title('Long vs short selection history at the same cutoff',loc='left',weight='bold',pad=12)
    fig.colorbar(im,ax=ax,fraction=.035,pad=.035,label='Joint loss change (%)')
    fig.text(.27,.02,'* R3 candidates use identical eligible feedback pools; = denotes structural equivalence.',fontsize=9)
    save(fig,figdir,'S7_selection_history')
    # E3 always displays the raw comparator adjacent to the reference-version contrast.
    ci=pd.read_csv(p/'tables/intervals.csv');ci=ci[ci.experiment.eq('E3')&ci.scope.eq('all')&ci.reference_block&ci.metric.eq('relative_loss_pct')]
    a=e[e.experiment.eq('E3')].merge(ci[F+['candidate','reference','lower','upper']],on=F+['candidate','reference'],validate='one_to_one').rename(columns={'relative_loss_pct':'point'})
    a.to_csv(out/'E3_joint_contrasts.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(9.1,5.2));fig.subplots_adjust(left=.24,right=.98,top=.9,bottom=.14,wspace=.22)
    for j,(ref,title) in enumerate([('old_reference','a  New reference vs old reference'),('Raw','b  New reference vs Raw')]):
        ax=axs[j];g=ordered(a[a.candidate.eq('new_reference')&a.reference.eq(ref)]);panel(ax,[label(r) for r in g.itertuples()],j==0);error(ax,g,np.arange(12),'#0072B2' if j==0 else '#D55E00');ax.set_title(title,loc='left',weight='bold',fontsize=10);ax.set_xlabel('Joint loss change (%)')
    save(fig,figdir,'S8_residual_reference')
    result=dict(passed=True,derived_single_quantile_intervals=48,interaction_intervals_recomputed=len(ii),max_interaction_error=maxerr,new_model_fits=0,scope='Derived displays from fixed cached paths; selected parameters not changed; interval uncertainty conditional on the selected paths.')
    (out/'CHECKS.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));ap.add_argument('--skip-score-audit',action='store_true');a=ap.parse_args();main(a.root,not a.skip_score_audit)
