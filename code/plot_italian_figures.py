from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
# Presentation adapter; archived scientific code and data are unchanged.
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'italian/work/sr_external_italy_20261028'))
from common import OUT as EXPERIMENT,TAUS
SRC=EXPERIMENT/'analysis'
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
def draw(src=SRC,out=OUT):
    d=pd.read_csv(src/'interaction_intervals.csv');d=d[(d.H==14)&(d.U==28)&(d.family=='lgbm')&(d.block==28)].sort_values('quantile')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6.5,2.45));ax.axvline(0,color='.5',lw=.8)
    for j,r in enumerate(d.itertuples()):
        ax.errorbar(r.interaction_percent,j,xerr=[[r.interaction_percent-r.lower97_5],[r.upper97_5-r.interaction_percent]],fmt='o',color='#234f6e',capsize=4,markersize=6)
        ax.annotate(f'{r.interaction_percent:.2f}% [{r.lower97_5:.2f}, {r.upper97_5:.2f}]',(r.interaction_percent,j),xytext=(0,15),textcoords='offset points',ha='center',fontsize=10)
    ax.set_yticks([0,1],['q90','q95']);ax.set_ylim(-.45,1.6);ax.invert_yaxis();ax.set_xlim(-1,36);ax.set_xlabel('Rule × sharing interaction (% of Raw scaled pinball loss)');ax.grid(axis='x',alpha=.12)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(out/f'Figure_4.{ext}',dpi=400,bbox_inches='tight')
    plt.close(fig)

def supplementary():
    a=SRC;fig=OUT;fig.mkdir(exist_ok=True)
    scores=pd.read_csv(a/'scores.csv');intervals=pd.read_csv(a/'interaction_intervals.csv');effects=pd.read_csv(a/'effect_intervals.csv')
    configs8=[(H,U,f) for f in ['lgbm','mlp'] for H in [7,14] for U in [28,56]];labels=[f'{f.upper()}  H{H} / U{U}' for H,U,f in configs8]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(name,f):
        f.savefig(fig/(name+'.pdf'),bbox_inches='tight');f.savefig(fig/(name+'.svg'),bbox_inches='tight');f.savefig(fig/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(f)
    f,axs=plt.subplots(1,2,figsize=(10,4.8),sharey=True)
    for tau,ax in zip(TAUS,axs):
        g=intervals[(intervals['quantile']==tau)&intervals.main_block]
        for j,(H,U,fam) in enumerate(configs8):
            r=g[(g.H==H)&(g.U==U)&(g.family==fam)].iloc[0];primary=(H,U,fam)==(14,28,'lgbm');lo=r.lower97_5 if primary else r.lower95;hi=r.upper97_5 if primary else r.upper95
            ax.errorbar(r.interaction_percent,j,xerr=[[r.interaction_percent-lo],[hi-r.interaction_percent]],fmt='s' if primary else 'o',color='#174d72' if primary else '#8796a5',capsize=3,markersize=6)
        ax.axvline(0,color='.35',lw=.8);ax.set_title(f'q{int(tau*100)}');ax.set_xlabel('Rule × sharing interaction\n(% of Raw scaled pinball loss)');ax.grid(axis='x',alpha=.15)
    axs[0].set_yticks(range(8),labels);axs[0].invert_yaxis();f.suptitle('Retrospective confirmation in one Italian grocery store',fontsize=13)
    f.text(.25,-.02,'Dark square: prespecified primary, 97.5% CI. Other intervals: descriptive 95% CI.\nCalendar blocks = 2H; 5,000 paired resamples; positive interaction does not imply superiority to Raw.',fontsize=9)
    f.tight_layout();save('S10_external_interactions',f)
    bridge=[]
    for H,U,fam in configs8:
        g=scores[(scores.H==H)&(scores.U==U)&(scores.family==fam)]
        raw=g[g.method=='Raw'].sort_values('quantile')
        for m in ['R2_reuse','R3_isolate']:
            p=g[g.method==m].sort_values('quantile');bridge.append(dict(H=H,U=U,family=fam,method=m,joint_scaled_pinball=p.scaled_pinball.mean(),joint_loss_change_percent=100*(p.scaled_pinball.mean()/raw.scaled_pinball.mean()-1),q90_weak=p.weak.iloc[0],q95_weak=p.weak.iloc[1],q90_weak_change_pp=100*(p.weak.iloc[0]-raw.weak.iloc[0]),q95_weak_change_pp=100*(p.weak.iloc[1]-raw.weak.iloc[1])))
    bridge=pd.DataFrame(bridge);bridge.to_csv(EXPERIMENT/'release_audit/bridge_joint_summary.csv',index=False)
    f,axs=plt.subplots(1,3,figsize=(12,5),sharey=True)
    for m,color,label,shift in [('R2_reuse','#146a80','Reuse',-.12),('R3_isolate','#bb6645','Isolate',.12)]:
        g=bridge[bridge.method==m]
        for ax,col in zip(axs,['joint_loss_change_percent','q90_weak_change_pp','q95_weak_change_pp']):
            ax.plot(g[col],np.arange(8)+shift,'o',color=color,label=label,markersize=5)
    for ax,title in zip(axs,['Joint scaled pinball change vs Raw (%)','q90 weak coverage change (pp)','q95 weak coverage change (pp)']):
        ax.axvline(0,color='.4',lw=.8);ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.15)
    axs[0].set_yticks(range(8),labels);axs[0].invert_yaxis();axs[-1].legend(frameon=False,loc='lower right');f.suptitle('A fixed calibrator: feedback reuse and isolation',fontsize=13)
    f.text(.2,-.02,'Point estimates, equal quantile joint loss. Negative loss change is better.\nAll per-quantile paired intervals, including intervals crossing zero, are supplied in effect_intervals.csv.',fontsize=9);f.tight_layout();save('S11_external_feedback',f)
    # Prespecified E1 sensitivity: univariate paths and post-hoc sorted paths both retained.
    short=scores[scores.selection_length==28];long=scores[scores.selection_length==84]
    e1=short.merge(long,on=['H','U','family','method','quantile'],suffixes=('_short','_long'));e1['long_history_loss_change_percent']=100*(e1.scaled_pinball_long/e1.scaled_pinball_short-1);e1.to_csv(EXPERIMENT/'release_audit/E1_history_sensitivity.csv',index=False)
    # Same estimands, predeclared origin-year summaries; 2016 has zero valid targets.

if __name__=='__main__':
    draw();supplementary()
