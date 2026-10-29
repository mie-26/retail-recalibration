"""Generate research tables, figures, and a bounded interpretation of the follow-up."""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
    tables=root/'tables';out_tables=root.parent/'reproduced/robustness_tables';out_tables.mkdir(parents=True,exist_ok=True);fig=root.parent/'reproduced/figures';fig.mkdir(parents=True,exist_ok=True)
    effects=pd.read_csv(tables/'effects.csv');ints=pd.read_csv(tables/'intervals.csv');seeds=pd.read_csv(tables/'seed_effects.csv');sel=pd.read_csv(tables/'selected_parameters.csv');q=pd.read_csv(tables/'quantile_summary.csv')
    ref=ints[(ints.metric=='relative_loss_pct')&ints.reference_block]
    keys=['source','H','U','family','analysis','candidate','reference']
    e=effects.merge(ref[keys+['lower','upper']],on=keys,validate='one_to_one')
    e.to_csv(out_tables/'paper_effects.csv',index=False)
    summaries=[]
    for (source,candidate),g in e[e.analysis=='selected_vs_raw'].groupby(['source','candidate']):
        summaries.append(dict(source=source,method=candidate,settings=len(g),lower_loss=int((g.relative_loss_pct<0).sum()),interval_below_zero=int((g.upper<0).sum()),interval_above_zero=int((g.lower>0).sum()),min_relative_pct=g.relative_loss_pct.min(),max_relative_pct=g.relative_loss_pct.max(),min_q90_change_pp=g.q90_weak_difference.min(),max_q90_change_pp=g.q90_weak_difference.max(),min_q95_change_pp=g.q95_weak_difference.min(),max_q95_change_pp=g.q95_weak_difference.max()))
    summary=pd.DataFrame(summaries);summary.to_csv(out_tables/'paper_parameter_summary.csv',index=False)
    grid=e[(e.analysis=='full_grid_sensitivity')&e.candidate.str.startswith('R5_')].groupby(['source','candidate']).agg(settings=('relative_loss_pct','size'),lower_loss=('relative_loss_pct',lambda v:int((v<0).sum())),min_relative_pct=('relative_loss_pct','min'),max_relative_pct=('relative_loss_pct','max'),interval_below_zero=('upper',lambda v:int((v<0).sum()))).reset_index()
    grid.to_csv(out_tables/'paper_recursive_grid.csv',index=False)
    mechanism=e[(e.analysis=='three_seed_mechanism')&(e.candidate=='Q2_output_clamp_played')&(e.reference=='Q1_output_clamp_shadow')].copy()
    ss=seeds[(seeds.analysis=='three_seed_mechanism')&(seeds.candidate=='Q2_output_clamp_played')&(seeds.reference=='Q1_output_clamp_shadow')]
    sign=ss.groupby(['source','H','U','family']).relative_loss_pct.agg(seed_min='min',seed_max='max',negative_seeds=lambda x:int((x<0).sum())).reset_index()
    mechanism=mechanism.merge(sign,on=['source','H','U','family']);mechanism.to_csv(out_tables/'paper_three_seed_mechanism.csv',index=False)
    sim=pd.read_csv(root/'simulation/effects.csv');support=pd.read_csv(root/'heterogeneity/support.csv');hetero=pd.read_csv(root/'heterogeneity/block_effects.csv')
    # Supplementary plots use separate axes; absolute losses from different sources are never pooled.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    names={'m5':'M5','retail':'Online Retail II','dominicks':"Dominick's"};colors={'R2_reuse_selected':'#2066A8','R3_isolate_selected':'#D87825','R5_multiqt_nonnegative_selected':'#338C63'}
    fig1,axs=plt.subplots(3,2,figsize=(10,10),layout='constrained')
    for i,source in enumerate(names):
        for j,family in enumerate(['lgbm','mlp']):
            ax=axs[i,j];part=e[(e.source==source)&(e.family==family)&(e.analysis=='selected_vs_raw')];settings=sorted(set(zip(part.H,part.U)));labels=[f'H{H}/U{U}' for H,U in settings]
            for n,(method,color) in enumerate(colors.items()):
                g=part[part.candidate==method].sort_values(['H','U']);yy=np.arange(len(g))+(n-1)*.19
                ax.errorbar(g.relative_loss_pct,yy,xerr=np.array([g.relative_loss_pct-g.lower,g.upper-g.relative_loss_pct]),fmt='o',ms=3,elinewidth=.8,c=color,label=method.split('_')[0])
            ax.axvline(0,color='.5',lw=.7,ls=':');ax.set_yticks(np.arange(len(settings)),labels);ax.invert_yaxis();ax.set_title(f'{names[source]} · {family.upper()}');ax.set_xlabel('Selected recalibrator vs Raw: loss change (%)')
    axs[0,0].legend(frameon=False,ncol=3,fontsize=8)
    save(fig1,fig/'S2_parameter_sensitivity')
    fig2,axs=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    for ax,(source,name) in zip(axs,names.items()):
        g=mechanism[mechanism.source==source].sort_values(['family','H']);yy=np.arange(len(g))
        ax.errorbar(g.relative_loss_pct,yy,xerr=np.array([g.relative_loss_pct-g.lower,g.upper-g.relative_loss_pct]),fmt='o',c='#2066A8',ms=4)
        for n,r in enumerate(g.itertuples()):ax.plot([r.seed_min,r.seed_max],[n+.14,n+.14],c='#D87825',lw=2)
        ax.set_yticks(yy,[f'{r.family.upper()} H{r.H}' for r in g.itertuples()]);ax.invert_yaxis();ax.axvline(0,c='.5',ls=':',lw=.7);ax.set_title(name);ax.set_xlabel('Q2 vs Q1: loss change (%)')
    save(fig2,fig/'S5_three_seed_feedback')
    fig3,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,pair in zip(axs,[('R2_reuse','R0_raw'),('R3_isolate','R2_reuse')]):
        for n,delay in enumerate([0,2]):
            g=sim[(sim.candidate==pair[0])&(sim.reference==pair[1])&(sim.delay==delay)].sort_values('world');yy=np.arange(len(g))+(n-.5)*.15
            ax.errorbar(g.relative_loss_pct,yy,xerr=np.array([g.relative_loss_pct-g.lower,g.upper-g.relative_loss_pct]),fmt='o',ms=4,label=f'extra delay {delay}')
        ax.set_yticks(np.arange(4),g.world);ax.invert_yaxis();ax.axvline(0,c='.5',ls=':',lw=.7);ax.set_title(f'{pair[0]} vs {pair[1]}');ax.set_xlabel('Joint loss change (%)');ax.legend(frameon=False,fontsize=8)
    save(fig3,fig/'S6_simulation')
    print("Rebuilt four robustness summary tables and supplementary figures S2, S5 and S6.")

def save(fig,p):
    for ext in ['png','pdf','svg']:fig.savefig(p.with_suffix('.'+ext),dpi=320,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();main(a.root)
