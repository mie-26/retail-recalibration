"""Publication figures from unchanged archived tables and explicit vector geometry."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'figure_source_data';OUT=ROOT/'reproduced/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
NAMES={'m5':'M5','retail':'Online Retail II','dominicks':"Dominick’s"};SOURCES=list(NAMES);FAMS={'lgbm':'LightGBM','mlp':'MLP'}
COL={'R0_raw':'#333333','R2_reuse':'#0072B2','R3_isolate':'#D55E00','R5_multiqt_nonnegative':'#009E73'}

def save(fig,n):
    name={1:'Figure_1',2:'S1_complete_policy_effects',3:'Figure_3',4:'S3_coverage',5:'S4_training_budget',6:'Figure_4','primary':'Figure_2'}[n]
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=320)
    plt.close(fig)

def box(ax,x,y,w,h,label,fc='#F2F5F7',fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',facecolor=fc,edgecolor='#66717C',linewidth=.7))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#4A535B','lw':1})
def clean(ax):ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
def title(ax,s):ax.text(0,1,s,ha='left',va='top',weight='bold',fontsize=9)
def bars(ax,a,ys,marker,color,label=None):
    # Percentile intervals need not contain the sample estimate in general.
    ax.hlines(ys,a.lower,a.upper,color=color,lw=.9)
    ax.scatter(a.relative_loss_pct,ys,marker=marker,s=15,color=color,label=label,zorder=3)
def zeroline(ax):ax.axvline(0,c='#888888',ls=':',lw=.7)

def f1():
    from figure1_design import draw_figure1
    draw_figure1(OUT)


def f2():
    d=pd.read_csv(DATA/'figure2_paired.csv');fig=plt.figure(figsize=(7.2,8.1));outer=fig.add_gridspec(3,2,left=.10,right=.985,bottom=.105,top=.905,hspace=.60,wspace=.31)
    for si,src in enumerate(SOURCES):
        for fi,fam in enumerate(FAMS):
            grid=outer[si,fi].subgridspec(1,2,width_ratios=[1.65,1],wspace=.18)
            ax=fig.add_subplot(grid[0]);sec=fig.add_subplot(grid[1],sharey=ax)
            sub=d[(d.source==src)&(d.family==fam)];order=sub[['H','U']].drop_duplicates().sort_values(['H','U']);labels=[f'H{int(r.H)}/U{int(r.U)}' for r in order.itertuples()]
            for method,marker,off in [('R2_reuse','o',-.12),('R3_isolate','s',.12)]:
                a=sub[sub.candidate==method].sort_values(['H','U']);bars(ax,a,np.arange(6)+off,marker,COL[method])
            a=sub[sub.candidate=='R5_multiqt_nonnegative'].sort_values(['H','U']);bars(sec,a,np.arange(6),'^',COL['R5_multiqt_nonnegative'])
            for v,methods in [(ax,['R2_reuse','R3_isolate']),(sec,['R5_multiqt_nonnegative'])]:
                z=d[(d.source==src)&d.candidate.isin(methods)];lo=min(0,z.lower.min());hi=max(0,z.upper.max());pad=max(.025,(hi-lo)*.08);v.set_xlim(lo-pad,hi+pad);zeroline(v);v.set_xlabel('Loss change (%)',fontsize=7)
            ax.set_yticks(range(6),labels);ax.set_ylim(5.5,-.5);sec.tick_params(left=False,labelleft=False)
            ax.text(0,1.19,f'{chr(97+si*2+fi)}  {NAMES[src]} | {FAMS[fam]}',transform=ax.transAxes,fontsize=9,weight='bold')
            ax.set_title('Primary contrasts',loc='left',fontsize=7.5);sec.set_title('R5 − Raw',fontsize=7.5)
    handles=[Line2D([0],[0],marker='o',color=COL['R2_reuse'],ls='',label='R2 − Raw'),Line2D([0],[0],marker='s',color=COL['R3_isolate'],ls='',label='R3 − R2'),Line2D([0],[0],marker='^',color=COL['R5_multiqt_nonnegative'],ls='',label='R5 − Raw (secondary)')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=7.5,bbox_to_anchor=(.53,.001));save(fig,2)

def f3():
    d=pd.read_csv(DATA/'figure3_delay.csv');a=pd.read_csv(DATA/'figure3_activation.csv')
    fig,axs=plt.subplots(3,2,figsize=(7.2,7.7));fig.subplots_adjust(left=.11,right=.97,top=.95,bottom=.115,hspace=.45,wspace=.28)
    for i,s in enumerate(SOURCES):
        H=2 if s=='dominicks' else 14
        for j,ax in enumerate(axs[i]):
            for m,mark in [('R2_reuse','o'),('R3_isolate','s')]:
                for kind,ls in [('calendar','-'),('mature','--')]:
                    if j==0:
                        z=d[(d.source==s)&(d.candidate==m)&(d.window_kind==kind)].sort_values('delay');x=np.r_[0,z.delay/H];y=np.r_[0,z.relative_loss_pct]
                    else:
                        z=a[(a.source==s)&(a.method==m)&(a.window_kind==kind)].sort_values('delay');x=z.delay/H;y=z.active_fraction
                    ax.plot(x,y,marker=mark,ms=3.5,ls=ls,lw=1,c=COL[m],label=('Reuse' if m=='R2_reuse' else 'Isolation')+' / '+kind)
            ax.axvline(1,c='#777777',ls=':',lw=.8);ax.set_xticks([0,.5,1,2]);ax.set_xlabel('Additional feedback delay d/H')
            ax.set_title(f'{chr(97+2*i+j)}  {NAMES[s]}',loc='left',weight='bold')
            ax.set_ylabel('Loss change vs d = 0 (%)' if j==0 else 'Active origin fraction')
            if j:ax.set_ylim(-.03,1.03)
            else:ax.axhline(0,c='#999999',ls=':',lw=.7)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,fontsize=7.5,bbox_to_anchor=(.5,.003));save(fig,3)

def f4():
    d=pd.read_csv(DATA/'figure4_coverage.csv');fig,axs=plt.subplots(3,2,figsize=(7.2,7.7));fig.subplots_adjust(left=.19,right=.98,top=.95,bottom=.085,hspace=.45,wspace=.35)
    methods=['R0_raw','R2_reuse','R3_isolate','R5_multiqt_nonnegative']
    for i,s in enumerate(SOURCES):
        for j,q in enumerate([90,95]):
            ax=axs[i,j];labs=[]
            for k,(f,m) in enumerate((f,m) for f in FAMS for m in methods):
                r=d[(d.source==s)&(d.family==f)&(d.method==m)].iloc[0];lo=100*r[f'q{q}_strict']-q;hi=100*r[f'q{q}_weak']-q
                ax.plot([lo,hi],[k,k],c=COL[m],lw=1.1);ax.scatter(lo,k,facecolor='white',edgecolor=COL[m],s=18,zorder=4);ax.scatter(hi,k,c=COL[m],s=11,zorder=5)
                labs.append(('GBM' if f=='lgbm' else 'MLP')+' '+('Raw' if m=='R0_raw' else m[:2]))
            ax.set_yticks(range(8),labs);ax.set_ylim(7.6,-.6);zeroline(ax);ax.set_title(f'{chr(97+2*i+j)}  {NAMES[s]} | q{q}',loc='left',weight='bold');ax.set_xlabel('Coverage minus nominal (pp)')
    fig.text(.5,.018,'Open: strict coverage     Filled: weak coverage     Segment: ties, not uncertainty',ha='center',fontsize=7.5);save(fig,4)

def f5():
    d=pd.read_csv(DATA/'figure5_training.csv');fig=plt.figure(figsize=(7.2,6.1));gs=fig.add_gridspec(2,2,height_ratios=[1.1,1],left=.18,right=.98,top=.94,bottom=.10,hspace=.55,wspace=.5)
    for i,H in enumerate([7,14]):
        ax=fig.add_subplot(gs[0,i]);pairs=[(h,m) for h in [42,84] for m in ['R2_reuse','R3_isolate']]
        for budget,mark,off,color in [('original_epochs','o',-.13,'#0072B2'),('equal_steps','s',.13,'#D55E00')]:
            a=pd.concat([d[(d.H==H)&(d.history==h)&(d.budget==budget)&(d.candidate==m)&(d.reference=='R0_raw')] for h,m in pairs]);bars(ax,a,np.arange(4)+off,mark,color)
        ax.set_yticks(range(4),[f'{h} d | {m[:2]}' for h,m in pairs]);ax.set_ylim(3.5,-.5);zeroline(ax);ax.set_title(f'{chr(97+i)}  H{H} | vs own Raw',loc='left',weight='bold');ax.set_xlabel('Relative loss change (%)')
    ax=fig.add_subplot(gs[1,:]);a=d[(d.comparison=='optimization_budget')&(d.candidate=='R0_raw')].sort_values(['H','history']);bars(ax,a,np.arange(4),'D','#555555');ax.set_yticks(range(4),[f'H{int(r.H)} | {int(r.history)} d' for r in a.itertuples()]);ax.set_ylim(3.5,-.5);zeroline(ax);ax.set_title('c  Matched-update Raw versus original-budget Raw',loc='left',weight='bold');ax.set_xlabel('Relative loss change (%)')
    fig.legend(handles=[Line2D([0],[0],color='#0072B2',marker='o',ls='',label='Original epochs'),Line2D([0],[0],color='#D55E00',marker='s',ls='',label='Matched optimizer updates')],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.005),fontsize=7.5);save(fig,5)

def f6():
    d=pd.read_csv(DATA/'figure6_multiqt.csv');fig=plt.figure(figsize=(7.2,7.9));gs=fig.add_gridspec(2,2,height_ratios=[1.05,2.2],left=.13,right=.98,top=.97,bottom=.07,hspace=.26,wspace=.5)
    ax=fig.add_subplot(gs[0,:]);clean(ax);title(ax,'a  Hold the base forecasts fixed and change the output interface')
    for i,(name,text,foot) in enumerate([('Q0','Untruncated forecast\n→ issue and feed back','Untruncated recursion'),('Q1','Same Q0 shadow path\n→ issue max(Q0, 0)','Output-only truncation'),('Q2','Nonnegative forecast\n→ issue and feed back','Issued-feedback recursion')]):
        x=.01+i*.33;box(ax,x,.24,.3,.54,name+'\n\n'+text,fs=8);ax.text(x+.15,.10,foot,ha='center',fontsize=7.5)
    ax.text(.5,-.035,'All retain the task-specific normalization, base forecasts, projection and step sizes.',ha='center',fontsize=7.2)
    for j,(ca,re,label) in enumerate([('Q1_output_clamp_shadow','Q0_unclamped','b  Q1 − Q0 | three seeds'),('Q2_output_clamp_played','Q1_output_clamp_shadow','c  Q2 − Q1 | three seeds')]):
        ax=fig.add_subplot(gs[1,j]);a=d[(d.candidate==ca)&(d.reference==re)].copy();a['source']=pd.Categorical(a.source,SOURCES,ordered=True);a=a.sort_values(['source','family','H']);bars(ax,a,np.arange(12),'o','#0072B2' if j==0 else '#D55E00');labs=[f"{ {'m5':'M5','retail':'Retail','dominicks':'Dom.'}[str(r.source)]} {('GBM' if r.family=='lgbm' else 'MLP')} H{int(r.H)}" for r in a.itertuples()];ax.set_yticks(range(12),labs);ax.set_ylim(11.6,-.6);zeroline(ax);ax.set_title(label,loc='left',weight='bold');ax.set_xlabel('Relative loss change (%)')
    save(fig,6)


def primary_effects():
    d=pd.read_csv(ROOT/'supplementary_data/Data1_primary_contrasts.csv')
    fig,axs=plt.subplots(3,2,figsize=(7.2,6.9))
    fig.subplots_adjust(left=.115,right=.98,top=.955,bottom=.115,hspace=.48,wspace=.28)
    for i,src in enumerate(SOURCES):
        common=d[d.source==src];lo=min(0,common.lower.min());hi=max(0,common.upper.max());pad=max(.04,(hi-lo)*.08)
        for j,fam in enumerate(FAMS):
            ax=axs[i,j];z=d[(d.source==src)&(d.family==fam)]
            settings=z[['H','U']].drop_duplicates().sort_values(['H','U'])
            for method,marker,off in [('R2_reuse','o',-.13),('R3_isolate','s',.13)]:
                a=z[z.candidate==method].sort_values(['H','U'])
                bars(ax,a,np.arange(len(a))+off,marker,COL[method])
            ax.set_yticks(range(len(settings)),[f'H{int(r.H)}/U{int(r.U)}' for r in settings.itertuples()])
            ax.set_ylim(5.5,-.5);ax.set_xlim(lo-pad,hi+pad);zeroline(ax)
            ax.set_title(f'{chr(97+2*i+j)}  {NAMES[src]} | {FAMS[fam]}',loc='left',weight='bold')
            ax.set_xlabel('Joint loss change (%)')
    fig.legend(handles=[Line2D([0],[0],color=COL['R2_reuse'],marker='o',ls='',label='R2 group reuse − R0 Raw'),Line2D([0],[0],color=COL['R3_isolate'],marker='s',ls='',label='R3 isolation − R2 group reuse')],loc='lower center',bbox_to_anchor=(.5,.012),ncol=2,frameon=False,fontsize=8)
    save(fig,'primary')

if __name__=='__main__':
    for f in [f1,f2,f3,f4,f5,primary_effects]:f()
    print('Earlier core figures saved; the controlled-analysis step supplies main Figure 4 as PNG, PDF and editable-text SVG; no new scores.')
