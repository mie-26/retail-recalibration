"""Check numerical links from final plot/table inputs to delivered analysis tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def equal(a,b,keys,tol=1e-9):
    a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[b.columns],b,check_dtype=False,check_exact=False,atol=tol,rtol=tol)

def main():
    keys=['source','H','U','family','candidate','reference','analysis']
    e=pd.read_csv(ROOT/'robustness/tables/paper_effects.csv');f=pd.read_csv(ROOT/'figure_source_data/figure6_multiqt.csv')
    equal(f,e[e.analysis=='three_seed_mechanism'],keys,1e-12);assert len(f)==48 and f.seed_count.eq(3).all()
    derived={'paper_effects.csv':keys,'paper_parameter_summary.csv':['source','method'],'paper_recursive_grid.csv':['source','candidate'],'paper_three_seed_mechanism.csv':keys}
    for name,k in derived.items():equal(pd.read_csv(ROOT/'reproduced/robustness_tables'/name),pd.read_csv(ROOT/'robustness/tables'/name),k)
    # Supplementary Table S1 retains the complete policy-loss summaries.
    ef=pd.read_csv(ROOT/'supplementary_data/Data1_core_effects.csv')
    t=pd.read_csv(ROOT/'figure_source_data/table2_summary.csv')
    for row in t.itertuples():
        a=ef[(ef.source==row.source)&(ef.family==row.family)&(ef.candidate==row.method)&(ef.reference=='R0_raw')].relative_loss_pct
        np.testing.assert_allclose([a.mean(),a.min(),a.max(),len(a)],[row.mean,row.minimum,row.maximum,row.configurations],rtol=1e-9,atol=1e-9)
    primary=pd.read_csv(ROOT/'figure_source_data/table2_primary_summary.csv')
    cp=pd.read_csv(ROOT/'supplementary_data/Data1_primary_contrasts.csv')
    for row in primary.itertuples():
        a=cp[(cp.source==row.source)&(cp.family==row.family)&(cp.candidate==row.candidate)&(cp.reference==row.reference)]
        np.testing.assert_allclose([a.relative_loss_pct.mean(),a.relative_loss_pct.min(),a.relative_loss_pct.max(),a.q90_weak_difference.mean(),a.q95_weak_difference.mean(),len(a)],[row.mean_loss_pct,row.min_loss_pct,row.max_loss_pct,row.mean_q90_weak_pp,row.mean_q95_weak_pp,row.settings],rtol=1e-9,atol=1e-9)
    names=[f'Figure_{i}' for i in range(1,5)]+['S1_complete_policy_effects','S2_parameter_sensitivity','S3_coverage','S4_training_budget','S5_three_seed_feedback','S6_simulation','S7_selection_history','S8_residual_reference','S9_discovery_sharing','S10_external_interactions','S11_external_feedback']
    for name in names:
        for ext in ['png','pdf','svg']:
            p=ROOT/'reproduced/figures'/f'{name}.{ext}';assert p.is_file() and p.stat().st_size>0,p
    for name in ['E2_quantile_sharing_intervals.csv','E2_interaction_intervals.csv','E3_joint_contrasts.csv']:
        expected=pd.read_csv(ROOT/'figure_source_data'/('controlled_'+name))
        actual=pd.read_csv(ROOT/'reproduced/controlled'/name)
        pd.testing.assert_frame_equal(expected,actual,check_exact=False,rtol=1e-10,atol=1e-10)
    italy=json.loads((ROOT/'reproduced/checks/italian_confirmation.json').read_text(encoding='utf8'));assert italy['passed'] and italy['identity_rows']==34416
    result=dict(passed=True,three_seed_figure_rows=48,three_seed_count=3,rebuilt_summary_tables_compared=len(derived),main_table2_rows=len(primary),supplement_table1_rows=len(t),figures_generated=len(names),formats_per_figure=3)
    p=ROOT/'reproduced/checks/result_consistency.json';p.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
