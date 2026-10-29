"""Single entry point for the supplied numerical reproduction, without model training."""
from pathlib import Path
import argparse, datetime, importlib.metadata, json, platform, subprocess, sys, time

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)

    a=ap.parse_args();out=ROOT/'reproduced';(out/'logs').mkdir(parents=True,exist_ok=True)
    steps=[('integrity',['code/verify_package.py']),('restore_data',['code/prepare_data.py']),('feedback_tests',['code/test_methods.py']),('representative_models',['code/check_models.py']),
      ('core_scores_and_reference_intervals',['code/audit_scores.py']),
      ('main_figures',['code/plot_figures.py']),
      ('supplement_figures',['robustness/code/plot_supplement.py','--root','robustness']),('controlled_analyses',['code/controlled_analysis.py']),('italian_confirmation',['code/italian_confirmation.py']),('output_consistency',['code/verify_results.py'])]
    record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mode='compact-core',python=sys.version,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib']},steps=[],passed=False,new_model_fits=0)
    start=time.perf_counter()
    for name,args in steps:
        print('RUN',name,flush=True);before=time.perf_counter();log=out/'logs'/f'{name}.txt'
        with log.open('w',encoding='utf8') as f:
            proc=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,encoding='utf8')
        record['steps'].append(dict(name=name,command=['python',*args],returncode=proc.returncode,seconds=time.perf_counter()-before,log=log.relative_to(ROOT).as_posix()))
        record['seconds']=time.perf_counter()-start
        (out/'RUN_SUMMARY.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        print(('PASS' if proc.returncode==0 else 'FAIL'),name,f"{record['steps'][-1]['seconds']:.2f}s",flush=True)
        if proc.returncode:
            print(log.read_text(encoding='utf8'),flush=True);raise SystemExit(proc.returncode)
    record['passed']=True;record['seconds']=time.perf_counter()-start
    (out/'RUN_SUMMARY.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print('PASS: compact core reproduction completed (scope and exclusions in SCOPE.json). See reproduced/RUN_SUMMARY.json',flush=True)

if __name__=='__main__':main()
