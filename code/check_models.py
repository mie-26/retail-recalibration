"""Check representative original weights on saved feature/prediction anchors. CPU only."""
from pathlib import Path
import hashlib,json
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]

def main():
    base=ROOT/'models';spec=json.loads((base/'INDEX.json').read_text(encoding='utf8'));checks=[]
    for item in spec['checkpoints']:
        for r in item['files']:
            p=base/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        folder=base/item['id']
        with np.load(folder/'inference_anchor.npz',allow_pickle=False) as a:
            x=a['x'];scale=a['scale'];expected=a['expected']
        if item['family']=='lgbm':
            q=np.column_stack([lgb.Booster(model_file=str(folder/f'q{tau}.txt')).predict(x,num_threads=1) for tau in [90,95]])
        else:
            # Exact float32 arrays exported from the included original PyTorch checkpoint.
            with np.load(folder/'weights.npz',allow_pickle=False) as w:
                y=np.clip((x-w['mean'])/w['sd'],-20,20)
                for layer in [0,2]:y=np.maximum(y@w[f'{layer}.weight'].T+w[f'{layer}.bias'],np.float32(0))
                y=y@w['4.weight'].T+w['4.bias']
                q=np.where(y>20,y,np.logaddexp(np.float32(0),y))
        actual=q*scale[:,None]
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
        checks.append(dict(id=item['id'],rows=len(x),max_absolute_error=float(np.max(np.abs(actual-expected))),passed=True))
    result=dict(passed=True,checkpoints=len(checks),logical_models=sum(i['models'] for i in spec['checkpoints']),checks=checks,new_model_fits=0,rtol=1e-5,atol=1e-5,scope=spec['scope'])
    p=ROOT/'reproduced/checks/model_inference.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
