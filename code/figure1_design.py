"""Restore the exact published PPT-authored mechanism schematic (not an empirical plot)."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
def draw_figure1(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    assets=ROOT/'figure_assets'
    checks=json.loads((assets/'EXPORTS.json').read_text())
    for name,digest in checks.items():
        p=assets/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Figure 1 export checksum mismatch: '+name)
        if p.suffix in {'.png','.pdf','.svg'}:shutil.copy2(p,output/name)
    return output
if __name__=='__main__':
    draw_figure1(ROOT/'reproduced/figures')
    print('Verified and restored PPT-authored Figure 1; no model fitting or score changes.')
