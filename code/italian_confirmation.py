"""Restore Italian confirmation inputs and audit saved scores without fitting."""
from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,subprocess,sys,shutil,os

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    subprocess.run([sys.executable,str(ROOT/'code/audit_italian_core.py')],cwd=ROOT,check=True)
    dest=ROOT/'reproduced/figures';dest.mkdir(exist_ok=True)
    # The preceding controlled step draws its original discovery figure.
    for ext in ['png','pdf','svg']:shutil.copy2(dest/f'Figure_4.{ext}',dest/f'S9_discovery_sharing.{ext}')
    subprocess.run([sys.executable,str(ROOT/'code/plot_italian_figures.py'),str(dest)],cwd=ROOT,check=True)
    report=json.loads((ROOT/'italian/outputs/sr_external_italy_20261028/release_audit/AUDIT.json').read_text(encoding='utf8'))
    (ROOT/'reproduced/checks/italian_confirmation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('ITALIAN_CORE_PASS: declared saved-score subset; no new model fits.',flush=True)

if __name__=='__main__':main()
