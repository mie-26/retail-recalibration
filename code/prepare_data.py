"""Verify and safely restore the compact scientific inputs. Standard library only."""
from pathlib import Path,PurePosixPath
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    specs=json.loads((ROOT/'CORE_ARCHIVES.json').read_text(encoding='utf8'))['archives'];count=0
    for spec in specs:
        arc=ROOT/spec['archive'];assert arc.parent==ROOT
        assert arc.stat().st_size==spec['bytes'] and sha(arc.read_bytes())==spec['sha256']
        rows={r['path']:r for r in spec['files']};seen=set()
        with tarfile.open(arc,'r|xz') as z:
            for m in z:
                n=m.name;r=rows[n];p=(ROOT/n).resolve();rel=PurePosixPath(n)
                assert m.isfile() and n not in seen and not rel.is_absolute() and '..' not in rel.parts and p.is_relative_to(ROOT)
                b=z.extractfile(m).read();assert len(b)==r['bytes'] and sha(b)==r['sha256']
                if p.exists():assert p.is_file() and sha(p.read_bytes())==r['sha256'],n
                else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
                seen.add(n)
        assert seen==set(rows);count+=len(seen)
    print('PASS: compact data restored and verified:',count,'files')
if __name__=='__main__':main()
