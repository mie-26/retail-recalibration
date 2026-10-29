"""Verify all published files before running any numerical analysis (standard library only)."""
from pathlib import Path
import hashlib
import json
import sys


def verify(root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    for record in manifest['files']:
        p = (root / record['path']).resolve()
        if not p.is_file() and record.get('optional', False) and p.is_relative_to(root):
            print('NOTE: optional Git configuration absent: ' + record['path'])
            continue
        if not p.is_relative_to(root) or not p.is_file():
            failures.append(record['path'] + ': missing or invalid path')
            continue
        with p.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if p.stat().st_size != record['bytes'] or actual != record['sha256']:
            failures.append(record['path'] + ': checksum mismatch')
    if failures:
        raise RuntimeError('\n'.join(failures))
    print(f"PASS: {len(manifest['files'])} manifest entries checked (mandatory files plus any present optional Git configuration).")
    print('The manifest is an integrity check, not a digital signature or independent validation.')
    return len(manifest['files'])


if __name__ == '__main__':
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
