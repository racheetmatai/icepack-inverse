"""Audit tracked source files before making a reproduction release."""
import ast
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
failures = []
python_count = 0
total = 0
for name in filter(None, names):
    path = root / name
    size = path.stat().st_size
    total += size
    if size >= 100 * 1024 * 1024:
        failures.append(f'File exceeds 100 MiB: {name}')
    if path.suffix == '.py':
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
        python_count += 1
    if path.suffix in {'.py', '.md', '.sh', '.json', '.yml', '.cff'}:
        content = path.read_text(encoding='utf-8-sig')
        if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----', content):
            failures.append(f'Private key marker: {name}')
        if re.search(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}', content):
            failures.append(f'Credential marker: {name}')
result = {'passed': not failures, 'python_files_parsed': python_count,
          'tracked_bytes': total, 'failures': failures}
print(json.dumps(result, indent=2))
if failures:
    raise SystemExit(1)
