#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the four official analysis modules from the project's GitHub branch.
Requires internet on first run; existing source files are not overwritten.
"""
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request
import sys
BASE = 'https://raw.githubusercontent.com/aishanchung/Statistical-Practice/'
BRANCH = '資料欄位相關文件'
FILES = [
    'vd_imputation_experiment.py',
    'vd_spatial_methods_benchmark.py',
    'vd_gp_kriging_experiment.py',
    'vd_robust_mask_report.py',
]
folder = Path(__file__).resolve().parent
for name in FILES:
    dest = folder / name
    if dest.exists() and dest.stat().st_size > 100:
        print('Already present:', name)
        continue
    url = BASE + quote(BRANCH, safe='') + '/analysis/' + name
    try:
        req = Request(url, headers={'User-Agent':'VD-validation-package/1.0'})
        with urlopen(req, timeout=35) as reply:
            content = reply.read()
        if len(content) < 100 or not content.startswith(b'#!/usr/bin/env python3'):
            raise RuntimeError('Unexpected downloaded contents')
        dest.write_bytes(content)
        print('Downloaded:', name, len(content), 'bytes')
    except Exception as exc:
        print('\nERROR: Could not download', name, '\n', exc, file=sys.stderr)
        print('Download it manually from the GitHub analysis folder and put it beside this script.', file=sys.stderr)
        sys.exit(1)
print('\nAll four source modules are present.')
