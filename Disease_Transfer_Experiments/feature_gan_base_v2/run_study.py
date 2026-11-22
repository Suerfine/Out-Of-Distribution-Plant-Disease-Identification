import json
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
seeds = (42, 43, 44)
for seed in seeds:
    subprocess.run([sys.executable,'-u',str(here/'train.py'),'--seed',str(seed)],check=True)
runs = [json.loads((here/'runs'/f'seed_{seed}'/'results.json').read_text()) for seed in seeds]
summary = {'seeds':seeds, 'epoch':3, 'baseline':runs[0]['baseline'], 'budgets':{}}
for budget in (32,96,192):
    arms = {}
    for arm in ('gan','real_only'):
        rows = [r[f'{budget}_{arm}'][-1] for r in runs]
        arms[arm] = {k:{'correct_by_seed':[r[k]['correct'] for r in rows], 'images':rows[0][k]['images']}
                     for k in rows[0]}
    summary['budgets'][budget] = arms
(here/'summary.json').write_text(json.dumps(summary,indent=2))
