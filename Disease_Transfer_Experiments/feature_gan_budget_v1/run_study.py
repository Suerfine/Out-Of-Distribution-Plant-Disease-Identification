import json
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
seeds = (42,43,44)
results = []
for seed in seeds:
    subprocess.run([sys.executable,'-u',str(here/'train.py'),'--seed',str(seed)],check=True)
    results.append(json.loads((here/'runs'/f'seed_{seed}'/'results.json').read_text()))
summary = {'seeds':seeds,'epoch':3,'baseline':results[0]['baseline'],'budgets':{}}
for budget in (48,96,192):
    summary['budgets'][budget] = {
        arm:{metric:[r[f'{budget}_{arm}'][-1][metric] for r in results]
             for metric in ('seen','held_out','held_out_macro')}
        for arm in ('gan','real_only')}
(here/'summary.json').write_text(json.dumps(summary,indent=2))
