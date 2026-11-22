import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
seeds = (42,43,44)
results = {}
for seed in seeds:
    subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--seed',str(seed)],check=True)
    results[seed] = json.loads((HERE/'runs'/f'seed_{seed}'/'results.json').read_text())
summary = {}
for arm in ('six_gan','six_real_only','all_gan','all_real_only'):
    rows = [results[s][arm][-1] for s in seeds]
    summary[arm] = {k:[r[k] for r in rows] for k in ('seen','held_out','subset_six','other_fourteen','held_out_macro')}
(HERE/'summary.json').write_text(json.dumps({'seeds':seeds,'epoch':3,'baseline':results[42]['baseline'],'arms':summary},indent=2))
