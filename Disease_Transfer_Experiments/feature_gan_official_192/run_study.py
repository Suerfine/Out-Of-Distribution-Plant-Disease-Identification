import json
from pathlib import Path
import subprocess
import sys
p = Path(__file__).resolve().parent
results = {}
for seed in (42,43,44):
    run = p/'runs'/f'seed_{seed}'
    if not (run/'results.json').exists():
        subprocess.run([sys.executable,'-u',str(p/'train.py'),'--seed',str(seed)],check=True)
    results[seed] = json.loads((run/'results.json').read_text())
(p/'summary.json').write_text(json.dumps(results,indent=2))
