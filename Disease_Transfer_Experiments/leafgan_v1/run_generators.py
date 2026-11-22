"""Run the registered generators sequentially and leave previews for review."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
config = json.loads((HERE/'study.json').read_text(encoding='utf-8-sig'))
for index, pair in enumerate(config['pairs']):
    slug = f"{pair['crop']}_{pair['disease']}"
    status = HERE/'multi_pair'/slug/'generator'/'status.json'
    if status.exists() and json.loads(status.read_text())['phase'] in ('complete','skipped_insufficient_masks'):
        continue
    subprocess.run([sys.executable,'-u',str(HERE/'train_multi.py'),'--pair',str(index)],check=True)

