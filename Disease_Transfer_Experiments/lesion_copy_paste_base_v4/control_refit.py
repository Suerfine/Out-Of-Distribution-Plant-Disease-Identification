from refit_all import HERE
from supervisor import run_controller
import subprocess
import sys
import json

if __name__ == '__main__':
    run = HERE/'full_data_refit'
    cfg = {'max_restarts':2,'supervisor_hours':24,'min_free_bytes':6*1024**3,
           'watchdog_seconds':900,'append_micro':False}
    run_controller([sys.executable,'-u',str(HERE/'refit_all.py')],run,cfg)
    if json.loads((run/'status.json').read_text())['phase']=='complete':
        subprocess.run([sys.executable,'-u',str(HERE/'evaluate_refit.py')],check=True)
