import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cnn_detail_base_v1'))
from supervisor import run_controller

if __name__ == '__main__':
    cfg = {'max_restarts':3,'supervisor_hours':24,'min_free_bytes':6*1024**3,
           'watchdog_seconds':900,'append_micro':False}
    run_controller([sys.executable,'-u',str(HERE/'train_full.py')],HERE/'full_run',cfg)
