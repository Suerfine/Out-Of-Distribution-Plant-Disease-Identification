"""Start expanded training only after the prior supervisor records completion."""
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cnn_detail_base_v1'))
from storage import atomic_json, now
from supervisor import run_controller
import msvcrt

if __name__ == '__main__':
    prior = HERE.parent/'lesion_copy_paste_base_v1/full_run'
    run = HERE/'full_run'
    with (run/'queue.lock').open('a+b') as lock:
        if not lock.seek(0,2):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        while True:
            state = json.loads((prior/'supervisor_status.json').read_text())['state']
            if state == 'complete':
                history = json.loads((prior/'history.json').read_text())
                assert len(history) == 15 and history[-1]['epoch'] == 15
                break
            if state != 'running' or (run/'STOP.request').exists():
                atomic_json(run/'queue_status.json',{'phase':'blocked','previous':state,'utc':now()})
                sys.exit(0)
            atomic_json(run/'queue_status.json',{'phase':'waiting','pid':os.getpid(),'utc':now()})
            time.sleep(15)
        atomic_json(run/'queue_status.json',{'phase':'released','pid':os.getpid(),'utc':now()})
        cfg = {'max_restarts':3,'supervisor_hours':24,'min_free_bytes':6*1024**3,
               'watchdog_seconds':900,'append_micro':False}
        run_controller([sys.executable,'-u',str(HERE/'train_full.py')],run,cfg)
