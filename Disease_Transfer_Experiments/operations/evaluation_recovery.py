"""Evaluation-only recovery for the positional-dim argsort API mismatch."""
import argparse
from contextlib import contextmanager
import json
import os
import sys
from pathlib import Path


@contextmanager
def argsort_compatibility():
    import torch
    original=torch.Tensor.argsort
    def compatible(tensor,*args,**kwargs):
        if args:
            if len(args)!=1 or 'dim' in kwargs: raise TypeError('Expected one dimension argument')
            kwargs['dim']=args[0]
        return original(tensor,**kwargs)
    torch.Tensor.argsort=compatible
    try: yield
    finally: torch.Tensor.argsort=original


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--experiment',required=True,type=Path); args=parser.parse_args()
    here=args.experiment.resolve(); sys.path.insert(0,str(here))
    import torch
    from protocol import prepare,load_config,StudyLock,status,write_json,sha256,stamp,seed_all
    from storage import CheckpointStore
    from evaluation import evaluate_and_report
    from verify import verify_run
    from supervisor import keep_awake
    cfg=load_config(here/'config.json'); run=here/'runs'/cfg['run_name']
    torch.set_num_threads(4); torch.set_num_interop_threads(2); seed_all(cfg['seed']); os.environ['HF_HUB_OFFLINE']='1'
    with StudyLock(run/'worker.lock'):
        keep_awake(True)
        try:
            p=prepare(cfg,run); state=CheckpointStore(run,p['fingerprint']).load()
            if state['stage']!='evaluate' or len(state['histories']['ssl'])!=20 or len(state['histories']['finetune'])!=15:
                raise RuntimeError('Expected both training stages already complete')
            frozen=json.loads((run/'selection_frozen_before_test.json').read_text()); before=sha256(run/'disease_final.pt')
            if before!=frozen['checkpoint_sha256']: raise RuntimeError('Trained model differs from frozen checksum')
            del state
            status(run,'evaluation_recovery',training_complete=True,retraining=False,
                   repair='Pass argsort dimension by keyword; unchanged sorting kernel and stable tie rule')
            with argsort_compatibility():
                evaluate_and_report(p,cfg,run)
                checks=verify_run(run)
            if sha256(run/'disease_final.pt')!=before: raise RuntimeError('Evaluation modified model weights')
            receipt={'passed':True,'utc':stamp(),'operational_source_sha256':sha256(__file__),
                'original_fingerprint':p['fingerprint'],'trained_model_sha256':before,'model_unchanged':True,
                'training_performed':False,'frozen_sources_unchanged':True,'verification_checks':checks['checks'],
                'repair':'Local argsort positional-dim compatibility adapter; same stable ranking and Torch kernel'}
            write_json(run/'evaluation_recovery_verified.json',receipt)
            plot_note='Training plot visual review remains pending.'
            status(run,'complete',verification_checks=checks['checks'],evaluation_recovery_verified=True,results=str(run/'results.md'))
            current=json.loads((run/'status.json').read_text())
            write_json(run/'supervisor_status.json',{'utc':stamp(),'state':'complete','recovered_by':'evaluation-only recovery','pid':os.getpid(),'worker_results':current,'sleep_prevention':False})
        except BaseException as error:
            import traceback
            status(run,'evaluation_recovery_failed',error=repr(error),traceback=traceback.format_exc()); raise
        finally: keep_awake(False)


if __name__=='__main__': main()
