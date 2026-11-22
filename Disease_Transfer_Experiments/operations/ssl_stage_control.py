"""User-requested pause after SSL; preserve the frozen numerical training code."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


class SSLBoundaryPause(Exception): pass


def verify_ssl_boundary(run,recipe,cfg,encoder):
    import torch
    from storage import CheckpointStore,IntegrityError,sha,atomic_json,now
    from protocol import tensor_hash
    run=Path(run); state=CheckpointStore(run,recipe['fingerprint']).load()
    if state['stage']!='ssl' or state['epoch']!=cfg['ssl_epochs'] or state['cursor']!=0:
        raise IntegrityError('SSL stop hook reached before the complete fixed SSL budget')
    if len(state['histories']['ssl'])!=cfg['ssl_epochs'] or state['histories']['finetune']:
        raise IntegrityError('Unexpected stage histories at SSL boundary')
    path=run/'ssl_encoder.pt'; saved=torch.load(path,map_location='cpu',weights_only=True)
    expected=tensor_hash(encoder)
    stored_encoder={k.removeprefix('encoder.'):v for k,v in state['model'].items() if k.startswith('encoder.')}
    if saved['epochs']!=cfg['ssl_epochs'] or saved['fingerprint']!=recipe['fingerprint'] or tensor_hash(saved['encoder'])!=expected or tensor_hash(stored_encoder)!=expected:
        raise IntegrityError('Saved SSL encoder does not match the final SSL checkpoint')
    value={'passed':True,'utc':now(),'reason':'User requested stop after SSL before supervised training',
        'ssl_epochs_completed':cfg['ssl_epochs'],'supervised_epochs_completed':0,'encoder_file':str(path),
        'encoder_sha256':sha(path),'encoder_state_hash':expected,'original_fingerprint':recipe['fingerprint'],
        'operational_source_sha256':sha(__file__),'frozen_training_sources_unchanged':True,'supervised_start_authorized':False}
    atomic_json(run/'ssl_stage_pause_verified.json',value)
    return value


def install_boundary_guard(train,run):
    original=train.run_training
    def do_not_start_supervised(recipe,cfg,encoder):
        value=verify_ssl_boundary(run,recipe,cfg,encoder)
        train.status(run,'paused',stage='ssl_complete',epoch=cfg['ssl_epochs'],supervised_epochs=0,
                     encoder=value['encoder_file'],encoder_sha256=value['encoder_sha256'])
        raise SSLBoundaryPause()
    train.initialise_finetune=do_not_start_supervised
    def ssl_only(*args,**kwargs):
        try: return original(*args,**kwargs)
        except SSLBoundaryPause: return 'paused'
    train.run_training=ssl_only


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['launch','supervise','worker'])
    parser.add_argument('--experiment',required=True,type=Path); parser.add_argument('--micro-batch',type=int)
    args=parser.parse_args(); here=args.experiment.resolve(); sys.path.insert(0,str(here))
    from storage import atomic_json,sha,now,IntegrityError
    from supervisor import native_python_environment,run_controller
    cfg=json.loads((here/'config.json').read_text()); run=here/'runs'/cfg['run_name']
    request=run/'PAUSE_AFTER_SSL.request.json'
    if not request.exists(): raise RuntimeError('Missing explicit user stage-pause request')
    if args.mode=='launch':
        status=json.loads((run/'supervisor_status.json').read_text())
        if status['state']!='paused': raise RuntimeError('Original supervisor must finish its safe pause first')
        if (run/'ssl_stage_pause_verified.json').exists():
            print('SSL is already complete and paused; no training started.'); return
        stop=run/'STOP.request'
        if stop.exists(): stop.replace(run/f'STOP.consumed_for_ssl_only_{time.time_ns()}.json')
        stdout=run/f'ssl_only_supervisor_{time.time_ns()}.stdout.log'; stderr=stdout.with_name(stdout.name.replace('stdout','stderr'))
        native,env=native_python_environment()
        flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB
        with stdout.open('w') as out,stderr.open('w') as err:
            child=subprocess.Popen([native,'-u',str(Path(__file__).resolve()),'supervise','--experiment',str(here)],
                cwd=here,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags)
        atomic_json(run/'ssl_only_launch.json',{'utc':now(),'supervisor_pid':child.pid,'stdout':str(stdout),'stderr':str(stderr),
            'operational_source':str(Path(__file__).resolve()),'operational_sha256':sha(__file__),
            'user_requested_boundary':'after20 SSL epochs; before supervised training','frozen_recipe_changed':False})
        print(json.dumps({'launched_ssl_only_supervisor':child.pid,'stops_after_ssl':True}),flush=True)
        return
    if args.mode=='supervise':
        import msvcrt
        with (run/'supervisor.lock').open('a+b') as lock:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            try: run_controller([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--experiment',str(here)],run,cfg)
            except BaseException as error:
                atomic_json(run/'supervisor_status.json',{'utc':now(),'state':'needs_attention','pid':os.getpid(),'error':repr(error),'checkpoints_preserved':True})
                raise
        return
    import train
    from protocol import source_hashes
    recipe=json.loads((run/'prepared.json').read_text())
    if recipe['source_hashes']!=source_hashes(): raise IntegrityError('Frozen training code changed')
    pointer=json.loads((run/'checkpoint_pointer.json').read_text())
    if pointer['active']['stage']!='ssl': raise IntegrityError('Refusing to start a supervised-stage checkpoint')
    if not (run/'preflight.json').exists(): raise IntegrityError('Original preflight must already have passed')
    # Match the deterministic backend settings of the original worker; saved
    # CPU/CUDA RNG states are still restored by the unchanged training loader.
    import torch
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    install_boundary_guard(train,run)
    sys.argv=[str(here/'train.py'),'--config',str(here/'config.json'),'--micro-batch',str(args.micro_batch or cfg['micro_batch'])]
    train.main()


if __name__=='__main__': main()
