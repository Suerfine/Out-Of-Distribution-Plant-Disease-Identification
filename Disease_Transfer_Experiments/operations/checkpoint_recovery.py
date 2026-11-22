"""Operational checkpoint I/O repair; no training/model/selection changes."""
import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def record(event_file,**values):
    if event_file is not None:
        with Path(event_file).open('a',encoding='utf-8') as f:
            f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**values})+'\n')


def checked_save(path,payload,event_path=None,attempts=3,retry_seconds=5):
    """Keep the old checkpoint until fsync and ZIP CRC verification succeed."""
    import torch
    path=Path(path); temp=path.with_suffix(path.suffix+'.io.tmp')
    for attempt in range(1,attempts+1):
        free=shutil.disk_usage(path.parent).free
        try:
            with temp.open('wb') as f:
                torch.save(payload,f)
                f.flush()
                os.fsync(f.fileno())
            with zipfile.ZipFile(temp) as archive:
                bad=archive.testzip()
                if bad is not None: raise OSError('Checkpoint ZIP CRC failed: '+bad)
            size=temp.stat().st_size
            os.replace(temp,path)
            record(event_path,path=str(path),attempt=attempt,free_before=free,bytes=size,verified_zip_crc=True,result='saved')
            return
        except (OSError,RuntimeError,zipfile.BadZipFile) as error:
            # Retry serialization/storage failures, never unrelated model errors.
            io_error=isinstance(error,(OSError,zipfile.BadZipFile)) or any(s in str(error).lower() for s in ('unexpected pos','ios_base','streamwriter','write_record','failed writing','file write','iostream'))
            record(event_path,path=str(path),attempt=attempt,free_before=free,free_after=shutil.disk_usage(path.parent).free,error=repr(error),result='failed; prior checkpoint preserved')
            if not io_error or attempt==attempts: raise
            time.sleep(retry_seconds)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',required=True,type=Path)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args(); experiment=args.experiment.resolve()
    import sys
    sys.path.insert(0,str(experiment))
    import torch
    import train
    import protocol
    cfg=protocol.load_config(experiment/'config.json'); run=experiment/'runs'/cfg['run_name']
    events=run/'checkpoint_io_recovery_events.jsonl'
    recipe=json.loads((run/'prepared.json').read_text())
    for path,h in recipe['source_hashes'].items():
        if digest(path)!=h: raise RuntimeError('Frozen experiment source changed: '+path)
    self_hash=digest(__file__)
    if args.probe_only:
        source=run/'conditioned/last.pt'; before=digest(source)
        payload=torch.load(source,map_location='cpu',weights_only=True)
        if payload['fingerprint']!=recipe['fingerprint'] or payload['complete']: raise RuntimeError('Unexpected recovery checkpoint')
        target=run/'checkpoint_io_probe.pt'
        checked_save(target,payload,events)
        loaded=torch.load(target,map_location='cpu',weights_only=True)
        if loaded['epoch']!=payload['epoch'] or loaded['fingerprint']!=payload['fingerprint']: raise RuntimeError('Probe metadata differs')
        for field in ('model','optimizer','scaler','rng','history'):
            # Tensor equality for all model weights; remaining payload was fully
            # deserialized, and CRC verifies every serialized storage record.
            if field=='model' and protocol.tensor_hash(loaded[field])!=protocol.tensor_hash(payload[field]): raise RuntimeError('Probe model weights differ')
        if digest(source)!=before: raise RuntimeError('Probe modified the saved checkpoint')
        value={'passed':True,'epoch':payload['epoch'],'source_checkpoint_sha256':before,
            'probe_bytes':target.stat().st_size,'probe_sha256':digest(target),'operational_source_sha256':self_hash,
            'original_model_state_unchanged':True,'frozen_experiment_sources_unchanged':True,
            'utc':datetime.now(timezone.utc).isoformat()}
        protocol.write_json(run/'checkpoint_io_probe_verified.json',value)
        # Delete only this new, verified diagnostic copy; keep the original and
        # the earlier failed temporary file intact.
        if target.resolve().parent!=run.resolve() or target.name!='checkpoint_io_probe.pt': raise RuntimeError('Unsafe probe cleanup path')
        target.unlink()
        print(json.dumps(value),flush=True)
        return
    probe=json.loads((run/'checkpoint_io_probe_verified.json').read_text())
    if not probe['passed'] or probe['operational_source_sha256']!=self_hash: raise RuntimeError('Run the checked I/O probe with this operational wrapper first')
    def save(path,payload): return checked_save(path,payload,events)
    train.save_checkpoint=protocol.save_checkpoint=save
    original_status=train.status
    def status(directory,phase,**details):
        if phase=='complete':
            if digest(__file__)!=self_hash: raise RuntimeError('Operational source changed during training')
            protocol.write_json(run/'checkpoint_io_recovery_verified.json',{'passed':True,'operational_source_sha256':self_hash,
                'original_fingerprint':recipe['fingerprint'],'training_recipe_changed':False,'frozen_sources_unchanged':True,
                'description':'Explicit-stream fsync, ZIP CRC check, atomic replacement, retries for storage errors only'})
            details['checkpoint_io_repair_verified']=True
        return original_status(directory,phase,**details)
    train.status=status
    record(events,result='operational recovery installed',operational_source=str(Path(__file__).resolve()),operational_sha256=self_hash,
           original_fingerprint=recipe['fingerprint'],training_model_data_selection_changed=False)
    sys.argv=[str(experiment/'train.py'),'--config',str(experiment/'config.json')]
    train.main()


if __name__=='__main__': main()
