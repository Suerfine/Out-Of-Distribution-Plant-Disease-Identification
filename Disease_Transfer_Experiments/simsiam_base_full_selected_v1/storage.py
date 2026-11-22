"""Checksummed two-slot checkpoints with transactional pointer updates."""
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone


class LowDiskSpace(OSError): pass
class IntegrityError(RuntimeError): pass


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def replace_with_retry(source,target,attempts=8):
    for attempt in range(attempts):
        try:
            os.replace(source,target)
            return
        except PermissionError:
            if attempt==attempts-1: raise
            time.sleep(min(.05*(2**attempt),.8))


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    replace_with_retry(temp,path)


def event(event_file,**value):
    try:
        with Path(event_file).open('a',encoding='utf-8') as f: f.write(json.dumps({'utc':now(),**value})+'\n')
    except OSError: pass  # A log failure must not invalidate a committed checkpoint.


def checked_save(path,payload,min_free=0,events=None,attempts=3,retry_seconds=5):
    import torch
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    for attempt in range(1,attempts+1):
        free=shutil.disk_usage(path.parent).free
        if free<min_free: raise LowDiskSpace(f'Free bytes {free} below reserve {min_free}')
        try:
            with temp.open('wb') as f:
                torch.save(payload,f); f.flush(); os.fsync(f.fileno())
            with zipfile.ZipFile(temp) as z:
                bad=z.testzip()
                if bad is not None: raise OSError('Checkpoint CRC failed: '+bad)
            digest=sha(temp); size=temp.stat().st_size
            replace_with_retry(temp,path)
            if events: event(events,result='saved',path=str(path),attempt=attempt,sha256=digest,bytes=size,free_before=free)
            return digest
        except (OSError,RuntimeError,zipfile.BadZipFile) as error:
            io=isinstance(error,(OSError,zipfile.BadZipFile)) or any(s in str(error).lower() for s in ('unexpected pos','ios_base','streamwriter','write_record','failed writing','iostream','file write'))
            if events: event(events,result='save_failed',path=str(path),attempt=attempt,error=repr(error),free_now=shutil.disk_usage(path.parent).free)
            if not io or attempt==attempts: raise
            time.sleep(retry_seconds)


class CheckpointStore:
    def __init__(self,run,fingerprint,min_free=0):
        self.run=Path(run); self.run.mkdir(parents=True,exist_ok=True)
        self.fingerprint=fingerprint; self.min_free=min_free
        self.pointer=self.run/'checkpoint_pointer.json'; self.events=self.run/'checkpoint_events.jsonl'

    def save(self,state):
        if state['fingerprint']!=self.fingerprint: raise IntegrityError('Checkpoint recipe differs')
        old=json.loads(self.pointer.read_text()) if self.pointer.exists() else {}
        target='checkpoint_b.pt' if old.get('active',{}).get('file')=='checkpoint_a.pt' else 'checkpoint_a.pt'
        digest=checked_save(self.run/target,state,self.min_free,self.events)
        active={'file':target,'sha256':digest,'stage':state['stage'],'epoch':state['epoch'],'cursor':state['cursor'],'saved_utc':now()}
        atomic_json(self.pointer,{'fingerprint':self.fingerprint,'active':active,'previous':old.get('active')})
        return active

    def load(self):
        import torch
        if not self.pointer.exists(): return None
        pointer=json.loads(self.pointer.read_text())
        if pointer['fingerprint']!=self.fingerprint: raise IntegrityError('Checkpoint pointer recipe differs')
        errors=[]
        for role in ('active','previous'):
            item=pointer.get(role)
            if not item: continue
            path=self.run/item['file']
            if path.parent.resolve()!=self.run.resolve() or item['file'] not in ('checkpoint_a.pt','checkpoint_b.pt'): raise IntegrityError('Unexpected checkpoint path')
            try:
                if sha(path)!=item['sha256']: raise IntegrityError('Checkpoint checksum differs')
                state=torch.load(path,map_location='cpu',weights_only=True)
                if state['fingerprint']!=self.fingerprint: raise IntegrityError('Checkpoint payload recipe differs')
                if role=='previous':
                    event(self.events,result='fallback_to_previous',errors=errors,checkpoint=item)
                    # Promote the intact fallback. The next transaction will write
                    # to the corrupt slot, preserving this good slot throughout.
                    atomic_json(self.pointer,{'fingerprint':self.fingerprint,'active':item,'previous':None})
                return state
            except (OSError,RuntimeError,EOFError,zipfile.BadZipFile) as error: errors.append(repr(error))
        raise IntegrityError('Both saved checkpoint slots are unusable: '+str(errors))
