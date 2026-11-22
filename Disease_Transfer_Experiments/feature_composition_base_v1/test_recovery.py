import json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from storage import CheckpointStore,checked_save,atomic_json
from supervisor import run_controller
from protocol import HERE

class StorageTests(unittest.TestCase):
    def payload(self,n): return {'fingerprint':'test','stage':'ssl','epoch':0,'cursor':n,'weight':torch.arange(9)+n}

    def test_corrupt_latest_falls_back_and_next_save_preserves_good_slot(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            store=CheckpointStore(tmp,'test'); store.save(self.payload(1)); store.save(self.payload(2))
            p=json.loads(store.pointer.read_text()); (Path(tmp)/p['active']['file']).write_bytes(b'corrupt')
            self.assertEqual(store.load()['cursor'],1)
            intact=json.loads(store.pointer.read_text())['active']; before=(Path(tmp)/intact['file']).read_bytes()
            store.save(self.payload(3)); self.assertEqual((Path(tmp)/intact['file']).read_bytes(),before)
            self.assertEqual(store.load()['cursor'],3)

    def test_failed_write_does_not_replace_pointer_or_valid_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            store=CheckpointStore(tmp,'test'); store.save(self.payload(1)); before=store.pointer.read_bytes()
            with patch('torch.save',side_effect=OSError('simulated disk error')),patch('storage.time.sleep'):
                with self.assertRaises(OSError): store.save(self.payload(2))
            self.assertEqual(store.pointer.read_bytes(),before); self.assertEqual(store.load()['cursor'],1)

    def test_transient_windows_replace_denial_recovers_without_losing_state(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp)/'status.json'; atomic_json(path,{'step':1}); actual=os.replace
            attempts=[]
            def transient(source,target):
                attempts.append(1)
                self.assertEqual(json.loads(path.read_text()),{'step':1})
                if len(attempts)<3: raise PermissionError('simulated Windows sharing violation')
                return actual(source,target)
            with patch('storage.os.replace',side_effect=transient),patch('storage.time.sleep'):
                atomic_json(path,{'step':2})
            self.assertEqual(len(attempts),3)
            self.assertEqual(json.loads(path.read_text()),{'step':2})

class SupervisorTests(unittest.TestCase):
    def config(self,watchdog=10): return {'max_restarts':2,'supervisor_hours':.02,'watchdog_seconds':watchdog,'min_free_bytes':0,'append_micro':False}

    def test_crashed_worker_is_restarted_from_committed_state(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp); script=path/'toy.py'
            script.write_text("import os,sys,json,time\nfrom pathlib import Path\np=Path(sys.argv[1]); f=p/'counter.json'\nn=json.loads(f.read_text()) if f.exists() else 0\nf.write_text(json.dumps(n+1))\n(p/'status.json').write_text(json.dumps({'pid':os.getpid(),'phase':'complete' if n else 'failed','updated_utc':str(time.time())}))\nsys.exit(0 if n else 17)\n")
            run_controller([sys.executable,str(script),str(path)],path,self.config(),poll=.05,retry_delay=.01,use_power=False)
            self.assertEqual(json.loads((path/'counter.json').read_text()),2)
            self.assertEqual(json.loads((path/'supervisor_status.json').read_text())['attempts'],2)

    def test_stalled_worker_is_killed_and_restarted(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp); script=path/'toy.py'
            script.write_text("import os,sys,json,time\nfrom pathlib import Path\np=Path(sys.argv[1]); f=p/'counter.json'\nn=json.loads(f.read_text()) if f.exists() else 0\nf.write_text(json.dumps(n+1))\n(p/'status.json').write_text(json.dumps({'pid':os.getpid(),'phase':'complete' if n else 'stalled','updated_utc':str(time.time())}))\nif not n: time.sleep(30)\n")
            run_controller([sys.executable,str(script),str(path)],path,self.config(.8),poll=.05,retry_delay=.01,use_power=False)
            self.assertEqual(json.loads((path/'counter.json').read_text()),2)
            events=[json.loads(s) for s in (path/'supervisor_events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e.get('event')=='watchdog_restart' for e in events))
