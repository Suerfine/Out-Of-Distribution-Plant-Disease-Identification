"""Gradient correctness and real fault-recovery tests before overnight fitting."""
import copy,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from model import projector,predictor,simsiam_loss,cache_backward
from storage import CheckpointStore,checked_save,atomic_json,IntegrityError
from data import TrainingRows,epoch_indices,training_loader
from supervisor import run_controller,native_python_environment
from protocol import HERE,load_config
from reuse import validate_ssl_settings


class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(8,8),nn.LayerNorm(8),nn.Tanh())
        self.projector=projector(8,16); self.predictor=predictor(16,4)


class GradientTests(unittest.TestCase):
    def test_cached_gradients_match_direct_full_batch_heads(self):
        torch.manual_seed(7); a=Toy(); b=copy.deepcopy(a); x=torch.randn(64,8); y=torch.randn(64,8)
        encoder_sizes=[]; head_sizes=[]
        hooks=[b.encoder.register_forward_pre_hook(lambda m,args:encoder_sizes.append(len(args[0]))),
               b.projector[1].register_forward_pre_hook(lambda m,args:head_sizes.append(len(args[0])))]
        f1=torch.cat([a.encoder(x[i:i+4]) for i in range(0,64,4)]); f2=torch.cat([a.encoder(y[i:i+4]) for i in range(0,64,4)])
        loss,*_=simsiam_loss(a,f1,f2); loss.backward()
        cached,details=cache_backward(b,x,y,4,'cpu',check_replay=True)
        self.assertAlmostEqual(float(loss.detach()),cached,places=6)
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()): self.assertTrue(torch.allclose(p.grad,q.grad,atol=2e-6,rtol=2e-5),name)
        self.assertEqual(set(encoder_sizes),{4}); self.assertEqual(head_sizes,[64,64])
        for h in hooks: h.remove()

    def test_stop_gradient_has_the_required_effect(self):
        torch.manual_seed(8); m=Toy(); a=torch.randn(64,8,requires_grad=True); b=torch.randn(64,8,requires_grad=True)
        loss,*_=simsiam_loss(m,a,b); ga=torch.autograd.grad(loss,a)[0]
        other=copy.deepcopy(m); aa=a.detach().clone().requires_grad_(); bb=b.detach().clone().requires_grad_()
        z1,z2=other.projector(aa),other.projector(bb); p1,p2=other.predictor(z1),other.predictor(z2)
        wrong=-.5*(F.cosine_similarity(p1,z2).mean()+F.cosine_similarity(p2,z1).mean())
        wrong_gradient=torch.autograd.grad(wrong,aa)[0]
        self.assertFalse(torch.allclose(ga,wrong_gradient,atol=1e-6))


class DataTests(unittest.TestCase):
    def test_ssl_needs_no_labels_and_augmentation_resumes_exactly(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp)/'leaf.jpg'; image=np.random.default_rng(3).integers(0,256,(256,320,3),dtype=np.uint8); Image.fromarray(image).save(path)
            rows=[{'path':str(path)}]; cfg={'seed':42}; base={'input':{'mean':[.485,.456,.406],'std':[.229,.224,.225]}}
            data=TrainingRows(rows,'ssl',3,cfg,base); before=torch.get_rng_state(); a,b,index=data[0]
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            again=TrainingRows(rows,'ssl',3,cfg,base)[0]
            self.assertTrue(torch.equal(a,again[0])); self.assertTrue(torch.equal(b,again[1])); self.assertFalse(torch.equal(a,b))
            self.assertEqual(index,0); self.assertEqual(tuple(a.shape),(3,224,224))

    def test_every_image_once_and_cursor_suffix(self):
        cfg=load_config(HERE/'config.json'); order=epoch_indices(8000,cfg,'ssl',5)
        self.assertEqual(sorted(order.tolist()),list(range(8000)))
        self.assertTrue(np.array_equal(order,epoch_indices(8000,cfg,'ssl',5)))
        self.assertFalse(np.array_equal(order,epoch_indices(8000,cfg,'ssl',6)))
        self.assertEqual(len(order[17*64:]),8000-17*64)


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


class ReuseTests(unittest.TestCase):
    def test_source_metadata_and_all_ssl_exposure_files_exist(self):
        cfg=load_config(HERE/'config.json'); source=Path(cfg['ssl_source_run'])
        self.assertTrue((source.parent.parent/'config.json').is_file())
        for name in ['prepared.json','ssl_encoder.pt','finetune_initialization.json','ssl_initialization.json',
                     'independent_verification.json','preflight.json','ssl/history.json']:
            self.assertTrue((source/name).is_file(),name)
        for epoch in range(1,21):
            self.assertTrue((source/'ssl'/f'exposures_epoch{epoch:02}.npz').is_file())

    def test_ssl_recipe_matches_source_and_development_and_rejects_changes(self):
        cfg=load_config(HERE/'config.json')
        for name in ['ssl_source_run','development_run']:
            old=json.loads((Path(cfg[name])/'prepared.json').read_text())
            validate_ssl_settings(cfg,old['config'])
        altered=copy.deepcopy(cfg); altered['ssl_epochs']=21
        with self.assertRaises(IntegrityError):validate_ssl_settings(cfg,altered)

    def test_supervised_budget_matches_saved_development_selection(self):
        cfg=load_config(HERE/'config.json')
        selected=json.loads((Path(cfg['development_run'])/'selection_frozen.json').read_text())
        self.assertEqual(cfg['finetune_epochs'],selected['best']['epoch'])
        self.assertEqual(cfg['finetune_epochs'],9)
        self.assertFalse(selected['official_scores_used'])

    def test_keyword_argsort_preserves_stable_ties(self):
        values=torch.tensor([[1.,2.,2.,0.]])
        self.assertEqual(values.argsort(dim=1,descending=True,stable=True).tolist(),[[1,2,0,3]])


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


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
