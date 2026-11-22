import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ssl_stage_control import verify_ssl_boundary,install_boundary_guard

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/'simsiam_base_v1'
sys.path.insert(0,str(STUDY))
from storage import CheckpointStore,checked_save,IntegrityError


class StageControlTests(unittest.TestCase):
    def test_encoder_is_verified_and_supervised_constructor_is_never_called(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            run=Path(tmp); encoder={'weight':torch.arange(6.)}; p={'fingerprint':'test'}; cfg={'ssl_epochs':20}
            state={'fingerprint':'test','stage':'ssl','epoch':20,'cursor':0,
                'histories':{'ssl':[{}]*20,'finetune':[]},'model':{'encoder.weight':encoder['weight']}}
            CheckpointStore(run,'test').save(state)
            checked_save(run/'ssl_encoder.pt',{'encoder':encoder,'fingerprint':'test','epochs':20})
            calls=[]
            train=SimpleNamespace(initialise_finetune=lambda *a:calls.append('supervised'),status=lambda *a,**k:calls.append(k))
            def original(*args): return train.initialise_finetune(p,cfg,encoder)
            train.run_training=original; install_boundary_guard(train,run)
            self.assertEqual(train.run_training(),'paused')
            self.assertNotIn('supervised',calls)
            receipt=json.loads((run/'ssl_stage_pause_verified.json').read_text())
            self.assertEqual(receipt['ssl_epochs_completed'],20); self.assertEqual(receipt['supervised_epochs_completed'],0)

    def test_incomplete_ssl_cannot_be_reported_as_finished(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            run=Path(tmp); state={'fingerprint':'test','stage':'ssl','epoch':19,'cursor':125}
            CheckpointStore(run,'test').save(state)
            with self.assertRaises(IntegrityError): verify_ssl_boundary(run,{'fingerprint':'test'},{'ssl_epochs':20},{})


if __name__=='__main__': unittest.main()
