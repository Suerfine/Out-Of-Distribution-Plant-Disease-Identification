import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from protocol import HERE,load_config
from data import image_views,cache_original,DetailImages,epoch_indices
from model import DetailCNN,GatedAttention,optimizer
from train import backward_batch
from evaluation import validation_metrics,composed,is_improvement,should_stop


class Toy(DetailCNN):
    def __init__(self,arm):
        nn.Module.__init__(self);self.arm=arm;self.local_chunk=1
        self.backbone=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Dropout(0.),nn.Linear(8,4))
        if arm=='global_local':self.add_local_modules(8,4,4,1745)


class ModelTests(unittest.TestCase):
    def test_zero_local_branch_is_exact_global_prediction(self):
        torch.manual_seed(7);a=Toy('resolution384').eval();b=Toy('global_local').eval()
        b.backbone.load_state_dict(a.backbone.state_dict());b.head.load_state_dict(a.head.state_dict())
        g=torch.randn(2,3,12,12);local=torch.randn(2,4,3,8,8)
        self.assertTrue(torch.equal(a(g),b(g,local)))

    def test_attention_is_normalized_and_patch_order_invariant(self):
        torch.manual_seed(8);m=GatedAttention(8,4);x=torch.randn(2,4,8);pooled,w=m(x)
        self.assertTrue(torch.allclose(w.sum(1),torch.ones(2)))
        perm=torch.tensor([3,1,0,2]);other,_=m(x[:,perm])
        self.assertTrue(torch.allclose(pooled,other,atol=1e-6))

    def test_local_pixels_can_change_prediction_after_branch_learns(self):
        torch.manual_seed(9);m=Toy('global_local').eval()
        nn.init.normal_(m.local_head[1].weight)
        g=torch.randn(2,3,12,12);local=torch.randn(2,4,3,8,8)
        self.assertFalse(torch.allclose(m(g,local),m(g,local+2)))

    def test_aggregate_image_loss_matches_direct_gradients(self):
        torch.manual_seed(10);a=Toy('global_local').eval();nn.init.normal_(a.local_head[1].weight,std=.1);b=copy.deepcopy(a)
        batch={'global':torch.randn(2,3,12,12),'local':torch.randn(2,4,3,8,8),'label':torch.tensor([0,3])}
        direct=F.cross_entropy(a(batch['global'],batch['local']),batch['label'],label_smoothing=.1);direct.backward()
        loss,_=backward_batch(b,batch,{'device':'cpu','amp':False,'label_smoothing':.1},1)
        self.assertAlmostEqual(float(direct.detach()),loss/2,places=6)
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertTrue(torch.allclose(p.grad,q.grad,atol=2e-6,rtol=1e-4),name)

    def test_extra_modules_preserve_rng_and_get_head_lr(self):
        m=Toy('global_local');before=torch.get_rng_state()
        with patch('torch.cuda.manual_seed_all') as cuda_seed:m.add_local_modules(8,4,4,1745)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));cuda_seed.assert_not_called()
        base={'optimizer':{'lr_head':5e-4,'lr_backbone':5e-5,'weight_decay':.05,'betas':[.9,.999],'eps':1e-8}}
        opt=optimizer(m,base);groups={id(p):g for g in opt.param_groups for p in g['params']}
        for name,p in m.named_parameters():
            self.assertEqual(groups[id(p)]['lr'],5e-5 if name.startswith('backbone.') else 5e-4)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_config(HERE/'config.json');cls.p=json.loads((Path(cls.cfg['source_development_run'])/'prepared.json').read_text())
        cls.base=cls.p['baseline_config']
    def image(self):return Image.fromarray(np.random.default_rng(4).integers(0,256,(512,640,3),dtype=np.uint8))

    def test_global_views_identical_in_both_arms_training_and_eval(self):
        image=self.image()
        for training in [True,False]:
            torch.manual_seed(12);a=image_views(image,self.base,self.cfg,training,False)
            torch.manual_seed(12);b=image_views(image,self.base,self.cfg,training,True)
            self.assertTrue(torch.equal(a['global'],b['global']))
            self.assertEqual(tuple(b['global'].shape),(3,384,384));self.assertEqual(tuple(b['local'].shape),(4,3,224,224))

    def test_cache_uses_original_and_preserves_aspect_ratio(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            source=Path(tmp)/'original.png';destination=Path(tmp)/'new512.jpg'
            self.image().resize((960,768)).save(source)
            info=cache_original(source,destination,512)
            self.assertEqual(info['original_size'],[960,768])
            self.assertEqual(info['original_path'],str(source))
            with Image.open(destination) as im:self.assertEqual(im.size,(640,512))

    def test_dataset_augments_deterministically_and_keeps_image_label_scalar(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp)/'leaf.jpg';self.image().save(path)
            rows=[{'path':str(path),'disease_id':7}];before=torch.get_rng_state()
            a=DetailImages(rows,self.base,self.cfg,'resolution384',True,2)[0]
            b=DetailImages(rows,self.base,self.cfg,'global_local',True,2)[0]
            self.assertTrue(torch.equal(a['global'],b['global']));self.assertEqual(b['label'],7)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_coverage_and_resume_suffix(self):
        a=epoch_indices(5798,self.cfg,2);b=epoch_indices(5798,self.cfg,2)
        self.assertTrue(np.array_equal(np.sort(a),np.arange(5798)));self.assertTrue(np.array_equal(a[17*32:],b[17*32:]))
        self.assertFalse(np.array_equal(a,epoch_indices(5798,self.cfg,3)))

    def test_original_reference_candidate_mask_and_selection(self):
        import pandas as pd
        with np.load(self.p['baseline_validation_logits']) as bank:
            value=validation_metrics(bank['disease'],self.p);scores=composed(bank['crop'],bank['disease'],pd.DataFrame(self.p['pairs']))
        self.assertAlmostEqual(value['objective']['harmonic_mean_top1'],.0857505367497348,places=12)
        self.assertTrue(np.all(np.isfinite(scores).sum(1)==323));self.assertTrue(is_improvement(0.,None))
        self.assertFalse(is_improvement(.1,{'harmonic_mean_top1':.1}));self.assertTrue(should_stop(15,8,self.cfg))


if __name__=='__main__':torch.set_num_threads(2);unittest.main()
