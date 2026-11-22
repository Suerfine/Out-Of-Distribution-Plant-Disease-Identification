import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from protocol import HERE, load_config, validate_partition, read_rows
from data import synthesis_plan, epoch_schedule, exposure_arrays, FittingImages
from model import FeatureComposition, composition_loss, cached_backward, optimizer
from evaluation import validation_metrics, is_improvement, should_stop, composed
from storage import IntegrityError


class Toy(FeatureComposition):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone=nn.Sequential(nn.Linear(8,8),nn.LayerNorm(8),nn.GELU(),nn.Dropout(.25))
        self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(8,4))
        self.add_factor_modules(8,4,12,3,4,1703)


def toy_meta():
    return {'primary_count':4,'primary_crop':torch.tensor([0,1,2,0]),'primary_disease':torch.tensor([0,1,2,3]),
        'crop_positions':torch.tensor([4,5,6,7]),'disease_positions':torch.tensor([8,9,10,11]),
        'target_crop':torch.tensor([0,1,2,1]),'target_disease':torch.tensor([1,2,3,0])}


class FeatureTests(unittest.TestCase):
    def config(self):
        return {**load_config(HERE/'config.json'),'device':'cpu','amp':False}

    def test_factor_initialization_preserves_cpu_rng_and_does_not_reseed_cuda(self):
        model=Toy(); before=torch.get_rng_state()
        with patch('torch.cuda.manual_seed_all') as cuda_seed:
            model.add_factor_modules(8,4,12,3,4,1703)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        cuda_seed.assert_not_called()

    def test_stochastic_cache_matches_direct_gradients_and_preserves_rng(self):
        torch.manual_seed(6); a=Toy().train(); b=copy.deepcopy(a); images=torch.randn(12,8); cfg=self.config(); meta=toy_meta()
        torch.manual_seed(123)
        features=torch.cat([a.backbone(images[i:i+4]) for i in range(0,12,4)])
        loss,_=composition_loss(a,features,meta,cfg,2); loss.backward(); after=torch.get_rng_state()
        torch.manual_seed(123)
        cached,detail=cached_backward(b,images,meta,cfg,2,4,check_replay=True)
        self.assertEqual(detail['replay_max_error'],0.)
        self.assertAlmostEqual(float(loss.detach()),cached,places=6)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertTrue(torch.allclose(p.grad,q.grad,atol=2e-6,rtol=1e-4),name)

    def test_synthetic_loss_updates_composer_factors_and_donor_features(self):
        torch.manual_seed(7); model=Toy().eval(); features=torch.randn(12,8,requires_grad=True); cfg=self.config(); meta=toy_meta()
        cfg.update(auxiliary_weight=0.,reconstruction_weight=0.,cycle_weight=0.,synthetic_weight=0.)
        loss,_=composition_loss(model,features,meta,cfg,3); loss.backward()
        self.assertEqual(sum(float(p.grad.abs().sum()) for p in model.composer.parameters()),0.)
        real_only_head_gradients=[p.grad.clone() for p in model.head.parameters()]
        model.zero_grad(set_to_none=True); features.grad=None; cfg['synthetic_weight']=1.
        loss,_=composition_loss(model,features,meta,cfg,3); loss.backward()
        for name in ['composer','crop_factor','disease_factor','head']:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters()),0.,name)
        self.assertTrue(any(not torch.equal(old,p.grad) for old,p in zip(real_only_head_gradients,model.head.parameters())))
        self.assertGreater(float(features.grad[4:].abs().sum()),0.)

    def test_composer_depends_on_both_sources(self):
        torch.manual_seed(8); m=Toy(); c=torch.randn(4,4); d=torch.randn(4,4)
        original=m.composer(torch.cat([c,d],1))
        self.assertFalse(torch.equal(original,m.composer(torch.cat([c+1,d],1))))
        self.assertFalse(torch.equal(original,m.composer(torch.cat([c,d+1],1))))

    def test_inference_uses_real_features_without_training_targets(self):
        torch.manual_seed(9); m=Toy().eval(); x=torch.randn(4,8); before=m(x)
        with torch.no_grad():
            for module in [m.composer,m.crop_factor,m.disease_factor,m.crop_classifier,m.factor_disease_classifier]:
                for p in module.parameters():p.add_(100.)
        self.assertTrue(torch.equal(before,m(x)))

    def test_all_new_modules_get_head_lr_and_biases_get_no_decay(self):
        m=Toy(); base={'optimizer':{'lr_backbone':5e-5,'lr_head':5e-4,'weight_decay':.05,'betas':[.9,.999],'eps':1e-8}}
        opt=optimizer(m,base); groups={id(p):g for g in opt.param_groups for p in g['params']}
        for name,p in m.named_parameters():
            self.assertEqual(groups[id(p)]['lr'],5e-5 if name.startswith('backbone.') else 5e-4)
            self.assertEqual(groups[id(p)]['weight_decay'],.05 if p.ndim>1 and not name.endswith('.bias') else 0.)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_config(HERE/'config.json')
        cls.source=json.loads((Path(cls.cfg['ssl_development_run'])/'prepared.json').read_text())
        cls.rows,cls.pairs=cls.source['roles']['fit'],cls.source['pairs']

    def test_every_novel_class_has_fit_only_crossed_donors(self):
        plan=synthesis_plan(self.rows,self.pairs)
        self.assertEqual(len(plan['novel_pair_ids']),20)
        self.assertEqual(len(plan['familiar_pair_ids']),161)
        held={r['composition_id'] for r in self.source['roles']['validation_unseen']}
        self.assertEqual(set(plan['novel_pair_ids']),held)
        schedule=epoch_schedule(self.rows,self.pairs,self.cfg,0)
        pairs={r['pair_id']:r for r in self.pairs}
        for step in schedule:
            self.assertTrue(np.all(step['unique_indices']<5798))
            self.assertEqual(int(step['novel'].sum()),8)
            self.assertEqual(len(step['target_pair']),16)
            for pid,ci,di in zip(step['target_pair'],step['crop_donor_indices'],step['disease_donor_indices']):
                p=pairs[int(pid)]; c,d=self.rows[int(ci)],self.rows[int(di)]
                self.assertFalse(p['is_unseen'])
                self.assertEqual(c['crop_id'],p['crop_id']); self.assertNotEqual(c['disease_id'],p['disease_id'])
                self.assertEqual(d['disease_id'],p['disease_id']); self.assertNotEqual(d['crop_id'],p['crop_id'])

    def test_primary_coverage_novel_balance_and_exact_resume_schedule(self):
        a=epoch_schedule(self.rows,self.pairs,self.cfg,2); b=epoch_schedule(self.rows,self.pairs,self.cfg,2)
        arrays=exposure_arrays(a,5798,333)
        self.assertTrue(np.all(arrays['primary_counts']==1))
        self.assertEqual(len(a[-1]['primary_indices']),6)
        self.assertGreater(int(arrays['encoded_counts'].sum()),5798)
        for x,y in zip(a[17:],b[17:]):
            for key in ['unique_indices','target_pair','crop_donor_indices','disease_donor_indices']:
                self.assertTrue(np.array_equal(x[key],y[key]))
            self.assertEqual(x['keys'],y['keys'])
        novel=synthesis_plan(self.rows,self.pairs)['novel_pair_ids']; counts=arrays['synthetic_pair_counts'][novel]
        self.assertLessEqual(int(counts.max()-counts.min()),1)

    def test_augmentations_resume_and_donor_views_can_change_by_step(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp)/'leaf.jpg'; Image.fromarray(np.random.default_rng(3).integers(0,256,(256,320,3),dtype=np.uint8)).save(path)
            data=FittingImages([{'path':str(path)}]); before=torch.get_rng_state()
            a,_=data[(0,123)]; b,_=data[(0,123)]; c,_=data[(0,456)]
            self.assertTrue(torch.equal(a,b)); self.assertFalse(torch.equal(a,c))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_exact_partition_and_pseudo_pair_leak_rejected(self):
        study=Path(self.cfg['development_study'])
        frames={k:read_rows(study/f'{k}.csv') for k in ['fit','validation_seen','validation_unseen','selection_excluded','all_train']}
        validate_partition(frames)
        frames['fit'][0]['pair_id']=frames['validation_unseen'][0]['pair_id']
        with self.assertRaises(IntegrityError):validate_partition(frames)

    def test_reference_objective_and_all_development_candidates(self):
        import pandas as pd
        p=self.source
        with np.load(p['baseline_validation_logits']) as bank:
            value=validation_metrics(bank['disease'],p)
            scores=composed(bank['crop'],bank['disease'],pd.DataFrame(p['pairs']))
        self.assertTrue(np.all(np.isfinite(scores).sum(1)==323))
        self.assertAlmostEqual(value['objective']['harmonic_mean_top1'],.0857505367497348,places=12)

    def test_selection_earliest_tie_and_declared_patience(self):
        self.assertTrue(is_improvement(0.,None)); self.assertFalse(is_improvement(.1,{'harmonic_mean_top1':.1}))
        self.assertFalse(should_stop(15,7,self.cfg)); self.assertTrue(should_stop(15,8,self.cfg))
        self.assertTrue(should_stop(30,0,self.cfg))


if __name__=='__main__':
    torch.set_num_threads(2); unittest.main()
