"""Mechanism tests for conditional disease inference and training safeguards."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import CropConditioner, conditional_scores
from protocol import HERE, EpochBatches, load_config, metrics


def ontology():
    return {'crop':[0,1], 'disease':[0,1], 'candidates':[
        {'composition_id':i,'crop_id':i//2,'disease_id':i%2} for i in range(4)]}


class ConditionalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.module=CropConditioner(2,8,4,.1)
        self.head=nn.Linear(8,2)
        self.features=torch.randn(3,8,requires_grad=True)

    def test_identity_exact_for_all_crops_and_selected_training_crop(self):
        baseline=self.head(self.features)
        self.assertTrue(torch.equal(self.module(self.features,self.head),baseline[:,None,:].expand(-1,2,-1)))
        self.assertTrue(torch.equal(self.module(self.features,self.head,torch.tensor([0,1,0])),baseline))

    def test_training_updates_shared_head_features_and_conditioner(self):
        optimizer=torch.optim.SGD(list(self.module.parameters())+list(self.head.parameters()),lr=.2)
        before=self.module(self.features,self.head).detach().clone()
        # (crop 1, disease 1) is absent, but its score can change through shared weights.
        loss=F.cross_entropy(self.module(self.features,self.head,torch.tensor([0,0,1])),torch.tensor([0,1,0]))
        loss.backward()
        for grad in (self.features.grad,self.head.weight.grad,self.module.affine.weight.grad):
            self.assertGreater(float(grad.norm()),0)
        optimizer.step()
        after=self.module(self.features,self.head)
        self.assertFalse(torch.allclose(before[:,1,1],after[:,1,1]))
        self.assertFalse(torch.equal(after[:,0,:],after[:,1,:]))

    def test_candidate_crop_normalizers_change_ranking(self):
        crop=torch.zeros(1,2)
        logits=torch.tensor([[[100.,101.],[0.,2.]]])
        pair,disease=conditional_scores(crop,logits,ontology())
        self.assertEqual(int(pair.argmax()),3)
        self.assertEqual(int(logits.flatten().argmax()),1)
        joint=crop.double().softmax(1)[:,:,None]*logits.double().softmax(2)
        self.assertTrue(torch.allclose(pair.exp(),joint.flatten(1),atol=1e-6,rtol=0))
        self.assertTrue(torch.allclose(disease.exp(),joint.sum(1),atol=1e-6,rtol=0))

    def test_marginal_includes_crops_without_a_candidate_for_the_disease(self):
        o=ontology(); o['candidates']=o['candidates'][:3]
        c=torch.tensor([[0.,3.]])
        d=torch.tensor([[[0.,0.],[0.,4.]]])
        _,marginal=conditional_scores(c,d,o)
        expected=(c.double().softmax(1)[:,:,None]*d.double().softmax(2)).sum(1)
        self.assertTrue(torch.allclose(marginal.exp(),expected,atol=1e-6,rtol=0))
        self.assertAlmostEqual(float(marginal.exp().sum()),1.,places=6)

    def test_zero_conditioning_recovers_independent_baseline_ranking(self):
        c=torch.randn(3,2); d=self.head(self.features).detach()
        pair,marginal=conditional_scores(c,d[:,None,:].expand(-1,2,-1),ontology())
        baseline_fp32=(c.log_softmax(1)[:,:,None]+d.log_softmax(1)[:,None,:]).flatten(1)
        self.assertTrue(torch.equal(pair,baseline_fp32.double()))
        independent=c.double()[:,:,None]+d.double()[:,None,:]
        self.assertTrue(torch.equal(pair.argsort(1,descending=True),independent.flatten(1).argsort(1,descending=True)))
        self.assertTrue(torch.allclose(marginal,F.log_softmax(d.double(),1),atol=1e-6,rtol=0))

    def test_predict_does_not_use_ground_truth_crop(self):
        from train import predict
        class Toy(nn.Module):
            def forward(self,images): return torch.tensor([[[0.,1.],[2.,0.]]]).expand(len(images),-1,-1)
        p={'ontology':ontology()}; cfg={'device':'cpu','amp':False,'eval_batch_size':1}
        c=torch.tensor([[1.,0.]])
        with patch('train.loader',return_value=[(torch.zeros(1,3,2,2),torch.tensor([[0,0,0]]))]):
            a=predict(Toy(),[{}],p,cfg,None,c)
        with patch('train.loader',return_value=[(torch.zeros(1,3,2,2),torch.tensor([[1,1,3]]))]):
            b=predict(Toy(),[{}],p,cfg,None,c)
        self.assertTrue(torch.equal(a['pair'],b['pair']))
        self.assertTrue(torch.equal(a['disease'],b['disease']))


class ProtocolTests(unittest.TestCase):
    def test_fixed_budget_replaces_validation_selection(self):
        cfg=load_config(HERE/'config.json')
        self.assertEqual(cfg['model'],'convnext_base.fb_in22k_ft_in1k')
        self.assertEqual(cfg['epochs'],15); self.assertEqual(cfg['schedule_epochs'],30)
        self.assertNotIn('patience',cfg)
        self.assertEqual(cfg['baseline_gamma'],6.2)

    def test_optimizer_preserves_base_decay_and_learning_rates(self):
        from train import optimizer_for
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.net=nn.Module(); self.net.backbone=nn.Sequential(nn.Linear(8,8),nn.LayerNorm(8))
                self.net.head=nn.Sequential(nn.Dropout(.2),nn.Linear(8,2)); self.conditioner=CropConditioner(2,8,4,.1)
        model=Toy(); cfg=load_config(HERE/'config.json'); opt=optimizer_for(model,cfg)
        groups={id(p):g for g in opt.param_groups for p in g['params']}
        for name,p in model.named_parameters():
            group=groups[id(p)]; head=name.startswith(('net.head.','conditioner.'))
            self.assertEqual(group['lr'],cfg['head_lr' if head else 'backbone_lr'])
            self.assertEqual(group['weight_decay'],cfg['weight_decay'] if p.ndim>1 and not name.endswith('.bias') else 0.)

    def test_exact_full_image_exposure(self):
        rows=[{'crop_id':i%3,'disease_id':i%5} for i in range(51)]
        cfg=load_config(HERE/'config.json'); a=EpochBatches(rows,cfg); b=EpochBatches(rows,cfg)
        self.assertEqual(a.indices(),b.indices()); self.assertEqual(sorted(a.indices()),list(range(51)))
        a.epoch=1; self.assertNotEqual(a.indices(),b.indices())

    def test_completed_checkpoint_recovers_final_weights(self):
        from train import recover_reports
        checkpoint={'complete':True,'epoch':15,'history':[{'epoch':15,'seconds':2.}],
            'initial_hash':'test','model':{'weight':torch.tensor([1.])},'fingerprint':'test'}
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            recover_reports(Path(temp),checkpoint)
            self.assertEqual(json.loads((Path(temp)/'summary.json').read_text())['selected_epoch'],15)
            final=torch.load(Path(temp)/'final.pt',weights_only=True)
            self.assertEqual(final['epoch'],15); self.assertTrue(torch.equal(final['model']['weight'],torch.tensor([1.])))



if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
