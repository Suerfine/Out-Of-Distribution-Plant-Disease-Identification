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
        self.assertTrue(torch.allclose(pair.exp(),joint.flatten(1),atol=1e-12,rtol=0))
        self.assertTrue(torch.allclose(disease.exp(),joint.sum(1),atol=1e-12,rtol=0))

    def test_marginal_includes_crops_without_a_candidate_for_the_disease(self):
        o=ontology(); o['candidates']=o['candidates'][:3]
        c=torch.tensor([[0.,3.]])
        d=torch.tensor([[[0.,0.],[0.,4.]]])
        _,marginal=conditional_scores(c,d,o)
        expected=(c.double().softmax(1)[:,:,None]*d.double().softmax(2)).sum(1)
        self.assertTrue(torch.allclose(marginal.exp(),expected,atol=1e-12,rtol=0))
        self.assertAlmostEqual(float(marginal.exp().sum()),1.,places=12)

    def test_zero_conditioning_recovers_independent_baseline_ranking(self):
        c=torch.randn(3,2); d=self.head(self.features).detach()
        pair,marginal=conditional_scores(c,d[:,None,:].expand(-1,2,-1),ontology())
        independent=c.double()[:,:,None]+d.double()[:,None,:]
        self.assertTrue(torch.equal(pair.argsort(1,descending=True),independent.flatten(1).argsort(1,descending=True)))
        self.assertTrue(torch.allclose(marginal,F.log_softmax(d.double(),1),atol=1e-12,rtol=0))

    def test_predict_does_not_use_ground_truth_crop(self):
        from train import predict
        class Toy(nn.Module):
            def forward(self,images): return torch.tensor([[[0.,1.],[2.,0.]]]).expand(len(images),-1,-1)
        p={'ontology':ontology()}; cfg={'device':'cpu','amp':False}
        c=torch.tensor([[1.,0.]])
        with patch('train.loader',return_value=[(torch.zeros(1,3,2,2),torch.tensor([[0,0,0]]))]):
            a=predict(Toy(),[{}],p,cfg,None,c)
        with patch('train.loader',return_value=[(torch.zeros(1,3,2,2),torch.tensor([[1,1,3]]))]):
            b=predict(Toy(),[{}],p,cfg,None,c)
        self.assertTrue(torch.equal(a['pair'],b['pair']))
        self.assertTrue(torch.equal(a['disease'],b['disease']))


class ProtocolTests(unittest.TestCase):
    def test_validation_selects_disease_marginal_not_pair_or_true_crop(self):
        from train import validation
        o=ontology(); rows=[{'crop_id':1,'disease_id':1,'composition_id':3}]
        out={'crop':torch.tensor([[5.,0.]]),'disease':torch.tensor([[-5.,-.01]],dtype=torch.float64),
             'pair':torch.tensor([[5.,0.,0.,0.]])}
        with patch('train.predict',return_value=out): v=validation(None,{'ontology':o,'roles':{'seen_validation':rows}},{},None,None)
        self.assertEqual(v['selection_score'],1.)
        self.assertEqual(v['metrics']['composition']['top1'],0.)
        self.assertAlmostEqual(v['mean_ce'],.01)

    def test_epoch_zero_and_earlier_exact_ties_win_selection(self):
        records=[(.85,-.8),(.84,-.7),(.85,-.8)]
        self.assertEqual(max(range(len(records)),key=lambda i:records[i]),0)
        records.append((.85,-.79))
        self.assertEqual(max(range(len(records)),key=lambda i:records[i]),3)

    def test_exact_full_image_exposure(self):
        rows=[{'crop_id':i%3,'disease_id':i%5} for i in range(51)]
        cfg=load_config(HERE/'config.json'); a=EpochBatches(rows,cfg); b=EpochBatches(rows,cfg)
        self.assertEqual(a.indices(),b.indices()); self.assertEqual(sorted(a.indices()),list(range(51)))
        a.epoch=1; self.assertNotEqual(a.indices(),b.indices())

    def test_completed_checkpoint_recovers_reports(self):
        from train import recover_reports
        checkpoint={'complete':True,'epoch':1,'best_epoch':0,'best_key':[.85,-.8],
                    'history':[{'epoch':1,'seconds':2.}],'initial_hash':'test'}
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            recover_reports(Path(temp),checkpoint)
            self.assertEqual(json.loads((Path(temp)/'history.json').read_text()),checkpoint['history'])
            self.assertEqual(json.loads((Path(temp)/'summary.json').read_text())['selected_epoch'],0)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
