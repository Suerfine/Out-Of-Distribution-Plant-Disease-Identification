"""Tests for the mechanisms needed for genuine unseen composition scoring."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from model import Composer,training_loss
from protocol import HERE,EpochBatches,metrics,classification_metrics,load_config


class CompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model=Composer(2,2,8,16)
        self.c=torch.tensor([0,0,1,1]); self.d=torch.tensor([0,1,0,1])

    def test_both_primitives_control_pair_vectors(self):
        original=self.model(self.c,self.d).detach()
        self.assertEqual(tuple(original.shape),(4,8))
        self.assertTrue(torch.allclose(original.norm(dim=1),torch.ones(4),atol=1e-6))
        with torch.no_grad(): self.model.crop.weight[0,0]+=.7
        changed=self.model(self.c,self.d).detach()
        self.assertFalse(torch.allclose(original[:2],changed[:2]))
        self.assertTrue(torch.equal(original[2:],changed[2:]))
        with torch.no_grad(): self.model.disease.weight[1,2]+=.6
        changed_d=self.model(self.c,self.d).detach()
        self.assertTrue(torch.equal(changed[[0,2]],changed_d[[0,2]]))
        self.assertFalse(torch.allclose(changed[[1,3]],changed_d[[1,3]]))

    def test_unseen_vector_changes_via_shared_factors(self):
        image=F.normalize(torch.randn(3,8),dim=1)
        before=self.model(self.c,self.d).detach().clone()
        optimizer=torch.optim.SGD(self.model.parameters(),lr=.1)
        vectors=self.model(self.c,self.d)
        loss=F.cross_entropy((image@vectors.T)[:,:3],torch.tensor([0,1,2]))
        loss.backward()
        for x in (self.model.crop.weight,self.model.disease.weight,self.model.combine[-1].weight):
            self.assertIsNotNone(x.grad); self.assertGreater(float(x.grad.norm()),0)
        optimizer.step(); after=self.model(self.c,self.d).detach()
        self.assertFalse(torch.allclose(before[3],after[3]))
        self.assertFalse(any(tuple(v.shape)==(4,8) for v in self.model.parameters()))

    def test_loss_does_not_penalize_unseen_pair_logits(self):
        pair=torch.randn(2,4,requires_grad=True)
        output={'pair':pair,'crop':torch.randn(2,2,requires_grad=True),'disease':torch.randn(2,2,requires_grad=True)}
        labels=torch.tensor([[0,0,0],[1,0,2]])
        cfg={'label_smoothing':.1,'crop_loss_weight':.5,'disease_loss_weight':.5}
        loss,parts=training_loss(output,labels,torch.tensor([0,1,2]),torch.arange(4),torch.tensor([0,1,2,-1]),cfg)
        expected=F.cross_entropy(pair[:,:3],torch.tensor([0,2]),label_smoothing=.1,reduction='sum')
        self.assertTrue(torch.allclose(parts['pair_ce'],expected))
        loss.backward(); self.assertTrue(torch.equal(pair.grad[:,3],torch.zeros(2)))
        self.assertGreater(float(pair.grad[:,:3].norm()),0)

    def test_heldout_training_target_is_rejected(self):
        cfg={'label_smoothing':.1,'crop_loss_weight':.5,'disease_loss_weight':.5}
        o={'pair':torch.randn(1,4),'crop':torch.randn(1,2),'disease':torch.randn(1,2)}
        with self.assertRaisesRegex(ValueError,'Held-out'):
            training_loss(o,torch.tensor([[1,1,3]]),torch.tensor([0,1,2]),torch.arange(4),torch.tensor([0,1,2,-1]),cfg)


class ProtocolTests(unittest.TestCase):
    def test_checkpoint_selection_uses_primary_pair_output(self):
        from unittest.mock import patch
        from train import validation
        ontology={'crop':[0,1],'disease':[0,1],
            'candidates':[{'composition_id':i,'crop_id':i//2,'disease_id':i%2} for i in range(4)]}
        p={'ontology':ontology,'roles':{'seen_validation':[{'crop_id':1,'disease_id':1,'composition_id':3}]}}
        output={'pair':torch.tensor([[0.,0.,0.,5.]]),'crop':torch.tensor([[5.,0.]]),'disease':torch.tensor([[5.,0.]])}
        with patch('train.predict',return_value=output): result=validation(None,p,{},None)
        self.assertEqual(result['selection_score'],1.)
        self.assertEqual(result['metrics']['disease']['top1'],0.)

    def test_inference_uses_pair_scores(self):
        o={'crop':[0,1],'disease':[0,1],'candidates':[{'composition_id':i,'crop_id':i//2,'disease_id':i%2} for i in range(4)]}
        row=[{'crop_id':1,'disease_id':1,'composition_id':3}]
        factor=torch.tensor([[10.,0.]])
        result=metrics(factor,factor,row,o,pair=torch.tensor([[0.,0.,0.,5.]]))
        self.assertEqual(result['composition']['top1'],1.)
        self.assertEqual(result['crop']['top1'],0.)

    def test_full_exposure_and_reproducibility(self):
        rows=[{'crop_id':i%3,'disease_id':i%5} for i in range(51)]
        cfg=load_config(HERE/'config.json')
        a=EpochBatches(rows,cfg); b=EpochBatches(rows,cfg)
        self.assertEqual(a.indices(),b.indices())
        self.assertEqual(sorted(a.indices()),list(range(51)))
        a.epoch=1; self.assertNotEqual(a.indices(),b.indices())

    def test_stable_rank_metrics_numpy(self):
        scores=torch.tensor([[1.,1.,0.],[2.,0.,2.]])
        y=torch.tensor([0,2]); m=classification_metrics(scores,y)
        ranking=np.argsort(-scores.numpy(),axis=1,kind='stable')
        self.assertEqual(m['top1'],float((ranking[:,0]==y.numpy()).mean()))
        self.assertEqual(m['top5'],1.)

    def test_completed_checkpoint_recovers_reports(self):
        from train import train_model
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            run=Path(temp); directory=run/'compositional'; directory.mkdir()
            torch.save({'complete':True,'fingerprint':'fixture','epoch':2,'history':[{'seconds':.5},{'seconds':.7}],
                        'best_epoch':1,'best_key':[.8,-.5],'initial_hash':'fixture'},directory/'last.pt')
            train_model({'fingerprint':'fixture'},{},run)
            value=json.loads((directory/'summary.json').read_text())
            self.assertEqual(value['selected_epoch'],1); self.assertEqual(value['epochs'],2)


if __name__=='__main__': unittest.main()
